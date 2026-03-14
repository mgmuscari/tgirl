"""Tests for tgirl.cache — KV cache wrapper factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch


# --- Mock MLX model for testing ---

@dataclass
class MockMLXCache:
    """Simulates an MLX KV cache object."""

    reset_count: int = 0


@dataclass
class MockMLXModel:
    """Mock MLX model that records forwarded input_ids.

    Tracks what tokens were forwarded and how many times make_cache was called.
    Returns deterministic logits based on the last token ID.
    """

    forwarded_inputs: list[list[int]] = field(default_factory=list)
    cache_created_count: int = 0
    vocab_size: int = 10

    def make_cache(self) -> list[MockMLXCache]:
        self.cache_created_count += 1
        return [MockMLXCache()]

    def __call__(
        self, input_ids: Any, cache: Any = None
    ) -> Any:
        """Simulate MLX model forward pass.

        Records what tokens were forwarded. Returns logits tensor
        where the value at each position equals the last input token ID.
        """
        # input_ids is expected to be like mx.array([token_ids])
        # For testing, we accept a list of lists or similar
        if hasattr(input_ids, "tolist"):
            tokens = input_ids.tolist()
        else:
            tokens = list(input_ids)

        # Flatten if nested
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]

        self.forwarded_inputs.append(tokens)

        # Return logits: a 3D tensor [batch=1, seq_len, vocab_size]
        seq_len = len(tokens)
        logits = torch.zeros(1, seq_len, self.vocab_size)
        # Make logits depend on last token for determinism
        if tokens:
            logits[0, -1, tokens[-1] % self.vocab_size] = 10.0
        return logits


class TestCacheStats:
    """CacheStats dataclass tracks hit/miss/reset/tokens_saved counts."""

    def test_default_values(self) -> None:
        from tgirl.cache import CacheStats

        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.resets == 0
        assert stats.tokens_saved == 0

    def test_is_dataclass(self) -> None:
        from tgirl.cache import CacheStats

        import dataclasses
        assert dataclasses.is_dataclass(CacheStats)

    def test_mutable_counters(self) -> None:
        from tgirl.cache import CacheStats

        stats = CacheStats()
        stats.hits += 1
        stats.misses += 2
        stats.resets += 3
        stats.tokens_saved += 100
        assert stats.hits == 1
        assert stats.misses == 2
        assert stats.resets == 3
        assert stats.tokens_saved == 100


class TestMakeMLXForwardFn:
    """make_mlx_forward_fn creates a cached forward function for MLX models."""

    def test_first_call_cache_miss_all_tokens_forwarded(self) -> None:
        """First call is always a cache miss — all tokens forwarded."""
        from tgirl.cache import CacheStats, make_mlx_forward_fn

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn(model, stats=stats)

        tokens = [1, 2, 3]
        logits = forward(tokens)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (model.vocab_size,)
        assert stats.misses == 1
        assert stats.hits == 0
        assert len(model.forwarded_inputs) == 1
        assert model.forwarded_inputs[0] == [1, 2, 3]

    def test_append_one_token_cache_hit(self) -> None:
        """Appending one token to previous input = cache hit, only new token forwarded."""
        from tgirl.cache import CacheStats, make_mlx_forward_fn

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn(model, stats=stats)

        forward([1, 2, 3])  # miss
        model.forwarded_inputs.clear()

        forward([1, 2, 3, 4])  # hit — only token 4 forwarded

        assert stats.hits == 1
        assert len(model.forwarded_inputs) == 1
        assert model.forwarded_inputs[0] == [4]
        assert stats.tokens_saved == 3

    def test_divergent_prefix_cache_miss_full_reset(self) -> None:
        """Different prefix = cache miss, full reset."""
        from tgirl.cache import CacheStats, make_mlx_forward_fn

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn(model, stats=stats)

        forward([1, 2, 3])  # miss
        model.forwarded_inputs.clear()
        initial_cache_count = model.cache_created_count

        forward([1, 9, 3])  # divergent — miss + reset

        assert stats.misses == 2
        assert stats.resets == 1
        assert model.cache_created_count == initial_cache_count + 1
        assert model.forwarded_inputs[0] == [1, 9, 3]

    def test_same_tokens_returns_cached_logits(self) -> None:
        """Same tokens as previous call = return cached logits without forward."""
        from tgirl.cache import CacheStats, make_mlx_forward_fn

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn(model, stats=stats)

        logits1 = forward([1, 2, 3])
        n_forwards_after_first = len(model.forwarded_inputs)

        logits2 = forward([1, 2, 3])  # same tokens

        # No additional forward call
        assert len(model.forwarded_inputs) == n_forwards_after_first
        assert torch.equal(logits1, logits2)
        assert stats.hits == 1

    def test_stats_counters_correct_sequence(self) -> None:
        """Stats counters accumulate correctly across multiple calls."""
        from tgirl.cache import CacheStats, make_mlx_forward_fn

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn(model, stats=stats)

        forward([1, 2, 3])        # miss
        forward([1, 2, 3, 4])     # hit, 3 tokens saved
        forward([1, 2, 3, 4, 5])  # hit, 4 tokens saved
        forward([9, 8, 7])        # miss (divergent), reset
        forward([9, 8, 7])        # hit (same tokens), 0 new forwarded

        assert stats.misses == 2
        assert stats.hits == 3
        assert stats.resets == 1
        assert stats.tokens_saved == 3 + 4  # from the two append-hits

    def test_stats_optional(self) -> None:
        """Forward function works without stats parameter."""
        from tgirl.cache import make_mlx_forward_fn

        model = MockMLXModel()
        forward = make_mlx_forward_fn(model)

        logits = forward([1, 2, 3])
        assert isinstance(logits, torch.Tensor)

    def test_returns_last_position_logits(self) -> None:
        """Returns logits for the last token position only."""
        from tgirl.cache import make_mlx_forward_fn

        model = MockMLXModel(vocab_size=10)
        forward = make_mlx_forward_fn(model)

        logits = forward([1, 2, 3])
        assert logits.shape == (10,)


# --- Mock HuggingFace model for testing ---

@dataclass
class MockHFPastKeyValues:
    """Simulates HuggingFace past_key_values (immutable pattern)."""

    token_count: int = 0


@dataclass
class MockHFModelOutput:
    """Simulates HuggingFace model output with logits and past_key_values."""

    logits: torch.Tensor
    past_key_values: MockHFPastKeyValues


@dataclass
class MockHFModel:
    """Mock HuggingFace model that records forwarded input_ids.

    Returns (logits, new_past_key_values) following the HF convention
    where model() returns an object with .logits and .past_key_values.
    """

    forwarded_inputs: list[list[int]] = field(default_factory=list)
    vocab_size: int = 10

    def __call__(
        self,
        input_ids: Any,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> MockHFModelOutput:
        """Simulate HF model forward pass."""
        # input_ids is a torch tensor of shape [batch, seq_len]
        if isinstance(input_ids, torch.Tensor):
            tokens = input_ids[0].tolist()
        elif hasattr(input_ids, "tolist"):
            tokens = input_ids.tolist()
        else:
            tokens = list(input_ids)

        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]

        self.forwarded_inputs.append(tokens)

        seq_len = len(tokens)
        logits = torch.zeros(1, seq_len, self.vocab_size)
        if tokens:
            logits[0, -1, tokens[-1] % self.vocab_size] = 10.0

        # Return new past_key_values tracking total tokens seen
        prev_count = past_key_values.token_count if past_key_values else 0
        new_pkv = MockHFPastKeyValues(token_count=prev_count + seq_len)

        return MockHFModelOutput(logits=logits, past_key_values=new_pkv)


class TestMakeHFForwardFn:
    """make_hf_forward_fn creates a cached forward function for HF models."""

    def test_first_call_cache_miss_all_tokens_forwarded(self) -> None:
        """First call is always a cache miss — all tokens forwarded."""
        from tgirl.cache import CacheStats, make_hf_forward_fn

        model = MockHFModel()
        stats = CacheStats()
        forward = make_hf_forward_fn(model, device="cpu", stats=stats)

        tokens = [1, 2, 3]
        logits = forward(tokens)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (model.vocab_size,)
        assert stats.misses == 1
        assert stats.hits == 0
        assert len(model.forwarded_inputs) == 1
        assert model.forwarded_inputs[0] == [1, 2, 3]

    def test_append_one_token_cache_hit(self) -> None:
        """Appending one token = cache hit, only new token forwarded."""
        from tgirl.cache import CacheStats, make_hf_forward_fn

        model = MockHFModel()
        stats = CacheStats()
        forward = make_hf_forward_fn(model, device="cpu", stats=stats)

        forward([1, 2, 3])  # miss
        model.forwarded_inputs.clear()

        forward([1, 2, 3, 4])  # hit

        assert stats.hits == 1
        assert len(model.forwarded_inputs) == 1
        assert model.forwarded_inputs[0] == [4]
        assert stats.tokens_saved == 3

    def test_divergent_prefix_cache_miss_full_reset(self) -> None:
        """Different prefix = cache miss, full reset."""
        from tgirl.cache import CacheStats, make_hf_forward_fn

        model = MockHFModel()
        stats = CacheStats()
        forward = make_hf_forward_fn(model, device="cpu", stats=stats)

        forward([1, 2, 3])  # miss
        model.forwarded_inputs.clear()

        forward([1, 9, 3])  # divergent — miss + reset

        assert stats.misses == 2
        assert stats.resets == 1
        assert model.forwarded_inputs[0] == [1, 9, 3]

    def test_same_tokens_returns_cached_logits(self) -> None:
        """Same tokens = cached logits without forward."""
        from tgirl.cache import CacheStats, make_hf_forward_fn

        model = MockHFModel()
        stats = CacheStats()
        forward = make_hf_forward_fn(model, device="cpu", stats=stats)

        logits1 = forward([1, 2, 3])
        n_forwards = len(model.forwarded_inputs)

        logits2 = forward([1, 2, 3])

        assert len(model.forwarded_inputs) == n_forwards
        assert torch.equal(logits1, logits2)
        assert stats.hits == 1

    def test_stats_counters_correct_sequence(self) -> None:
        """Stats counters accumulate correctly."""
        from tgirl.cache import CacheStats, make_hf_forward_fn

        model = MockHFModel()
        stats = CacheStats()
        forward = make_hf_forward_fn(model, device="cpu", stats=stats)

        forward([1, 2, 3])        # miss
        forward([1, 2, 3, 4])     # hit, 3 saved
        forward([1, 2, 3, 4, 5])  # hit, 4 saved
        forward([9, 8, 7])        # miss, reset
        forward([9, 8, 7])        # hit (same)

        assert stats.misses == 2
        assert stats.hits == 3
        assert stats.resets == 1
        assert stats.tokens_saved == 7

    def test_stats_optional(self) -> None:
        """Forward function works without stats."""
        from tgirl.cache import make_hf_forward_fn

        model = MockHFModel()
        forward = make_hf_forward_fn(model)

        logits = forward([1, 2, 3])
        assert isinstance(logits, torch.Tensor)

    def test_past_key_values_passed_on_continuation(self) -> None:
        """On cache hit, past_key_values from previous call is reused."""
        from tgirl.cache import make_hf_forward_fn

        model = MockHFModel()
        forward = make_hf_forward_fn(model, device="cpu")

        forward([1, 2, 3])      # miss: 3 tokens
        forward([1, 2, 3, 4])   # hit: 1 token with past_key_values

        # The second call should have only forwarded [4]
        assert model.forwarded_inputs[-1] == [4]


class TestZeroCoupling:
    """cache.py must not import from other tgirl modules."""

    def test_no_tgirl_imports(self) -> None:
        """Verify cache.py has zero tgirl imports via source inspection."""
        import inspect

        import tgirl.cache as mod

        source = inspect.getsource(mod)
        lines = source.split("\n")
        tgirl_imports = [
            line
            for line in lines
            if ("from tgirl" in line or "import tgirl" in line)
            and not line.strip().startswith("#")
        ]
        assert tgirl_imports == [], (
            f"cache.py has tgirl imports: {tgirl_imports}"
        )
