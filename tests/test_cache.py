"""Tests for tgirl.cache — KV cache wrapper factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np
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
    Returns MLX arrays matching the real MLX model interface:
    input is mx.array([token_ids]), output is mx.array of shape [1, seq_len, vocab].
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

        Accepts mx.array([token_ids]) as input.
        Records the token list for test assertions.
        Returns mx.array of shape [1, seq_len, vocab_size].
        """
        # input_ids is mx.array([token_ids]) — shape [1, seq_len]
        tokens = input_ids.tolist()
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]

        self.forwarded_inputs.append(tokens)

        # Return MLX array: shape [batch=1, seq_len, vocab_size]
        seq_len = len(tokens)
        logits = np.zeros((1, seq_len, self.vocab_size), dtype=np.float32)
        if tokens:
            logits[0, -1, tokens[-1] % self.vocab_size] = 10.0
        return mx.array(logits)


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
    """make_mlx_forward_fn creates a cached forward function returning mx.array."""

    def test_returns_mx_array(self) -> None:
        """Return type is mx.array, not torch.Tensor."""
        from tgirl.cache import CacheStats, make_mlx_forward_fn

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn(model, stats=stats)

        logits = forward([1, 2, 3])

        assert isinstance(logits, mx.array)
        assert logits.shape == (model.vocab_size,)
        assert stats.misses == 1

    def test_first_call_cache_miss_all_tokens_forwarded(self) -> None:
        """First call is always a cache miss — all tokens forwarded."""
        from tgirl.cache import CacheStats, make_mlx_forward_fn

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn(model, stats=stats)

        tokens = [1, 2, 3]
        logits = forward(tokens)

        assert isinstance(logits, mx.array)
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
        assert mx.array_equal(logits1, logits2)
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
        assert isinstance(logits, mx.array)

    def test_returns_last_position_logits(self) -> None:
        """Returns logits for the last token position only."""
        from tgirl.cache import make_mlx_forward_fn

        model = MockMLXModel(vocab_size=10)
        forward = make_mlx_forward_fn(model)

        logits = forward([1, 2, 3])
        assert logits.shape == (10,)


class TestMakeMLXForwardFnTorch:
    """make_mlx_forward_fn_torch returns torch.Tensor (compat wrapper)."""

    def test_returns_torch_tensor(self) -> None:
        from tgirl.cache import CacheStats, make_mlx_forward_fn_torch

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn_torch(model, stats=stats)

        logits = forward([1, 2, 3])

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (model.vocab_size,)
        assert stats.misses == 1

    def test_cache_hit_continuation(self) -> None:
        from tgirl.cache import CacheStats, make_mlx_forward_fn_torch

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn_torch(model, stats=stats)

        forward([1, 2, 3])
        model.forwarded_inputs.clear()
        forward([1, 2, 3, 4])

        assert stats.hits == 1
        assert model.forwarded_inputs[0] == [4]

    def test_same_tokens_cached(self) -> None:
        from tgirl.cache import CacheStats, make_mlx_forward_fn_torch

        model = MockMLXModel()
        stats = CacheStats()
        forward = make_mlx_forward_fn_torch(model, stats=stats)

        logits1 = forward([1, 2, 3])
        logits2 = forward([1, 2, 3])

        assert torch.equal(logits1, logits2)
        assert stats.hits == 1


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


@pytest.fixture(scope="module")
def qwen_model_and_tok():
    """Load Qwen3.5-0.8B once for all tests in this module."""
    import mlx_lm

    model, tok = mlx_lm.load("Qwen/Qwen3.5-0.8B")
    return model, tok


class TestBottleneckHook:
    """_BottleneckHook captures and injects at a specific layer."""

    def test_capture_shape(self, qwen_model_and_tok) -> None:
        from tgirl.cache import _BottleneckHook

        model, tok = qwen_model_and_tok
        layers = model
        for attr in "language_model.model.layers".split("."):
            layers = getattr(layers, attr)

        hook = _BottleneckHook(layers, layer_idx=14)
        hook.install()
        try:
            tokens = mx.array([[tok.encode("Hello world")[-1]]])
            _ = model(tokens)
            captured = hook.get_captured()
            assert captured is not None
            assert captured.ndim == 1
            # d_model for Qwen3.5-0.8B is 1024
            assert captured.shape[0] == 1024
        finally:
            hook.uninstall()

    def test_probe_reading_shape(self, qwen_model_and_tok) -> None:
        from tgirl.cache import _BottleneckHook

        model, tok = qwen_model_and_tok
        layers = model
        for attr in "language_model.model.layers".split("."):
            layers = getattr(layers, attr)

        K = 11
        V_basis = mx.random.normal((1024, K))

        hook = _BottleneckHook(layers, layer_idx=14)
        hook.install()
        try:
            tokens = mx.array([[tok.encode("Hello world")[-1]]])
            _ = model(tokens)
            probe = hook.get_probe(V_basis)
            assert probe is not None
            assert probe.shape == (K,)
        finally:
            hook.uninstall()

    def test_injection_modifies_output(self, qwen_model_and_tok) -> None:
        from tgirl.cache import _BottleneckHook

        model, tok = qwen_model_and_tok
        layers = model
        for attr in "language_model.model.layers".split("."):
            layers = getattr(layers, attr)

        K = 11
        V_basis = mx.random.normal((1024, K))
        delta = mx.ones((K,)) * 5.0  # large delta to ensure visible effect

        # Run without injection
        cache1 = model.make_cache()
        tokens = mx.array([[tok.encode("Hello world")[-1]]])
        logits_baseline = model(tokens, cache=cache1)
        mx.eval(logits_baseline)

        # Run with injection
        hook = _BottleneckHook(layers, layer_idx=14)
        hook.install()
        try:
            hook.set_steering(V_basis, delta)
            cache2 = model.make_cache()
            logits_steered = model(tokens, cache=cache2)
            mx.eval(logits_steered)

            diff = mx.max(mx.abs(logits_baseline - logits_steered))
            mx.eval(diff)
            assert float(diff.item()) > 0.0
        finally:
            hook.uninstall()

    def test_uninstall_restores_original(self, qwen_model_and_tok) -> None:
        from tgirl.cache import _BottleneckHook

        model, tok = qwen_model_and_tok
        layers = model
        for attr in "language_model.model.layers".split("."):
            layers = getattr(layers, attr)

        original_call = type(layers[14]).__call__

        hook = _BottleneckHook(layers, layer_idx=14)
        hook.install()
        assert type(layers[14]).__call__ is not original_call
        hook.uninstall()
        assert type(layers[14]).__call__ is original_call

    def test_raw_correction_modifies_output(self, qwen_model_and_tok) -> None:
        """set_raw_correction injects a (d_model,) vector directly."""
        from tgirl.cache import _BottleneckHook

        model, tok = qwen_model_and_tok
        layers = model
        for attr in "language_model.model.layers".split("."):
            layers = getattr(layers, attr)

        hook = _BottleneckHook(layers, layer_idx=14)
        hook.install()
        try:
            tokens = mx.array([[tok.encode("Hello world")[-1]]])

            # Baseline
            cache1 = model.make_cache()
            hook.clear_steering()
            logits_baseline = model(tokens, cache=cache1)
            mx.eval(logits_baseline)

            # With raw correction
            cache2 = model.make_cache()
            hook.set_raw_correction(mx.ones((1024,)) * 0.5)
            logits_steered = model(tokens, cache=cache2)
            mx.eval(logits_steered)

            diff = mx.max(mx.abs(logits_baseline - logits_steered))
            mx.eval(diff)
            assert float(diff.item()) > 0.0
        finally:
            hook.uninstall()

    def test_probe_feedback_loop(self, qwen_model_and_tok) -> None:
        """v_steer(n+1) = alpha * v_probe(n) produces valid output."""
        from tgirl.cache import _BottleneckHook

        model, tok = qwen_model_and_tok
        layers = model
        for attr in "language_model.model.layers".split("."):
            layers = getattr(layers, attr)

        hook = _BottleneckHook(layers, layer_idx=14)
        hook.install()
        try:
            cache = model.make_cache()
            token_ids = tok.encode("Hello")

            # Token 0: no steering, capture probe
            hook.clear_steering()
            logits = model(mx.array([token_ids]), cache=cache)
            v_probe = hook.get_captured()
            assert v_probe is not None
            assert v_probe.shape == (1024,)

            # Token 1: steer with alpha * v_probe(0)
            alpha = 0.1
            hook.set_raw_correction(alpha * v_probe)
            next_input = mx.array([[int(mx.argmax(logits[0, -1, :]).item())]])
            logits2 = model(next_input, cache=cache)
            v_probe2 = hook.get_captured()
            mx.eval(logits2)

            # Should still produce valid logits and a new probe
            assert logits2.shape[-1] > 0
            assert v_probe2 is not None
            assert v_probe2.shape == (1024,)
        finally:
            hook.uninstall()

    # --- Band steering (β + skewed-Gaussian layer spreading) ---
    #
    # These tests use a tiny fake-layer list (not the real Qwen model)
    # because band *weighting math* is architectural and doesn't need
    # a real transformer. Semantic validity of off-bottleneck codebook
    # injection is an empirical question handled by the smoke test,
    # not by unit tests. .item() on MLX scalars forces materialization
    # without an explicit mx.eval call.

    def _make_fake_layers(self, n: int) -> list[Any]:
        """Build a list of identity layers sharing one class.

        The hook patches `type(target).__call__`, so the patched call
        fires for every instance of this class — exactly what the band
        needs to test multi-layer injection.
        """

        class FakeLayer:
            def __init__(self, idx: int) -> None:
                self.idx = idx

            def __call__(self, x: Any) -> Any:
                return x  # identity — leaves correction visible

        return [FakeLayer(i) for i in range(n)]

    def test_default_is_single_layer_injection_bit_compatible(self) -> None:
        """Without set_band, only the bottleneck layer gets correction —
        matches pre-band behavior bit-for-bit.
        """
        from tgirl.cache import _BottleneckHook

        layers = self._make_fake_layers(7)
        hook = _BottleneckHook(layers, layer_idx=3)
        hook.install()
        try:
            hook.set_raw_correction(mx.ones((4,)))
            x = mx.zeros((1, 1, 4))
            outs = [layers[i](x) for i in range(7)]
            max_per_layer = [float(mx.max(o).item()) for o in outs]
            expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            assert max_per_layer == pytest.approx(expected)
        finally:
            hook.uninstall()

    def test_set_band_spreads_correction_across_layers(self) -> None:
        """With a finite β, correction lands on the bottleneck and
        its neighbors with weights that sum to 1 across the band.
        """
        from tgirl.cache import _BottleneckHook

        layers = self._make_fake_layers(11)
        hook = _BottleneckHook(layers, layer_idx=5)
        hook.install()
        try:
            from tgirl.band import band_weights

            hook.set_band(
                band_weights(
                    n_layers=len(layers),
                    bottleneck_idx=hook._layer_idx,
                    beta=0.7,
                    skew=1.0,
                )
            )
            hook.set_raw_correction(mx.ones((4,)))
            x = mx.zeros((1, 1, 4))
            outs = [layers[i](x) for i in range(11)]
            max_per_layer = [float(mx.max(o).item()) for o in outs]

            # Weights sum to 1 → max values sum to 1 too
            assert sum(max_per_layer) == pytest.approx(1.0, abs=1e-5)
            # Peak at bottleneck
            assert max_per_layer[5] == pytest.approx(max(max_per_layer))
            # Symmetric (skew=1)
            assert max_per_layer[4] == pytest.approx(max_per_layer[6])
            assert max_per_layer[3] == pytest.approx(max_per_layer[7])
        finally:
            hook.uninstall()

    def test_set_band_none_restores_single_layer(self) -> None:
        """set_band(None) after setting a finite β returns the hook
        to single-layer injection.
        """
        from tgirl.cache import _BottleneckHook

        layers = self._make_fake_layers(7)
        hook = _BottleneckHook(layers, layer_idx=3)
        hook.install()
        try:
            from tgirl.band import band_weights

            hook.set_band(
                band_weights(
                    n_layers=len(layers),
                    bottleneck_idx=hook._layer_idx,
                    beta=0.5,
                    skew=1.0,
                )
            )
            hook.set_band(None)
            hook.set_raw_correction(mx.ones((4,)))
            x = mx.zeros((1, 1, 4))
            outs = [layers[i](x) for i in range(7)]
            max_per_layer = [float(mx.max(o).item()) for o in outs]
            expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            assert max_per_layer == pytest.approx(expected)
        finally:
            hook.uninstall()

    def test_band_does_not_broaden_capture_site(self) -> None:
        """Probe capture is bottleneck-only regardless of band width —
        the band is an *injection* distribution; capture stays the
        source of behavioral state.
        """
        from tgirl.cache import _BottleneckHook

        layers = self._make_fake_layers(11)
        hook = _BottleneckHook(layers, layer_idx=5)
        hook.install()
        try:
            from tgirl.band import band_weights

            hook.set_band(
                band_weights(
                    n_layers=len(layers),
                    bottleneck_idx=hook._layer_idx,
                    beta=0.7,
                    skew=1.0,
                )
            )
            # Each layer receives a distinct marker; capture should
            # only record the one that hit the bottleneck layer.
            for i, layer in enumerate(layers):
                marker = mx.ones((1, 1, 4)) * float(i + 1)
                _ = layer(marker)

            captured = hook.get_captured()
            assert captured is not None
            # Last capture at layer 5 means marker=6.0 (i+1 for i=5)
            captured_max = float(mx.max(captured).item())
            assert captured_max == pytest.approx(6.0)
        finally:
            hook.uninstall()


class TestSteerableMLXForwardFn:
    """make_steerable_mlx_forward_fn: forward with optional probe/inject."""

    def test_no_steering_returns_forward_result(self, qwen_model_and_tok) -> None:
        from tgirl.cache import ForwardResult, make_steerable_mlx_forward_fn

        model, tok = qwen_model_and_tok
        fwd = make_steerable_mlx_forward_fn(
            model, bottleneck_layer=14,
            layer_path="language_model.model.layers",
        )
        token_ids = tok.encode("Hello")
        result = fwd(token_ids)
        assert isinstance(result, ForwardResult)
        assert result.probe_alpha is None  # no steering → no probe
        assert result.logits.ndim == 1

    def test_with_steering_returns_probe(self, qwen_model_and_tok) -> None:
        from tgirl.cache import ForwardResult, make_steerable_mlx_forward_fn
        from tgirl.estradiol import SteeringState

        model, tok = qwen_model_and_tok
        K = 11
        V = mx.random.normal((1024, K))
        fwd = make_steerable_mlx_forward_fn(
            model, bottleneck_layer=14,
            layer_path="language_model.model.layers",
        )
        steering = SteeringState(V_basis=V, delta_alpha=mx.zeros((K,)), bottleneck_layer=14)
        token_ids = tok.encode("Hello")
        result = fwd(token_ids, steering=steering)
        assert isinstance(result, ForwardResult)
        assert result.probe_alpha is not None
        assert result.probe_alpha.shape == (K,)

    def test_injection_modifies_logits(self, qwen_model_and_tok) -> None:
        from tgirl.cache import make_steerable_mlx_forward_fn
        from tgirl.estradiol import SteeringState

        model, tok = qwen_model_and_tok
        K = 11
        V = mx.random.normal((1024, K))
        fwd = make_steerable_mlx_forward_fn(
            model, bottleneck_layer=14,
            layer_path="language_model.model.layers",
        )
        token_ids = tok.encode("Hello")

        # Baseline (no injection)
        steering_zero = SteeringState(V_basis=V, delta_alpha=mx.zeros((K,)), bottleneck_layer=14)
        r1 = fwd(token_ids, steering=steering_zero)

        # Reset cache by passing different tokens then back
        fwd(tok.encode("Reset"))

        # With injection
        steering_big = SteeringState(V_basis=V, delta_alpha=mx.ones((K,)) * 10.0, bottleneck_layer=14)
        r2 = fwd(token_ids, steering=steering_big)

        diff = mx.max(mx.abs(r1.logits - r2.logits))
        mx.eval(diff)
        assert float(diff.item()) > 0.0

    def test_cache_continuation_works(self, qwen_model_and_tok) -> None:
        from tgirl.cache import CacheStats, make_steerable_mlx_forward_fn

        model, tok = qwen_model_and_tok
        stats = CacheStats()
        fwd = make_steerable_mlx_forward_fn(
            model, bottleneck_layer=14,
            layer_path="language_model.model.layers",
            stats=stats,
        )
        tokens = tok.encode("Hello world")
        fwd(tokens)
        # Extend by one token — should be a cache hit
        fwd(tokens + [tokens[-1]])
        assert stats.hits >= 1
