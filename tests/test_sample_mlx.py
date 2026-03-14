"""Tests for tgirl.sample_mlx — MLX-native sampling functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

from tgirl.types import ModelIntervention


class TestApplyPenaltiesMlx:
    """apply_penalties_mlx modifies logits based on penalties."""

    def test_repetition_penalty_modifies_correct_positions(self) -> None:
        from tgirl.sample_mlx import apply_penalties_mlx

        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        intervention = ModelIntervention(repetition_penalty=2.0)
        token_history = [1, 3]
        result = apply_penalties_mlx(logits, intervention, token_history)
        result_np = np.array(result)
        # Positive logits divided by penalty
        assert result_np[1] == pytest.approx(1.0, abs=1e-5)  # 2.0 / 2.0
        assert result_np[3] == pytest.approx(2.0, abs=1e-5)  # 4.0 / 2.0
        # Unpenalized positions unchanged
        assert result_np[0] == pytest.approx(1.0, abs=1e-5)
        assert result_np[2] == pytest.approx(3.0, abs=1e-5)
        assert result_np[4] == pytest.approx(5.0, abs=1e-5)

    def test_no_penalties_noop(self) -> None:
        from tgirl.sample_mlx import apply_penalties_mlx

        logits = mx.array([1.0, 2.0, 3.0])
        intervention = ModelIntervention()
        result = apply_penalties_mlx(logits, intervention, [])
        np.testing.assert_allclose(np.array(result), np.array(logits), atol=1e-6)

    def test_presence_penalty(self) -> None:
        from tgirl.sample_mlx import apply_penalties_mlx

        logits = mx.array([5.0, 5.0, 5.0])
        intervention = ModelIntervention(presence_penalty=1.0)
        result = apply_penalties_mlx(logits, intervention, [0, 2])
        result_np = np.array(result)
        assert result_np[0] == pytest.approx(4.0, abs=1e-5)
        assert result_np[1] == pytest.approx(5.0, abs=1e-5)
        assert result_np[2] == pytest.approx(4.0, abs=1e-5)

    def test_frequency_penalty(self) -> None:
        from tgirl.sample_mlx import apply_penalties_mlx

        logits = mx.array([5.0, 5.0, 5.0])
        intervention = ModelIntervention(frequency_penalty=0.5)
        result = apply_penalties_mlx(logits, intervention, [0, 0, 2])
        result_np = np.array(result)
        assert result_np[0] == pytest.approx(4.0, abs=1e-5)  # 5 - 0.5*2
        assert result_np[1] == pytest.approx(5.0, abs=1e-5)
        assert result_np[2] == pytest.approx(4.5, abs=1e-5)  # 5 - 0.5*1

    def test_logit_bias(self) -> None:
        from tgirl.sample_mlx import apply_penalties_mlx

        logits = mx.array([1.0, 2.0, 3.0])
        intervention = ModelIntervention(logit_bias={0: 10.0, 2: -5.0})
        result = apply_penalties_mlx(logits, intervention, [])
        result_np = np.array(result)
        assert result_np[0] == pytest.approx(11.0, abs=1e-5)
        assert result_np[1] == pytest.approx(2.0, abs=1e-5)
        assert result_np[2] == pytest.approx(-2.0, abs=1e-5)

    def test_negative_logit_repetition_penalty(self) -> None:
        from tgirl.sample_mlx import apply_penalties_mlx

        logits = mx.array([-2.0, 3.0])
        intervention = ModelIntervention(repetition_penalty=2.0)
        result = apply_penalties_mlx(logits, intervention, [0, 1])
        result_np = np.array(result)
        # Negative logits multiplied by penalty
        assert result_np[0] == pytest.approx(-4.0, abs=1e-5)
        # Positive logits divided by penalty
        assert result_np[1] == pytest.approx(1.5, abs=1e-5)


class TestApplyShapingMlx:
    """apply_shaping_mlx applies temperature, top-k, top-p."""

    def test_temperature_division(self) -> None:
        from tgirl.sample_mlx import apply_shaping_mlx

        logits = mx.array([2.0, 4.0, 6.0])
        intervention = ModelIntervention(temperature=2.0)
        result = apply_shaping_mlx(logits, intervention)
        result_np = np.array(result)
        np.testing.assert_allclose(result_np, [1.0, 2.0, 3.0], atol=1e-5)

    def test_top_k_masking(self) -> None:
        from tgirl.sample_mlx import apply_shaping_mlx

        logits = mx.array([1.0, 5.0, 3.0, 2.0, 4.0])
        intervention = ModelIntervention(top_k=2)
        result = apply_shaping_mlx(logits, intervention)
        result_np = np.array(result)
        # Top 2 are indices 1 (5.0) and 4 (4.0)
        assert np.isfinite(result_np[1])
        assert np.isfinite(result_np[4])
        # Rest should be -inf
        assert result_np[0] == float("-inf")
        assert result_np[2] == float("-inf")
        assert result_np[3] == float("-inf")

    def test_top_p_nucleus(self) -> None:
        from tgirl.sample_mlx import apply_shaping_mlx

        # Logits designed so softmax gives clear top-p boundary
        logits = mx.array([10.0, 0.0, 0.0, -10.0, -10.0])
        intervention = ModelIntervention(top_p=0.5)
        result = apply_shaping_mlx(logits, intervention)
        result_np = np.array(result)
        # Token 0 has nearly all the probability mass
        assert np.isfinite(result_np[0])

    def test_greedy_temperature_zero(self) -> None:
        from tgirl.sample_mlx import apply_shaping_mlx

        logits = mx.array([1.0, 5.0, 3.0, 2.0, 4.0])
        intervention = ModelIntervention(temperature=0.0)
        result = apply_shaping_mlx(logits, intervention)
        result_np = np.array(result)
        # Only the max (index 1) should be finite
        assert np.isfinite(result_np[1])
        finite_count = np.isfinite(result_np).sum()
        assert finite_count == 1

    def test_no_intervention_noop(self) -> None:
        from tgirl.sample_mlx import apply_shaping_mlx

        logits = mx.array([1.0, 2.0, 3.0])
        intervention = ModelIntervention()
        result = apply_shaping_mlx(logits, intervention)
        np.testing.assert_allclose(np.array(result), np.array(logits), atol=1e-6)


class TestGrammarTemperatureHookMlx:
    """GrammarTemperatureHookMlx computes temperature from valid_mask."""

    def test_single_valid_returns_zero_temp(self) -> None:
        from tgirl.sample_mlx import GrammarTemperatureHookMlx

        hook = GrammarTemperatureHookMlx()
        valid_mask = mx.array([False, True, False, False, False])
        logits = mx.zeros(5)
        result = hook.pre_forward(0, valid_mask, [], logits)
        assert result.temperature == 0.0

    def test_all_valid_returns_base_temp(self) -> None:
        from tgirl.sample_mlx import GrammarTemperatureHookMlx

        hook = GrammarTemperatureHookMlx(base_temperature=0.3, scaling_exponent=0.5)
        valid_mask = mx.ones(100, dtype=mx.bool_)
        logits = mx.zeros(100)
        result = hook.pre_forward(0, valid_mask, [], logits)
        # freedom = 1.0, temp = 0.3 * 1.0^0.5 = 0.3
        assert result.temperature == pytest.approx(0.3, abs=1e-5)

    def test_partial_valid_scales_temp(self) -> None:
        from tgirl.sample_mlx import GrammarTemperatureHookMlx

        hook = GrammarTemperatureHookMlx(base_temperature=1.0, scaling_exponent=0.5)
        # 25 valid out of 100 → freedom = 0.25
        valid_mask = mx.concatenate([
            mx.ones(25, dtype=mx.bool_),
            mx.zeros(75, dtype=mx.bool_),
        ])
        logits = mx.zeros(100)
        result = hook.pre_forward(0, valid_mask, [], logits)
        # temp = 1.0 * 0.25^0.5 = 0.5
        assert result.temperature == pytest.approx(0.5, abs=1e-5)

    def test_protocol_compliance(self) -> None:
        from tgirl.sample_mlx import GrammarTemperatureHookMlx, InferenceHookMlx

        hook = GrammarTemperatureHookMlx()
        assert isinstance(hook, InferenceHookMlx)


class TestRunConstrainedGenerationMlx:
    """run_constrained_generation_mlx produces tokens from mock model."""

    def _make_mock_grammar_mlx(self, accept_after: int = 3, vocab_size: int = 10):
        """Create a mock MLX-native grammar that accepts after N advances."""
        gs = MagicMock()
        gs.get_valid_mask_mx = MagicMock(
            return_value=mx.ones(vocab_size, dtype=mx.bool_)
        )
        call_count = [0]

        def advance(token_id):
            call_count[0] += 1

        def is_accepting():
            return call_count[0] >= accept_after

        gs.advance.side_effect = advance
        gs.is_accepting.side_effect = is_accepting
        return gs

    def _make_mock_forward(self, vocab_size: int = 10):
        """Create forward_fn returning mx.array logits."""
        def forward(tokens):
            logits = mx.zeros((vocab_size,))
            # Favor token 5
            logits = logits.at[5].add(mx.array(10.0))
            return logits
        return forward

    def test_basic_produces_tokens(self) -> None:
        from tgirl.sample_mlx import run_constrained_generation_mlx
        from tgirl.transport import TransportConfig

        gs = self._make_mock_grammar_mlx(accept_after=3)
        forward = self._make_mock_forward()
        decode = lambda tokens: "".join(str(t) for t in tokens)
        embeddings = mx.array(
            np.random.randn(10, 8).astype(np.float32)
        )

        result = run_constrained_generation_mlx(
            grammar_state=gs,
            forward_fn=forward,
            tokenizer_decode=decode,
            embeddings=embeddings,
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        assert len(result.tokens) == 3
        assert isinstance(result.tokens, list)
        assert all(isinstance(t, int) for t in result.tokens)

    def test_stops_when_grammar_accepts(self) -> None:
        from tgirl.sample_mlx import run_constrained_generation_mlx
        from tgirl.transport import TransportConfig

        gs = self._make_mock_grammar_mlx(accept_after=2)
        forward = self._make_mock_forward()
        decode = lambda tokens: "test"
        embeddings = mx.array(
            np.random.randn(10, 8).astype(np.float32)
        )

        result = run_constrained_generation_mlx(
            grammar_state=gs,
            forward_fn=forward,
            tokenizer_decode=decode,
            embeddings=embeddings,
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=100,
        )

        assert len(result.tokens) == 2

    def test_stops_at_max_tokens(self) -> None:
        from tgirl.sample_mlx import run_constrained_generation_mlx
        from tgirl.transport import TransportConfig

        gs = self._make_mock_grammar_mlx(accept_after=999)
        forward = self._make_mock_forward()
        decode = lambda tokens: "test"
        embeddings = mx.array(
            np.random.randn(10, 8).astype(np.float32)
        )

        result = run_constrained_generation_mlx(
            grammar_state=gs,
            forward_fn=forward,
            tokenizer_decode=decode,
            embeddings=embeddings,
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=5,
        )

        assert len(result.tokens) == 5

    def test_returns_constrained_generation_result(self) -> None:
        from tgirl.sample import ConstrainedGenerationResult
        from tgirl.sample_mlx import run_constrained_generation_mlx
        from tgirl.transport import TransportConfig

        gs = self._make_mock_grammar_mlx(accept_after=2)
        forward = self._make_mock_forward()
        decode = lambda tokens: "55"
        embeddings = mx.array(
            np.random.randn(10, 8).astype(np.float32)
        )

        result = run_constrained_generation_mlx(
            grammar_state=gs,
            forward_fn=forward,
            tokenizer_decode=decode,
            embeddings=embeddings,
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        assert isinstance(result, ConstrainedGenerationResult)
        assert len(result.grammar_valid_counts) == 2
        assert len(result.temperatures_applied) == 2
        assert len(result.wasserstein_distances) == 2

    def test_with_mlx_hook(self) -> None:
        from tgirl.sample_mlx import (
            GrammarTemperatureHookMlx,
            run_constrained_generation_mlx,
        )
        from tgirl.transport import TransportConfig

        gs = self._make_mock_grammar_mlx(accept_after=2)
        forward = self._make_mock_forward()
        decode = lambda tokens: "55"
        embeddings = mx.array(
            np.random.randn(10, 8).astype(np.float32)
        )
        hook = GrammarTemperatureHookMlx()

        result = run_constrained_generation_mlx(
            grammar_state=gs,
            forward_fn=forward,
            tokenizer_decode=decode,
            embeddings=embeddings,
            hooks=[hook],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        assert len(result.tokens) == 2
        # Hook should have set temperatures
        assert all(t >= 0 for t in result.temperatures_applied)

    def test_torch_grammar_fallback(self) -> None:
        """torch-based grammar state still works via fallback path."""
        import torch

        from tgirl.sample_mlx import run_constrained_generation_mlx
        from tgirl.transport import TransportConfig

        # Use spec to prevent MagicMock from having get_valid_mask_mx
        gs = MagicMock(spec=["get_valid_mask", "is_accepting", "advance"])
        gs.get_valid_mask = MagicMock(
            return_value=torch.ones(10, dtype=torch.bool)
        )
        call_count = [0]

        def advance(token_id):
            call_count[0] += 1

        def is_accepting():
            return call_count[0] >= 2

        gs.advance.side_effect = advance
        gs.is_accepting.side_effect = is_accepting

        forward = self._make_mock_forward()
        decode = lambda tokens: "test"
        embeddings = mx.array(
            np.random.randn(10, 8).astype(np.float32)
        )

        result = run_constrained_generation_mlx(
            grammar_state=gs,
            forward_fn=forward,
            tokenizer_decode=decode,
            embeddings=embeddings,
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        assert len(result.tokens) == 2


class TestZeroTorchInSampleMlx:
    """sample_mlx.py must not import torch at module level."""

    def test_no_torch_import(self) -> None:
        import inspect

        import tgirl.sample_mlx as mod

        source = inspect.getsource(mod)
        lines = source.split("\n")
        torch_imports = [
            line.strip()
            for line in lines
            if ("import torch" in line)
            and not line.strip().startswith("#")
        ]
        assert torch_imports == [], (
            f"sample_mlx.py imports torch: {torch_imports}"
        )

    def test_no_numpy_import(self) -> None:
        import inspect

        import tgirl.sample_mlx as mod

        source = inspect.getsource(mod)
        lines = source.split("\n")
        numpy_imports = [
            line.strip()
            for line in lines
            if ("import numpy" in line)
            and not line.strip().startswith("#")
        ]
        assert numpy_imports == [], (
            f"sample_mlx.py imports numpy: {numpy_imports}"
        )
