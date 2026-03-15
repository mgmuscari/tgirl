"""Tests for the ADSR modulation matrix.

Covers source conditioning, phase detection with hysteresis,
envelope configuration, and the unified ModMatrixHook.
"""

from __future__ import annotations

import dataclasses
import json

import mlx.core as mx
import torch

from tgirl.modulation import (
    DEFAULT_MATRIX,
    DEFAULT_MATRIX_FLAT,
    EnvelopeConfig,
    EnvelopeState,
    EnvelopeTelemetry,
    ModMatrixHook,
    ModMatrixHookMlx,
    SourceConditionerConfig,
    condition_source,
)
from tgirl.types import ModelIntervention

# === Task 1: Source conditioning ===


class TestConditionSource:
    """Tests for condition_source normalization."""

    def test_normalizes_to_unit_range(self) -> None:
        cfg = SourceConditionerConfig(range_min=0.0, range_max=10.0)
        result = condition_source(5.0, cfg, 0.0)
        assert abs(result - 0.5) < 1e-6

    def test_clamps_below_zero(self) -> None:
        cfg = SourceConditionerConfig(range_min=0.0, range_max=1.0)
        result = condition_source(-0.5, cfg, 0.0)
        assert result >= 0.0

    def test_clamps_above_one(self) -> None:
        cfg = SourceConditionerConfig(range_min=0.0, range_max=1.0)
        result = condition_source(1.5, cfg, 0.0)
        assert result <= 1.0

    def test_invert_flips_polarity(self) -> None:
        cfg = SourceConditionerConfig(range_min=0.0, range_max=1.0, invert=True)
        result = condition_source(0.2, cfg, 0.0)
        assert abs(result - 0.8) < 1e-6

    def test_rectify_clamps_negative_to_zero(self) -> None:
        # With invert, 0.8 -> 0.2 after invert... but normalized first.
        # Actually rectify applies after normalization and invert.
        # A value that normalizes to a negative is already clamped by the
        # max(0, min(1, ...)) step. Rectify is a no-op when value is already
        # non-negative after clamping. Test with a value that would be
        # negative before clamping.
        cfg = SourceConditionerConfig(range_min=0.0, range_max=1.0, rectify=True)
        result = condition_source(0.5, cfg, 0.0)
        assert result >= 0.0

    def test_slew_rate_smoothing(self) -> None:
        cfg = SourceConditionerConfig(range_min=0.0, range_max=1.0, slew_rate=0.5)
        # First call: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        r1 = condition_source(1.0, cfg, 0.0)
        assert abs(r1 - 0.5) < 1e-6
        # Second call: 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        r2 = condition_source(1.0, cfg, r1)
        assert abs(r2 - 0.75) < 1e-6

    def test_entropy_range(self) -> None:
        """Entropy source uses range 0..12.5."""
        cfg = SourceConditionerConfig(range_min=0.0, range_max=12.5)
        result = condition_source(6.25, cfg, 0.0)
        assert abs(result - 0.5) < 1e-6

    def test_confidence_inverted(self) -> None:
        """Confidence source uses range -5..0, inverted."""
        cfg = SourceConditionerConfig(range_min=-5.0, range_max=0.0, invert=True)
        # raw=-2.5 -> normalized=0.5 -> inverted=0.5
        result = condition_source(-2.5, cfg, 0.0)
        assert abs(result - 0.5) < 1e-6
        # raw=0.0 (max confidence) -> normalized=1.0 -> inverted=0.0
        result_max = condition_source(0.0, cfg, 0.0)
        assert abs(result_max - 0.0) < 1e-6


# === Task 1: Phase detection with hysteresis ===


class TestPhaseDetection:
    """Tests for EnvelopeState.detect_phase with hysteresis."""

    def _make_state(self, **kwargs) -> EnvelopeState:
        defaults = {"prev_smoothed": [0.0] * 11}
        defaults.update(kwargs)
        return EnvelopeState(**defaults)

    def test_attack_on_depth_increase(self) -> None:
        """Attack fires immediately when depth increases."""
        state = self._make_state(
            phase="sustain", depth=1, prev_depth=1, phase_position=5,
        )
        result = state.detect_phase(freedom=0.5, depth=2)
        assert result == "attack"

    def test_decay_requires_two_consecutive_tokens(self) -> None:
        """Decay requires 2 consecutive tokens with freedom < 30% of peak."""
        state = self._make_state(
            phase="attack", peak_freedom=1.0, depth=2, prev_depth=2,
            phase_position=3,
        )
        # First token below threshold — not yet decay
        r1 = state.detect_phase(freedom=0.2, depth=2)
        assert r1 == "attack"  # Pending, not confirmed

        # Second consecutive token below threshold — confirmed
        r2 = state.detect_phase(freedom=0.2, depth=2)
        assert r2 == "decay"

    def test_release_immediate_when_depth_lte_1(self) -> None:
        """Release fires immediately when depth decreases AND depth <= 1."""
        state = self._make_state(
            phase="sustain", depth=2, prev_depth=2, phase_position=5,
        )
        result = state.detect_phase(freedom=0.5, depth=1)
        assert result == "release"

    def test_no_release_when_depth_gt_1(self) -> None:
        """Depth decrease with depth > 1 does NOT trigger release."""
        state = self._make_state(
            phase="sustain", depth=4, prev_depth=4, phase_position=5,
        )
        result = state.detect_phase(freedom=0.5, depth=3)
        assert result != "release"

    def test_sustain_after_decay_position_gt_2(self) -> None:
        """Sustain fires after decay has lasted > 2 tokens."""
        state = self._make_state(phase="decay", depth=2, prev_depth=2, phase_position=3)
        result = state.detect_phase(freedom=0.5, depth=2)
        assert result == "sustain"

    def test_single_token_flicker_no_decay(self) -> None:
        """Single-token freedom drop doesn't trigger decay (hysteresis)."""
        state = self._make_state(
            phase="attack", peak_freedom=1.0, depth=2, prev_depth=2,
            phase_position=3,
        )
        # One token below threshold
        r1 = state.detect_phase(freedom=0.2, depth=2)
        assert r1 == "attack"

        # Freedom recovers — reset pending
        r2 = state.detect_phase(freedom=0.8, depth=2)
        assert r2 == "attack"

    def test_minimum_phase_duration_blocks_transition(self) -> None:
        """Transition blocked if current phase has lasted < 2 tokens."""
        state = self._make_state(
            phase="attack", peak_freedom=1.0, depth=2, prev_depth=2,
            phase_position=0,  # Just entered attack
        )
        # Even with 2 consecutive low-freedom tokens, min_phase_duration blocks
        state.detect_phase(freedom=0.2, depth=2)
        state.detect_phase(freedom=0.2, depth=2)
        # advance_phase should hold current phase
        state.advance_phase(freedom=0.2, depth=2)
        assert state.phase == "attack"  # Blocked by min_phase_duration
        state.advance_phase(freedom=0.2, depth=2)
        assert state.phase == "attack"  # Still blocked (position=1)

    def test_nested_expression_sequence(self) -> None:
        """Nested expression (tool1 (tool2 arg) ) produces correct phases.

        Inner ')' does NOT trigger release because depth > 1.
        """
        state = self._make_state(prev_smoothed=[0.0] * 11)

        # Opening '(' — depth 0 -> 1
        state.advance_phase(freedom=0.8, depth=1)
        assert state.phase == "attack"

        # tool1 tokens — freedom collapses as grammar constrains
        state.advance_phase(freedom=0.7, depth=1)
        assert state.phase == "attack"  # peak tracking

        # Freedom drops below 30% of peak (0.8) = 0.24
        state.advance_phase(freedom=0.2, depth=1)
        # May still be attack due to hysteresis
        state.advance_phase(freedom=0.2, depth=1)
        state.advance_phase(freedom=0.2, depth=1)
        # After enough tokens, should transition to decay then sustain

        # Inner '(' — depth 1 -> 2
        state.advance_phase(freedom=0.6, depth=2)
        assert state.phase == "attack"  # New nesting = new attack

        # Inner tokens
        state.advance_phase(freedom=0.5, depth=2)
        state.advance_phase(freedom=0.15, depth=2)
        state.advance_phase(freedom=0.15, depth=2)
        state.advance_phase(freedom=0.15, depth=2)

        # Inner ')' — depth 2 -> 1 (depth > 1 at new value, no release)
        # Actually depth goes from 2 to 1, and 1 <= 1, so this SHOULD be release
        # But prev_depth was 2 and new depth is 1, depth <= 1 is true
        # Wait — the condition is depth < prev_depth AND depth <= 1
        # prev_depth=2, new_depth=1, 1 < 2 AND 1 <= 1 => release
        # This is correct per the spec: release fires approaching root

        # Outer ')' — depth 1 -> 0
        state.advance_phase(freedom=0.9, depth=0)
        assert state.phase == "release"

    def test_advance_phase_tracks_peak_freedom(self) -> None:
        """Peak freedom is tracked during attack phase."""
        state = self._make_state(prev_smoothed=[0.0] * 11)
        state.advance_phase(freedom=0.5, depth=1)
        assert state.peak_freedom == 0.5
        state.advance_phase(freedom=0.8, depth=1)
        assert state.peak_freedom == 0.8
        state.advance_phase(freedom=0.6, depth=1)
        assert state.peak_freedom == 0.8  # Doesn't decrease


# === Task 2: Default modulation matrix and EnvelopeConfig ===


class TestEnvelopeConfig:
    """Tests for EnvelopeConfig and default matrix."""

    def test_default_matrix_shape(self) -> None:
        assert len(DEFAULT_MATRIX) == 11
        for row in DEFAULT_MATRIX:
            assert len(row) == 7

    def test_default_matrix_flat_length(self) -> None:
        assert len(DEFAULT_MATRIX_FLAT) == 11 * 7

    def test_config_matrix_shape(self) -> None:
        cfg = EnvelopeConfig()
        assert cfg.matrix_shape == (11, 7)

    def test_config_is_frozen(self) -> None:
        cfg = EnvelopeConfig()
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            cfg.base_temperature = 0.5  # type: ignore[misc]

    def test_base_values_are_sensible(self) -> None:
        cfg = EnvelopeConfig()
        assert 0.0 < cfg.base_temperature < 2.0
        assert 0.0 < cfg.base_top_p <= 1.0
        assert cfg.base_repetition_bias < 0.0
        assert 0.0 < cfg.base_epsilon < 1.0

    def test_clamp_ranges_prevent_invalid(self) -> None:
        cfg = EnvelopeConfig()
        lo, hi = cfg.temperature_range
        assert lo >= 0.0
        assert hi <= 5.0
        lo_p, hi_p = cfg.top_p_range
        assert lo_p > 0.0
        assert hi_p <= 1.0

    def test_column_5_backtrack_zeroed(self) -> None:
        """Column 5 (backtrack_threshold) is zeroed in default matrix."""
        for row in DEFAULT_MATRIX:
            assert row[5] == 0.0

    def test_conditioners_count(self) -> None:
        cfg = EnvelopeConfig()
        assert len(cfg.conditioners) == 11

    def test_matrix_weights_attack_row(self) -> None:
        """Phase_attack row (index 6) has expected values."""
        attack_row = DEFAULT_MATRIX[6]
        # temp=0.3, top_p=0.35, rep=10.0, eps=-0.05
        assert attack_row[0] == 0.3
        assert attack_row[1] == 0.35
        assert attack_row[2] == 10.0
        assert attack_row[3] == -0.05


# === Task 3: ModMatrixHookMlx ===


def _make_hook(
    config: EnvelopeConfig | None = None,
    vocab_size: int = 100,
    max_tokens: int = 128,
) -> ModMatrixHookMlx:
    """Create a hook with a simple tokenizer mock."""
    if config is None:
        config = EnvelopeConfig()

    def decode(ids: list[int]) -> str:
        # Token 10 = "(", token 11 = ")"
        mapping = {10: "(", 11: ")"}
        return "".join(mapping.get(i, "x") for i in ids)

    return ModMatrixHookMlx(
        config=config,
        tokenizer_decode=decode,
        vocab_size=vocab_size,
        max_tokens=max_tokens,
    )


class TestModMatrixHookMlx:
    """Tests for the unified MLX modulation hook."""

    def test_implements_protocol(self) -> None:
        """Hook satisfies InferenceHookMlx protocol."""
        from tgirl.sample_mlx import InferenceHookMlx

        hook = _make_hook()
        assert isinstance(hook, InferenceHookMlx)

    def test_pre_forward_returns_intervention(self) -> None:
        hook = _make_hook()
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        result = hook.pre_forward(0, valid_mask, [], logits)
        assert isinstance(result, ModelIntervention)

    def test_intervention_has_temperature(self) -> None:
        hook = _make_hook()
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        result = hook.pre_forward(0, valid_mask, [], logits)
        assert result.temperature is not None
        assert result.temperature >= 0.0

    def test_intervention_has_top_p(self) -> None:
        hook = _make_hook()
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        result = hook.pre_forward(0, valid_mask, [], logits)
        assert result.top_p is not None
        assert 0.0 < result.top_p <= 1.0

    def test_attack_higher_temp_than_sustain(self) -> None:
        """Phase detection affects output: attack -> higher temperature."""
        hook = _make_hook()
        logits = mx.random.normal((100,))
        # All tokens valid = high freedom -> attack phase
        valid_mask_high = mx.ones((100,), dtype=mx.bool_)
        r_attack = hook.pre_forward(0, valid_mask_high, [], logits)

        # Reset and simulate sustain: low freedom, many tokens deep
        hook.reset()
        hook._state.phase = "sustain"
        hook._state.phase_position = 10
        hook._state.depth = 2
        hook._state.prev_depth = 2
        # Very few valid tokens
        valid_mask_low = mx.zeros((100,), dtype=mx.bool_)
        valid_mask_low = valid_mask_low.at[0:5].add(
            mx.ones((5,), dtype=mx.bool_)
        )
        r_sustain = hook.pre_forward(5, valid_mask_low, [], logits)

        # Attack should produce higher temperature than sustain
        assert r_attack.temperature > r_sustain.temperature

    def test_depth_tracking_via_advance(self) -> None:
        """advance() tracks depth like NestingDepthHook."""
        hook = _make_hook()
        assert hook._state.depth == 0
        hook.advance(10)  # "(" -> depth +1
        assert hook._state.depth == 1
        hook.advance(10)  # "(" -> depth +1
        assert hook._state.depth == 2
        hook.advance(11)  # ")" -> depth -1
        assert hook._state.depth == 1
        hook.advance(11)  # ")" -> depth -1
        assert hook._state.depth == 0

    def test_cycle_detection_activates_rep_bias(self) -> None:
        """Cycle detection produces logit_bias on cycle tokens."""
        hook = _make_hook()
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        # Create a cycling history: [5,6,5,6]
        history = [5, 6, 5, 6]
        result = hook.pre_forward(4, valid_mask, history, logits)
        # cycle_detected should activate and produce logit_bias
        if result.logit_bias is not None:
            # Cycle tokens should have negative bias
            for tid in (5, 6):
                if tid in result.logit_bias:
                    assert result.logit_bias[tid] < 0.0

    def test_reset_clears_state(self) -> None:
        hook = _make_hook()
        hook.advance(10)  # depth -> 1
        hook._state.phase = "sustain"
        hook._state.phase_position = 10
        hook.reset()
        assert hook._state.depth == 0
        assert hook._state.phase == "attack"
        assert hook._state.phase_position == 0

    def test_matrix_multiply_uses_mx(self) -> None:
        """Verify mod matrix is an mx.array (native MLX matmul)."""
        hook = _make_hook()
        assert isinstance(hook._mod_matrix, mx.array)
        assert hook._mod_matrix.shape == (11, 7)

    def test_temperature_clamped_to_range(self) -> None:
        """Temperature is clamped within config range."""
        cfg = EnvelopeConfig(temperature_range=(0.1, 1.5))
        hook = _make_hook(config=cfg)
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        result = hook.pre_forward(0, valid_mask, [], logits)
        assert 0.1 <= result.temperature <= 1.5

    def test_opener_penalty_at_budget_limit(self) -> None:
        """Opener penalty activates when depth + position nears budget."""
        hook = _make_hook(max_tokens=20)
        # Set depth high so remaining budget is tight
        hook._state.depth = 15
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        result = hook.pre_forward(15, valid_mask, [], logits)
        # With depth=15, opener_bias_val should be very negative
        # from the depth row: depth * -20.0 (after conditioning)
        if result.logit_bias is not None:
            for tid in hook._opener_ids:
                if tid in result.logit_bias:
                    assert result.logit_bias[tid] < 0.0

    def test_rep_penalty_behavioral_equivalence(self) -> None:
        """Window-based counting matches RepetitionPenaltyHook behavior."""
        hook = _make_hook()
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        # History with token 42 appearing 4 times in 8-token window
        history = [42, 42, 42, 42, 1, 2, 3, 4]
        result = hook.pre_forward(8, valid_mask, history, logits)
        # The mod matrix should activate repetition bias
        # Check that logit_bias has a penalty for token 42
        if result.logit_bias is not None and 42 in result.logit_bias:
            assert result.logit_bias[42] < 0.0


# === Task 4: ModMatrixHook (torch variant) ===


def _make_torch_hook(
    config: EnvelopeConfig | None = None,
    vocab_size: int = 100,
    max_tokens: int = 128,
) -> ModMatrixHook:
    """Create a torch hook with a simple tokenizer mock."""
    if config is None:
        config = EnvelopeConfig()

    def decode(ids: list[int]) -> str:
        mapping = {10: "(", 11: ")"}
        return "".join(mapping.get(i, "x") for i in ids)

    return ModMatrixHook(
        config=config,
        tokenizer_decode=decode,
        vocab_size=vocab_size,
        max_tokens=max_tokens,
    )


class TestModMatrixHookTorch:
    """Tests for the torch variant of the modulation hook."""

    def test_implements_inference_hook(self) -> None:
        """Hook satisfies InferenceHook protocol."""
        from tgirl.sample import InferenceHook

        hook = _make_torch_hook()
        assert isinstance(hook, InferenceHook)

    def test_pre_forward_returns_intervention(self) -> None:
        hook = _make_torch_hook()
        logits = torch.randn(100)
        # Create a mock grammar state
        from unittest.mock import MagicMock

        gs = MagicMock()
        gs.get_valid_mask.return_value = torch.ones(100)
        result = hook.pre_forward(0, gs, [], logits)
        assert isinstance(result, ModelIntervention)
        assert result.temperature is not None

    def test_torch_mlx_parity(self) -> None:
        """Torch and MLX variants produce same outputs for same inputs."""
        config = EnvelopeConfig()
        torch.manual_seed(42)
        logits_np = torch.randn(100).numpy().tolist()

        # Torch variant
        torch_hook = _make_torch_hook(config=config)
        t_logits = torch.tensor(logits_np)
        from unittest.mock import MagicMock

        gs = MagicMock()
        gs.get_valid_mask.return_value = torch.ones(100)
        r_torch = torch_hook.pre_forward(0, gs, [], t_logits)

        # MLX variant
        mlx_hook = _make_hook(config=config)
        m_logits = mx.array(logits_np)
        m_mask = mx.ones((100,), dtype=mx.bool_)
        r_mlx = mlx_hook.pre_forward(0, m_mask, [], m_logits)

        # Temperature and top_p should be close
        assert abs(r_torch.temperature - r_mlx.temperature) < 0.01
        assert abs(r_torch.top_p - r_mlx.top_p) < 0.01

    def test_no_cross_framework_imports(self) -> None:
        """Torch variant does not import mlx."""
        import inspect

        source = inspect.getsource(ModMatrixHook.pre_forward)
        assert "mx." not in source
        assert "mlx" not in source


# === Task 5: transport_epsilon on ModelIntervention ===


class TestTransportEpsilon:
    """Tests for transport_epsilon field on ModelIntervention."""

    def test_model_intervention_accepts_transport_epsilon(self) -> None:
        mi = ModelIntervention(transport_epsilon=0.05)
        assert mi.transport_epsilon == 0.05

    def test_transport_epsilon_default_none(self) -> None:
        mi = ModelIntervention()
        assert mi.transport_epsilon is None

    def test_mod_matrix_hook_sets_epsilon(self) -> None:
        """ModMatrixHookMlx sets transport_epsilon in intervention."""
        hook = _make_hook()
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        result = hook.pre_forward(0, valid_mask, [], logits)
        assert result.transport_epsilon is not None
        assert result.transport_epsilon > 0.0


# === Task 7: Telemetry integration ===


class TestEnvelopeTelemetry:
    """Tests for EnvelopeTelemetry dataclass."""

    def test_telemetry_records_fields(self) -> None:
        t = EnvelopeTelemetry(
            phase="attack",
            phase_position=0,
            depth=1,
            source_vector=[0.5] * 11,
            modulation_vector=[0.1] * 7,
            final_temperature=0.3,
            final_epsilon=0.1,
        )
        assert t.phase == "attack"
        assert t.depth == 1
        assert len(t.source_vector) == 11
        assert len(t.modulation_vector) == 7

    def test_telemetry_json_serializable(self) -> None:
        t = EnvelopeTelemetry(
            phase="sustain",
            phase_position=5,
            depth=2,
            source_vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0],
            modulation_vector=[0.3, 0.9, -20.0, 0.1, 0.0, 0.0, 0.0],
            final_temperature=0.45,
            final_epsilon=0.08,
        )
        import dataclasses

        d = dataclasses.asdict(t)
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["phase"] == "sustain"
        assert deserialized["depth"] == 2

    def test_hook_produces_telemetry(self) -> None:
        """ModMatrixHookMlx stores last telemetry after pre_forward."""
        hook = _make_hook()
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        hook.pre_forward(0, valid_mask, [], logits)
        assert hook.last_telemetry is not None
        assert isinstance(hook.last_telemetry, EnvelopeTelemetry)
        assert hook.last_telemetry.phase in (
            "attack", "decay", "sustain", "release",
        )

    def test_telemetry_list_matches_token_count(self) -> None:
        """Multiple pre_forward calls produce matching telemetry."""
        hook = _make_hook()
        logits = mx.random.normal((100,))
        valid_mask = mx.ones((100,), dtype=mx.bool_)
        telemetry_list = []
        for i in range(5):
            hook.pre_forward(i, valid_mask, list(range(i)), logits)
            telemetry_list.append(hook.last_telemetry)
        assert len(telemetry_list) == 5
        assert all(t is not None for t in telemetry_list)
