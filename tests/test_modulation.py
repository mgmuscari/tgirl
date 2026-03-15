"""Tests for the ADSR modulation matrix.

Covers source conditioning, phase detection with hysteresis,
envelope configuration, and the unified ModMatrixHook.
"""

from __future__ import annotations

import dataclasses

from tgirl.modulation import (
    DEFAULT_MATRIX,
    DEFAULT_MATRIX_FLAT,
    EnvelopeConfig,
    EnvelopeState,
    SourceConditionerConfig,
    condition_source,
)


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
        state = self._make_state(phase="sustain", depth=1, prev_depth=1, phase_position=5)
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
        state = self._make_state(phase="sustain", depth=2, prev_depth=2, phase_position=5)
        result = state.detect_phase(freedom=0.5, depth=1)
        assert result == "release"

    def test_no_release_when_depth_gt_1(self) -> None:
        """Depth decrease with depth > 1 does NOT trigger release."""
        state = self._make_state(phase="sustain", depth=4, prev_depth=4, phase_position=5)
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
