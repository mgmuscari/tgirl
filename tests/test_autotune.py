"""Tests for tgirl.autotune — rule-based steering controller."""

from __future__ import annotations

import pytest


def _obs(**kwargs):
    """Builder for Observables with sensible defaults."""
    from tgirl.autotune import Observables

    defaults = dict(
        repeat_rate=0.0,
        bigram_novelty=0.7,
        token_entropy=0.7,
        n_tokens=200,
        finish_reason="stop",
        mean_entropy=2.0,
        mean_top1_prob=0.5,
        mean_top1_margin=0.3,
        alpha=0.5,
        beta=2.0,
        temperature=0.0,
    )
    defaults.update(kwargs)
    return Observables(**defaults)


class TestClassify:
    """Regime labels from the 2D sweep + earlier α sweep data."""

    def test_baseline_signal(self) -> None:
        from tgirl.autotune import classify

        # Mid entropy, no repeats, healthy length → signal
        assert classify(_obs(token_entropy=0.7, repeat_rate=0.0, n_tokens=200)) == "signal"

    def test_severe_loop_collapse_high_repeat(self) -> None:
        from tgirl.autotune import classify

        # The α=0.9 / β=1.0 disaster: rep=0.83, H=0.08
        assert classify(_obs(repeat_rate=0.83, token_entropy=0.08)) == "loop_collapse"

    def test_severe_loop_collapse_low_entropy(self) -> None:
        from tgirl.autotune import classify

        # Low entropy alone is a collapse signal even if repeats haven't peaked
        assert classify(_obs(repeat_rate=0.15, token_entropy=0.25)) == "loop_collapse"

    def test_loop_emergence_moderate(self) -> None:
        from tgirl.autotune import classify

        # α=0.7 region: repeats climbing, H softening
        assert classify(_obs(repeat_rate=0.16, token_entropy=0.55)) == "loop_emergence"

    def test_sycophant_trap_short_stop_high_entropy(self) -> None:
        from tgirl.autotune import classify

        # The α=0.4 sycophant: 49 tokens, H=0.93, finish=stop
        assert classify(_obs(
            n_tokens=49, finish_reason="stop", token_entropy=0.93,
            repeat_rate=0.0,
        )) == "sycophant_trap"

    def test_sycophant_classifier_requires_short_AND_stop(self) -> None:
        """A short response that hit `length` (no EOS) is NOT sycophant —
        it's a different shape entirely."""
        from tgirl.autotune import classify

        result = classify(_obs(
            n_tokens=49, finish_reason="length", token_entropy=0.93,
            repeat_rate=0.0,
        ))
        assert result != "sycophant_trap"


class TestAutotuneActions:
    """Per-regime control responses, calibrated against the 2D sweep."""

    def test_signal_regime_no_change(self) -> None:
        from tgirl.autotune import autotune

        obs = _obs(token_entropy=0.7, repeat_rate=0.0, n_tokens=200)
        action = autotune(obs)
        assert action.regime == "signal"
        assert action.next_alpha == obs.alpha
        assert action.next_beta == obs.beta
        assert action.next_temperature == obs.temperature

    def test_loop_collapse_reduces_alpha_and_widens_beta_floor(self) -> None:
        """Severe collapse: pull α down hard, ensure β isn't in the
        dangerous <2.0 zone where the α=0.9/β=1.0 disaster lives.
        """
        from tgirl.autotune import autotune

        obs = _obs(alpha=0.9, beta=1.0, repeat_rate=0.83, token_entropy=0.08)
        action = autotune(obs)
        assert action.regime == "loop_collapse"
        assert action.next_alpha < obs.alpha
        # β floored at 2.0 (safe across α range per 2D sweep)
        assert action.next_beta is None or action.next_beta >= 2.0

    def test_loop_emergence_gentle_alpha_reduction(self) -> None:
        from tgirl.autotune import autotune

        obs = _obs(alpha=0.7, beta=2.0, repeat_rate=0.16, token_entropy=0.55)
        action = autotune(obs)
        assert action.regime == "loop_emergence"
        assert action.next_alpha < obs.alpha
        # Smaller step than collapse — gentler intervention
        assert obs.alpha - action.next_alpha <= 0.15

    def test_sycophant_trap_escapes_basin(self) -> None:
        """At low α (basin attractor at α=0.3-0.4), push UP past 0.5.
        At higher α (we landed in the basin from above), back DOWN.
        """
        from tgirl.autotune import autotune

        # In the basin from below
        obs_low = _obs(
            alpha=0.4, beta=None, n_tokens=49, finish_reason="stop",
            token_entropy=0.93, repeat_rate=0.0,
        )
        action_low = autotune(obs_low)
        assert action_low.regime == "sycophant_trap"
        assert action_low.next_alpha > obs_low.alpha

        # In the basin from above (e.g. swung past from a loop reduction)
        obs_high = _obs(
            alpha=0.55, beta=None, n_tokens=49, finish_reason="stop",
            token_entropy=0.93, repeat_rate=0.0,
        )
        action_high = autotune(obs_high)
        assert action_high.regime == "sycophant_trap"
        assert action_high.next_alpha < obs_high.alpha

    def test_alpha_clipped_to_unit_interval(self) -> None:
        """Even compounded reductions never go negative; pushes never exceed 1."""
        from tgirl.autotune import autotune

        # Already low + collapse signal would naively go negative
        obs = _obs(alpha=0.05, repeat_rate=0.83, token_entropy=0.08)
        assert 0.0 <= autotune(obs).next_alpha <= 1.0

        # Already high + sycophant push-up signal
        obs = _obs(
            alpha=0.95, n_tokens=49, finish_reason="stop",
            token_entropy=0.93, repeat_rate=0.0,
        )
        assert 0.0 <= autotune(obs).next_alpha <= 1.0

    def test_beta_never_drops_below_safe_floor(self) -> None:
        """β=1.0 at high α is catastrophic per the 2D sweep — controller
        must never drive β below 2.0 (or stay at None for single-layer).
        """
        from tgirl.autotune import autotune

        obs = _obs(beta=2.0, repeat_rate=0.83, token_entropy=0.08)
        action = autotune(obs)
        assert action.next_beta is None or action.next_beta >= 2.0


class TestActionShape:
    """Action carries everything the serve layer + training-set logger need."""

    def test_action_carries_rationale_and_regime(self) -> None:
        from tgirl.autotune import autotune

        action = autotune(_obs())
        assert isinstance(action.rationale, str) and len(action.rationale) > 0
        assert action.regime in {
            "signal", "sycophant_trap", "loop_emergence", "loop_collapse"
        }

    def test_observables_serialize_to_dict(self) -> None:
        """Training-set logger needs (obs, action) tuples as JSON.
        Both must be dict-convertible.
        """
        from tgirl.autotune import autotune, observables_to_dict

        obs = _obs(repeat_rate=0.1, token_entropy=0.6)
        d = observables_to_dict(obs)
        assert d["repeat_rate"] == pytest.approx(0.1)
        assert d["token_entropy"] == pytest.approx(0.6)
        assert "alpha" in d and "beta" in d and "temperature" in d
