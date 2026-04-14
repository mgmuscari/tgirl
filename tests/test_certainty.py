"""Tests for tgirl.certainty — pre-sampling logit-distribution signals."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest


class TestStepCertainty:
    """Per-token signals derived from the logit distribution. These
    complement the post-generation coherence triple by measuring what
    the model *was considering* at each decision point, not what it
    ultimately emitted. Used by the autotuner alongside coherence.
    """

    def test_one_hot_logits_are_maximally_confident(self) -> None:
        """Logits with one very-high value → softmax is ~one-hot →
        entropy≈0, top1≈1, margin≈1.
        """
        from tgirl.certainty import step_certainty

        vocab = 32
        logits = mx.zeros((vocab,))
        logits = logits.at[5].add(mx.array(100.0))
        s = step_certainty(logits)
        assert s["entropy"] == pytest.approx(0.0, abs=1e-3)
        assert s["top1_prob"] == pytest.approx(1.0, abs=1e-3)
        assert s["top1_margin"] == pytest.approx(1.0, abs=1e-3)

    def test_uniform_logits_are_maximally_diffuse(self) -> None:
        """All-zero logits → uniform softmax → entropy=log(vocab),
        top1=1/vocab, margin=0.
        """
        from tgirl.certainty import step_certainty

        vocab = 16
        logits = mx.zeros((vocab,))
        s = step_certainty(logits)
        assert s["entropy"] == pytest.approx(math.log(vocab), abs=1e-3)
        assert s["top1_prob"] == pytest.approx(1.0 / vocab, abs=1e-3)
        assert s["top1_margin"] == pytest.approx(0.0, abs=1e-4)

    def test_two_way_tie_has_entropy_one_bit(self) -> None:
        """Two tokens equally preferred far above the rest → softmax
        concentrates on those two with probability ~0.5 each → entropy
        ≈ ln(2), margin ≈ 0.
        """
        from tgirl.certainty import step_certainty

        vocab = 32
        logits = mx.zeros((vocab,))
        logits = logits.at[3].add(mx.array(100.0))
        logits = logits.at[7].add(mx.array(100.0))
        s = step_certainty(logits)
        assert s["entropy"] == pytest.approx(math.log(2), abs=1e-3)
        assert s["top1_prob"] == pytest.approx(0.5, abs=1e-3)
        assert s["top1_margin"] == pytest.approx(0.0, abs=1e-3)

    def test_return_shape_is_stable(self) -> None:
        """Autotuner reads fixed key set — lock it in."""
        from tgirl.certainty import step_certainty

        s = step_certainty(mx.zeros((8,)))
        assert set(s.keys()) == {"entropy", "top1_prob", "top1_margin"}


class TestAggregateCertainty:
    """End-of-turn mean aggregation for autotuner input."""

    def test_mean_certainty_empty_returns_safe_defaults(self) -> None:
        """No tokens accumulated (n=0) → neutral signal that won't
        trigger classifier alarms.
        """
        from tgirl.certainty import mean_certainty

        m = mean_certainty([])
        assert m["mean_entropy"] == 0.0
        assert m["mean_top1_prob"] == 1.0
        assert m["mean_top1_margin"] == 1.0
        assert m["n_steps"] == 0

    def test_mean_over_steps_is_arithmetic(self) -> None:
        """Aggregate just averages the per-step dicts."""
        from tgirl.certainty import mean_certainty

        steps = [
            {"entropy": 0.1, "top1_prob": 0.9, "top1_margin": 0.8},
            {"entropy": 0.3, "top1_prob": 0.7, "top1_margin": 0.4},
        ]
        m = mean_certainty(steps)
        assert m["n_steps"] == 2
        assert m["mean_entropy"] == pytest.approx(0.2)
        assert m["mean_top1_prob"] == pytest.approx(0.8)
        assert m["mean_top1_margin"] == pytest.approx(0.6)
