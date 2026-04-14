"""Tests for tgirl.band — skewed-Gaussian layer-band weighting."""

from __future__ import annotations

import math

import pytest


class TestBandWeights:
    """Mass-preserving skewed-Gaussian weights over a band of layers,
    centered at the bottleneck. β is sharpness (inverse σ); skew is the
    ratio σ_up / σ_down with parametrization that preserves mean σ=1/β.
    """

    def test_beta_none_degenerates_to_single_layer(self) -> None:
        """β=None means 'infinite sharpness' → all weight at the
        bottleneck layer. This keeps pre-band callers bit-compatible.
        """
        from tgirl.band import band_weights

        w = band_weights(n_layers=28, bottleneck_idx=14, beta=None)
        assert w == {14: 1.0}

    def test_symmetric_band_has_peak_at_bottleneck(self) -> None:
        """skew=1.0 → symmetric Gaussian. Peak is at the bottleneck
        and neighbors fall off monotonically.
        """
        from tgirl.band import band_weights

        w = band_weights(
            n_layers=28, bottleneck_idx=14, beta=0.5, skew=1.0
        )
        # Peak
        assert max(w.values()) == pytest.approx(w[14])
        # Symmetric around bottleneck
        for offset in (1, 2, 3):
            assert w.get(14 - offset) == pytest.approx(w.get(14 + offset))
        # Monotonic falloff
        assert w[14] > w[13] > w[12] > w[11]
        assert w[14] > w[15] > w[16] > w[17]

    def test_symmetric_band_weights_sum_to_one(self) -> None:
        """Mass preservation: band weights sum to 1 when the support
        fits inside the layer stack (middle of stack, not truncated).
        α controls magnitude; β controls only distribution.
        """
        from tgirl.band import band_weights

        w = band_weights(
            n_layers=40, bottleneck_idx=20, beta=0.5, skew=1.0
        )
        total = sum(w.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_truncated_support_still_sums_to_one(self) -> None:
        """Bottleneck near an edge truncates half the Gaussian. The
        remaining weights must still normalize to 1 so α keeps the
        same semantic scale regardless of where the bottleneck sits.
        """
        from tgirl.band import band_weights

        w = band_weights(
            n_layers=28, bottleneck_idx=0, beta=0.5, skew=1.0
        )
        # Bottleneck is at the bottom edge — no layers below
        assert all(idx >= 0 for idx in w)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        # Peak stays at the bottleneck
        assert max(w.values()) == pytest.approx(w[0])

    def test_skew_above_one_puts_more_mass_upstream(self) -> None:
        """skew > 1 → σ_up > σ_down → more mass distributed above
        the bottleneck (toward output) than below. This handles the
        "bottleneck is late in the stack" case without spilling weight
        past the top edge.
        """
        from tgirl.band import band_weights

        w = band_weights(
            n_layers=40, bottleneck_idx=20, beta=0.5, skew=2.0
        )
        mass_above = sum(v for idx, v in w.items() if idx > 20)
        mass_below = sum(v for idx, v in w.items() if idx < 20)
        assert mass_above > mass_below
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_skew_below_one_puts_more_mass_downstream(self) -> None:
        """skew < 1 → σ_up < σ_down → more mass below the bottleneck."""
        from tgirl.band import band_weights

        w = band_weights(
            n_layers=40, bottleneck_idx=20, beta=0.5, skew=0.5
        )
        mass_above = sum(v for idx, v in w.items() if idx > 20)
        mass_below = sum(v for idx, v in w.items() if idx < 20)
        assert mass_below > mass_above
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_support_truncated_at_three_sigma(self) -> None:
        """Weights past ~3σ from the peak are negligible — we drop
        them entirely rather than carrying floating-point noise.
        Keeps hook forward-pass overhead bounded.
        """
        from tgirl.band import band_weights

        # beta=1.0 → sigma=1. Support should be within [bottleneck±3].
        w = band_weights(
            n_layers=40, bottleneck_idx=20, beta=1.0, skew=1.0
        )
        included = sorted(w.keys())
        # Everything outside the 3σ window should be absent
        assert all(abs(idx - 20) <= 3 for idx in included), included
        # And the weight dict shouldn't be just the peak (something
        # spread out)
        assert len(included) > 1

    def test_larger_beta_narrows_the_band(self) -> None:
        """Higher β = sharper (narrower distribution). Measured by
        how much mass lands at the peak.
        """
        from tgirl.band import band_weights

        wide = band_weights(
            n_layers=40, bottleneck_idx=20, beta=0.3, skew=1.0
        )
        narrow = band_weights(
            n_layers=40, bottleneck_idx=20, beta=1.0, skew=1.0
        )
        assert narrow[20] > wide[20]

    def test_weight_math_matches_reference_formula(self) -> None:
        """Spot-check against a hand-computed value. Symmetric, β=1, peak
        at offset 0 before normalization: gaussian(0, σ=1) = 1.
        Two neighbors at ±1: exp(-0.5) ≈ 0.6065.
        Normalized peak = 1 / (1 + 2*exp(-0.5) + 2*exp(-2) + 2*exp(-4.5)).
        """
        from tgirl.band import band_weights

        w = band_weights(
            n_layers=40, bottleneck_idx=20, beta=1.0, skew=1.0
        )
        unnorm = {
            0: 1.0,
            1: math.exp(-0.5),
            2: math.exp(-2),
            3: math.exp(-4.5),
        }
        denom = unnorm[0] + 2 * (unnorm[1] + unnorm[2] + unnorm[3])
        expected_peak = unnorm[0] / denom
        assert w[20] == pytest.approx(expected_peak, rel=1e-6)
