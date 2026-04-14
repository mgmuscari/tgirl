"""Skewed-Gaussian band-weighting over transformer layers.

Given a bottleneck layer and a sharpness β (with optional skew), produce
a dict mapping layer indices to non-negative weights that sum to 1. The
steering-correction injection site uses this to spread a single
correction across a *band* of layers instead of applying it at the
bottleneck alone.

Parametrization (ESTRADIOL-v2):

- ``beta`` — sharpness (inverse σ, in layers⁻¹). ``None`` or ``math.inf``
  degenerates to a single-layer injection at the bottleneck (bit-
  compatible with the pre-band behavior).
- ``skew`` — ratio ``σ_up / σ_down``. ``1.0`` is symmetric. Values > 1
  put more mass above the bottleneck (toward output); values < 1 put
  more mass below (toward input). The mean σ is held at ``1/beta``
  regardless of skew, so β and skew are independent knobs.

The Gaussian support is truncated at 3σ on each side. Layers outside
that window get no injection at all — keeps the per-token forward-pass
overhead bounded.

Weights are renormalized to sum to 1 *after* clamping to the layer-
stack bounds, so a bottleneck near an edge still contributes the
same total correction magnitude (α) as a bottleneck in the middle.
The α knob controls total injected correction; β controls only how
that total is distributed across layers.
"""

from __future__ import annotations

import math


def _sigma_up_down(beta: float, skew: float) -> tuple[float, float]:
    """σ_up and σ_down derived from (β, skew) with mean σ = 1/β.

    Formula: σ_up = (2·skew / (1+skew)) / β
             σ_down = (2 / (1+skew)) / β

    At skew=1, both equal 1/β (symmetric).
    As skew grows, σ_up grows and σ_down shrinks, with their mean
    held at 1/β — so β and skew are independent distributional knobs.
    """
    mean_sigma = 1.0 / beta
    sigma_up = (2.0 * skew / (1.0 + skew)) * mean_sigma
    sigma_down = (2.0 / (1.0 + skew)) * mean_sigma
    return sigma_up, sigma_down


def band_weights(
    n_layers: int,
    bottleneck_idx: int,
    beta: float | None,
    skew: float = 1.0,
) -> dict[int, float]:
    """Mass-preserving skewed-Gaussian weights over a band of layers.

    Args:
        n_layers: Total number of layers in the model.
        bottleneck_idx: Layer index of the bottleneck (peak of the band).
        beta: Sharpness (inverse σ). ``None`` or ``math.inf`` → single
            layer. Must be positive if not ``None``.
        skew: Ratio ``σ_up / σ_down``. Default ``1.0`` (symmetric).
            Must be positive.

    Returns:
        Dict mapping layer index → weight in ``(0, 1]``. Sums to 1.

    Raises:
        ValueError: If ``bottleneck_idx`` is out of range, or ``beta`` /
            ``skew`` are non-positive.
    """
    if not (0 <= bottleneck_idx < n_layers):
        msg = (
            f"bottleneck_idx={bottleneck_idx} out of range for "
            f"n_layers={n_layers}"
        )
        raise ValueError(msg)
    if skew <= 0:
        msg = f"skew must be positive (got {skew})"
        raise ValueError(msg)

    # Degenerate: infinite sharpness ⇒ single-layer injection.
    # This is the default (beta=None) and the pre-band behavior.
    if beta is None or math.isinf(beta):
        return {bottleneck_idx: 1.0}
    if beta <= 0:
        msg = f"beta must be positive (got {beta})"
        raise ValueError(msg)

    sigma_up, sigma_down = _sigma_up_down(beta, skew)

    # 3σ cutoff on each side, clamped to the layer stack.
    lo = max(0, bottleneck_idx - math.ceil(3.0 * sigma_down))
    hi = min(n_layers - 1, bottleneck_idx + math.ceil(3.0 * sigma_up))

    unnormalized: dict[int, float] = {}
    for idx in range(lo, hi + 1):
        offset = idx - bottleneck_idx
        sigma = sigma_up if offset >= 0 else sigma_down
        unnormalized[idx] = math.exp(-0.5 * (offset / sigma) ** 2)

    total = sum(unnormalized.values())
    return {idx: w / total for idx, w in unnormalized.items()}
