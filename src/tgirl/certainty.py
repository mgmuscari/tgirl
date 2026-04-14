"""Pre-sampling logit-distribution signals for the ESTRADIOL-v2 autotuner.

Complements ``tgirl.coherence`` (post-generation output structure) by
measuring what the model was *considering* at each decision point:

- **entropy** — Shannon entropy of the softmax distribution, in nats.
  Low = model committed to one token; high = distribution is diffuse.
- **top1_prob** — probability assigned to the chosen (argmax) token.
  Low = model unsure; high = locked in.
- **top1_margin** — ``p_top1 − p_top2``. A different slice of confidence:
  the model can be commit-strong (top1≈1) or commit-weakly (top1≈0.5
  with top2≈0.49 = tiny margin, big ambiguity).

The three are correlated but not redundant. Together they characterize
the *structure* of uncertainty, not just its magnitude — which matters
because loop collapse (low entropy AND low margin: model is locked into
the same repeated token) looks different from fracture (high entropy
AND near-zero margin: model is uniformly unsure).

Usage::

    # Per token, inside the generation loop:
    steps.append(step_certainty(logits))

    # At end of turn, surface means as autotuner input:
    stats = mean_certainty(steps)
"""

from __future__ import annotations

from typing import Any


def step_certainty(logits: Any) -> dict[str, float]:
    """One-token logit-distribution signals.

    Args:
        logits: ``mx.array`` of shape ``(vocab,)``.

    Returns:
        Dict with ``entropy``, ``top1_prob``, ``top1_margin``.
    """
    import mlx.core as mx

    # Softmax stability: subtract max before exp (no behavioral change,
    # keeps exp()'s inputs in a safe range even for 100+ magnitudes used
    # in tests).
    shifted = logits - mx.max(logits)
    probs = mx.softmax(shifted)

    # Entropy in nats; clamp input to log to avoid log(0). MLX's softmax
    # yields non-negative values; the clamp just hardens against any
    # numerical drift at extreme magnitudes.
    logp = mx.log(mx.maximum(probs, mx.array(1e-20)))
    entropy = float(-mx.sum(probs * logp).item())

    # Top-2 by sorted order (mx.sort is ascending; take the last two).
    sorted_probs = mx.sort(probs)
    top1 = float(sorted_probs[-1].item())
    top2 = float(sorted_probs[-2].item())
    return {
        "entropy": entropy,
        "top1_prob": top1,
        "top1_margin": top1 - top2,
    }


def mean_certainty(steps: list[dict[str, float]]) -> dict[str, Any]:
    """Arithmetic mean of per-step certainty dicts.

    Safe defaults for ``n_steps == 0`` (neutral signal that won't trip
    autotuner classifier thresholds): entropy=0, top1_prob=1, margin=1.

    Returns the same key set with a ``mean_`` prefix plus ``n_steps``.
    """
    n = len(steps)
    if n == 0:
        return {
            "n_steps": 0,
            "mean_entropy": 0.0,
            "mean_top1_prob": 1.0,
            "mean_top1_margin": 1.0,
        }
    mean_e = sum(s["entropy"] for s in steps) / n
    mean_p = sum(s["top1_prob"] for s in steps) / n
    mean_m = sum(s["top1_margin"] for s in steps) / n
    return {
        "n_steps": n,
        "mean_entropy": mean_e,
        "mean_top1_prob": mean_p,
        "mean_top1_margin": mean_m,
    }
