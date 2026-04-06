"""Offline calibration pipeline for ESTRADIOL v2.

Discovers the bottleneck layer, extracts behavioral vectors via
contrastive generation, builds the low-rank codebook via SVD,
and validates scaffold/complement decomposition.

Adapted from platonic-experiments (experiments 27a, 27d, 27c).
"""

from __future__ import annotations

import math
import time
from typing import Any

import mlx.core as mx
import structlog

from tgirl.cache import _BottleneckHook

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Effective rank metrics
# ---------------------------------------------------------------------------


def effective_rank(X: mx.array) -> float:
    """Effective rank via Shannon entropy of singular values.

    exp(-Sigma p_i log p_i) where p_i = s_i / Sigma_s.
    Returns 1.0 for rank-1, d for full rank (identity).

    Args:
        X: (n_samples, d_features) matrix.
    """
    _, S, _ = mx.linalg.svd(X, stream=mx.cpu)
    mx.eval(S)
    # Filter near-zero singular values
    s = [float(v) for v in S.tolist() if v > 1e-10]
    if not s:
        return 0.0
    total = sum(s)
    entropy = -sum((si / total) * math.log(si / total) for si in s)
    return math.exp(entropy)


def participation_ratio(s: mx.array) -> float:
    """Participation ratio: (Sigma_s)^2 / Sigma(s^2).

    Used for codebook compression analysis (experiment27d).
    Returns 1.0 for single dominant value, N for uniform spectrum.
    """
    s_sum = float(mx.sum(s).item())
    s_sq_sum = float(mx.sum(s * s).item())
    if s_sq_sum < 1e-30:
        return 0.0
    return (s_sum * s_sum) / s_sq_sum


# ---------------------------------------------------------------------------
# Bottleneck discovery
# ---------------------------------------------------------------------------


class _AllPositionsHook:
    """Captures activations at ALL token positions for rank analysis.

    Unlike _BottleneckHook (which captures last-token only for runtime),
    this hook captures the full (seq_len, d_model) output at the target
    layer. Used only during calibration.
    """

    def __init__(self, layers: Any, layer_idx: int) -> None:
        self._target = layers[layer_idx]
        self._layer_type = type(self._target)
        self._original_call: Any = None
        self._captured: Any = None  # (seq_len, d_model)
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._original_call = self._layer_type.__call__
        hook = self

        def _patched(layer_self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
            result = hook._original_call(layer_self, x, *args, **kwargs)
            if layer_self is hook._target:
                # Capture all positions: (batch=1, seq_len, d_model) → (seq_len, d_model)
                hook._captured = result[0, :, :].astype(mx.float32)
            return result

        self._layer_type.__call__ = _patched
        self._installed = True

    def uninstall(self) -> None:
        if self._installed and self._original_call is not None:
            self._layer_type.__call__ = self._original_call
            self._installed = False
            self._original_call = None


def discover_bottleneck(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layer_path: str = "model.layers",
) -> tuple[int, list[float]]:
    """Sweep effective rank across all layers to find the bottleneck.

    For each layer, captures activations at ALL token positions across
    all texts, producing a (total_tokens, d_model) matrix. Effective
    rank of this matrix measures the dimensionality of the layer's
    representational space.

    Args:
        model: MLX model with make_cache() and __call__.
        tokenizer: Tokenizer with encode().
        texts: Calibration texts (10+ diverse sentences recommended).
        layer_path: Dot-separated path to model layers list.

    Returns:
        (bottleneck_layer_idx, effective_ranks_per_layer)
    """
    # Navigate to layers
    layers = model
    for attr in layer_path.split("."):
        layers = getattr(layers, attr)
    n_layers = len(layers)

    logger.info("discover_bottleneck_start", n_layers=n_layers, n_texts=len(texts))
    t0 = time.monotonic()

    ranks: list[float] = []
    for layer_idx in range(n_layers):
        hook = _AllPositionsHook(layers, layer_idx=layer_idx)
        hook.install()
        try:
            all_activations = []
            for text in texts:
                token_ids = tokenizer.encode(text)
                cache = model.make_cache()
                input_ids = mx.array([token_ids])
                _ = model(input_ids, cache=cache)
                if hook._captured is not None:
                    all_activations.append(hook._captured)
                hook._captured = None

            if all_activations:
                # Concatenate: (total_tokens, d_model)
                M = mx.concatenate(all_activations, axis=0)
                mx.eval(M)
                rank = effective_rank(M)
            else:
                rank = float("inf")
            ranks.append(rank)
        finally:
            hook.uninstall()

    bn_layer = int(min(range(n_layers), key=lambda i: ranks[i]))
    elapsed = time.monotonic() - t0
    logger.info(
        "discover_bottleneck_done",
        bottleneck_layer=bn_layer,
        bottleneck_rank=round(ranks[bn_layer], 2),
        elapsed_s=round(elapsed, 1),
    )
    return bn_layer, ranks
