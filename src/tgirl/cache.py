"""KV cache wrapper factories for inference backends.

Provides forward-function wrappers that transparently manage KV cache
state for MLX and HuggingFace Transformers models. Each wrapper detects
prefix continuations and only forwards new tokens, reusing cached state
for the shared prefix.

This module has ZERO coupling to any other tgirl module — it operates
on model objects and token ID lists only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch


class ForwardResult(NamedTuple):
    """Logits plus optional probe reading from steerable forward functions."""

    logits: Any  # mx.array or torch.Tensor (vocab_size,)
    probe_alpha: Any | None = None  # mx.array (K,) or None


class _BottleneckHook:
    """Monkey-patches a transformer layer to capture and inject activations.

    Ported from platonic-experiments LayerHook (experiment26n). Patches
    the layer type's __call__ so the target layer's output can be read
    (probe) and modified (steer) each forward pass.

    Usage::

        hook = _BottleneckHook(model_layers_list, layer_idx=14)
        hook.install()
        # ... run model forward ...
        h = hook.get_captured()       # (d_model,) last-token activation
        alpha = hook.get_probe(V)     # (K,) = V.T @ h
        hook.set_steering(V, delta)   # inject V @ delta next forward
        # ...
        hook.uninstall()
    """

    def __init__(self, layers: Any, layer_idx: int) -> None:
        self._layers = layers
        self._layer_idx = layer_idx
        self._target = layers[layer_idx]
        self._layer_type = type(self._target)
        self._original_call: Any = None
        self._captured: Any = None  # (d_model,) last-token activation
        self._V_basis: Any = None  # (d_model, K) for codebook injection
        self._delta_alpha: Any = None  # (K,) correction for codebook injection
        self._raw_correction: Any = None  # (d_model,) raw injection vector
        # Residual-relative steering: probe direction + alpha stored
        # separately so the correction magnitude can be rescaled
        # against |residual_last| at forward time. Makes α a structural
        # fraction of the residual stream being overwritten, not an
        # absolute scalar that varies with |v_probe|.
        self._probe_direction: Any = None  # unit (d_model,) vector
        self._probe_alpha: float = 0.0
        # Band injection: {id(layer) -> weight}. None ⇒ single-layer
        # (weight 1.0 at the bottleneck only). When set by set_band,
        # correction is spread across a neighborhood of layers with
        # mass-preserving weights that sum to 1 — so α controls total
        # correction magnitude independently of β.
        self._band_weights_by_id: dict[int, float] | None = None
        self._installed = False

    def install(self) -> None:
        """Patch the layer type's __call__."""
        if self._installed:
            return
        self._original_call = self._layer_type.__call__
        hook = self

        def _patched(layer_self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
            result = hook._original_call(layer_self, x, *args, **kwargs)

            # Capture stays bottleneck-only regardless of band config:
            # the band is an *injection* distribution, not a capture one.
            if layer_self is hook._target:
                import mlx.core as mx

                hook._captured = result[:, -1, :].astype(mx.float32).reshape(-1)

            # Determine the injection weight at this layer.
            if hook._band_weights_by_id is None:
                # Single-layer (pre-band) default: full weight at target only.
                weight = 1.0 if layer_self is hook._target else None
            else:
                weight = hook._band_weights_by_id.get(id(layer_self))

            if weight is not None:
                import mlx.core as mx

                if hook._raw_correction is not None:
                    result = result + (
                        weight * hook._raw_correction
                    ).reshape(1, 1, -1)
                elif (
                    hook._V_basis is not None
                    and hook._delta_alpha is not None
                ):
                    # Codebook projection; (d_model,) correction spread
                    # across the band per the user's "keep the codebook"
                    # hunch that V_basis stays approximately valid in a
                    # small neighborhood of the calibration layer.
                    correction = hook._V_basis @ hook._delta_alpha
                    result = result + (weight * correction).reshape(1, 1, -1)
                elif hook._probe_direction is not None:
                    # Residual-relative steering: scale the unit-direction
                    # probe by α * |residual_last| so the correction's
                    # magnitude is proportional to the local signal power.
                    # Makes α a structural fraction of the residual stream
                    # rather than an absolute scalar.
                    r_last = result[:, -1, :]
                    r_norm = mx.linalg.norm(r_last)
                    mag = weight * hook._probe_alpha * r_norm
                    correction = mag * hook._probe_direction
                    result = result + correction.reshape(1, 1, -1)
            return result

        self._layer_type.__call__ = _patched
        self._installed = True

    def uninstall(self) -> None:
        """Restore the original __call__."""
        if self._installed and self._original_call is not None:
            self._layer_type.__call__ = self._original_call
            self._installed = False
            self._original_call = None

    def get_captured(self) -> Any:
        """Return the last captured activation (d_model,), or None."""
        return self._captured

    def get_probe(self, V_basis: Any) -> Any:
        """Project captured activation onto codebook basis: V.T @ h → (K,)."""
        if self._captured is None:
            return None
        return V_basis.T @ self._captured

    def set_steering(self, V_basis: Any, delta_alpha: Any) -> None:
        """Configure injection for subsequent forward passes."""
        self._V_basis = V_basis
        self._delta_alpha = delta_alpha

    def set_raw_correction(self, correction: Any) -> None:
        """Set a raw (d_model,) vector to inject. Bypasses codebook."""
        self._raw_correction = correction
        # Mutually exclusive with residual-relative mode.
        self._probe_direction = None
        self._probe_alpha = 0.0

    def set_probe_steering(self, v_probe: Any, alpha: float) -> None:
        """Configure residual-relative steering.

        The injected correction at each forward pass will be
        ``alpha * |residual_last| * v_probe / |v_probe|``, scaled
        further by any active band weight. α is a structural fraction
        of the residual stream — stable across models, layers, and
        contexts. ``v_probe`` supplies direction only; its magnitude
        is stripped.

        Args:
            v_probe: (d_model,) direction vector. Any non-zero norm is
                acceptable; the hook normalizes it. Passing the cached
                probe vector from a prior turn is the intended use.
            alpha: Structural fraction in ``[0, ~1]`` (higher values
                over-steer past the residual's own magnitude; use at
                your own risk). ``alpha=0`` disables injection.

        Mutually exclusive with ``set_raw_correction`` and
        ``set_steering`` — the last ``set_*`` call wins.
        """
        import mlx.core as mx

        v_norm = mx.linalg.norm(v_probe)
        # Guard against zero-vector inputs; treat as "no steering".
        if float(v_norm.item()) == 0.0:
            self._probe_direction = None
            self._probe_alpha = 0.0
        else:
            self._probe_direction = v_probe / v_norm
            self._probe_alpha = float(alpha)
        # Mutually exclusive with other injection modes.
        self._raw_correction = None
        self._V_basis = None
        self._delta_alpha = None

    def set_band(self, weights: dict[int, float] | None) -> None:
        """Configure multi-layer injection via a pre-computed weight map.

        Args:
            weights: ``{layer_idx: weight}``. ``None`` restores the
                single-layer default (correction applies at the
                bottleneck only with weight 1.0 — bit-compatible with
                the pre-band hook). When provided, correction is
                spread across every named layer scaled by its weight.
                Callers typically produce this via
                ``tgirl.band.band_weights(...)``; the hook stays
                dependency-free per the cache.py zero-coupling
                invariant.
        """
        if weights is None:
            self._band_weights_by_id = None
            return
        self._band_weights_by_id = {
            id(self._layers[idx]): w for idx, w in weights.items()
        }

    def clear_steering(self) -> None:
        """Disable injection."""
        self._V_basis = None
        self._raw_correction = None
        self._delta_alpha = None
        self._probe_direction = None
        self._probe_alpha = 0.0


@dataclass
class CacheStats:
    """Tracks KV cache hit/miss/reset statistics."""

    hits: int = 0
    misses: int = 0
    resets: int = 0
    tokens_saved: int = 0


def make_mlx_forward_fn(
    model: Any,
    *,
    stats: CacheStats | None = None,
) -> Callable[[list[int]], Any]:
    """Create a cached forward function for MLX models (MLX-native).

    The returned closure maintains KV cache state between calls.
    When consecutive calls share a prefix, only new tokens are forwarded.

    Returns ``mx.array`` logits directly — no torch conversion.
    Use ``make_mlx_forward_fn_torch`` if you need ``torch.Tensor`` output.

    Args:
        model: MLX model with ``make_cache()`` and
            ``__call__(input_ids, cache=)`` interface.
        stats: Optional CacheStats to record hit/miss/reset/tokens_saved.

    Returns:
        A function ``(token_ids: list[int]) -> mx.array`` returning
        last-position logits of shape ``(vocab_size,)``.
    """
    import mlx.core as mx

    _cache: list[Any] = model.make_cache()
    _prev_tokens: list[int] = []
    _last_logits: Any = None

    def forward(token_ids: list[int]) -> Any:
        nonlocal _cache, _prev_tokens, _last_logits

        # Same tokens as last call — return cached logits
        if token_ids == _prev_tokens and _last_logits is not None:
            if stats is not None:
                stats.hits += 1
            return _last_logits

        # Check if this is a prefix continuation
        prev_len = len(_prev_tokens)
        is_continuation = (
            prev_len > 0
            and len(token_ids) > prev_len
            and token_ids[:prev_len] == _prev_tokens
        )

        if is_continuation:
            # Cache hit: only forward the new tokens
            new_tokens = token_ids[prev_len:]
            if stats is not None:
                stats.hits += 1
                stats.tokens_saved += prev_len

            input_ids = mx.array([new_tokens])
            mlx_logits = model(input_ids, cache=_cache)
        else:
            # Cache miss: reset and forward all tokens
            _cache = model.make_cache()
            if stats is not None:
                stats.misses += 1
                if _prev_tokens:
                    stats.resets += 1

            input_ids = mx.array([token_ids])
            mlx_logits = model(input_ids, cache=_cache)

        result = mlx_logits[0, -1, :].astype(mx.float32)
        mx.eval(result)

        _prev_tokens = list(token_ids)
        _last_logits = result
        return result

    return forward


def make_steerable_mlx_forward_fn(
    model: Any,
    bottleneck_layer: int,
    layer_path: str = "model.layers",
    *,
    stats: CacheStats | None = None,
) -> Callable[..., Any]:
    """Create a cached, steerable forward function for MLX models.

    Like ``make_mlx_forward_fn`` but supports optional activation-level
    probe and injection at the bottleneck layer via a steering argument.

    When called as ``fn(token_ids)`` (no steering), returns
    ``ForwardResult(logits, None)`` — same logits as the non-steerable variant.

    When called as ``fn(token_ids, steering=state)`` where ``state`` has
    ``.V_basis``, ``.delta_alpha``, and ``.bottleneck_layer`` attributes:
    probes the bottleneck activation and optionally injects a correction,
    returning ``ForwardResult(logits, probe_alpha)``.

    Args:
        model: MLX model with ``make_cache()`` and ``__call__`` interface.
        bottleneck_layer: Layer index for probe/injection.
        layer_path: Dot-separated path to the layers list (e.g.
            ``"language_model.model.layers"``).
        stats: Optional CacheStats for hit/miss tracking.

    Returns:
        ``(token_ids, steering=None) -> ForwardResult``
    """
    import mlx.core as mx

    # Navigate to layers and install hook
    layers = model
    for attr in layer_path.split("."):
        layers = getattr(layers, attr)

    hook = _BottleneckHook(layers, layer_idx=bottleneck_layer)
    hook.install()

    _cache: list[Any] = model.make_cache()
    _prev_tokens: list[int] = []
    _last_result: ForwardResult | None = None

    _mx_eval = mx.eval

    def forward(token_ids: list[int], *, steering: Any = None) -> ForwardResult:
        nonlocal _cache, _prev_tokens, _last_result

        # Same tokens as last call — return cached result
        if token_ids == _prev_tokens and _last_result is not None and steering is None:
            if stats is not None:
                stats.hits += 1
            return _last_result

        # Configure hook for this forward pass
        if steering is not None:
            hook.set_steering(steering.V_basis, steering.delta_alpha)
        else:
            hook.clear_steering()

        # Check prefix continuation
        prev_len = len(_prev_tokens)
        is_continuation = (
            prev_len > 0
            and len(token_ids) > prev_len
            and token_ids[:prev_len] == _prev_tokens
        )

        if is_continuation:
            new_tokens = token_ids[prev_len:]
            if stats is not None:
                stats.hits += 1
                stats.tokens_saved += prev_len
            input_ids = mx.array([new_tokens])
            mlx_logits = model(input_ids, cache=_cache)
        else:
            _cache = model.make_cache()
            if stats is not None:
                stats.misses += 1
                if _prev_tokens:
                    stats.resets += 1
            input_ids = mx.array([token_ids])
            mlx_logits = model(input_ids, cache=_cache)

        logits = mlx_logits[0, -1, :].astype(mx.float32)

        # Read probe if steering was active
        probe_alpha = None
        if steering is not None:
            probe_alpha = hook.get_probe(steering.V_basis)

        _mx_eval(logits)
        if probe_alpha is not None:
            _mx_eval(probe_alpha)

        result = ForwardResult(logits=logits, probe_alpha=probe_alpha)
        _prev_tokens = list(token_ids)
        _last_result = result
        return result

    return forward


def make_mlx_forward_fn_torch(
    model: Any,
    *,
    stats: CacheStats | None = None,
) -> Callable[[list[int]], torch.Tensor]:
    """Create a cached forward function for MLX models (torch output).

    Compatibility wrapper: same as ``make_mlx_forward_fn`` but converts
    the MLX output to ``torch.Tensor`` via numpy. Use this if your
    downstream code expects ``torch.Tensor`` logits.

    Args:
        model: MLX model with ``make_cache()`` and
            ``__call__(input_ids, cache=)`` interface.
        stats: Optional CacheStats to record hit/miss/reset/tokens_saved.

    Returns:
        A function ``(token_ids: list[int]) -> torch.Tensor`` returning
        last-position logits of shape ``(vocab_size,)``.
    """
    import mlx.core as mx
    import numpy as np

    _cache: list[Any] = model.make_cache()
    _prev_tokens: list[int] = []
    _last_logits: torch.Tensor | None = None

    _mx_eval = mx.eval

    def _mlx_to_torch(mlx_logits: Any) -> torch.Tensor:
        """Extract last-position logits from MLX output, convert to torch."""
        last = mlx_logits[0, -1, :].astype(mx.float32)
        _mx_eval(last)
        return torch.from_numpy(np.array(last, copy=False))

    def forward(token_ids: list[int]) -> torch.Tensor:
        nonlocal _cache, _prev_tokens, _last_logits

        # Same tokens as last call — return cached logits
        if token_ids == _prev_tokens and _last_logits is not None:
            if stats is not None:
                stats.hits += 1
            return _last_logits

        # Check if this is a prefix continuation
        prev_len = len(_prev_tokens)
        is_continuation = (
            prev_len > 0
            and len(token_ids) > prev_len
            and token_ids[:prev_len] == _prev_tokens
        )

        if is_continuation:
            # Cache hit: only forward the new tokens
            new_tokens = token_ids[prev_len:]
            if stats is not None:
                stats.hits += 1
                stats.tokens_saved += prev_len

            input_ids = mx.array([new_tokens])
            mlx_logits = model(input_ids, cache=_cache)
        else:
            # Cache miss: reset and forward all tokens
            _cache = model.make_cache()
            if stats is not None:
                stats.misses += 1
                if _prev_tokens:
                    stats.resets += 1

            input_ids = mx.array([token_ids])
            mlx_logits = model(input_ids, cache=_cache)

        result = _mlx_to_torch(mlx_logits)

        _prev_tokens = list(token_ids)
        _last_logits = result
        return result

    return forward


def make_hf_forward_fn(
    model: Any,
    *,
    device: str = "cpu",
    stats: CacheStats | None = None,
) -> Callable[[list[int]], torch.Tensor]:
    """Create a cached forward function for HuggingFace Transformers models.

    The returned closure maintains past_key_values state between calls.
    Uses the immutable cache pattern: stores the returned past_key_values
    from each forward call.

    Args:
        model: HuggingFace model with
            ``__call__(input_ids, past_key_values=, use_cache=True)``
            returning an object with ``.logits`` and ``.past_key_values``.
        device: Device for input tensors (default "cpu").
        stats: Optional CacheStats to record hit/miss/reset/tokens_saved.

    Returns:
        A function ``(token_ids: list[int]) -> torch.Tensor`` returning
        last-position logits of shape ``(vocab_size,)``.
    """
    _past_key_values: Any = None
    _prev_tokens: list[int] = []
    _last_logits: torch.Tensor | None = None

    def forward(token_ids: list[int]) -> torch.Tensor:
        nonlocal _past_key_values, _prev_tokens, _last_logits

        # Same tokens as last call — return cached logits
        if token_ids == _prev_tokens and _last_logits is not None:
            if stats is not None:
                stats.hits += 1
            return _last_logits

        # Check if this is a prefix continuation
        prev_len = len(_prev_tokens)
        is_continuation = (
            prev_len > 0
            and len(token_ids) > prev_len
            and token_ids[:prev_len] == _prev_tokens
        )

        if is_continuation:
            # Cache hit: only forward new tokens with past_key_values
            new_tokens = token_ids[prev_len:]
            if stats is not None:
                stats.hits += 1
                stats.tokens_saved += prev_len

            input_ids = torch.tensor([new_tokens], device=device)
            output = model(
                input_ids,
                past_key_values=_past_key_values,
                use_cache=True,
            )
        else:
            # Cache miss: reset and forward all tokens
            if stats is not None:
                stats.misses += 1
                if _prev_tokens:
                    stats.resets += 1

            _past_key_values = None
            input_ids = torch.tensor([token_ids], device=device)
            output = model(
                input_ids,
                past_key_values=None,
                use_cache=True,
            )

        # Store new past_key_values (immutable pattern)
        _past_key_values = output.past_key_values

        # Extract last-position logits
        logits_out: torch.Tensor = output.logits
        if logits_out.dim() == 3:
            result = logits_out[0, -1, :]
        elif logits_out.dim() == 2:
            result = logits_out[-1, :]
        else:
            result = logits_out

        _prev_tokens = list(token_ids)
        _last_logits = result
        return result

    return forward
