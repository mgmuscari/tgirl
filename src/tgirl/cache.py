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
        self._V_basis: Any = None  # (d_model, K) for injection
        self._delta_alpha: Any = None  # (K,) correction to inject
        self._installed = False

    def install(self) -> None:
        """Patch the layer type's __call__."""
        if self._installed:
            return
        self._original_call = self._layer_type.__call__
        hook = self

        def _patched(layer_self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
            result = hook._original_call(layer_self, x, *args, **kwargs)
            if layer_self is hook._target:
                import mlx.core as mx

                # Capture last-token activation (float32)
                hook._captured = result[:, -1, :].astype(mx.float32).reshape(-1)

                # Inject if steering is configured
                if hook._V_basis is not None and hook._delta_alpha is not None:
                    correction = hook._V_basis @ hook._delta_alpha  # (d_model,)
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

    def clear_steering(self) -> None:
        """Disable injection."""
        self._V_basis = None
        self._delta_alpha = None


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
