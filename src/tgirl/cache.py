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
from typing import Any

import torch


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

        _prev_tokens = list(token_ids)
        _last_logits = result
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
