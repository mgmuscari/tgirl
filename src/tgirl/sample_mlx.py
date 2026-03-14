"""MLX-native sampling functions for constrained generation.

Ports apply_penalties, apply_shaping, and run_constrained_generation
from sample.py to MLX, keeping all per-token math in mx.array.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import structlog
import torch

from tgirl.sample import (
    ConstrainedGenerationResult,
    GrammarState,
    InferenceHook,
    merge_interventions,
)
from tgirl.transport import TransportConfig
from tgirl.transport_mlx import redistribute_logits_mlx
from tgirl.types import ModelIntervention

logger = structlog.get_logger()


def apply_penalties_mlx(
    logits: mx.array,
    intervention: ModelIntervention,
    token_history: list[int],
) -> mx.array:
    """Pre-OT: apply repetition, presence, frequency penalties and logit bias.

    MLX-native version of sample.apply_penalties.
    """
    result_np = np.array(logits, copy=True)

    # Repetition penalty
    if (
        intervention.repetition_penalty is not None
        and intervention.repetition_penalty != 1.0
    ):
        for token_id in set(token_history):
            if result_np[token_id] > 0:
                result_np[token_id] /= intervention.repetition_penalty
            else:
                result_np[token_id] *= intervention.repetition_penalty

    # Presence penalty
    if (
        intervention.presence_penalty is not None
        and intervention.presence_penalty != 0.0
    ):
        for token_id in set(token_history):
            result_np[token_id] -= intervention.presence_penalty

    # Frequency penalty
    if (
        intervention.frequency_penalty is not None
        and intervention.frequency_penalty != 0.0
    ):
        counts = Counter(token_history)
        for token_id, count in counts.items():
            result_np[token_id] -= intervention.frequency_penalty * count

    # Logit bias
    if intervention.logit_bias is not None:
        for token_id, bias in intervention.logit_bias.items():
            result_np[token_id] += bias

    return mx.array(result_np)


def apply_shaping_mlx(
    logits: mx.array,
    intervention: ModelIntervention,
) -> mx.array:
    """Post-OT: apply temperature, top-k, top-p to redistributed logits."""
    result = logits

    # Temperature
    if intervention.temperature is not None and intervention.temperature > 0:
        result = result / intervention.temperature
    elif (
        intervention.temperature is not None
        and intervention.temperature == 0
    ):
        # Greedy: set all but max to -inf
        max_idx = int(mx.argmax(result).item())
        result_np = np.full(result.shape, float("-inf"), dtype=np.float32)
        result_np[max_idx] = float(result[max_idx].item())
        result = mx.array(result_np)

    # Top-k
    if intervention.top_k is not None and intervention.top_k > 0:
        k = min(intervention.top_k, result.shape[-1])
        # Get top-k values to find threshold
        top_vals = mx.topk(result, k)
        threshold = float(mx.min(top_vals).item())
        result = mx.where(result >= threshold, result, float("-inf"))

    # Top-p (nucleus)
    if intervention.top_p is not None and intervention.top_p < 1.0:
        result_np = np.array(result)
        sorted_indices = np.argsort(-result_np)
        sorted_logits = result_np[sorted_indices]
        probs = np.exp(sorted_logits - sorted_logits.max())
        probs = probs / probs.sum()
        cumulative = np.cumsum(probs)
        cutoff = cumulative - probs > intervention.top_p
        sorted_logits[cutoff] = float("-inf")
        # Unsort
        output = np.empty_like(sorted_logits)
        output[sorted_indices] = sorted_logits
        result = mx.array(output)

    return result


def run_constrained_generation_mlx(
    grammar_state: GrammarState,
    forward_fn: Callable[[list[int]], mx.array],
    tokenizer_decode: Callable[[list[int]], str],
    embeddings: mx.array,
    hooks: list[InferenceHook],
    transport_config: TransportConfig,
    max_tokens: int = 512,
    context_tokens: list[int] | None = None,
) -> ConstrainedGenerationResult:
    """Run constrained token generation until grammar accepts or max_tokens.

    MLX-native version: all per-token math stays in mx.array.
    Grammar mask converted from torch once per token. Hook calls receive
    torch tensors (hooks are user-provided and may depend on torch).
    """
    start_time = time.monotonic()
    tokens: list[int] = []
    token_history = list(context_tokens) if context_tokens else []
    grammar_valid_counts: list[int] = []
    temperatures_applied: list[float] = []
    wasserstein_distances: list[float] = []
    top_p_applied: list[float] = []
    token_log_probs: list[float] = []
    ot_computation_total_ms = 0.0
    ot_bypassed_count = 0

    vocab_size = embeddings.shape[0]

    for position in range(max_tokens):
        # 1. Forward pass (returns mx.array)
        raw_logits = forward_fn(token_history)

        # 2. Grammar mask (torch -> numpy -> mx)
        valid_mask_torch = grammar_state.get_valid_mask(vocab_size)
        valid_mask_np = valid_mask_torch.numpy()
        valid_mask_mx = mx.array(valid_mask_np)
        valid_count = int(valid_mask_np.sum())
        grammar_valid_counts.append(valid_count)

        # 3. Hooks (convert to torch for hook interface)
        if hooks:
            raw_logits_torch = torch.from_numpy(np.array(raw_logits))
            interventions = [
                hook.pre_forward(
                    position, grammar_state, token_history, raw_logits_torch
                )
                for hook in hooks
            ]
            merged = merge_interventions(interventions)
        else:
            merged = ModelIntervention()

        # 4. Pre-OT penalties
        adjusted = apply_penalties_mlx(raw_logits, merged, token_history)

        # 5. OT redistribution
        ot_start = time.monotonic()
        ot_result = redistribute_logits_mlx(
            adjusted, valid_mask_mx, embeddings, config=transport_config
        )
        ot_elapsed_ms = (time.monotonic() - ot_start) * 1000
        ot_computation_total_ms += ot_elapsed_ms
        wasserstein_distances.append(ot_result.wasserstein_distance)
        if ot_result.bypassed:
            ot_bypassed_count += 1

        # 6. Post-OT shaping
        shaped = apply_shaping_mlx(ot_result.logits, merged)

        # Record temperature and top_p applied
        temperatures_applied.append(
            merged.temperature
            if merged.temperature is not None
            else -1.0
        )
        top_p_applied.append(
            merged.top_p if merged.top_p is not None else -1.0
        )

        # 7. Sample token (mx.random.categorical takes logits)
        token_id = int(mx.random.categorical(shaped).item())
        tokens.append(token_id)
        token_history.append(token_id)

        # Record log prob
        probs = mx.softmax(shaped, axis=-1)
        log_prob = (
            float(mx.log(probs[token_id]).item())
            if float(probs[token_id].item()) > 0
            else float("-inf")
        )
        token_log_probs.append(log_prob)

        # 8. Advance grammar
        grammar_state.advance(token_id)

        # Check if grammar accepts
        if grammar_state.is_accepting():
            break

    elapsed_ms = (time.monotonic() - start_time) * 1000
    hy_source = tokenizer_decode(tokens)

    return ConstrainedGenerationResult(
        tokens=tokens,
        hy_source=hy_source,
        grammar_valid_counts=grammar_valid_counts,
        temperatures_applied=temperatures_applied,
        wasserstein_distances=wasserstein_distances,
        top_p_applied=top_p_applied,
        token_log_probs=token_log_probs,
        ot_computation_total_ms=ot_computation_total_ms,
        ot_bypassed_count=ot_bypassed_count,
        grammar_generation_ms=elapsed_ms,
    )
