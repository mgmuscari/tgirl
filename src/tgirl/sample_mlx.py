"""MLX-native sampling functions for constrained generation.

All per-token math stays in mx.array. Zero torch, zero numpy in the
hot loop. Only conversion: int() on token ID scalar (unavoidable).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import mlx.core as mx
import structlog

from tgirl.sample import (
    ConstrainedGenerationResult,
    merge_interventions,
)
from tgirl.transport import TransportConfig
from tgirl.transport_mlx import redistribute_logits_mlx
from tgirl.types import ModelIntervention

logger = structlog.get_logger()


@runtime_checkable
class GrammarStateMlx(Protocol):
    """Protocol for MLX-native grammar state trackers."""

    def get_valid_mask_mx(self, tokenizer_vocab_size: int) -> mx.array: ...
    def is_accepting(self) -> bool: ...
    def advance(self, token_id: int) -> None: ...


@runtime_checkable
class InferenceHookMlx(Protocol):
    """Protocol for MLX-native per-token inference hooks.

    Unlike InferenceHook (torch), receives the pre-computed valid_mask
    as mx.array rather than the grammar state object. This avoids
    re-computing the mask inside each hook.
    """

    def pre_forward(
        self,
        position: int,
        valid_mask: mx.array,
        token_history: list[int],
        logits: mx.array,
    ) -> ModelIntervention: ...


class GrammarTemperatureHookMlx:
    """MLX-native grammar-implied temperature scheduling.

    Receives the valid_mask as a parameter (already computed by the main
    loop), avoiding redundant grammar state queries.
    """

    def __init__(
        self, base_temperature: float = 0.3, scaling_exponent: float = 0.5
    ) -> None:
        self.base_temperature = base_temperature
        self.scaling_exponent = scaling_exponent

    def pre_forward(
        self,
        position: int,
        valid_mask: mx.array,
        token_history: list[int],
        logits: mx.array,
    ) -> ModelIntervention:
        vocab_size = logits.shape[-1]
        valid_count = int(mx.sum(valid_mask).item())
        if valid_count <= 1:
            return ModelIntervention(temperature=0.0)
        freedom = valid_count / vocab_size
        temp = self.base_temperature * (freedom**self.scaling_exponent)
        return ModelIntervention(temperature=temp)


def apply_penalties_mlx(
    logits: mx.array,
    intervention: ModelIntervention,
    token_history: list[int],
) -> mx.array:
    """Pre-OT: apply repetition, presence, frequency penalties and logit bias.

    Pure MLX — uses functional scatter via .at[indices].add(deltas).
    Index/value construction from small Python collections (tens of elements).
    """
    result = logits

    # Repetition penalty
    if (
        intervention.repetition_penalty is not None
        and intervention.repetition_penalty != 1.0
    ):
        unique_ids = list(set(token_history))
        if unique_ids:
            indices = mx.array(unique_ids)
            vals = result[indices]
            pos_result = vals / intervention.repetition_penalty
            neg_result = vals * intervention.repetition_penalty
            penalized = mx.where(vals > 0, pos_result, neg_result)
            result = result.at[indices].add(penalized - vals)

    # Presence penalty
    if (
        intervention.presence_penalty is not None
        and intervention.presence_penalty != 0.0
    ):
        unique_ids = list(set(token_history))
        if unique_ids:
            indices = mx.array(unique_ids)
            deltas = mx.full((len(unique_ids),), -intervention.presence_penalty)
            result = result.at[indices].add(deltas)

    # Frequency penalty
    if (
        intervention.frequency_penalty is not None
        and intervention.frequency_penalty != 0.0
    ):
        counts = Counter(token_history)
        if counts:
            ids = list(counts.keys())
            freq_deltas = [-intervention.frequency_penalty * counts[tid] for tid in ids]
            indices = mx.array(ids)
            deltas = mx.array(freq_deltas)
            result = result.at[indices].add(deltas)

    # Logit bias
    if intervention.logit_bias is not None:
        ids = list(intervention.logit_bias.keys())
        if ids:
            biases = [intervention.logit_bias[tid] for tid in ids]
            indices = mx.array(ids)
            deltas = mx.array(biases)
            result = result.at[indices].add(deltas)

    return result


def apply_shaping_mlx(
    logits: mx.array,
    intervention: ModelIntervention,
) -> mx.array:
    """Post-OT: apply temperature, top-k, top-p to redistributed logits.

    Pure MLX — no numpy conversions.
    """
    result = logits

    # Temperature
    if intervention.temperature is not None and intervention.temperature > 0:
        result = result / intervention.temperature
    elif (
        intervention.temperature is not None
        and intervention.temperature == 0
    ):
        # Greedy: keep only max value(s), set rest to -inf
        max_val = mx.max(result)
        result = mx.where(result >= max_val, result, mx.array(float("-inf")))

    # Top-k
    if intervention.top_k is not None and intervention.top_k > 0:
        k = min(intervention.top_k, result.shape[-1])
        top_vals = mx.topk(result, k)
        threshold = mx.min(top_vals)
        result = mx.where(result >= threshold, result, mx.array(float("-inf")))

    # Top-p (nucleus) — pure MLX
    if intervention.top_p is not None and intervention.top_p < 1.0:
        sorted_indices = mx.argsort(-result)
        sorted_logits = result[sorted_indices]
        probs = mx.softmax(sorted_logits, axis=-1)
        cumulative = mx.cumsum(probs)
        cutoff = (cumulative - probs) > intervention.top_p
        sorted_logits = mx.where(cutoff, mx.array(float("-inf")), sorted_logits)
        # Unsort: scatter back to original positions
        unsort_indices = mx.argsort(sorted_indices)
        result = sorted_logits[unsort_indices]

    return result


def run_constrained_generation_mlx(
    grammar_state: GrammarStateMlx,
    forward_fn: Callable[[list[int]], mx.array],
    tokenizer_decode: Callable[[list[int]], str],
    embeddings: mx.array,
    hooks: list[InferenceHookMlx],
    transport_config: TransportConfig,
    max_tokens: int = 512,
    context_tokens: list[int] | None = None,
) -> ConstrainedGenerationResult:
    """Run constrained token generation until grammar accepts or max_tokens.

    Fully MLX-native loop. Zero torch, zero numpy in computation.
    Only Python scalar extraction: int() on token ID.
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

    # Detect grammar state type once
    has_mlx_mask = hasattr(grammar_state, "get_valid_mask_mx")

    for position in range(max_tokens):
        _t0 = time.monotonic()

        # 1. Forward pass (returns mx.array)
        raw_logits = forward_fn(token_history)
        _t1 = time.monotonic()

        # 2. Grammar mask — pure MLX via llguidance.mlx (or fallback)
        if has_mlx_mask:
            valid_mask = grammar_state.get_valid_mask_mx(vocab_size)
        else:
            # Fallback for torch-based grammar states
            valid_mask_torch = grammar_state.get_valid_mask(vocab_size)
            valid_mask = mx.array(valid_mask_torch.numpy())
        valid_count = int(mx.sum(valid_mask).item())
        grammar_valid_counts.append(valid_count)
        _t2 = time.monotonic()

        # 3. Hooks — pure MLX, receive mx.array + pre-computed mask
        if hooks:
            interventions = [
                hook.pre_forward(position, valid_mask, token_history, raw_logits)
                for hook in hooks
            ]
            merged = merge_interventions(interventions)
        else:
            merged = ModelIntervention()
        _t3 = time.monotonic()

        # 4. Pre-OT penalties — pure MLX scatter
        adjusted = apply_penalties_mlx(raw_logits, merged, token_history)
        _t4 = time.monotonic()

        # 5. OT redistribution — pure MLX
        ot_start = time.monotonic()
        ot_result = redistribute_logits_mlx(
            adjusted, valid_mask, embeddings, config=transport_config
        )
        ot_elapsed_ms = (time.monotonic() - ot_start) * 1000
        ot_computation_total_ms += ot_elapsed_ms
        wasserstein_distances.append(ot_result.wasserstein_distance)
        if ot_result.bypassed:
            ot_bypassed_count += 1
        _t5 = time.monotonic()

        # 6. Post-OT shaping — pure MLX
        shaped = apply_shaping_mlx(ot_result.logits, merged)
        _t6 = time.monotonic()

        # Record temperature and top_p applied
        temperatures_applied.append(
            merged.temperature
            if merged.temperature is not None
            else -1.0
        )
        top_p_applied.append(
            merged.top_p if merged.top_p is not None else -1.0
        )

        # 7. Sample token — materialize graph, then sample
        mx.eval(shaped)
        probs_check = mx.softmax(shaped, axis=-1)
        prob_sum = float(mx.sum(probs_check).item())
        if prob_sum > 0:
            token_id = int(mx.random.categorical(shaped).item())
        else:
            # Fallback: uniform over valid tokens
            uniform = valid_mask.astype(mx.float32)
            uniform = uniform / mx.sum(uniform)
            token_id = int(
                mx.random.categorical(mx.log(uniform)).item()
            )
        tokens.append(token_id)
        token_history.append(token_id)
        _t7 = time.monotonic()

        # Record log prob
        probs = mx.softmax(shaped, axis=-1)
        log_prob = (
            float(mx.log(probs[token_id]).item())
            if float(probs[token_id].item()) > 0
            else float("-inf")
        )
        token_log_probs.append(log_prob)
        _t8 = time.monotonic()

        # 8. Advance grammar
        grammar_state.advance(token_id)
        _t9 = time.monotonic()

        if position < 5 or position % 20 == 0:
            logger.info(
                "constrained_mlx_timing",
                pos=position,
                forward_ms=round((_t1 - _t0) * 1000, 1),
                grammar_ms=round((_t2 - _t1) * 1000, 1),
                hooks_ms=round((_t3 - _t2) * 1000, 1),
                penalties_ms=round((_t4 - _t3) * 1000, 1),
                ot_ms=round((_t5 - _t4) * 1000, 1),
                shaping_ms=round((_t6 - _t5) * 1000, 1),
                sample_ms=round((_t7 - _t6) * 1000, 1),
                logprob_ms=round((_t8 - _t7) * 1000, 1),
                advance_ms=round((_t9 - _t8) * 1000, 1),
                total_ms=round((_t9 - _t0) * 1000, 1),
            )

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
