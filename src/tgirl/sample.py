"""Constrained sampling engine for dual-mode inference.

Integrates grammar constraints, optimal transport redistribution,
and Hy pipeline execution into a dual-mode sampling loop.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import structlog
import torch
from pydantic import BaseModel, ConfigDict

from tgirl.transport import TransportConfig, redistribute_logits
from tgirl.types import ModelIntervention

logger = structlog.get_logger()


@runtime_checkable
class GrammarState(Protocol):
    """Protocol for grammar state trackers."""

    def get_valid_mask(self, tokenizer_vocab_size: int) -> torch.Tensor: ...
    def is_accepting(self) -> bool: ...
    def advance(self, token_id: int) -> None: ...


@runtime_checkable
class InferenceHook(Protocol):
    """Protocol for per-token inference hooks."""

    def pre_forward(
        self,
        position: int,
        grammar_state: GrammarState,
        token_history: list[int],
        logits: torch.Tensor,
    ) -> ModelIntervention: ...


def merge_interventions(interventions: list[ModelIntervention]) -> ModelIntervention:
    """Merge multiple hook interventions. Last non-None value wins per field."""
    merged: dict[str, object] = {}
    for intervention in interventions:
        for field_name in ModelIntervention.model_fields:
            val = getattr(intervention, field_name)
            if val is not None:
                merged[field_name] = val
    return ModelIntervention(**merged)


class GrammarTemperatureHook:
    """Default hook: grammar-implied temperature scheduling.

    Adjusts temperature based on the fraction of valid tokens (freedom).
    When few tokens are valid, temperature is low (more deterministic).
    When many tokens are valid, temperature approaches base_temperature.
    """

    def __init__(
        self, base_temperature: float = 0.3, scaling_exponent: float = 0.5
    ) -> None:
        self.base_temperature = base_temperature
        self.scaling_exponent = scaling_exponent

    def pre_forward(
        self,
        position: int,
        grammar_state: GrammarState,
        token_history: list[int],
        logits: torch.Tensor,
    ) -> ModelIntervention:
        vocab_size = logits.shape[-1]
        valid_mask = grammar_state.get_valid_mask(vocab_size)
        valid_count = valid_mask.sum().item()
        if valid_count <= 1:
            return ModelIntervention(temperature=0.0)
        freedom = valid_count / vocab_size
        temp = self.base_temperature * (freedom**self.scaling_exponent)
        return ModelIntervention(temperature=temp)


def apply_penalties(
    logits: torch.Tensor,
    intervention: ModelIntervention,
    token_history: list[int],
) -> torch.Tensor:
    """Pre-OT: apply repetition, presence, frequency penalties and logit bias.

    These modify the model's original logit distribution before OT redistribution,
    so that OT operates on a penalty-adjusted signal.
    """
    result = logits.clone()

    # Repetition penalty
    if (
        intervention.repetition_penalty is not None
        and intervention.repetition_penalty != 1.0
    ):
        for token_id in set(token_history):
            if result[token_id] > 0:
                result[token_id] /= intervention.repetition_penalty
            else:
                result[token_id] *= intervention.repetition_penalty

    # Presence penalty
    if (
        intervention.presence_penalty is not None
        and intervention.presence_penalty != 0.0
    ):
        for token_id in set(token_history):
            result[token_id] -= intervention.presence_penalty

    # Frequency penalty
    if (
        intervention.frequency_penalty is not None
        and intervention.frequency_penalty != 0.0
    ):
        counts = Counter(token_history)
        for token_id, count in counts.items():
            result[token_id] -= intervention.frequency_penalty * count

    # Logit bias
    if intervention.logit_bias is not None:
        for token_id, bias in intervention.logit_bias.items():
            result[token_id] += bias

    return result


def apply_shaping(
    logits: torch.Tensor,
    intervention: ModelIntervention,
) -> torch.Tensor:
    """Post-OT: apply temperature, top-k, top-p to redistributed logits."""
    result = logits.clone()

    # Temperature
    if intervention.temperature is not None and intervention.temperature > 0:
        result = result / intervention.temperature
    elif intervention.temperature is not None and intervention.temperature == 0:
        # Greedy: set all but max to -inf. torch.argmax returns the first
        # (lowest-index) maximum, which is deterministic and consistent.
        max_idx = result.argmax()
        mask = torch.ones_like(result, dtype=torch.bool)
        mask[max_idx] = False
        result[mask] = float("-inf")

    # Top-k
    if intervention.top_k is not None and intervention.top_k > 0:
        top_k_vals, _ = torch.topk(result, min(intervention.top_k, result.shape[-1]))
        threshold = top_k_vals[-1]
        result[result < threshold] = float("-inf")

    # Top-p (nucleus)
    if intervention.top_p is not None and intervention.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(result, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff_mask = cumulative - probs > intervention.top_p
        sorted_logits[cutoff_mask] = float("-inf")
        # Unsort: scatter modified sorted values back to original positions
        result = torch.empty_like(sorted_logits)
        result.scatter_(0, sorted_indices, sorted_logits)

    return result


class ConstrainedGenerationResult(BaseModel):
    """Result of a constrained generation pass."""

    model_config = ConfigDict(frozen=True)
    tokens: list[int]
    hy_source: str
    grammar_valid_counts: list[int]
    temperatures_applied: list[float]
    wasserstein_distances: list[float]
    top_p_applied: list[float]
    token_log_probs: list[float]
    ot_computation_total_ms: float
    ot_bypassed_count: int
    grammar_generation_ms: float


def run_constrained_generation(
    grammar_state: GrammarState,
    forward_fn: Callable[[list[int]], torch.Tensor],
    tokenizer_decode: Callable[[list[int]], str],
    embeddings: torch.Tensor,
    hooks: list[InferenceHook],
    transport_config: TransportConfig,
    max_tokens: int = 512,
    context_tokens: list[int] | None = None,
) -> ConstrainedGenerationResult:
    """Run constrained token generation until grammar accepts or max_tokens.

    Per-token processing order (TGIRL.md 8.5, penalties moved pre-OT):
    1. forward_fn(context_tokens) -> raw logits
    2. grammar_state.get_valid_mask(vocab_size) -> valid_mask
    3. call all InferenceHooks -> merge_interventions() -> merged
    4. apply_penalties(logits, intervention, token_history) -> adjusted
    5. redistribute_logits(adjusted, valid_mask, embeddings, config) -> OT
    6. apply_shaping(ot_logits, intervention) -> shaped logits
    7. sample token from shaped logits
    8. grammar_state.advance(token_id)
    9. record telemetry
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
        # 1. Forward pass
        raw_logits = forward_fn(token_history)

        # 2. Grammar mask
        valid_mask = grammar_state.get_valid_mask(vocab_size)
        valid_count = int(valid_mask.sum().item())
        grammar_valid_counts.append(valid_count)

        # 3. Hooks
        interventions = [
            hook.pre_forward(position, grammar_state, token_history, raw_logits)
            for hook in hooks
        ]
        merged = merge_interventions(interventions)

        # 4. Pre-OT penalties
        adjusted = apply_penalties(raw_logits, merged, token_history)

        # 5. OT redistribution
        ot_start = time.monotonic()
        ot_result = redistribute_logits(
            adjusted, valid_mask, embeddings, config=transport_config
        )
        ot_elapsed_ms = (time.monotonic() - ot_start) * 1000
        ot_computation_total_ms += ot_elapsed_ms
        wasserstein_distances.append(ot_result.wasserstein_distance)
        if ot_result.bypassed:
            ot_bypassed_count += 1

        # 6. Post-OT shaping
        shaped = apply_shaping(ot_result.logits, merged)

        # Record temperature and top_p applied
        temperatures_applied.append(
            merged.temperature if merged.temperature is not None else -1.0
        )
        top_p_applied.append(
            merged.top_p if merged.top_p is not None else -1.0
        )

        # 7. Sample token
        probs = torch.softmax(shaped, dim=-1)
        probs = torch.clamp(probs, min=0.0)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            # Fallback: uniform over valid tokens
            probs = valid_mask.float()
            probs = probs / probs.sum()

        token_id = int(torch.multinomial(probs, 1).item())
        tokens.append(token_id)
        token_history.append(token_id)

        # Record log prob
        log_prob = (
            torch.log(probs[token_id]).item()
            if probs[token_id] > 0
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
