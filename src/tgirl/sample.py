"""Constrained sampling engine for dual-mode inference.

Integrates grammar constraints, optimal transport redistribution,
and Hy pipeline execution into a dual-mode sampling loop.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol, runtime_checkable

import structlog
import torch

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
