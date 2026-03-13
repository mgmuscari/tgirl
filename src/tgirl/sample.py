"""Constrained sampling engine for dual-mode inference.

Integrates grammar constraints, optimal transport redistribution,
and Hy pipeline execution into a dual-mode sampling loop.
"""

from __future__ import annotations

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
