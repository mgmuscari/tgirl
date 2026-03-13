"""Optimal transport logit redistribution for constrained generation.

Redistributes probability mass from invalid to valid tokens using optimal
transport, with cost defined by semantic distance in embedding space.
The model's intent is preserved as much as the grammar allows.

This module has ZERO coupling to any other tgirl module — it operates
on raw logit tensors and valid token masks only.
"""

from __future__ import annotations

from typing import NamedTuple

import structlog
import torch
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger()


class TransportConfig(BaseModel):
    """Configuration for optimal transport logit redistribution."""

    model_config = ConfigDict(frozen=True)

    epsilon: float = 0.1
    max_iterations: int = 20
    convergence_threshold: float = 1e-6
    valid_ratio_threshold: float = 0.5
    invalid_mass_threshold: float = 0.01


class TransportResult(NamedTuple):
    """Result of logit redistribution.

    Unpacks as 2-tuple (logits, wasserstein_distance) for spec compatibility.
    Named access available for all fields.
    """

    logits: torch.Tensor
    wasserstein_distance: float
    bypassed: bool
    bypass_reason: str | None
    iterations: int


def _check_bypass(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    config: TransportConfig,
) -> tuple[bool, str | None]:
    """Check whether optimal transport should be bypassed.

    Three conditions checked in priority order:
    1. valid_mask.sum() <= 1 -> "forced_decode"
    2. valid ratio > threshold -> "valid_ratio_high"
    3. invalid probability mass < threshold -> "invalid_mass_low"

    Returns (should_bypass, reason) where reason is None if no bypass.
    """
    n_valid = valid_mask.sum().item()

    # Condition 1: forced decode (0 or 1 valid token)
    if n_valid <= 1:
        return True, "forced_decode"

    # Condition 2: high valid ratio
    valid_ratio = valid_mask.float().mean().item()
    if valid_ratio > config.valid_ratio_threshold:
        return True, "valid_ratio_high"

    # Condition 3: low invalid mass
    probs = torch.softmax(logits, dim=-1)
    invalid_mass = probs[~valid_mask].sum().item()
    if invalid_mass < config.invalid_mass_threshold:
        return True, "invalid_mass_low"

    return False, None


def _standard_masking(
    logits: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Set invalid logits to -inf, preserving valid logits unchanged.

    This is the bypass fast path used when optimal transport is skipped.
    Does not mutate the input tensor.
    """
    result = logits.clone()
    result[~valid_mask] = float("-inf")
    return result
