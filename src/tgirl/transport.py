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
