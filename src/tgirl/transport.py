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


def _compute_cost_submatrix(
    embeddings: torch.Tensor,
    invalid_indices: torch.Tensor,
    valid_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine distance submatrix between invalid and valid tokens.

    Cost = 1 - cosine_similarity. Only allocates the (n_invalid, n_valid)
    submatrix, never the full V x V matrix.

    Returns tensor of shape (n_invalid, n_valid) with values in [0, 2].
    """
    invalid_emb = torch.nn.functional.normalize(
        embeddings[invalid_indices], dim=-1
    )
    valid_emb = torch.nn.functional.normalize(
        embeddings[valid_indices], dim=-1
    )
    similarity = invalid_emb @ valid_emb.T
    return 1.0 - similarity


def _sinkhorn_log_domain(
    cost_matrix: torch.Tensor,
    source_mass: torch.Tensor,
    target_capacity: torch.Tensor,
    epsilon: float,
    max_iterations: int,
    convergence_threshold: float,
) -> tuple[torch.Tensor, float, int]:
    """Log-domain Sinkhorn algorithm for optimal transport.

    Uses log-domain computation for numerical stability:
    log_kernel = -cost / epsilon, with logsumexp for scaling updates.

    Args:
        cost_matrix: (n_source, n_target) cost matrix.
        source_mass: (n_source,) source probability distribution.
        target_capacity: (n_target,) target probability distribution.
        epsilon: Entropic regularization parameter.
        max_iterations: Maximum Sinkhorn iterations.
        convergence_threshold: Convergence on marginal error.

    Returns:
        (transport_plan, wasserstein_distance, iterations)
    """
    n_source, n_target = cost_matrix.shape
    log_kernel = -cost_matrix / epsilon

    # Initialize log scaling vectors
    log_u = torch.zeros(n_source, dtype=cost_matrix.dtype)
    log_v = torch.zeros(n_target, dtype=cost_matrix.dtype)

    log_source = torch.log(source_mass)
    log_target = torch.log(target_capacity)

    iterations = 0
    for i in range(max_iterations):
        iterations = i + 1

        # Update log_u: row scaling
        log_u = log_source - torch.logsumexp(log_kernel + log_v[None, :], dim=1)

        # Update log_v: column scaling
        log_v = log_target - torch.logsumexp(log_kernel + log_u[:, None], dim=0)

        # Check convergence: marginal error
        log_plan_row_sums = torch.logsumexp(
            log_kernel + log_u[:, None] + log_v[None, :], dim=1
        )
        marginal_error = (
            torch.exp(log_plan_row_sums) - source_mass
        ).abs().max().item()

        if marginal_error < convergence_threshold:
            break

    # Compute final plan
    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = torch.exp(log_plan)

    # Wasserstein distance = (plan * cost).sum()
    wasserstein = (plan * cost_matrix).sum().item()

    return plan, wasserstein, iterations


def _apply_transport_plan(
    plan: torch.Tensor,
    valid_indices: torch.Tensor,
    original_logits: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Apply transport plan to produce redistributed log-space logits.

    Steps:
    1. Sum plan columns -> redistributed probability per valid token
    2. Add existing valid probabilities from original logits
    3. Convert to log-space
    4. Set invalid tokens to -inf

    Args:
        plan: (n_invalid, n_valid) transport plan matrix.
        valid_indices: indices of valid tokens.
        original_logits: original logit tensor (full vocab).
        vocab_size: total vocabulary size.

    Returns:
        Log-space logits tensor of shape (vocab_size,).
    """
    # Get original probabilities for valid tokens
    original_probs = torch.softmax(original_logits, dim=-1)
    valid_probs = original_probs[valid_indices]

    # Sum plan columns: mass redistributed to each valid token
    redistributed = plan.sum(dim=0)

    # Combined probability = original valid + redistributed
    combined_probs = valid_probs + redistributed

    # Build output: -inf everywhere, then fill valid positions
    result = torch.full((vocab_size,), float("-inf"), dtype=original_logits.dtype)
    result[valid_indices] = torch.log(combined_probs)

    return result
