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
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger()


class TransportConfig(BaseModel):
    """Configuration for optimal transport logit redistribution."""

    model_config = ConfigDict(frozen=True)

    epsilon: float = Field(default=0.1, gt=0)
    max_iterations: int = Field(default=20, ge=1, le=1000)
    convergence_threshold: float = Field(default=1e-6, gt=0)
    valid_ratio_threshold: float = Field(default=0.5, ge=0, le=1)
    invalid_mass_threshold: float = Field(default=0.01, ge=0, le=1)
    max_problem_size: int = Field(default=1_000_000, ge=1)


class TransportResult(NamedTuple):
    """Result of logit redistribution.

    First two positional fields are logits and wasserstein_distance
    for spec compatibility. Named access available for all fields.
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


def redistribute_logits(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    embeddings: torch.Tensor,
    epsilon: float = 0.1,
    max_iterations: int = 20,
    convergence_threshold: float = 1e-6,
    config: TransportConfig | None = None,
    reachable_tokens: frozenset[int] | None = None,
) -> TransportResult:
    """Redistribute probability mass from invalid to valid tokens using OT.

    Accepts either individual parameters or a TransportConfig object.
    Config takes precedence over individual params when provided.

    Args:
        logits: Raw logit tensor of shape (vocab_size,).
        valid_mask: Boolean mask of valid tokens.
        embeddings: Token embedding matrix of shape (vocab_size, embed_dim).
        epsilon: Entropic regularization (ignored if config provided).
        max_iterations: Max Sinkhorn iterations (ignored if config provided).
        convergence_threshold: Convergence threshold (ignored if config provided).
        config: TransportConfig object (overrides individual params).

    Returns:
        TransportResult with redistributed logits.
    """
    cfg = config or TransportConfig(
        epsilon=epsilon,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    )

    # Work on clones to avoid mutating inputs
    logits = logits.clone()

    # Check bypass conditions
    should_bypass, bypass_reason = _check_bypass(logits, valid_mask, cfg)
    if should_bypass:
        logger.debug(
            "transport_bypass",
            reason=bypass_reason,
            n_valid=valid_mask.sum().item(),
        )
        masked = _standard_masking(logits, valid_mask)
        return TransportResult(
            logits=masked,
            wasserstein_distance=0.0,
            bypassed=True,
            bypass_reason=bypass_reason,
            iterations=0,
        )

    # Full OT path
    vocab_size = logits.shape[0]

    # Project to reachable subset if provided
    if reachable_tokens is not None and len(reachable_tokens) < vocab_size:
        reachable_idx = torch.tensor(
            sorted(reachable_tokens), dtype=torch.long
        )
        valid_mask_R = valid_mask[reachable_idx]
        valid_indices_R = torch.where(valid_mask_R)[0]
        invalid_indices_R = torch.where(~valid_mask_R)[0]

        problem_size = len(invalid_indices_R) * len(valid_indices_R)
        if (
            problem_size > cfg.max_problem_size
            or len(valid_indices_R) == 0
            or len(invalid_indices_R) == 0
        ):
            masked = _standard_masking(logits, valid_mask)
            return TransportResult(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason="problem_size_exceeded",
                iterations=0,
            )

        # Source/target from full-vocab softmax (YP1)
        probs_full = torch.softmax(logits, dim=-1)
        full_invalid_idx = reachable_idx[invalid_indices_R]
        full_valid_idx = reachable_idx[valid_indices_R]

        source_mass = probs_full[full_invalid_idx]
        source_mass = source_mass / source_mass.sum().clamp(min=1e-30)
        source_mass = torch.clamp(source_mass, min=1e-30)

        target_capacity = probs_full[full_valid_idx]
        target_capacity = target_capacity / target_capacity.sum().clamp(
            min=1e-30
        )
        target_capacity = torch.clamp(target_capacity, min=1e-30)

        embeddings_R = embeddings[reachable_idx]
        cost = _compute_cost_submatrix(
            embeddings_R, invalid_indices_R, valid_indices_R
        )

        logger.debug(
            "transport_sinkhorn_start",
            n_invalid=len(invalid_indices_R),
            n_valid=len(valid_indices_R),
            reachable_size=len(reachable_tokens),
            epsilon=cfg.epsilon,
        )
        plan, wasserstein, iterations = _sinkhorn_log_domain(
            cost, source_mass, target_capacity,
            cfg.epsilon, cfg.max_iterations, cfg.convergence_threshold,
        )

        total_invalid_mass = probs_full[full_invalid_idx].sum()
        plan = plan * total_invalid_mass
        result_logits = _apply_transport_plan(
            plan, full_valid_idx, logits, vocab_size
        )
    else:
        # No projection — original full-vocab path
        invalid_indices = torch.where(~valid_mask)[0]
        valid_indices = torch.where(valid_mask)[0]

        problem_size = len(invalid_indices) * len(valid_indices)
        if problem_size > cfg.max_problem_size:
            logger.debug(
                "transport_bypass",
                reason="problem_size_exceeded",
                problem_size=problem_size,
            )
            masked = _standard_masking(logits, valid_mask)
            return TransportResult(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason="problem_size_exceeded",
                iterations=0,
            )

        probs = torch.softmax(logits, dim=-1)
        source_mass = probs[invalid_indices]
        source_mass = source_mass / source_mass.sum()
        source_mass = torch.clamp(source_mass, min=1e-30)

        target_capacity = probs[valid_indices]
        target_capacity = target_capacity / target_capacity.sum()
        target_capacity = torch.clamp(target_capacity, min=1e-30)

        cost = _compute_cost_submatrix(
            embeddings, invalid_indices, valid_indices
        )

        logger.debug(
            "transport_sinkhorn_start",
            n_invalid=len(invalid_indices),
            n_valid=len(valid_indices),
            epsilon=cfg.epsilon,
        )
        plan, wasserstein, iterations = _sinkhorn_log_domain(
            cost, source_mass, target_capacity,
            cfg.epsilon, cfg.max_iterations, cfg.convergence_threshold,
        )

        total_invalid_mass = probs[invalid_indices].sum()
        plan = plan * total_invalid_mass

        result_logits = _apply_transport_plan(
            plan, valid_indices, logits, vocab_size
        )

    logger.debug(
        "transport_complete",
        wasserstein_distance=wasserstein,
        iterations=iterations,
    )

    return TransportResult(
        logits=result_logits,
        wasserstein_distance=wasserstein,
        bypassed=False,
        bypass_reason=None,
        iterations=iterations,
    )
