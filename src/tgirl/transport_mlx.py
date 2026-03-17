"""MLX-native optimal transport logit redistribution.

Mirrors transport.py for Apple Silicon — all per-token math stays in mx.array.
This module has ZERO coupling to any other tgirl module except importing
TransportConfig (a plain Pydantic model with no torch dependency).
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx
import numpy as np
import structlog

from tgirl.transport import TransportConfig

logger = structlog.get_logger()


class TransportResultMlx(NamedTuple):
    """Result of logit redistribution (MLX-native).

    Mirrors TransportResult field-for-field except logits is mx.array.
    """

    logits: mx.array
    wasserstein_distance: float
    bypassed: bool
    bypass_reason: str | None
    iterations: int


def _check_bypass_mlx(
    logits: mx.array,
    valid_mask: mx.array,
    config: TransportConfig,
) -> tuple[bool, str | None]:
    """Check whether optimal transport should be bypassed.

    Three conditions checked in priority order:
    1. valid_mask.sum() <= 1 -> "forced_decode"
    2. valid ratio > threshold -> "valid_ratio_high"
    3. invalid probability mass < threshold -> "invalid_mass_low"
    """
    n_valid = int(mx.sum(valid_mask).item())

    if n_valid <= 1:
        return True, "forced_decode"

    valid_ratio = float(mx.mean(valid_mask.astype(mx.float32)).item())
    if valid_ratio > config.valid_ratio_threshold:
        return True, "valid_ratio_high"

    probs = mx.softmax(logits, axis=-1)
    invalid_mask = mx.logical_not(valid_mask)
    invalid_mass = float(mx.sum(probs * invalid_mask.astype(mx.float32)).item())
    if invalid_mass < config.invalid_mass_threshold:
        return True, "invalid_mass_low"

    return False, None


def _standard_masking_mlx(
    logits: mx.array, valid_mask: mx.array
) -> mx.array:
    """Set invalid logits to -inf, preserving valid logits unchanged."""
    return mx.where(valid_mask, logits, mx.array(float("-inf")))


def _compute_cost_submatrix_mlx(
    embeddings: mx.array,
    invalid_indices: mx.array,
    valid_indices: mx.array,
) -> mx.array:
    """Compute cosine distance submatrix between invalid and valid tokens.

    Cost = 1 - cosine_similarity. Only allocates the (n_invalid, n_valid)
    submatrix.
    """
    invalid_emb = embeddings[invalid_indices]
    valid_emb = embeddings[valid_indices]

    # L2 normalize
    invalid_norm = mx.sqrt(mx.sum(invalid_emb * invalid_emb, axis=-1, keepdims=True))
    invalid_norm = mx.maximum(invalid_norm, mx.array(1e-8))
    invalid_emb = invalid_emb / invalid_norm

    valid_norm = mx.sqrt(mx.sum(valid_emb * valid_emb, axis=-1, keepdims=True))
    valid_norm = mx.maximum(valid_norm, mx.array(1e-8))
    valid_emb = valid_emb / valid_norm

    similarity = invalid_emb @ mx.transpose(valid_emb)
    return 1.0 - similarity


def _sinkhorn_log_domain_mlx(
    cost_matrix: mx.array,
    source_mass: mx.array,
    target_capacity: mx.array,
    epsilon: float,
    max_iterations: int,
    convergence_threshold: float,
) -> tuple[mx.array, float, int]:
    """Log-domain Sinkhorn algorithm for optimal transport (MLX-native)."""
    n_source = cost_matrix.shape[0]
    n_target = cost_matrix.shape[1]
    log_kernel = -cost_matrix / epsilon

    log_u = mx.zeros((n_source,), dtype=cost_matrix.dtype)
    log_v = mx.zeros((n_target,), dtype=cost_matrix.dtype)

    log_source = mx.log(source_mass)
    log_target = mx.log(target_capacity)

    iterations = 0
    for i in range(max_iterations):
        iterations = i + 1

        # Update log_u: row scaling
        log_u = log_source - mx.logsumexp(
            log_kernel + mx.expand_dims(log_v, axis=0), axis=1
        )

        # Update log_v: column scaling
        log_v = log_target - mx.logsumexp(
            log_kernel + mx.expand_dims(log_u, axis=1), axis=0
        )

        # Materialize graph to prevent unbounded growth
        mx.eval(log_u, log_v)

        # Check convergence
        log_plan_row_sums = mx.logsumexp(
            log_kernel + mx.expand_dims(log_u, axis=1) + mx.expand_dims(log_v, axis=0),
            axis=1,
        )
        marginal_error = float(
            mx.max(mx.abs(mx.exp(log_plan_row_sums) - source_mass)).item()
        )

        if marginal_error < convergence_threshold:
            break

    # Compute final plan
    log_plan = (
        mx.expand_dims(log_u, axis=1) + log_kernel + mx.expand_dims(log_v, axis=0)
    )
    plan = mx.exp(log_plan)

    # Wasserstein distance
    wasserstein = float(mx.sum(plan * cost_matrix).item())

    return plan, wasserstein, iterations


def _apply_transport_plan_mlx(
    plan: mx.array,
    valid_indices: mx.array,
    original_logits: mx.array,
    vocab_size: int,
) -> mx.array:
    """Apply transport plan to produce redistributed log-space logits."""
    original_probs = mx.softmax(original_logits, axis=-1)
    valid_probs = original_probs[valid_indices]

    redistributed = mx.sum(plan, axis=0)
    combined_probs = valid_probs + redistributed

    # Build result: -inf everywhere, then place valid log-probs
    log_valid = mx.log(combined_probs)
    result = mx.full((vocab_size,), float("-inf"), dtype=original_logits.dtype)
    # Scatter valid log-probs into result positions
    valid_log_full = mx.full((vocab_size,), 0.0, dtype=original_logits.dtype)
    valid_log_full = valid_log_full.at[valid_indices].add(log_valid)
    # Build boolean mask for valid positions
    valid_flag = mx.zeros((vocab_size,), dtype=original_logits.dtype)
    valid_flag = valid_flag.at[valid_indices].add(
        mx.ones((len(valid_indices),), dtype=original_logits.dtype)
    )
    is_valid = valid_flag > 0.5
    result = mx.where(is_valid, valid_log_full, result)

    return result


def redistribute_logits_mlx(
    logits: mx.array,
    valid_mask: mx.array,
    embeddings: mx.array,
    epsilon: float = 0.1,
    max_iterations: int = 20,
    convergence_threshold: float = 1e-6,
    config: TransportConfig | None = None,
    reachable_tokens: frozenset[int] | None = None,
) -> TransportResultMlx:
    """Redistribute probability mass from invalid to valid tokens using OT (MLX-native).

    Returns TransportResultMlx with mx.array logits.
    """
    cfg = config or TransportConfig(
        epsilon=epsilon,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    )

    # Check bypass conditions
    # When reachable set is provided, only check forced_decode (n_valid<=1)
    # on full vocab. Other bypasses check projected tensors instead.
    if reachable_tokens is None or len(reachable_tokens) >= logits.shape[0]:
        should_bypass, bypass_reason = _check_bypass_mlx(
            logits, valid_mask, cfg
        )
        if should_bypass:
            logger.debug(
                "transport_bypass_mlx",
                reason=bypass_reason,
                n_valid=int(mx.sum(valid_mask).item()),
            )
            masked = _standard_masking_mlx(logits, valid_mask)
            return TransportResultMlx(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason=bypass_reason,
                iterations=0,
            )
    else:
        # With reachable set: only check forced_decode on full vocab
        n_valid = int(mx.sum(valid_mask).item())
        if n_valid <= 1:
            masked = _standard_masking_mlx(logits, valid_mask)
            return TransportResultMlx(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason="forced_decode",
                iterations=0,
            )

    # Full OT path
    vocab_size = logits.shape[0]

    # Project to reachable subset if provided
    if reachable_tokens is not None and len(reachable_tokens) < vocab_size:
        reachable_idx = mx.array(sorted(reachable_tokens))
        valid_mask_R = valid_mask[reachable_idx]

        # Materialize projected mask
        mx.eval(valid_mask_R)
        mask_np_R = np.array(valid_mask_R)
        valid_indices_R = mx.array(
            np.where(mask_np_R)[0].astype(np.int32)
        )
        invalid_indices_R = mx.array(
            np.where(~mask_np_R)[0].astype(np.int32)
        )

        # Bypass checks on projected tensors
        n_valid_R = len(valid_indices_R)
        n_invalid_R = len(invalid_indices_R)
        n_reachable = len(reachable_tokens)

        if n_valid_R == 0 or n_invalid_R == 0:
            masked = _standard_masking_mlx(logits, valid_mask)
            return TransportResultMlx(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason="forced_decode",
                iterations=0,
            )

        # Valid ratio on projected space
        valid_ratio_R = n_valid_R / n_reachable if n_reachable > 0 else 1.0
        if valid_ratio_R > cfg.valid_ratio_threshold:
            masked = _standard_masking_mlx(logits, valid_mask)
            return TransportResultMlx(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason="valid_ratio_high",
                iterations=0,
            )

        # Invalid mass on projected space
        probs_check = mx.softmax(logits, axis=-1)
        invalid_mass_R = float(
            mx.sum(probs_check[reachable_idx[invalid_indices_R]]).item()
        )
        if invalid_mass_R < cfg.invalid_mass_threshold:
            masked = _standard_masking_mlx(logits, valid_mask)
            return TransportResultMlx(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason="invalid_mass_low",
                iterations=0,
            )

        # Problem size on projected tensors
        problem_size = n_invalid_R * n_valid_R
        if problem_size > cfg.max_problem_size:
            masked = _standard_masking_mlx(logits, valid_mask)
            return TransportResultMlx(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason="problem_size_exceeded",
                iterations=0,
            )

        # Source/target mass from full-vocab softmax (YP1)
        probs_full = mx.softmax(logits, axis=-1)
        full_invalid_idx = reachable_idx[invalid_indices_R]
        full_valid_idx = reachable_idx[valid_indices_R]

        source_mass = probs_full[full_invalid_idx]
        sm_sum = mx.sum(source_mass)
        source_mass = source_mass / mx.maximum(sm_sum, mx.array(1e-30))
        source_mass = mx.clip(source_mass, 1e-30, None)

        target_capacity = probs_full[full_valid_idx]
        tc_sum = mx.sum(target_capacity)
        target_capacity = target_capacity / mx.maximum(
            tc_sum, mx.array(1e-30)
        )
        target_capacity = mx.clip(target_capacity, 1e-30, None)

        # Cost submatrix on projected embeddings
        embeddings_R = embeddings[reachable_idx]
        cost = _compute_cost_submatrix_mlx(
            embeddings_R, invalid_indices_R, valid_indices_R
        )

        logger.debug(
            "transport_sinkhorn_start_mlx",
            n_invalid=len(invalid_indices_R),
            n_valid=len(valid_indices_R),
            reachable_size=len(reachable_tokens),
            epsilon=cfg.epsilon,
        )
        plan, wasserstein, iterations = _sinkhorn_log_domain_mlx(
            cost, source_mass, target_capacity,
            cfg.epsilon, cfg.max_iterations, cfg.convergence_threshold,
        )

        # Scale and apply — remap to full vocab
        total_invalid_mass = mx.sum(probs_full[full_invalid_idx])
        plan = plan * total_invalid_mass
        result_logits = _apply_transport_plan_mlx(
            plan, full_valid_idx, logits, vocab_size
        )
    else:
        # No projection — original full-vocab path
        mx.eval(valid_mask)
        mask_np = np.array(valid_mask)
        valid_indices = mx.array(
            np.where(mask_np)[0].astype(np.int32)
        )
        invalid_indices = mx.array(
            np.where(~mask_np)[0].astype(np.int32)
        )

        problem_size = len(invalid_indices) * len(valid_indices)
        if problem_size > cfg.max_problem_size:
            logger.debug(
                "transport_bypass_mlx",
                reason="problem_size_exceeded",
                problem_size=problem_size,
            )
            masked = _standard_masking_mlx(logits, valid_mask)
            return TransportResultMlx(
                logits=masked,
                wasserstein_distance=0.0,
                bypassed=True,
                bypass_reason="problem_size_exceeded",
                iterations=0,
            )

        probs = mx.softmax(logits, axis=-1)
        source_mass = probs[invalid_indices]
        source_mass = source_mass / mx.sum(source_mass)
        source_mass = mx.clip(source_mass, 1e-30, None)

        target_capacity = probs[valid_indices]
        target_capacity = target_capacity / mx.sum(target_capacity)
        target_capacity = mx.clip(target_capacity, 1e-30, None)

        cost = _compute_cost_submatrix_mlx(
            embeddings, invalid_indices, valid_indices
        )

        logger.debug(
            "transport_sinkhorn_start_mlx",
            n_invalid=len(invalid_indices),
            n_valid=len(valid_indices),
            epsilon=cfg.epsilon,
        )
        plan, wasserstein, iterations = _sinkhorn_log_domain_mlx(
            cost, source_mass, target_capacity,
            cfg.epsilon, cfg.max_iterations, cfg.convergence_threshold,
        )

        total_invalid_mass = mx.sum(probs[invalid_indices])
        plan = plan * total_invalid_mass

        result_logits = _apply_transport_plan_mlx(
            plan, valid_indices, logits, vocab_size
        )

    logger.debug(
        "transport_complete_mlx",
        wasserstein_distance=wasserstein,
        iterations=iterations,
    )

    return TransportResultMlx(
        logits=result_logits,
        wasserstein_distance=wasserstein,
        bypassed=False,
        bypass_reason=None,
        iterations=iterations,
    )
