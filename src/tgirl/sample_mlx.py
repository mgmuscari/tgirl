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
from tgirl.state_machine import TransitionSignal
from tgirl.transport import TransportConfig
from tgirl.transport_mlx import redistribute_logits_mlx
from tgirl.types import ModelIntervention

logger = structlog.get_logger()


def apply_cycle_gate(
    valid_mask: mx.array,
    token_history: list[int],
    hold_count: int = 3,
    max_period: int = 16,
) -> mx.array:
    """Remove cycling tokens from valid_mask after K repetitions.

    Gated reverb for token repetition: detects suffix cycles in
    token_history and hard-gates the cycling tokens (sets them to
    False in the mask) after hold_count full cycle repetitions.

    Args:
        valid_mask: Current valid token mask (bool, shape (vocab_size,)).
        token_history: Generated tokens so far (constrained gen only).
        hold_count: Number of full cycle repetitions before gate closes.
        max_period: Maximum cycle period to detect.

    Returns:
        Modified valid_mask with cycling tokens removed, or original
        mask if no cycle detected or insufficient repetitions.
    """
    from tgirl.sample import detect_cycle

    n = len(token_history)
    cycle_len = detect_cycle(token_history, max_period)
    if cycle_len is None:
        return valid_mask

    # Count how many full repetitions of the cycle exist at the suffix
    repetitions = 0
    for i in range(1, n // cycle_len + 1):
        start = n - i * cycle_len
        if start < 0:
            break
        if token_history[start : start + cycle_len] == token_history[-cycle_len:]:
            repetitions += 1
        else:
            break

    if repetitions < hold_count:
        return valid_mask

    # Gate closes: remove cycling tokens from valid_mask
    cycle_tokens = set(token_history[-cycle_len:])
    gate_mask = mx.ones(valid_mask.shape, dtype=mx.bool_)
    indices = mx.array(list(cycle_tokens))
    gate_mask[indices] = False
    return valid_mask & gate_mask


def compute_transition_signal_mlx(
    token_position: int,
    logits: mx.array,
    grammar_valid_mask: mx.array,
    vocab_size: int,
    sampled_token_id: int | None = None,
) -> TransitionSignal:
    """Compute TransitionSignal using native MLX operations.

    All math stays in mx.array — no Python list iteration over tensors.
    """
    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(mx.clip(probs, 1e-30, None))

    grammar_mask_overlap = float(mx.sum(probs * grammar_valid_mask).item())

    p_log_p = probs * log_probs
    token_entropy = -float(mx.sum(p_log_p).item())

    if sampled_token_id is not None:
        token_log_prob = float(log_probs[sampled_token_id].item())
    else:
        token_log_prob = float(mx.max(log_probs).item())

    grammar_freedom = float(mx.sum(grammar_valid_mask).item()) / vocab_size

    return TransitionSignal(
        token_position=token_position,
        grammar_mask_overlap=grammar_mask_overlap,
        token_entropy=token_entropy,
        token_log_prob=token_log_prob,
        grammar_freedom=grammar_freedom,
    )


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


class BacktrackSteeringHookMlx:
    """MLX-native pre-OT hook that applies negative logit bias on dead-end tokens.

    Mirror of BacktrackSteeringHook from sample.py but conforms to
    InferenceHookMlx protocol (receives valid_mask: mx.array).
    """

    def __init__(
        self,
        dead_end_tokens: frozenset[int],
        bias_strength: float = -100.0,
    ) -> None:
        self.dead_end_tokens = dead_end_tokens
        self.bias_strength = bias_strength

    def pre_forward(
        self,
        position: int,
        valid_mask: mx.array,
        token_history: list[int],
        logits: mx.array,
    ) -> ModelIntervention:
        if not self.dead_end_tokens:
            return ModelIntervention()
        bias = {tid: self.bias_strength for tid in self.dead_end_tokens}
        return ModelIntervention(logit_bias=bias)


class NestingDepthHookMlx:
    """Prevents unclosable s-expressions by tracking nesting depth.

    Pre-computes a paren delta for every token in the vocabulary.
    During generation, tracks depth and penalizes open-paren tokens
    when remaining budget is too small to close all open parens.
    """

    def __init__(
        self,
        max_tokens: int,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
        margin: int = 2,
        bias: float = -100.0,
        open_char: str = "(",
        close_char: str = ")",
    ) -> None:
        self.max_tokens = max_tokens
        self.margin = margin
        self.bias = bias
        self._depth = 0

        # Pre-compute per-token paren delta and classify openers
        self._delta: dict[int, int] = {}
        self._opener_ids: set[int] = set()
        for tid in range(vocab_size):
            text = tokenizer_decode([tid])
            opens = text.count(open_char)
            closes = text.count(close_char)
            delta = opens - closes
            if delta != 0:
                self._delta[tid] = delta
            if opens > closes:
                self._opener_ids.add(tid)

    def advance(self, token_id: int) -> None:
        """Update nesting depth after a token is sampled."""
        self._depth += self._delta.get(token_id, 0)
        self._depth = max(self._depth, 0)

    def reset(self) -> None:
        """Reset depth for a new constrained generation pass."""
        self._depth = 0

    def pre_forward(
        self,
        position: int,
        valid_mask: mx.array,
        token_history: list[int],
        logits: mx.array,
    ) -> ModelIntervention:
        remaining = self.max_tokens - position
        needed = self._depth + self.margin
        if remaining > needed:
            return ModelIntervention()
        # Budget too tight — penalize all tokens that open new parens
        penalized = {tid: self.bias for tid in self._opener_ids}
        return ModelIntervention(logit_bias=penalized)


class RepetitionPenaltyHookMlx:
    """Penalizes degenerate token repetition via cycle detection.

    MLX-native variant. Detects repeating cycles in the token suffix
    and applies escalating penalties. Also penalizes individual tokens
    that exceed max_repeats in a window.
    """

    def __init__(
        self,
        window: int = 8,
        max_repeats: int = 2,
        bias: float = -20.0,
        max_period: int = 16,
        cycle_bias: float = -50.0,
    ) -> None:
        self.window = window
        self.max_repeats = max_repeats
        self.bias = bias
        self.max_period = max_period
        self.cycle_bias = cycle_bias

    def pre_forward(
        self,
        position: int,
        valid_mask: mx.array,
        token_history: list[int],
        logits: mx.array,
    ) -> ModelIntervention:
        from tgirl.sample import detect_cycle

        penalized: dict[int, float] = {}

        # 1. Window-based: penalize tokens exceeding max_repeats
        recent = token_history[-self.window:]
        counts: dict[int, int] = {}
        for tid in recent:
            counts[tid] = counts.get(tid, 0) + 1
        for tid, count in counts.items():
            if count > self.max_repeats:
                penalized[tid] = self.bias * (count - self.max_repeats)

        # 2. Cycle detection: penalize all tokens in a detected cycle
        cycle_len = detect_cycle(token_history, self.max_period)
        if cycle_len is not None and self.cycle_bias != 0.0:
            cycle_tokens = set(token_history[-cycle_len:])
            for tid in cycle_tokens:
                existing = penalized.get(tid, 0.0)
                penalized[tid] = existing + self.cycle_bias

        if not penalized:
            return ModelIntervention()
        return ModelIntervention(logit_bias=penalized)


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
    confidence_monitor: object | None = None,
    grammar_guide_factory: Callable[[str], Any] | None = None,
    grammar_text: str | None = None,
    stop_token_ids: list[int] | None = None,
    reachable_tokens: frozenset[int] | None = None,
    controller: object | None = None,
) -> ConstrainedGenerationResult:
    """Run constrained token generation until grammar accepts or max_tokens.

    Fully MLX-native loop. Zero torch, zero numpy in computation.
    Only Python scalar extraction: int() on token ID.

    If confidence_monitor is provided, tracks log probs and checkpoints
    at high-freedom positions. When confidence drops below threshold,
    returns early with backtrack_requested=True and the checkpoint.
    """
    from tgirl.state_machine import BacktrackEvent, Checkpoint

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
    ot_bypass_reasons: list[str | None] = []
    ot_iterations: list[int] = []
    backtrack_events: list[BacktrackEvent] = []
    estradiol_alphas: list[list[float]] | None = None
    estradiol_deltas: list[list[float]] | None = None

    vocab_size = embeddings.shape[0]
    last_checkpoint: Checkpoint | None = None
    backtrack_requested = False

    # ESTRADIOL controller setup
    _steering: object | None = None
    if controller is not None:
        if hasattr(controller, "reset"):
            controller.reset()
        K = controller.V_basis.shape[1]
        _steering = controller.make_steering_state(mx.zeros((K,)))
        estradiol_alphas = []
        estradiol_deltas = []

    # Reset stateful hooks for this generation pass
    for hook in hooks:
        if hasattr(hook, "reset"):
            hook.reset()

    # Detect grammar state type once
    has_mlx_mask = hasattr(grammar_state, "get_valid_mask_mx")

    for position in range(max_tokens):
        _t0 = time.monotonic()

        # 1. Forward pass — steerable if controller is active
        if controller is not None and _steering is not None:
            _fwd_result = forward_fn(token_history, steering=_steering)
            raw_logits = _fwd_result.logits
            if _fwd_result.probe_alpha is not None:
                _delta = controller.step(_fwd_result.probe_alpha)
                _steering = controller.make_steering_state(_delta)
                estradiol_alphas.append(controller.alpha_current.tolist())
                estradiol_deltas.append(_delta.tolist())
        else:
            raw_logits = forward_fn(token_history)
        _t1 = time.monotonic()

        # 2. Grammar mask — pure MLX via llguidance.mlx (or fallback)
        if has_mlx_mask:
            valid_mask = grammar_state.get_valid_mask_mx(vocab_size)
        else:
            # Fallback for torch-based grammar states
            valid_mask_torch = grammar_state.get_valid_mask(vocab_size)
            valid_mask = mx.array(valid_mask_torch.numpy())

        # Mask out stop tokens while grammar is not yet accepting —
        # prevents premature EOS. Once the grammar accepts (complete
        # expression), stop tokens are allowed so the model can terminate.
        if stop_token_ids and not grammar_state.is_accepting():
            stop_mask = mx.ones(valid_mask.shape, dtype=mx.bool_)
            indices = mx.array(stop_token_ids)
            stop_mask[indices] = False
            valid_mask = valid_mask & stop_mask

        # Cycle gate: hard-remove cycling tokens from valid_mask after
        # K repetitions (gated reverb — the tail stops, not fades).
        valid_mask = apply_cycle_gate(valid_mask, tokens)

        valid_count = int(mx.sum(valid_mask).item())
        grammar_valid_counts.append(valid_count)
        _t2 = time.monotonic()

        # Dead end: no valid token can continue the parse — abort
        if valid_count == 0:
            logger.warning("grammar_dead_end", position=position)
            break

        # Checkpoint if monitor says so
        if (
            confidence_monitor is not None
            and grammar_text is not None
            and confidence_monitor.should_checkpoint(valid_count)
        ):
            last_checkpoint = Checkpoint(
                position=position,
                tokens_so_far=tuple(tokens),
                context_tokens=tuple(
                    context_tokens if context_tokens else []
                ),
                grammar_text=grammar_text,
                dead_end_tokens=frozenset(),
            )

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
        # Use per-token epsilon if set by a hook
        if merged.transport_epsilon is not None:
            token_transport_config = TransportConfig(
                epsilon=merged.transport_epsilon,
                max_iterations=transport_config.max_iterations,
                convergence_threshold=transport_config.convergence_threshold,
                valid_ratio_threshold=transport_config.valid_ratio_threshold,
                invalid_mass_threshold=transport_config.invalid_mass_threshold,
                max_problem_size=transport_config.max_problem_size,
            )
        else:
            token_transport_config = transport_config

        ot_start = time.monotonic()
        ot_result = redistribute_logits_mlx(
            adjusted, valid_mask, embeddings,
            config=token_transport_config,
            reachable_tokens=reachable_tokens,
        )
        ot_elapsed_ms = (time.monotonic() - ot_start) * 1000
        ot_computation_total_ms += ot_elapsed_ms
        wasserstein_distances.append(ot_result.wasserstein_distance)
        ot_bypass_reasons.append(ot_result.bypass_reason)
        ot_iterations.append(ot_result.iterations)
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

        # Update stateful hooks (e.g., nesting depth tracking)
        for hook in hooks:
            if hasattr(hook, "advance"):
                hook.advance(token_id)

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

        # Confidence monitoring and backtrack check
        if confidence_monitor is not None:
            confidence_monitor.record_log_prob(log_prob)
            if (
                confidence_monitor.should_backtrack()
                and confidence_monitor.backtracks_remaining > 0
                and last_checkpoint is not None
            ):
                # Divergent token is the one chosen at checkpoint
                cp_pos = last_checkpoint.position
                divergent_token = (
                    tokens[cp_pos]
                    if cp_pos < len(tokens)
                    else token_id
                )
                # Track best-so-far before backtracking
                mean_lp = (
                    sum(token_log_probs) / len(token_log_probs)
                    if token_log_probs
                    else float("-inf")
                )
                last_checkpoint = last_checkpoint.with_attempt(
                    tokens=tuple(tokens), mean_log_prob=mean_lp
                )
                event = BacktrackEvent(
                    checkpoint_position=cp_pos,
                    trigger_position=position,
                    trigger_log_prob=log_prob,
                    dead_end_tokens_added=frozenset(
                        {divergent_token}
                    ),
                )
                backtrack_events.append(event)
                last_checkpoint = (
                    last_checkpoint.with_added_dead_end(
                        divergent_token
                    )
                )
                confidence_monitor.record_backtrack()
                backtrack_requested = True
                break

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
        ot_bypass_reasons=ot_bypass_reasons,
        ot_iterations=ot_iterations,
        grammar_generation_ms=elapsed_ms,
        backtrack_requested=backtrack_requested,
        backtrack_checkpoint=(
            last_checkpoint if backtrack_requested else None
        ),
        backtrack_events=backtrack_events,
        estradiol_alphas=estradiol_alphas,
        estradiol_deltas=estradiol_deltas,
    )
