"""Probabilistic state machine for SamplingSession.

Pure Python + Pydantic. Zero torch/mlx imports.
Controls state transitions in the dual-mode sampling loop.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class SessionState(str, Enum):
    """States in the dual-mode sampling loop."""

    FREEFORM = "freeform"
    ROUTE = "route"
    CONSTRAINED = "constrained"
    BACKTRACK = "backtrack"
    EXECUTE = "execute"
    INJECT = "inject"
    DONE = "done"


class TransitionSignal(BaseModel):
    """Per-token signal data for transition policy evaluation."""

    model_config = ConfigDict(frozen=True)

    token_position: int
    grammar_mask_overlap: float
    token_entropy: float
    token_log_prob: float
    grammar_freedom: float
    trend_window: tuple[float, ...] = ()


class TransitionDecision(BaseModel):
    """Output of a transition policy evaluation."""

    model_config = ConfigDict(frozen=True)

    should_transition: bool
    target_state: SessionState | None
    reason: str
    confidence: float


@runtime_checkable
class TransitionPolicy(Protocol):
    """Protocol for pluggable state transition policies."""

    def evaluate(
        self,
        current_state: SessionState,
        signal: TransitionSignal,
        **kwargs: object,
    ) -> TransitionDecision: ...


class DelimiterTransitionPolicy:
    """Transition policy that wraps delimiter detection.

    Backward-compatible with the existing DelimiterDetector behavior.
    Only triggers transitions from FREEFORM state.
    """

    def __init__(
        self,
        delimiter: str,
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self._delimiter = delimiter
        self._decode = tokenizer_decode
        self._decoded_window: str = ""
        self._max_window = len(delimiter) * 2

    def evaluate(
        self,
        current_state: SessionState,
        signal: TransitionSignal,
        **kwargs: object,
    ) -> TransitionDecision:
        """Evaluate whether to transition based on delimiter detection.

        Requires token_id as a keyword argument to feed the detector.
        Only triggers in FREEFORM state.
        """
        if current_state != SessionState.FREEFORM:
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="not in freeform state",
                confidence=0.0,
            )

        token_id = kwargs.get("token_id")
        if token_id is None:
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="no token_id provided",
                confidence=0.0,
            )

        # Feed token to internal delimiter detector
        new_text = self._decode([int(token_id)])
        self._decoded_window += new_text
        if len(self._decoded_window) > self._max_window:
            self._decoded_window = self._decoded_window[-self._max_window :]

        if self._delimiter in self._decoded_window:
            return TransitionDecision(
                should_transition=True,
                target_state=SessionState.ROUTE,
                reason="delimiter detected",
                confidence=1.0,
            )

        return TransitionDecision(
            should_transition=False,
            target_state=None,
            reason="delimiter not detected",
            confidence=0.0,
        )

    def reset(self) -> None:
        """Clear the detection window."""
        self._decoded_window = ""


class BudgetTransitionPolicy:
    """Force transition after a fixed number of freeform tokens.

    Useful for small models where bounded thinking improves accuracy.
    """

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._tokens_seen = 0

    def evaluate(
        self,
        current_state: SessionState,
        signal: TransitionSignal,
        **kwargs: object,
    ) -> TransitionDecision:
        if current_state != SessionState.FREEFORM:
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="not in freeform state",
                confidence=0.0,
            )

        self._tokens_seen += 1
        if self._tokens_seen > self._budget:
            return TransitionDecision(
                should_transition=True,
                target_state=SessionState.ROUTE,
                reason=f"budget exhausted ({self._budget} tokens)",
                confidence=1.0,
            )

        return TransitionDecision(
            should_transition=False,
            target_state=None,
            reason=f"budget remaining ({self._budget - self._tokens_seen + 1})",
            confidence=0.0,
        )

    def reset(self) -> None:
        """Reset token counter for next cycle."""
        self._tokens_seen = 0


class ImmediateTransitionPolicy:
    """Skip freeform generation entirely (equivalent to budget=0).

    Useful for pure tool-calling benchmarks like BFCL.
    """

    def evaluate(
        self,
        current_state: SessionState,
        signal: TransitionSignal,
        **kwargs: object,
    ) -> TransitionDecision:
        if current_state != SessionState.FREEFORM:
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="not in freeform state",
                confidence=0.0,
            )

        return TransitionDecision(
            should_transition=True,
            target_state=SessionState.ROUTE,
            reason="immediate transition",
            confidence=1.0,
        )


# --- Signal Computation ---


def compute_transition_signal(
    token_position: int,
    logits: Any,
    grammar_valid_mask: Any,
    softmax_fn: Callable[..., Any],
    sum_fn: Callable[..., Any],
    log_fn: Callable[..., Any],
    vocab_size: int,
    sampled_token_id: int | None = None,
) -> TransitionSignal:
    """Compute TransitionSignal from raw logits and grammar mask.

    Framework-agnostic: all operations are done via the provided
    callables (softmax_fn, sum_fn, log_fn). No framework imports.

    Args:
        token_position: Current token position in the sequence.
        logits: Raw logit values (framework-dependent type).
        grammar_valid_mask: Binary mask of grammar-valid tokens.
        softmax_fn: Computes softmax probabilities from logits.
        sum_fn: Sums elements of a sequence.
        log_fn: Computes element-wise log of a sequence.
        vocab_size: Total vocabulary size.
        sampled_token_id: ID of the actually sampled token. If
            provided, token_log_prob is the log prob of that token.
            If None, falls back to max log prob.

    Returns:
        TransitionSignal with computed metrics.
    """
    probs = softmax_fn(logits)
    log_probs = log_fn(probs)

    # grammar_mask_overlap: probability mass on grammar-valid tokens
    weighted = [p * m for p, m in zip(probs, grammar_valid_mask)]
    grammar_mask_overlap = float(sum_fn(weighted))

    # token_entropy: -sum(p * log(p))
    p_log_p = [
        p * lp if lp != float("-inf") else 0.0
        for p, lp in zip(probs, log_probs)
    ]
    token_entropy = -float(sum_fn(p_log_p))

    # token_log_prob: log prob of the sampled token
    if sampled_token_id is not None:
        token_log_prob = float(log_probs[sampled_token_id])
    else:
        # Fallback: max log prob
        finite = [lp for lp in log_probs if lp != float("-inf")]
        token_log_prob = float(max(finite)) if finite else float("-inf")

    # grammar_freedom: fraction of vocab that is grammar-valid
    grammar_freedom = float(sum_fn(grammar_valid_mask)) / vocab_size

    return TransitionSignal(
        token_position=token_position,
        grammar_mask_overlap=grammar_mask_overlap,
        token_entropy=token_entropy,
        token_log_prob=token_log_prob,
        grammar_freedom=grammar_freedom,
    )


# --- Backtracking Infrastructure ---


class Checkpoint(BaseModel):
    """Saved state at a high-freedom position for potential backtracking."""

    model_config = ConfigDict(frozen=True)

    position: int
    tokens_so_far: tuple[int, ...]
    context_tokens: tuple[int, ...]
    grammar_text: str
    dead_end_tokens: frozenset[int]
    best_tokens: tuple[int, ...] = ()
    best_mean_log_prob: float = float("-inf")
    attempts: int = 0

    def with_added_dead_end(self, token_id: int) -> Checkpoint:
        """Return a new Checkpoint with an additional dead-end token."""
        return Checkpoint(
            position=self.position,
            tokens_so_far=self.tokens_so_far,
            context_tokens=self.context_tokens,
            grammar_text=self.grammar_text,
            dead_end_tokens=self.dead_end_tokens | {token_id},
            best_tokens=self.best_tokens,
            best_mean_log_prob=self.best_mean_log_prob,
            attempts=self.attempts,
        )

    def with_attempt(
        self, tokens: tuple[int, ...], mean_log_prob: float
    ) -> Checkpoint:
        """Return new Checkpoint with updated best if better."""
        new_attempts = self.attempts + 1
        if mean_log_prob > self.best_mean_log_prob:
            return Checkpoint(
                position=self.position,
                tokens_so_far=self.tokens_so_far,
                context_tokens=self.context_tokens,
                grammar_text=self.grammar_text,
                dead_end_tokens=self.dead_end_tokens,
                best_tokens=tokens,
                best_mean_log_prob=mean_log_prob,
                attempts=new_attempts,
            )
        return Checkpoint(
            position=self.position,
            tokens_so_far=self.tokens_so_far,
            context_tokens=self.context_tokens,
            grammar_text=self.grammar_text,
            dead_end_tokens=self.dead_end_tokens,
            best_tokens=self.best_tokens,
            best_mean_log_prob=self.best_mean_log_prob,
            attempts=new_attempts,
        )


class BacktrackEvent(BaseModel):
    """Record of a single backtrack occurrence."""

    model_config = ConfigDict(frozen=True)

    checkpoint_position: int
    trigger_position: int
    trigger_log_prob: float
    dead_end_tokens_added: frozenset[int]


@runtime_checkable
class ConfidenceMonitorProto(Protocol):
    """Public surface of a confidence monitor for constrained generation.

    Documents the duck-typed contract that sample_mlx.py relies on.
    Concrete implementation: ``ConstrainedConfidenceMonitor`` (below).
    """

    backtracks_remaining: int  # property in concrete impl

    def should_checkpoint(self, grammar_valid_count: int) -> bool: ...

    def record_log_prob(self, log_prob: float) -> None: ...

    def should_backtrack(self) -> bool: ...

    def record_backtrack(self) -> None: ...


class ConstrainedConfidenceMonitor:
    """Monitors confidence during constrained generation.

    Tracks log probs at high-freedom positions and signals
    when backtracking should occur.
    """

    def __init__(
        self,
        log_prob_threshold: float = -1.0,
        window_size: int = 3,
        freedom_threshold: int = 5,
        max_backtracks: int = 3,
        exhaustion_fraction: float = 0.5,
    ) -> None:
        self.log_prob_threshold = log_prob_threshold
        self.window_size = window_size
        self.freedom_threshold = freedom_threshold
        self.max_backtracks = max_backtracks
        self.exhaustion_fraction = exhaustion_fraction
        self._log_probs: list[float] = []
        self._backtrack_count = 0

    def should_checkpoint(self, grammar_valid_count: int) -> bool:
        """Whether to create a checkpoint at this position."""
        return grammar_valid_count >= self.freedom_threshold

    def record_log_prob(self, log_prob: float) -> None:
        """Record a log prob observation."""
        self._log_probs.append(log_prob)

    def should_backtrack(self) -> bool:
        """Whether confidence has declined enough to trigger backtrack."""
        if len(self._log_probs) < self.window_size:
            return False
        window = self._log_probs[-self.window_size :]
        mean_log_prob = sum(window) / len(window)
        return mean_log_prob < self.log_prob_threshold

    def record_backtrack(self) -> None:
        """Record that a backtrack occurred."""
        self._backtrack_count += 1

    @property
    def backtracks_remaining(self) -> int:
        """Number of backtracks still allowed."""
        return max(0, self.max_backtracks - self._backtrack_count)

    def is_checkpoint_sealed(
        self, dead_end_count: int, valid_count: int
    ) -> bool:
        """Whether a checkpoint is sealed (too many dead ends)."""
        if valid_count == 0:
            return True
        return dead_end_count / valid_count > self.exhaustion_fraction

    def reset(self) -> None:
        """Reset monitor state for a new constrained generation pass."""
        self._log_probs.clear()
        self._backtrack_count = 0


# --- Confidence-Based Transition Policy ---


class ConfidenceTransitionPolicy:
    """Markov chain on model signals for freeform-to-constrained transition.

    Accumulates belief that the model is converging on structured output.
    Signals: grammar mask overlap (rising), entropy (dropping),
    log prob at grammar-valid tokens, and trend over a sliding window.

    Belief update: belief = clamp(belief + weighted_signal, 0, 1)
    Transition when belief exceeds threshold.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        w_readiness: float = 0.4,
        w_certainty: float = 0.3,
        w_quality: float = 0.2,
        w_trend: float = 0.1,
        decay: float = 0.9,
    ) -> None:
        self.threshold = threshold
        self.w_readiness = w_readiness
        self.w_certainty = w_certainty
        self.w_quality = w_quality
        self.w_trend = w_trend
        self.decay = decay
        self._belief = 0.0
        self._overlap_history: list[float] = []

    def evaluate(
        self,
        current_state: SessionState,
        signal: TransitionSignal,
        **kwargs: object,
    ) -> TransitionDecision:
        if current_state != SessionState.FREEFORM:
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="not in freeform state",
                confidence=0.0,
            )

        # Compute signal components
        readiness = signal.grammar_mask_overlap  # 0-1
        certainty = max(0.0, 1.0 - signal.token_entropy / 10.0)  # normalize
        quality = max(0.0, 1.0 + signal.token_log_prob)  # log_prob near 0 = good

        # Trend: rising overlap
        self._overlap_history.append(signal.grammar_mask_overlap)
        if len(self._overlap_history) >= 3:
            recent = self._overlap_history[-3:]
            trend = (recent[-1] - recent[0]) / 2.0  # positive = rising
        else:
            trend = 0.0

        # Weighted signal update
        delta = (
            self.w_readiness * readiness
            + self.w_certainty * certainty
            + self.w_quality * quality
            + self.w_trend * max(0.0, trend)
        )

        # Belief update with decay
        self._belief = min(1.0, max(0.0, self._belief * self.decay + delta))

        if self._belief >= self.threshold:
            return TransitionDecision(
                should_transition=True,
                target_state=SessionState.ROUTE,
                reason=f"confidence threshold reached ({self._belief:.3f})",
                confidence=self._belief,
            )

        return TransitionDecision(
            should_transition=False,
            target_state=None,
            reason=f"belief below threshold ({self._belief:.3f}/{self.threshold})",
            confidence=self._belief,
        )

    def reset(self) -> None:
        """Reset belief for next cycle."""
        self._belief = 0.0
        self._overlap_history.clear()


class LatchedTransitionPolicy:
    """Latch on high confidence, transition on sentence terminal.

    Two-phase regime transition for models that don't emit delimiters:
    1. LATCH: when token entropy drops below threshold (distribution is
       sharp — the model knows what it wants to do), set a one-way latch
    2. COMPLETE: wait for a sentence terminal token, then transition
    3. FALLBACK: if max_freeform_after_latch tokens pass without terminal,
       force transition anyway

    Confidence is measured by distribution sharpness (low entropy = high
    confidence), not by matching tool name strings in the output.

    The terminal character set is configurable for polyglot support:
    English {'.', '!', '?'}, Chinese {'。', '！', '？'}, etc.
    """

    def __init__(
        self,
        entropy_threshold: float = 2.0,
        terminal_chars: set[str] | None = None,
        max_freeform_after_latch: int = 64,
        tokenizer_decode: Callable[[list[int]], str] | None = None,
    ) -> None:
        self._entropy_threshold = entropy_threshold
        self._terminals = terminal_chars or {'.', '!', '?'}
        self._max_after_latch = max_freeform_after_latch
        self._decode = tokenizer_decode
        self._latched = False
        self._tokens_since_latch = 0

    def evaluate(
        self,
        current_state: SessionState,
        signal: TransitionSignal,
        **kwargs: object,
    ) -> TransitionDecision:
        if current_state != SessionState.FREEFORM:
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="not in freeform state",
                confidence=0.0,
            )

        # Decode token text for terminal detection
        token_text = str(kwargs.get("token_text", ""))
        token_id = kwargs.get("token_id")
        if token_id is not None and self._decode is not None:
            token_text = self._decode([int(token_id)])

        # Phase 1: latch on low entropy (high confidence)
        if not self._latched and signal.token_entropy < self._entropy_threshold:
            self._latched = True
            self._tokens_since_latch = 0

        if not self._latched:
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="not latched (entropy too high)",
                confidence=0.0,
            )

        self._tokens_since_latch += 1

        # Phase 3: budget exhaustion fallback
        if self._tokens_since_latch >= self._max_after_latch:
            return TransitionDecision(
                should_transition=True,
                target_state=SessionState.ROUTE,
                reason="budget exhaustion after latch",
                confidence=1.0,
            )

        # Phase 2: terminal detection
        stripped = token_text.strip()
        if stripped:
            last_char = stripped[-1]
            if last_char in self._terminals:
                return TransitionDecision(
                    should_transition=True,
                    target_state=SessionState.ROUTE,
                    reason="sentence terminal after latch",
                    confidence=0.9,
                )

        return TransitionDecision(
            should_transition=False,
            target_state=None,
            reason="latched, awaiting terminal",
            confidence=signal.token_entropy,
        )

    def reset(self) -> None:
        """Clear latch state for next cycle."""
        self._latched = False
        self._tokens_since_latch = 0


class CompositeTransitionPolicy:
    """Combine multiple policies with OR or AND logic.

    OR mode: transition if ANY policy triggers (first match wins).
    AND mode: transition only if ALL policies agree.
    """

    def __init__(
        self,
        policies: list[TransitionPolicy],
        mode: str = "or",
    ) -> None:
        self.policies = policies
        self.mode = mode

    def evaluate(
        self,
        current_state: SessionState,
        signal: TransitionSignal,
        **kwargs: object,
    ) -> TransitionDecision:
        if not self.policies:
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="no policies configured",
                confidence=0.0,
            )

        if self.mode == "or":
            # Short-circuit: return first policy that triggers
            max_confidence = 0.0
            for p in self.policies:
                d = p.evaluate(current_state, signal, **kwargs)
                if d.should_transition:
                    return d
                max_confidence = max(max_confidence, d.confidence)
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="no policy triggered (OR mode)",
                confidence=max_confidence,
            )

        # AND mode: evaluate all
        decisions = [
            p.evaluate(current_state, signal, **kwargs) for p in self.policies
        ]
        if True:  # "and"
            if all(d.should_transition for d in decisions):
                # Use highest confidence decision
                best = max(decisions, key=lambda d: d.confidence)
                return best
            return TransitionDecision(
                should_transition=False,
                target_state=None,
                reason="not all policies agree (AND mode)",
                confidence=min(d.confidence for d in decisions),
            )

    def reset(self) -> None:
        """Reset all sub-policies."""
        for p in self.policies:
            if hasattr(p, "reset"):
                p.reset()
