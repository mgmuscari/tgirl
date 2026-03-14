"""Probabilistic state machine for SamplingSession.

Pure Python + Pydantic. Zero torch/mlx imports.
Controls state transitions in the dual-mode sampling loop.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Protocol, runtime_checkable

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


# --- Backtracking Infrastructure ---


class Checkpoint(BaseModel):
    """Saved state at a high-freedom position for potential backtracking."""

    model_config = ConfigDict(frozen=True)

    position: int
    tokens_so_far: tuple[int, ...]
    context_tokens: tuple[int, ...]
    grammar_text: str
    dead_end_tokens: frozenset[int]

    def with_added_dead_end(self, token_id: int) -> Checkpoint:
        """Return a new Checkpoint with an additional dead-end token."""
        return Checkpoint(
            position=self.position,
            tokens_so_far=self.tokens_so_far,
            context_tokens=self.context_tokens,
            grammar_text=self.grammar_text,
            dead_end_tokens=self.dead_end_tokens | {token_id},
        )


class BacktrackEvent(BaseModel):
    """Record of a single backtrack occurrence."""

    model_config = ConfigDict(frozen=True)

    checkpoint_position: int
    trigger_position: int
    trigger_log_prob: float
    dead_end_tokens_added: frozenset[int]


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
