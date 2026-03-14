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
