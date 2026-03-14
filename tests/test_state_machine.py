"""Tests for tgirl.state_machine — Probabilistic state machine types."""

from __future__ import annotations

import pytest


class TestSessionState:
    """SessionState enum with all required states."""

    def test_has_all_required_states(self) -> None:
        from tgirl.state_machine import SessionState

        assert hasattr(SessionState, "FREEFORM")
        assert hasattr(SessionState, "ROUTE")
        assert hasattr(SessionState, "CONSTRAINED")
        assert hasattr(SessionState, "BACKTRACK")
        assert hasattr(SessionState, "EXECUTE")
        assert hasattr(SessionState, "INJECT")
        assert hasattr(SessionState, "DONE")

    def test_states_are_distinct(self) -> None:
        from tgirl.state_machine import SessionState

        states = list(SessionState)
        assert len(states) == len(set(states))
        assert len(states) == 7

    def test_state_values_are_string(self) -> None:
        from tgirl.state_machine import SessionState

        for state in SessionState:
            assert isinstance(state.value, str)


class TestTransitionSignal:
    """TransitionSignal carries per-token signals for policy evaluation."""

    def test_construction_with_required_fields(self) -> None:
        from tgirl.state_machine import TransitionSignal

        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.5,
            token_entropy=2.3,
            token_log_prob=-0.1,
            grammar_freedom=0.01,
        )
        assert signal.token_position == 0
        assert signal.grammar_mask_overlap == 0.5
        assert signal.token_entropy == 2.3
        assert signal.token_log_prob == -0.1
        assert signal.grammar_freedom == 0.01

    def test_trend_window_defaults_empty(self) -> None:
        from tgirl.state_machine import TransitionSignal

        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        assert signal.trend_window == ()

    def test_trend_window_accepts_values(self) -> None:
        from tgirl.state_machine import TransitionSignal

        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
            trend_window=(0.1, 0.2, 0.3),
        )
        assert signal.trend_window == (0.1, 0.2, 0.3)

    def test_is_frozen(self) -> None:
        from pydantic import ValidationError

        from tgirl.state_machine import TransitionSignal

        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        with pytest.raises(ValidationError):
            signal.token_position = 1  # type: ignore[misc]


class TestTransitionDecision:
    """TransitionDecision captures policy output."""

    def test_construction(self) -> None:
        from tgirl.state_machine import SessionState, TransitionDecision

        decision = TransitionDecision(
            should_transition=True,
            target_state=SessionState.CONSTRAINED,
            reason="delimiter detected",
            confidence=1.0,
        )
        assert decision.should_transition is True
        assert decision.target_state == SessionState.CONSTRAINED
        assert decision.reason == "delimiter detected"
        assert decision.confidence == 1.0

    def test_no_transition(self) -> None:
        from tgirl.state_machine import TransitionDecision

        decision = TransitionDecision(
            should_transition=False,
            target_state=None,
            reason="no signal",
            confidence=0.0,
        )
        assert decision.should_transition is False
        assert decision.target_state is None

    def test_is_frozen(self) -> None:
        from pydantic import ValidationError

        from tgirl.state_machine import TransitionDecision

        decision = TransitionDecision(
            should_transition=False,
            target_state=None,
            reason="no signal",
            confidence=0.0,
        )
        with pytest.raises(ValidationError):
            decision.should_transition = True  # type: ignore[misc]


class TestTransitionPolicy:
    """TransitionPolicy is a runtime-checkable protocol."""

    def test_conforming_class_accepted(self) -> None:
        from tgirl.state_machine import (
            SessionState,
            TransitionDecision,
            TransitionPolicy,
            TransitionSignal,
        )

        class MyPolicy:
            def evaluate(
                self,
                current_state: SessionState,
                signal: TransitionSignal,
            ) -> TransitionDecision:
                return TransitionDecision(
                    should_transition=False,
                    target_state=None,
                    reason="noop",
                    confidence=0.0,
                )

        assert isinstance(MyPolicy(), TransitionPolicy)

    def test_non_conforming_class_rejected(self) -> None:
        from tgirl.state_machine import TransitionPolicy

        class NotAPolicy:
            pass

        assert not isinstance(NotAPolicy(), TransitionPolicy)


class TestDelimiterTransitionPolicy:
    """DelimiterTransitionPolicy wraps delimiter detection for backward compat."""

    @staticmethod
    def _decode_alpha(ids: list[int]) -> str:
        return "".join(chr(65 + i) for i in ids)

    def test_no_transition_before_delimiter(self) -> None:
        from tgirl.state_machine import (
            DelimiterTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        decode = self._decode_alpha
        policy = DelimiterTransitionPolicy(
            delimiter="<tool>",
            tokenizer_decode=decode,
        )

        # Feed non-delimiter tokens via signal with token_id
        for token_id in [0, 1, 2]:
            signal = TransitionSignal(
                token_position=token_id,
                grammar_mask_overlap=0.0,
                token_entropy=0.0,
                token_log_prob=0.0,
                grammar_freedom=0.0,
            )
            decision = policy.evaluate(
                SessionState.FREEFORM,
                signal,
                token_id=token_id,
            )
            assert decision.should_transition is False

    def test_transition_on_delimiter(self) -> None:
        from tgirl.state_machine import (
            DelimiterTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        # Single-char delimiter for simplicity
        decode = self._decode_alpha
        policy = DelimiterTransitionPolicy(
            delimiter="C",
            tokenizer_decode=decode,
        )

        # Feed tokens until delimiter char
        for token_id in [0, 1]:  # "A", "B"
            signal = TransitionSignal(
                token_position=token_id,
                grammar_mask_overlap=0.0,
                token_entropy=0.0,
                token_log_prob=0.0,
                grammar_freedom=0.0,
            )
            decision = policy.evaluate(
                SessionState.FREEFORM,
                signal,
                token_id=token_id,
            )
            assert decision.should_transition is False

        # Token 2 = "C" = delimiter
        signal = TransitionSignal(
            token_position=2,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(
            SessionState.FREEFORM,
            signal,
            token_id=2,
        )
        assert decision.should_transition is True
        assert decision.target_state == SessionState.ROUTE
        assert decision.confidence == 1.0

    def test_reset_clears_state(self) -> None:
        from tgirl.state_machine import (
            DelimiterTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        decode = self._decode_alpha
        policy = DelimiterTransitionPolicy(
            delimiter="AB",
            tokenizer_decode=decode,
        )

        # Feed "A"
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        policy.evaluate(SessionState.FREEFORM, signal, token_id=0)

        # Reset
        policy.reset()

        # "B" alone should not trigger
        signal = TransitionSignal(
            token_position=1,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal, token_id=1)
        assert decision.should_transition is False

    def test_only_evaluates_in_freeform_state(self) -> None:
        from tgirl.state_machine import (
            DelimiterTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        decode = self._decode_alpha
        policy = DelimiterTransitionPolicy(
            delimiter="A",
            tokenizer_decode=decode,
        )

        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        # When not in FREEFORM, should not transition
        decision = policy.evaluate(SessionState.CONSTRAINED, signal, token_id=0)
        assert decision.should_transition is False

    def test_conforms_to_transition_policy_protocol(self) -> None:
        from tgirl.state_machine import (
            DelimiterTransitionPolicy,
            TransitionPolicy,
        )

        def decode(ids: list[int]) -> str:
            return ""
        policy = DelimiterTransitionPolicy(
            delimiter="<tool>",
            tokenizer_decode=decode,
        )
        assert isinstance(policy, TransitionPolicy)


class TestStateMachineModuleConstraints:
    """Verify state_machine.py has zero torch/mlx imports."""

    def test_no_torch_import(self) -> None:
        import sys

        # Clear any cached import
        if "tgirl.state_machine" in sys.modules:
            del sys.modules["tgirl.state_machine"]

        import tgirl.state_machine as sm

        source_file = sm.__file__
        assert source_file is not None
        with open(source_file) as f:
            source = f.read()
        assert "import torch" not in source
        assert "from torch" not in source

    def test_no_mlx_import(self) -> None:
        import tgirl.state_machine as sm

        source_file = sm.__file__
        assert source_file is not None
        with open(source_file) as f:
            source = f.read()
        assert "import mlx" not in source
        assert "from mlx" not in source
