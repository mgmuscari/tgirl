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


class TestBudgetTransitionPolicy:
    """BudgetTransitionPolicy forces transition after N freeform tokens."""

    def test_no_transition_before_budget(self) -> None:
        from tgirl.state_machine import (
            BudgetTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = BudgetTransitionPolicy(budget=3)
        for pos in range(3):
            signal = TransitionSignal(
                token_position=pos,
                grammar_mask_overlap=0.0,
                token_entropy=0.0,
                token_log_prob=0.0,
                grammar_freedom=0.0,
            )
            decision = policy.evaluate(SessionState.FREEFORM, signal)
            assert decision.should_transition is False

    def test_transition_at_budget(self) -> None:
        from tgirl.state_machine import (
            BudgetTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = BudgetTransitionPolicy(budget=3)
        # Feed 3 tokens (positions 0, 1, 2)
        for pos in range(3):
            signal = TransitionSignal(
                token_position=pos,
                grammar_mask_overlap=0.0,
                token_entropy=0.0,
                token_log_prob=0.0,
                grammar_freedom=0.0,
            )
            policy.evaluate(SessionState.FREEFORM, signal)

        # Position 3 = budget reached
        signal = TransitionSignal(
            token_position=3,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal)
        assert decision.should_transition is True
        assert decision.target_state == SessionState.ROUTE
        assert decision.confidence == 1.0

    def test_only_evaluates_in_freeform_state(self) -> None:
        from tgirl.state_machine import (
            BudgetTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = BudgetTransitionPolicy(budget=0)
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.CONSTRAINED, signal)
        assert decision.should_transition is False

    def test_reset_resets_counter(self) -> None:
        from tgirl.state_machine import (
            BudgetTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = BudgetTransitionPolicy(budget=2)
        # Feed 2 tokens
        for pos in range(2):
            signal = TransitionSignal(
                token_position=pos,
                grammar_mask_overlap=0.0,
                token_entropy=0.0,
                token_log_prob=0.0,
                grammar_freedom=0.0,
            )
            policy.evaluate(SessionState.FREEFORM, signal)

        policy.reset()

        # After reset, position 0 should not trigger
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal)
        assert decision.should_transition is False

    def test_conforms_to_protocol(self) -> None:
        from tgirl.state_machine import BudgetTransitionPolicy, TransitionPolicy

        assert isinstance(BudgetTransitionPolicy(budget=5), TransitionPolicy)


class TestImmediateTransitionPolicy:
    """ImmediateTransitionPolicy transitions immediately (budget=0)."""

    def test_transitions_on_first_token(self) -> None:
        from tgirl.state_machine import (
            ImmediateTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = ImmediateTransitionPolicy()
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal)
        assert decision.should_transition is True
        assert decision.target_state == SessionState.ROUTE

    def test_only_evaluates_in_freeform_state(self) -> None:
        from tgirl.state_machine import (
            ImmediateTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = ImmediateTransitionPolicy()
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.CONSTRAINED, signal)
        assert decision.should_transition is False

    def test_conforms_to_protocol(self) -> None:
        from tgirl.state_machine import ImmediateTransitionPolicy, TransitionPolicy

        assert isinstance(ImmediateTransitionPolicy(), TransitionPolicy)


class TestSamplingSessionTransitionPolicy:
    """Phase 1B: SamplingSession accepts transition_policy param."""

    def _make_session_with_policy(
        self,
        transition_policy=None,
        tool_delimiter_tokens: list[int] | None = None,
        freeform_tokens: int = 5,
    ):
        """Build a SamplingSession, optionally with a custom transition policy."""
        import torch

        from tgirl.registry import ToolRegistry
        from tgirl.sample import SamplingSession
        from tgirl.transport import TransportConfig
        from tgirl.types import SessionConfig

        vocab = [chr(i) for i in range(256)]

        def decode(ids: list[int]) -> str:
            return "".join(vocab[i] for i in ids)

        def encode(text: str) -> list[int]:
            return [ord(c) for c in text]

        registry = ToolRegistry()

        @registry.tool(quota=5, cost=1.0)
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        freeform_tok = list(range(freeform_tokens))
        if tool_delimiter_tokens is not None:
            all_tokens = freeform_tok + tool_delimiter_tokens
        else:
            all_tokens = freeform_tok

        token_idx = [0]

        def forward_fn(ctx: list[int]) -> torch.Tensor:
            idx = token_idx[0]
            token_idx[0] += 1
            logits = torch.zeros(256)
            if idx < len(all_tokens):
                logits[all_tokens[idx]] = 100.0
            else:
                logits[ord("x")] = 100.0
            return logits

        embeddings = torch.eye(256)

        def grammar_guide_factory(grammar_text: str):
            class MockGS:
                def __init__(self):
                    self._done = False

                def get_valid_mask(self, vocab_size: int) -> torch.Tensor:
                    mask = torch.zeros(vocab_size, dtype=torch.bool)
                    for c in '(greet "hi")':
                        mask[ord(c)] = True
                    return mask

                def is_accepting(self) -> bool:
                    return self._done

                def advance(self, token_id: int) -> None:
                    self._done = True

            return MockGS()

        config = SessionConfig(
            max_tool_cycles=10,
            freeform_max_tokens=20,
            constrained_max_tokens=10,
            session_timeout=30.0,
        )

        kwargs = dict(
            registry=registry,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            tokenizer_encode=encode,
            embeddings=embeddings,
            grammar_guide_factory=grammar_guide_factory,
            config=config,
            transport_config=TransportConfig(),
        )
        if transition_policy is not None:
            kwargs["transition_policy"] = transition_policy

        return SamplingSession(**kwargs)

    def test_default_transition_policy_is_delimiter(self) -> None:
        """SamplingSession defaults to DelimiterTransitionPolicy when none specified."""
        from tgirl.state_machine import DelimiterTransitionPolicy

        session = self._make_session_with_policy()
        assert isinstance(session._transition_policy, DelimiterTransitionPolicy)

    def test_custom_transition_policy_accepted(self) -> None:
        """SamplingSession accepts a custom transition policy."""
        from tgirl.state_machine import (
            SessionState,
            TransitionDecision,
            TransitionSignal,
        )

        class AlwaysTransitionPolicy:
            def evaluate(
                self,
                current_state: SessionState,
                signal: TransitionSignal,
                **kwargs: object,
            ) -> TransitionDecision:
                return TransitionDecision(
                    should_transition=True,
                    target_state=SessionState.ROUTE,
                    reason="always",
                    confidence=1.0,
                )

        session = self._make_session_with_policy(
            transition_policy=AlwaysTransitionPolicy()
        )
        assert isinstance(session._transition_policy, AlwaysTransitionPolicy)

    def test_default_policy_backward_compat_freeform_only(self) -> None:
        """With default policy, freeform-only sessions work identically."""
        session = self._make_session_with_policy(tool_delimiter_tokens=None)
        result = session.run(prompt_tokens=[])
        assert result.total_cycles == 0
        assert len(result.tool_calls) == 0
        assert result.total_tokens > 0

    def test_default_policy_backward_compat_with_delimiter(self) -> None:
        """With default policy, delimiter detection triggers constrained mode."""
        delimiter_tokens = [ord(c) for c in "<tool>"]
        session = self._make_session_with_policy(
            tool_delimiter_tokens=delimiter_tokens
        )
        result = session.run(prompt_tokens=[])
        assert result.total_cycles == 1
        assert len(result.tool_calls) == 1


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
