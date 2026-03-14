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


class TestCheckpoint:
    """Checkpoint tracks state for backtracking."""

    def test_construction(self) -> None:
        from tgirl.state_machine import Checkpoint

        cp = Checkpoint(
            position=5,
            tokens_so_far=(1, 2, 3, 4, 5),
            context_tokens=(100, 101, 102),
            grammar_text="(tool_call)",
            dead_end_tokens=frozenset(),
        )
        assert cp.position == 5
        assert cp.tokens_so_far == (1, 2, 3, 4, 5)
        assert cp.context_tokens == (100, 101, 102)
        assert cp.grammar_text == "(tool_call)"
        assert cp.dead_end_tokens == frozenset()

    def test_is_frozen(self) -> None:
        from pydantic import ValidationError

        from tgirl.state_machine import Checkpoint

        cp = Checkpoint(
            position=5,
            tokens_so_far=(1, 2, 3),
            context_tokens=(100,),
            grammar_text="x",
            dead_end_tokens=frozenset(),
        )
        with pytest.raises(ValidationError):
            cp.position = 10  # type: ignore[misc]

    def test_dead_end_tokens_is_frozenset(self) -> None:
        from tgirl.state_machine import Checkpoint

        cp = Checkpoint(
            position=0,
            tokens_so_far=(),
            context_tokens=(),
            grammar_text="",
            dead_end_tokens=frozenset({42, 99}),
        )
        assert isinstance(cp.dead_end_tokens, frozenset)
        assert 42 in cp.dead_end_tokens

    def test_with_added_dead_end(self) -> None:
        from tgirl.state_machine import Checkpoint

        cp = Checkpoint(
            position=0,
            tokens_so_far=(),
            context_tokens=(),
            grammar_text="",
            dead_end_tokens=frozenset({42}),
        )
        cp2 = cp.with_added_dead_end(99)
        assert cp2.dead_end_tokens == frozenset({42, 99})
        # Original unchanged
        assert cp.dead_end_tokens == frozenset({42})


class TestBacktrackEvent:
    """BacktrackEvent records a single backtrack occurrence."""

    def test_construction(self) -> None:
        from tgirl.state_machine import BacktrackEvent

        event = BacktrackEvent(
            checkpoint_position=5,
            trigger_position=10,
            trigger_log_prob=-2.5,
            dead_end_tokens_added=frozenset({42}),
        )
        assert event.checkpoint_position == 5
        assert event.trigger_position == 10
        assert event.trigger_log_prob == -2.5
        assert event.dead_end_tokens_added == frozenset({42})

    def test_is_frozen(self) -> None:
        from pydantic import ValidationError

        from tgirl.state_machine import BacktrackEvent

        event = BacktrackEvent(
            checkpoint_position=0,
            trigger_position=1,
            trigger_log_prob=-1.0,
            dead_end_tokens_added=frozenset(),
        )
        with pytest.raises(ValidationError):
            event.checkpoint_position = 5  # type: ignore[misc]


class TestConstrainedConfidenceMonitor:
    """ConstrainedConfidenceMonitor tracks confidence during constrained gen."""

    def test_construction_defaults(self) -> None:
        from tgirl.state_machine import ConstrainedConfidenceMonitor

        monitor = ConstrainedConfidenceMonitor()
        assert monitor.log_prob_threshold < 0
        assert monitor.window_size > 0
        assert monitor.freedom_threshold > 0
        assert monitor.max_backtracks > 0

    def test_custom_params(self) -> None:
        from tgirl.state_machine import ConstrainedConfidenceMonitor

        monitor = ConstrainedConfidenceMonitor(
            log_prob_threshold=-1.5,
            window_size=5,
            freedom_threshold=10,
            max_backtracks=2,
        )
        assert monitor.log_prob_threshold == -1.5
        assert monitor.window_size == 5
        assert monitor.freedom_threshold == 10
        assert monitor.max_backtracks == 2

    def test_should_checkpoint_at_high_freedom(self) -> None:
        from tgirl.state_machine import ConstrainedConfidenceMonitor

        monitor = ConstrainedConfidenceMonitor(freedom_threshold=5)
        assert monitor.should_checkpoint(grammar_valid_count=10) is True
        assert monitor.should_checkpoint(grammar_valid_count=3) is False

    def test_should_backtrack_below_threshold(self) -> None:
        from tgirl.state_machine import ConstrainedConfidenceMonitor

        monitor = ConstrainedConfidenceMonitor(
            log_prob_threshold=-1.0,
            window_size=3,
        )
        # Feed log probs that decline below threshold
        monitor.record_log_prob(-0.5)  # good
        monitor.record_log_prob(-0.8)  # ok
        monitor.record_log_prob(-1.5)  # bad
        # Window mean: (-0.5 + -0.8 + -1.5) / 3 = -0.933... above -1.0
        assert monitor.should_backtrack() is False

        # Another bad one pushes window below threshold
        monitor.record_log_prob(-2.0)
        # Window of last 3: (-0.8 + -1.5 + -2.0) / 3 = -1.433 < -1.0
        assert monitor.should_backtrack() is True

    def test_reset_clears_state(self) -> None:
        from tgirl.state_machine import ConstrainedConfidenceMonitor

        monitor = ConstrainedConfidenceMonitor(
            log_prob_threshold=-0.5,
            window_size=2,
        )
        monitor.record_log_prob(-5.0)
        monitor.record_log_prob(-5.0)
        assert monitor.should_backtrack() is True

        monitor.reset()
        assert monitor.should_backtrack() is False

    def test_backtrack_count_tracking(self) -> None:
        from tgirl.state_machine import ConstrainedConfidenceMonitor

        monitor = ConstrainedConfidenceMonitor(max_backtracks=2)
        assert monitor.backtracks_remaining == 2
        monitor.record_backtrack()
        assert monitor.backtracks_remaining == 1
        monitor.record_backtrack()
        assert monitor.backtracks_remaining == 0

    def test_exhaustion_detection(self) -> None:
        from tgirl.state_machine import ConstrainedConfidenceMonitor

        monitor = ConstrainedConfidenceMonitor(
            freedom_threshold=5,
            max_backtracks=1,
        )
        # Check sealed: dead_end_tokens > 50% of valid count
        assert monitor.is_checkpoint_sealed(
            dead_end_count=6, valid_count=10
        ) is True
        assert monitor.is_checkpoint_sealed(
            dead_end_count=4, valid_count=10
        ) is False


class TestBacktrackSteeringHook:
    """BacktrackSteeringHook applies negative logit bias on dead-end tokens."""

    def test_no_dead_ends_returns_no_bias(self) -> None:
        import torch

        from tgirl.sample import BacktrackSteeringHook

        hook = BacktrackSteeringHook(dead_end_tokens=frozenset())
        logits = torch.tensor([1.0, 2.0, 3.0])

        class MockGS:
            def get_valid_mask(self, vs: int) -> torch.Tensor:
                return torch.ones(vs, dtype=torch.bool)

            def is_accepting(self) -> bool:
                return False

            def advance(self, tid: int) -> None:
                pass

        result = hook.pre_forward(0, MockGS(), [], logits)
        assert result.logit_bias is None

    def test_dead_end_tokens_get_negative_bias(self) -> None:
        import torch

        from tgirl.sample import BacktrackSteeringHook

        hook = BacktrackSteeringHook(
            dead_end_tokens=frozenset({1, 2}),
            bias_strength=-100.0,
        )
        logits = torch.tensor([1.0, 2.0, 3.0])

        class MockGS:
            def get_valid_mask(self, vs: int) -> torch.Tensor:
                return torch.ones(vs, dtype=torch.bool)

            def is_accepting(self) -> bool:
                return False

            def advance(self, tid: int) -> None:
                pass

        result = hook.pre_forward(0, MockGS(), [], logits)
        assert result.logit_bias is not None
        assert result.logit_bias[1] == -100.0
        assert result.logit_bias[2] == -100.0
        assert 0 not in result.logit_bias

    def test_conforms_to_inference_hook_protocol(self) -> None:
        from tgirl.sample import BacktrackSteeringHook, InferenceHook

        hook = BacktrackSteeringHook(dead_end_tokens=frozenset())
        assert isinstance(hook, InferenceHook)


class TestBacktrackSteeringHookMlx:
    """BacktrackSteeringHookMlx applies negative logit bias via MLX."""

    def test_no_dead_ends_returns_no_bias(self) -> None:
        import mlx.core as mx

        from tgirl.sample_mlx import BacktrackSteeringHookMlx

        hook = BacktrackSteeringHookMlx(dead_end_tokens=frozenset())
        logits = mx.array([1.0, 2.0, 3.0])
        valid_mask = mx.ones((3,))

        result = hook.pre_forward(0, valid_mask, [], logits)
        assert result.logit_bias is None

    def test_dead_end_tokens_get_negative_bias(self) -> None:
        import mlx.core as mx

        from tgirl.sample_mlx import BacktrackSteeringHookMlx

        hook = BacktrackSteeringHookMlx(
            dead_end_tokens=frozenset({1, 2}),
            bias_strength=-100.0,
        )
        logits = mx.array([1.0, 2.0, 3.0])
        valid_mask = mx.ones((3,))

        result = hook.pre_forward(0, valid_mask, [], logits)
        assert result.logit_bias is not None
        assert result.logit_bias[1] == -100.0
        assert result.logit_bias[2] == -100.0
        assert 0 not in result.logit_bias

    def test_conforms_to_inference_hook_mlx_protocol(self) -> None:
        from tgirl.sample_mlx import (
            BacktrackSteeringHookMlx,
            InferenceHookMlx,
        )

        hook = BacktrackSteeringHookMlx(dead_end_tokens=frozenset())
        assert isinstance(hook, InferenceHookMlx)

    def test_custom_bias_strength(self) -> None:
        import mlx.core as mx

        from tgirl.sample_mlx import BacktrackSteeringHookMlx

        hook = BacktrackSteeringHookMlx(
            dead_end_tokens=frozenset({0}),
            bias_strength=-50.0,
        )
        logits = mx.array([1.0, 2.0])
        valid_mask = mx.ones((2,))

        result = hook.pre_forward(0, valid_mask, [], logits)
        assert result.logit_bias is not None
        assert result.logit_bias[0] == -50.0


class TestConfidenceTransitionPolicy:
    """ConfidenceTransitionPolicy uses Markov chain on model signals."""

    def test_construction_defaults(self) -> None:
        from tgirl.state_machine import ConfidenceTransitionPolicy

        policy = ConfidenceTransitionPolicy()
        assert policy.threshold > 0
        assert policy.w_readiness >= 0
        assert policy.w_certainty >= 0

    def test_no_transition_at_low_belief(self) -> None:
        from tgirl.state_machine import (
            ConfidenceTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = ConfidenceTransitionPolicy(threshold=0.8)
        # Low overlap, high entropy = low belief
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.01,
            token_entropy=5.0,
            token_log_prob=-3.0,
            grammar_freedom=0.5,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal)
        assert decision.should_transition is False

    def test_transition_at_high_belief(self) -> None:
        from tgirl.state_machine import (
            ConfidenceTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = ConfidenceTransitionPolicy(threshold=0.3)
        # High overlap, low entropy = high belief
        # Feed enough signals to accumulate belief
        for _ in range(5):
            signal = TransitionSignal(
                token_position=0,
                grammar_mask_overlap=0.9,
                token_entropy=0.5,
                token_log_prob=-0.1,
                grammar_freedom=0.01,
            )
            decision = policy.evaluate(SessionState.FREEFORM, signal)

        # After accumulation, should transition
        assert decision.should_transition is True
        assert decision.target_state == SessionState.ROUTE

    def test_only_evaluates_in_freeform_state(self) -> None:
        from tgirl.state_machine import (
            ConfidenceTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = ConfidenceTransitionPolicy()
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=1.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.CONSTRAINED, signal)
        assert decision.should_transition is False

    def test_reset_clears_belief(self) -> None:
        from tgirl.state_machine import (
            ConfidenceTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = ConfidenceTransitionPolicy(threshold=0.3)
        # Build up belief
        for _ in range(10):
            signal = TransitionSignal(
                token_position=0,
                grammar_mask_overlap=0.9,
                token_entropy=0.1,
                token_log_prob=-0.01,
                grammar_freedom=0.0,
            )
            policy.evaluate(SessionState.FREEFORM, signal)

        policy.reset()

        # After reset, low signal should not transition
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.01,
            token_entropy=5.0,
            token_log_prob=-3.0,
            grammar_freedom=0.5,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal)
        assert decision.should_transition is False

    def test_conforms_to_protocol(self) -> None:
        from tgirl.state_machine import ConfidenceTransitionPolicy, TransitionPolicy

        assert isinstance(ConfidenceTransitionPolicy(), TransitionPolicy)


class TestCompositeTransitionPolicy:
    """CompositeTransitionPolicy combines policies with OR/AND modes."""

    def test_or_mode_any_triggers(self) -> None:
        from tgirl.state_machine import (
            CompositeTransitionPolicy,
            ImmediateTransitionPolicy,
            SessionState,
            TransitionDecision,
            TransitionSignal,
        )

        class NeverPolicy:
            def evaluate(self, cs, sig, **kw):
                return TransitionDecision(
                    should_transition=False,
                    target_state=None,
                    reason="never",
                    confidence=0.0,
                )

        policy = CompositeTransitionPolicy(
            policies=[NeverPolicy(), ImmediateTransitionPolicy()],
            mode="or",
        )
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal)
        assert decision.should_transition is True

    def test_and_mode_requires_all(self) -> None:
        from tgirl.state_machine import (
            CompositeTransitionPolicy,
            ImmediateTransitionPolicy,
            SessionState,
            TransitionDecision,
            TransitionSignal,
        )

        class NeverPolicy:
            def evaluate(self, cs, sig, **kw):
                return TransitionDecision(
                    should_transition=False,
                    target_state=None,
                    reason="never",
                    confidence=0.0,
                )

        policy = CompositeTransitionPolicy(
            policies=[NeverPolicy(), ImmediateTransitionPolicy()],
            mode="and",
        )
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal)
        assert decision.should_transition is False

    def test_and_mode_all_agree(self) -> None:
        from tgirl.state_machine import (
            CompositeTransitionPolicy,
            ImmediateTransitionPolicy,
            SessionState,
            TransitionSignal,
        )

        policy = CompositeTransitionPolicy(
            policies=[ImmediateTransitionPolicy(), ImmediateTransitionPolicy()],
            mode="and",
        )
        signal = TransitionSignal(
            token_position=0,
            grammar_mask_overlap=0.0,
            token_entropy=0.0,
            token_log_prob=0.0,
            grammar_freedom=0.0,
        )
        decision = policy.evaluate(SessionState.FREEFORM, signal)
        assert decision.should_transition is True

    def test_conforms_to_protocol(self) -> None:
        from tgirl.state_machine import CompositeTransitionPolicy, TransitionPolicy

        policy = CompositeTransitionPolicy(policies=[], mode="or")
        assert isinstance(policy, TransitionPolicy)


class TestMakeTransitionPolicy:
    """Test the BFCL benchmark policy parser."""

    def test_delimiter_policy(self) -> None:
        from benchmarks.run_bfcl import make_transition_policy
        from tgirl.state_machine import DelimiterTransitionPolicy

        def decode(ids: list[int]) -> str:
            return ""

        policy = make_transition_policy("delimiter", tokenizer_decode=decode)
        assert isinstance(policy, DelimiterTransitionPolicy)

    def test_immediate_policy(self) -> None:
        from benchmarks.run_bfcl import make_transition_policy
        from tgirl.state_machine import ImmediateTransitionPolicy

        policy = make_transition_policy("immediate")
        assert isinstance(policy, ImmediateTransitionPolicy)

    def test_budget_policy(self) -> None:
        from benchmarks.run_bfcl import make_transition_policy
        from tgirl.state_machine import BudgetTransitionPolicy

        policy = make_transition_policy("budget:5")
        assert isinstance(policy, BudgetTransitionPolicy)
        assert policy._budget == 5

    def test_invalid_policy_raises(self) -> None:
        from benchmarks.run_bfcl import make_transition_policy

        with pytest.raises(ValueError, match="Unknown transition policy"):
            make_transition_policy("nonexistent")

    def test_invalid_budget_raises(self) -> None:
        from benchmarks.run_bfcl import make_transition_policy

        with pytest.raises(ValueError, match="Invalid budget policy"):
            make_transition_policy("budget:abc")


class TestComputeTransitionSignal:
    """compute_transition_signal computes framework-agnostic signal from raw logits."""

    def test_returns_transition_signal(self) -> None:
        """Returns a TransitionSignal with correct type."""
        from tgirl.state_machine import TransitionSignal, compute_transition_signal

        # Use plain Python lists as stand-in for framework tensors
        logits = [1.0, 2.0, 3.0, 4.0]
        grammar_valid_mask = [1.0, 0.0, 1.0, 0.0]

        import math

        def softmax_fn(x):
            max_x = max(x)
            exps = [math.exp(v - max_x) for v in x]
            s = sum(exps)
            return [e / s for e in exps]

        def sum_fn(x):
            return sum(x)

        def log_fn(x):
            return [math.log(v) if v > 0 else float("-inf") for v in x]

        signal = compute_transition_signal(
            token_position=5,
            logits=logits,
            grammar_valid_mask=grammar_valid_mask,
            softmax_fn=softmax_fn,
            sum_fn=sum_fn,
            log_fn=log_fn,
            vocab_size=4,
        )
        assert isinstance(signal, TransitionSignal)
        assert signal.token_position == 5

    def test_grammar_mask_overlap_computed(self) -> None:
        """grammar_mask_overlap = sum(softmax(logits) * grammar_valid_mask)."""
        from tgirl.state_machine import compute_transition_signal

        import math

        # Uniform logits, half of tokens valid
        logits = [0.0, 0.0, 0.0, 0.0]
        grammar_valid_mask = [1.0, 1.0, 0.0, 0.0]

        def softmax_fn(x):
            max_x = max(x)
            exps = [math.exp(v - max_x) for v in x]
            s = sum(exps)
            return [e / s for e in exps]

        def sum_fn(x):
            return sum(x)

        def log_fn(x):
            return [math.log(v) if v > 0 else float("-inf") for v in x]

        signal = compute_transition_signal(
            token_position=0,
            logits=logits,
            grammar_valid_mask=grammar_valid_mask,
            softmax_fn=softmax_fn,
            sum_fn=sum_fn,
            log_fn=log_fn,
            vocab_size=4,
        )
        # Uniform over 4 tokens, 2 valid: overlap = 0.5
        assert abs(signal.grammar_mask_overlap - 0.5) < 1e-6

    def test_grammar_freedom_computed(self) -> None:
        """grammar_freedom = sum(grammar_valid_mask) / vocab_size."""
        from tgirl.state_machine import compute_transition_signal

        import math

        logits = [0.0, 0.0, 0.0, 0.0, 0.0]
        grammar_valid_mask = [1.0, 0.0, 1.0, 0.0, 1.0]

        def softmax_fn(x):
            max_x = max(x)
            exps = [math.exp(v - max_x) for v in x]
            s = sum(exps)
            return [e / s for e in exps]

        def sum_fn(x):
            return sum(x)

        def log_fn(x):
            return [math.log(v) if v > 0 else float("-inf") for v in x]

        signal = compute_transition_signal(
            token_position=0,
            logits=logits,
            grammar_valid_mask=grammar_valid_mask,
            softmax_fn=softmax_fn,
            sum_fn=sum_fn,
            log_fn=log_fn,
            vocab_size=5,
        )
        # 3 valid out of 5
        assert abs(signal.grammar_freedom - 0.6) < 1e-6

    def test_token_entropy_computed(self) -> None:
        """token_entropy = -sum(p * log(p)) over the distribution."""
        from tgirl.state_machine import compute_transition_signal

        import math

        # Uniform distribution over 4 tokens -> entropy = log(4)
        logits = [0.0, 0.0, 0.0, 0.0]
        grammar_valid_mask = [1.0, 1.0, 1.0, 1.0]

        def softmax_fn(x):
            max_x = max(x)
            exps = [math.exp(v - max_x) for v in x]
            s = sum(exps)
            return [e / s for e in exps]

        def sum_fn(x):
            return sum(x)

        def log_fn(x):
            return [math.log(v) if v > 0 else float("-inf") for v in x]

        signal = compute_transition_signal(
            token_position=0,
            logits=logits,
            grammar_valid_mask=grammar_valid_mask,
            softmax_fn=softmax_fn,
            sum_fn=sum_fn,
            log_fn=log_fn,
            vocab_size=4,
        )
        expected_entropy = math.log(4)  # ~1.386
        assert abs(signal.token_entropy - expected_entropy) < 1e-5

    def test_zero_torch_mlx_in_function(self) -> None:
        """compute_transition_signal itself has no torch/mlx imports."""
        import inspect

        from tgirl.state_machine import compute_transition_signal

        source = inspect.getsource(compute_transition_signal)
        assert "torch" not in source
        assert "mlx" not in source


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
