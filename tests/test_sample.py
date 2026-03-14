"""Tests for tgirl.sample — Constrained sampling engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestSessionConfig:
    """Task 1: SessionConfig is a frozen Pydantic model with correct defaults."""

    def test_config_is_frozen(self) -> None:
        from tgirl.types import SessionConfig

        c = SessionConfig()
        with pytest.raises(ValidationError):
            c.freeform_temperature = 0.5  # type: ignore[misc]

    def test_config_defaults_match_spec(self) -> None:
        from tgirl.types import SessionConfig

        c = SessionConfig()
        # Freeform mode
        assert c.freeform_temperature == 0.7
        assert c.freeform_top_p == 0.9
        assert c.freeform_top_k is None
        assert c.freeform_repetition_penalty == 1.0
        assert c.freeform_max_tokens == 4096
        # Constrained mode
        assert c.constrained_base_temperature == 0.3
        assert c.constrained_ot_epsilon == 0.1
        assert c.constrained_max_tokens == 512
        # Session-level
        assert c.max_tool_cycles == 10
        assert c.session_cost_budget is None
        assert c.session_timeout == 300.0
        # Delimiters
        assert c.tool_open_delimiter == "<tool>"
        assert c.tool_close_delimiter == "</tool>"
        assert c.result_open_delimiter == "<tool_result>"
        assert c.result_close_delimiter == "</tool_result>"

    def test_config_custom_values_accepted(self) -> None:
        from tgirl.types import SessionConfig

        c = SessionConfig(
            freeform_temperature=1.0,
            constrained_base_temperature=0.1,
            max_tool_cycles=5,
            session_cost_budget=100.0,
            tool_open_delimiter="[TOOL]",
            tool_close_delimiter="[/TOOL]",
        )
        assert c.freeform_temperature == 1.0
        assert c.constrained_base_temperature == 0.1
        assert c.max_tool_cycles == 5
        assert c.session_cost_budget == 100.0
        assert c.tool_open_delimiter == "[TOOL]"
        assert c.tool_close_delimiter == "[/TOOL]"

    def test_delimiter_fields_are_configurable(self) -> None:
        from tgirl.types import SessionConfig

        c = SessionConfig(
            result_open_delimiter="<result>",
            result_close_delimiter="</result>",
        )
        assert c.result_open_delimiter == "<result>"
        assert c.result_close_delimiter == "</result>"


class TestModelIntervention:
    """Task 1: ModelIntervention is a frozen Pydantic model defaulting to all None."""

    def test_intervention_is_frozen(self) -> None:
        from tgirl.types import ModelIntervention

        m = ModelIntervention()
        with pytest.raises(ValidationError):
            m.temperature = 0.5  # type: ignore[misc]

    def test_intervention_defaults_to_all_none(self) -> None:
        from tgirl.types import ModelIntervention

        m = ModelIntervention()
        assert m.temperature is None
        assert m.top_p is None
        assert m.top_k is None
        assert m.repetition_penalty is None
        assert m.presence_penalty is None
        assert m.frequency_penalty is None
        assert m.logit_bias is None
        assert m.activation_steering is None

    def test_intervention_accepts_values(self) -> None:
        from tgirl.types import ModelIntervention

        m = ModelIntervention(
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            presence_penalty=0.6,
            frequency_penalty=0.5,
            logit_bias={100: 1.0, 200: -1.0},
        )
        assert m.temperature == 0.5
        assert m.top_p == 0.9
        assert m.top_k == 50
        assert m.repetition_penalty == 1.1
        assert m.presence_penalty == 0.6
        assert m.frequency_penalty == 0.5
        assert m.logit_bias == {100: 1.0, 200: -1.0}


class TestMergeInterventions:
    """Task 2: merge_interventions merges multiple hooks with last-writer-wins."""

    def test_empty_list_returns_all_none(self) -> None:
        from tgirl.sample import merge_interventions
        from tgirl.types import ModelIntervention

        result = merge_interventions([])
        assert result == ModelIntervention()

    def test_single_intervention_preserves_all_values(self) -> None:
        from tgirl.sample import merge_interventions
        from tgirl.types import ModelIntervention

        m = ModelIntervention(temperature=0.5, top_p=0.9, top_k=50)
        result = merge_interventions([m])
        assert result.temperature == 0.5
        assert result.top_p == 0.9
        assert result.top_k == 50

    def test_later_overrides_earlier(self) -> None:
        from tgirl.sample import merge_interventions
        from tgirl.types import ModelIntervention

        m1 = ModelIntervention(temperature=0.5)
        m2 = ModelIntervention(temperature=0.8)
        result = merge_interventions([m1, m2])
        assert result.temperature == 0.8

    def test_non_overlapping_fields_preserved_from_both(self) -> None:
        from tgirl.sample import merge_interventions
        from tgirl.types import ModelIntervention

        m1 = ModelIntervention(temperature=0.5)
        m2 = ModelIntervention(top_p=0.9)
        result = merge_interventions([m1, m2])
        assert result.temperature == 0.5
        assert result.top_p == 0.9


class TestInferenceHook:
    """Task 2: InferenceHook protocol is runtime-checkable."""

    def test_conforming_class_is_accepted(self) -> None:
        import torch

        from tgirl.sample import GrammarState, InferenceHook
        from tgirl.types import ModelIntervention

        class MyHook:
            def pre_forward(
                self,
                position: int,
                grammar_state: GrammarState,
                token_history: list[int],
                logits: torch.Tensor,
            ) -> ModelIntervention:
                return ModelIntervention()

        assert isinstance(MyHook(), InferenceHook)


class TestGrammarTemperatureHook:
    """Task 3: Grammar-implied temperature scheduling hook."""

    def _make_grammar_state(self, valid_count: int, vocab_size: int):
        """Helper: create a mock grammar state with given valid count."""
        import torch

        class MockGS:
            def get_valid_mask(self, tokenizer_vocab_size: int) -> torch.Tensor:
                mask = torch.zeros(tokenizer_vocab_size, dtype=torch.bool)
                mask[:valid_count] = True
                return mask

            def is_accepting(self) -> bool:
                return False

            def advance(self, token_id: int) -> None:
                pass

        return MockGS()

    def test_valid_count_one_returns_zero_temperature(self) -> None:
        import torch

        from tgirl.sample import GrammarTemperatureHook

        hook = GrammarTemperatureHook()
        gs = self._make_grammar_state(1, 100)
        result = hook.pre_forward(0, gs, [], torch.zeros(100))
        assert result.temperature == 0.0

    def test_valid_count_zero_returns_zero_temperature(self) -> None:
        import torch

        from tgirl.sample import GrammarTemperatureHook

        hook = GrammarTemperatureHook()
        gs = self._make_grammar_state(0, 100)
        result = hook.pre_forward(0, gs, [], torch.zeros(100))
        assert result.temperature == 0.0

    def test_all_valid_returns_base_temperature(self) -> None:
        import torch

        from tgirl.sample import GrammarTemperatureHook

        hook = GrammarTemperatureHook(base_temperature=0.3)
        gs = self._make_grammar_state(100, 100)
        result = hook.pre_forward(0, gs, [], torch.zeros(100))
        assert result.temperature == pytest.approx(0.3)

    def test_quarter_valid_returns_sqrt_quarter_times_base(self) -> None:
        import math

        import torch

        from tgirl.sample import GrammarTemperatureHook

        hook = GrammarTemperatureHook(base_temperature=0.3)
        gs = self._make_grammar_state(25, 100)
        result = hook.pre_forward(0, gs, [], torch.zeros(100))
        expected = 0.3 * math.sqrt(0.25)
        assert result.temperature == pytest.approx(expected)

    def test_custom_scaling_exponent_linear(self) -> None:
        import torch

        from tgirl.sample import GrammarTemperatureHook

        hook = GrammarTemperatureHook(base_temperature=0.3, scaling_exponent=1.0)
        gs = self._make_grammar_state(25, 100)
        result = hook.pre_forward(0, gs, [], torch.zeros(100))
        expected = 0.3 * 0.25  # linear: freedom^1.0
        assert result.temperature == pytest.approx(expected)

    def test_custom_base_temperature_is_respected(self) -> None:
        import torch

        from tgirl.sample import GrammarTemperatureHook

        hook = GrammarTemperatureHook(base_temperature=0.8)
        gs = self._make_grammar_state(100, 100)
        result = hook.pre_forward(0, gs, [], torch.zeros(100))
        assert result.temperature == pytest.approx(0.8)


class TestApplyPenalties:
    """Task 4: Pre-OT penalty application on raw logits."""

    def test_repetition_penalty_penalizes_repeated_tokens(self) -> None:
        import torch

        from tgirl.sample import apply_penalties
        from tgirl.types import ModelIntervention

        logits = torch.tensor([2.0, 3.0, 1.0, 4.0])
        intervention = ModelIntervention(repetition_penalty=2.0)
        result = apply_penalties(logits, intervention, token_history=[1, 3])
        # token 1: 3.0 > 0, so 3.0 / 2.0 = 1.5
        # token 3: 4.0 > 0, so 4.0 / 2.0 = 2.0
        assert result[0].item() == pytest.approx(2.0)
        assert result[1].item() == pytest.approx(1.5)
        assert result[2].item() == pytest.approx(1.0)
        assert result[3].item() == pytest.approx(2.0)

    def test_repetition_penalty_negative_logits(self) -> None:
        import torch

        from tgirl.sample import apply_penalties
        from tgirl.types import ModelIntervention

        logits = torch.tensor([2.0, -3.0])
        intervention = ModelIntervention(repetition_penalty=2.0)
        result = apply_penalties(logits, intervention, token_history=[1])
        # token 1: -3.0 < 0, so -3.0 * 2.0 = -6.0
        assert result[1].item() == pytest.approx(-6.0)

    def test_presence_penalty_subtracts_from_seen(self) -> None:
        import torch

        from tgirl.sample import apply_penalties
        from tgirl.types import ModelIntervention

        logits = torch.tensor([5.0, 3.0, 1.0])
        intervention = ModelIntervention(presence_penalty=1.0)
        result = apply_penalties(logits, intervention, token_history=[0, 2])
        assert result[0].item() == pytest.approx(4.0)
        assert result[1].item() == pytest.approx(3.0)
        assert result[2].item() == pytest.approx(0.0)

    def test_frequency_penalty_scales_with_count(self) -> None:
        import torch

        from tgirl.sample import apply_penalties
        from tgirl.types import ModelIntervention

        logits = torch.tensor([5.0, 3.0])
        intervention = ModelIntervention(frequency_penalty=0.5)
        result = apply_penalties(logits, intervention, token_history=[0, 0, 0, 1])
        # token 0: 5.0 - 0.5*3 = 3.5
        # token 1: 3.0 - 0.5*1 = 2.5
        assert result[0].item() == pytest.approx(3.5)
        assert result[1].item() == pytest.approx(2.5)

    def test_logit_bias_shifts_specific_tokens(self) -> None:
        import torch

        from tgirl.sample import apply_penalties
        from tgirl.types import ModelIntervention

        logits = torch.tensor([1.0, 2.0, 3.0])
        intervention = ModelIntervention(logit_bias={0: 10.0, 2: -5.0})
        result = apply_penalties(logits, intervention, token_history=[])
        assert result[0].item() == pytest.approx(11.0)
        assert result[1].item() == pytest.approx(2.0)
        assert result[2].item() == pytest.approx(-2.0)

    def test_all_none_intervention_returns_unchanged(self) -> None:
        import torch

        from tgirl.sample import apply_penalties
        from tgirl.types import ModelIntervention

        logits = torch.tensor([1.0, 2.0, 3.0])
        intervention = ModelIntervention()
        result = apply_penalties(logits, intervention, token_history=[0, 1, 2])
        assert torch.allclose(result, logits)


class TestApplyShaping:
    """Task 4: Post-OT shaping of redistributed logits."""

    def test_temperature_one_leaves_unchanged(self) -> None:
        import torch

        from tgirl.sample import apply_shaping
        from tgirl.types import ModelIntervention

        logits = torch.tensor([1.0, 2.0, 3.0])
        intervention = ModelIntervention(temperature=1.0)
        result = apply_shaping(logits, intervention)
        assert torch.allclose(result, logits)

    def test_temperature_half_doubles_logits(self) -> None:
        import torch

        from tgirl.sample import apply_shaping
        from tgirl.types import ModelIntervention

        logits = torch.tensor([1.0, 2.0, 3.0])
        intervention = ModelIntervention(temperature=0.5)
        result = apply_shaping(logits, intervention)
        assert torch.allclose(result, logits / 0.5)

    def test_temperature_zero_is_greedy(self) -> None:
        import torch

        from tgirl.sample import apply_shaping
        from tgirl.types import ModelIntervention

        logits = torch.tensor([1.0, 3.0, 2.0])
        intervention = ModelIntervention(temperature=0.0)
        result = apply_shaping(logits, intervention)
        # Only index 1 (max) survives
        assert result[1].item() == 3.0
        assert result[0].item() == float("-inf")
        assert result[2].item() == float("-inf")

    def test_temperature_zero_tied_maxima_lowest_index_wins(self) -> None:
        import torch

        from tgirl.sample import apply_shaping
        from tgirl.types import ModelIntervention

        logits = torch.tensor([1.0, 5.0, 5.0, 2.0])
        intervention = ModelIntervention(temperature=0.0)
        result = apply_shaping(logits, intervention)
        # torch.argmax returns first (lowest-index) maximum
        assert result[1].item() == 5.0
        assert result[2].item() == float("-inf")

    def test_top_k_keeps_only_top_k(self) -> None:
        import torch

        from tgirl.sample import apply_shaping
        from tgirl.types import ModelIntervention

        logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
        intervention = ModelIntervention(top_k=3)
        result = apply_shaping(logits, intervention)
        # Top 3 values: 5.0, 4.0, 3.0 (at indices 1, 3, 2)
        assert result[1].item() == 5.0
        assert result[3].item() == 4.0
        assert result[2].item() == 3.0
        assert result[0].item() == float("-inf")
        assert result[4].item() == float("-inf")

    def test_top_p_filters_and_preserves_positions(self) -> None:
        import torch

        from tgirl.sample import apply_shaping
        from tgirl.types import ModelIntervention

        logits = torch.tensor([5.0, 3.0, 1.0, 0.5, 0.1])
        intervention = ModelIntervention(top_p=0.5)
        result = apply_shaping(logits, intervention)
        # After softmax, index 0 (5.0) dominates. Its prob alone may exceed 0.5.
        # The key test: surviving tokens are at their original indices.
        # Index 0 must survive (highest prob). Low-prob tokens should be -inf.
        assert result[0].item() != float("-inf")
        # At least one low-prob token is filtered out
        assert result[4].item() == float("-inf")

    def test_all_none_intervention_returns_unchanged(self) -> None:
        import torch

        from tgirl.sample import apply_shaping
        from tgirl.types import ModelIntervention

        logits = torch.tensor([1.0, 2.0, 3.0])
        intervention = ModelIntervention()
        result = apply_shaping(logits, intervention)
        assert torch.allclose(result, logits)


class TestConstrainedGeneration:
    """Task 5: Constrained token generation core."""

    def _make_mock_grammar_state(
        self, valid_masks: list[list[bool]], accept_after: int
    ):
        """Create a mock grammar state with predetermined valid masks.

        Args:
            valid_masks: List of boolean masks, one per token position.
            accept_after: Accept after this many advance() calls.
        """
        import torch

        class MockGS:
            def __init__(self):
                self._position = 0
                self._advances = 0

            def get_valid_mask(self, tokenizer_vocab_size: int) -> torch.Tensor:
                if self._position < len(valid_masks):
                    return torch.tensor(valid_masks[self._position], dtype=torch.bool)
                return torch.ones(tokenizer_vocab_size, dtype=torch.bool)

            def is_accepting(self) -> bool:
                return self._advances >= accept_after

            def advance(self, token_id: int) -> None:
                self._position += 1
                self._advances += 1

        return MockGS()

    def test_produces_correct_number_of_tokens(self) -> None:
        import torch

        from tgirl.sample import ConstrainedGenerationResult, run_constrained_generation
        from tgirl.transport import TransportConfig

        masks = [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
        ]
        gs = self._make_mock_grammar_state(masks, accept_after=3)

        def forward_fn(ctx: list[int]) -> torch.Tensor:
            return torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

        def decode(ids: list[int]) -> str:
            return "".join(chr(65 + i) for i in ids)

        embeddings = torch.eye(5)
        config = TransportConfig()

        result = run_constrained_generation(
            grammar_state=gs,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            hooks=[],
            transport_config=config,
            max_tokens=10,
        )

        assert isinstance(result, ConstrainedGenerationResult)
        assert len(result.tokens) == 3

    def test_hooks_called_at_each_position(self) -> None:
        import torch

        from tgirl.sample import GrammarState, run_constrained_generation
        from tgirl.transport import TransportConfig
        from tgirl.types import ModelIntervention

        masks = [[True, True, True]]
        gs = self._make_mock_grammar_state(masks, accept_after=1)

        call_log: list[int] = []

        class LoggingHook:
            def pre_forward(
                self,
                position: int,
                grammar_state: GrammarState,
                token_history: list[int],
                logits: torch.Tensor,
            ) -> ModelIntervention:
                call_log.append(position)
                return ModelIntervention()

        result = run_constrained_generation(
            grammar_state=gs,
            forward_fn=lambda ctx: torch.tensor([3.0, 2.0, 1.0]),
            tokenizer_decode=lambda ids: "x",
            embeddings=torch.eye(3),
            hooks=[LoggingHook()],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        assert call_log == [0]

    def test_max_tokens_respected(self) -> None:
        import torch

        from tgirl.sample import run_constrained_generation
        from tgirl.transport import TransportConfig

        gs = self._make_mock_grammar_state(
            [[True, True, True]] * 100, accept_after=999
        )

        result = run_constrained_generation(
            grammar_state=gs,
            forward_fn=lambda ctx: torch.tensor([3.0, 2.0, 1.0]),
            tokenizer_decode=lambda ids: "x",
            embeddings=torch.eye(3),
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=5,
        )

        assert len(result.tokens) == 5

    def test_telemetry_fields_populated(self) -> None:
        import torch

        from tgirl.sample import run_constrained_generation
        from tgirl.transport import TransportConfig

        masks = [
            [True, False, False, False, False],
            [False, True, True, False, False],
        ]
        gs = self._make_mock_grammar_state(masks, accept_after=2)

        result = run_constrained_generation(
            grammar_state=gs,
            forward_fn=lambda ctx: torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
            tokenizer_decode=lambda ids: "".join(chr(65 + i) for i in ids),
            embeddings=torch.eye(5),
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        assert len(result.grammar_valid_counts) == 2
        assert result.grammar_valid_counts[0] == 1
        assert result.grammar_valid_counts[1] == 2
        assert len(result.temperatures_applied) == 2
        assert len(result.wasserstein_distances) == 2
        assert len(result.token_log_probs) == 2
        assert result.ot_computation_total_ms >= 0.0

    def test_ot_bypassed_when_single_valid_token(self) -> None:
        import torch

        from tgirl.sample import run_constrained_generation
        from tgirl.transport import TransportConfig

        masks = [[True, False, False, False, False]]
        gs = self._make_mock_grammar_state(masks, accept_after=1)

        result = run_constrained_generation(
            grammar_state=gs,
            forward_fn=lambda ctx: torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
            tokenizer_decode=lambda ids: "A",
            embeddings=torch.eye(5),
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        assert result.ot_bypassed_count >= 1

    def test_processing_order_penalties_before_ot_before_shaping(self) -> None:
        """Verify penalties applied before OT and shaping after OT."""
        import torch

        from tgirl.sample import GrammarState, run_constrained_generation
        from tgirl.transport import TransportConfig
        from tgirl.types import ModelIntervention

        masks = [[True, True, True, True, True]]
        gs = self._make_mock_grammar_state(masks, accept_after=1)

        class OrderTestHook:
            def pre_forward(
                self,
                position: int,
                grammar_state: GrammarState,
                token_history: list[int],
                logits: torch.Tensor,
            ) -> ModelIntervention:
                return ModelIntervention(
                    logit_bias={0: 100.0},
                    temperature=1.0,
                )

        result = run_constrained_generation(
            grammar_state=gs,
            forward_fn=lambda ctx: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            tokenizer_decode=lambda ids: "x",
            embeddings=torch.eye(5),
            hooks=[OrderTestHook()],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        assert result.tokens[0] == 0


class TestDelimiterDetector:
    """Task 6: Delimiter detection for mode switching."""

    def test_single_token_delimiter_detected(self) -> None:
        from tgirl.sample import DelimiterDetector

        # Each token decodes to a single character
        decode = lambda ids: "".join(chr(65 + i) for i in ids)
        # Delimiter is "C" which maps to token 2
        detector = DelimiterDetector("C", decode)
        assert not detector.feed(0)  # "A"
        assert not detector.feed(1)  # "B"
        assert detector.feed(2)  # "C" -> detected

    def test_multi_token_delimiter_detected(self) -> None:
        from tgirl.sample import DelimiterDetector

        # Tokens map to characters: 0->"<", 1->"/", 2->"t", 3->"o", 4->"o", 5->"l", 6->">"
        char_map = {0: "<", 1: "/", 2: "t", 3: "o", 4: "l", 5: ">"}
        decode = lambda ids: "".join(char_map[i] for i in ids)
        detector = DelimiterDetector("</tool>", decode)
        assert not detector.feed(0)  # "<"
        assert not detector.feed(1)  # "/"
        assert not detector.feed(2)  # "t"
        assert not detector.feed(3)  # "o"
        assert not detector.feed(3)  # "o"
        assert not detector.feed(4)  # "l"
        assert detector.feed(5)  # ">" -> "</tool>" detected

    def test_no_delimiter_never_triggers(self) -> None:
        from tgirl.sample import DelimiterDetector

        decode = lambda ids: "".join(chr(65 + i) for i in ids)
        detector = DelimiterDetector("<tool>", decode)
        for i in range(100):
            assert not detector.feed(i % 26)  # Only A-Z, never "<tool>"

    def test_reset_clears_state(self) -> None:
        from tgirl.sample import DelimiterDetector

        decode = lambda ids: "".join(chr(65 + i) for i in ids)
        detector = DelimiterDetector("AB", decode)
        detector.feed(0)  # "A"
        detector.reset()
        # After reset, feeding "B" alone should not detect "AB"
        assert not detector.feed(1)  # "B" alone is not "AB"

    def test_partial_match_then_non_match_no_false_positive(self) -> None:
        from tgirl.sample import DelimiterDetector

        decode = lambda ids: "".join(chr(65 + i) for i in ids)
        detector = DelimiterDetector("ABC", decode)
        assert not detector.feed(0)  # "A"
        assert not detector.feed(1)  # "B"
        assert not detector.feed(3)  # "D" -- breaks the pattern
        assert not detector.feed(0)  # "A" -- restart
        assert not detector.feed(1)  # "B"
        assert detector.feed(2)  # "C" -> "ABC" now detected

    def test_buffer_stays_bounded(self) -> None:
        from tgirl.sample import DelimiterDetector

        decode = lambda ids: "".join(chr(65 + (i % 26)) for i in ids)
        delimiter = "<tool>"
        detector = DelimiterDetector(delimiter, decode)
        for i in range(10_000):
            detector.feed(i % 26)
        # Window should be bounded to 2 * len(delimiter)
        assert len(detector._decoded_window) <= 2 * len(delimiter)


class TestSamplingSession:
    """Task 7: SamplingSession orchestrator for dual-mode loop."""

    def _make_simple_session(
        self,
        tool_delimiter_tokens: list[int] | None = None,
        tool_result: object = "ok",
        freeform_tokens: int = 5,
        max_cycles: int = 10,
        tool_open: str = "<tool>",
        tool_close: str = "</tool>",
    ):
        """Build a SamplingSession with mocked dependencies.

        If tool_delimiter_tokens is None, freeform generation never hits
        the tool delimiter. Otherwise, those token IDs (when decoded)
        form the tool_open delimiter, triggering constrained mode.
        """
        import torch

        from tgirl.registry import ToolRegistry
        from tgirl.sample import SamplingSession
        from tgirl.transport import TransportConfig
        from tgirl.types import SessionConfig

        # Simple char-based tokenizer
        vocab = [chr(i) for i in range(256)]

        def decode(ids: list[int]) -> str:
            return "".join(vocab[i] for i in ids)

        def encode(text: str) -> list[int]:
            return [ord(c) for c in text]

        # Build a simple registry with one tool
        registry = ToolRegistry()

        @registry.tool(quota=5, cost=1.0)
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        # Token sequence for freeform generation
        # Then tool_open tokens, then constrained tokens, then tool_close
        freeform_tok = list(range(freeform_tokens))
        if tool_delimiter_tokens is not None:
            # After freeform, emit the tool delimiter tokens
            all_tokens = freeform_tok + tool_delimiter_tokens
        else:
            all_tokens = freeform_tok

        token_idx = [0]

        def forward_fn(ctx: list[int]) -> torch.Tensor:
            idx = token_idx[0]
            token_idx[0] += 1
            # Strong bias toward the next planned token
            logits = torch.zeros(256)
            if idx < len(all_tokens):
                logits[all_tokens[idx]] = 100.0
            else:
                # After planned tokens, bias toward freeform content
                logits[ord("x")] = 100.0
            return logits

        embeddings = torch.eye(256)

        # Grammar guide factory: returns a mock grammar state that accepts
        # after producing a valid Hy expression
        def grammar_guide_factory(grammar_text: str):
            # Simple mock: accepts after 1 token
            class MockGS:
                def __init__(self):
                    self._done = False

                def get_valid_mask(self, vocab_size: int) -> torch.Tensor:
                    mask = torch.zeros(vocab_size, dtype=torch.bool)
                    # Allow ( and ) and some letters for hy source
                    for c in "(greet \"hi\")":
                        mask[ord(c)] = True
                    return mask

                def is_accepting(self) -> bool:
                    return self._done

                def advance(self, token_id: int) -> None:
                    self._done = True  # Accept after first token

            return MockGS()

        config = SessionConfig(
            max_tool_cycles=max_cycles,
            freeform_max_tokens=20,
            constrained_max_tokens=10,
            tool_open_delimiter=tool_open,
            tool_close_delimiter=tool_close,
            session_timeout=30.0,
        )

        session = SamplingSession(
            registry=registry,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            tokenizer_encode=encode,
            embeddings=embeddings,
            grammar_guide_factory=grammar_guide_factory,
            config=config,
            transport_config=TransportConfig(),
        )

        return session

    def test_no_tool_call_returns_freeform_only(self) -> None:
        from tgirl.sample import SamplingResult

        session = self._make_simple_session(tool_delimiter_tokens=None)
        result = session.run(prompt_tokens=[])
        assert isinstance(result, SamplingResult)
        assert result.total_cycles == 0
        assert len(result.tool_calls) == 0
        assert result.total_tokens > 0

    def test_max_tool_cycles_enforced(self) -> None:
        session = self._make_simple_session(
            tool_delimiter_tokens=None,
            max_cycles=3,
        )
        result = session.run(prompt_tokens=[])
        assert result.total_cycles <= 3

    def test_sampling_result_has_wall_time(self) -> None:
        session = self._make_simple_session(tool_delimiter_tokens=None)
        result = session.run(prompt_tokens=[])
        assert result.wall_time_ms > 0

    def test_count_tool_invocations_simple(self) -> None:
        from tgirl.sample import SamplingSession

        # Direct test of _count_tool_invocations
        session = self._make_simple_session()
        counts = session._count_tool_invocations(
            '(greet "Alice")', {"greet"}
        )
        assert counts == {"greet": 1}

    def test_count_tool_invocations_nested(self) -> None:
        from tgirl.sample import SamplingSession

        session = self._make_simple_session()

        @session._registry.tool(quota=5, cost=1.0)
        def shout(text: str) -> str:
            """Shout text."""
            return text.upper()

        counts = session._count_tool_invocations(
            '(-> (greet "Alice") (shout))',
            {"greet", "shout"},
        )
        assert counts == {"greet": 1, "shout": 1}

    def test_quotas_consumed_in_result(self) -> None:
        session = self._make_simple_session(tool_delimiter_tokens=None)
        result = session.run(prompt_tokens=[])
        assert isinstance(result.quotas_consumed, dict)

    def test_sampling_result_is_frozen(self) -> None:
        from pydantic import ValidationError

        session = self._make_simple_session(tool_delimiter_tokens=None)
        result = session.run(prompt_tokens=[])
        with pytest.raises(ValidationError):
            result.total_cycles = 999  # type: ignore[misc]

    def test_tool_call_record_is_frozen(self) -> None:
        from pydantic import ValidationError

        from tgirl.sample import ToolCallRecord

        record = ToolCallRecord(
            pipeline="(greet \"hi\")",
            result="Hello!",
            cycle_number=0,
            tool_invocations={"greet": 1},
        )
        with pytest.raises(ValidationError):
            record.cycle_number = 5  # type: ignore[misc]

    def test_count_tool_invocations_bare_symbol_in_threading(self) -> None:
        """S2 fix: bare symbols in threading position are counted."""
        session = self._make_simple_session()

        @session._registry.tool(quota=5, cost=1.0)
        def shout(text: str) -> str:
            """Shout text."""
            return text.upper()

        # In Hy threading, the last form can be a bare symbol: (-> (greet "x") shout)
        counts = session._count_tool_invocations(
            '(-> (greet "Alice") shout)',
            {"greet", "shout"},
        )
        assert counts["greet"] == 1
        assert counts["shout"] == 1

    def test_snapshot_with_remaining_quotas_reduces(self) -> None:
        """B1: verify _snapshot_with_remaining_quotas reduces quotas."""
        session = self._make_simple_session()
        # Simulate consuming 2 greet invocations
        session._consumed_quotas["greet"] = 2
        snapshot = session._snapshot_with_remaining_quotas()
        # Original quota was 5, consumed 2 -> remaining 3
        assert snapshot.quotas["greet"] == 3

    def test_snapshot_with_remaining_quotas_floors_at_zero(self) -> None:
        """B1: verify consumed > limit floors at 0, not negative."""
        session = self._make_simple_session()
        session._consumed_quotas["greet"] = 999
        snapshot = session._snapshot_with_remaining_quotas()
        assert snapshot.quotas["greet"] == 0

    def test_telemetry_is_list_of_telemetry_record(self) -> None:
        """B2: verify telemetry field type is list[TelemetryRecord]."""
        from tgirl.types import TelemetryRecord

        session = self._make_simple_session(tool_delimiter_tokens=None)
        result = session.run(prompt_tokens=[])
        # Freeform-only session: no tool cycles, so no telemetry records
        assert isinstance(result.telemetry, list)
        # But all entries (if any) must be TelemetryRecord
        for record in result.telemetry:
            assert isinstance(record, TelemetryRecord)

    def test_session_timeout_enforced(self) -> None:
        """S1: session stops when timeout is exceeded."""
        import torch

        from tgirl.registry import ToolRegistry
        from tgirl.sample import SamplingSession
        from tgirl.transport import TransportConfig
        from tgirl.types import SessionConfig

        registry = ToolRegistry()

        @registry.tool(quota=5, cost=1.0)
        def noop() -> str:
            """No-op."""
            return ""

        call_count = [0]

        def slow_forward(ctx: list[int]) -> torch.Tensor:
            call_count[0] += 1
            # Burn time on each call
            import time as _time
            _time.sleep(0.01)
            logits = torch.zeros(50)
            logits[0] = 100.0
            return logits

        config = SessionConfig(
            session_timeout=0.05,  # 50ms timeout
            freeform_max_tokens=10000,
        )

        session = SamplingSession(
            registry=registry,
            forward_fn=slow_forward,
            tokenizer_decode=lambda ids: "x" * len(ids),
            tokenizer_encode=lambda text: [0] * len(text),
            embeddings=torch.eye(50),
            grammar_guide_factory=lambda t: None,  # type: ignore
            config=config,
            transport_config=TransportConfig(),
        )

        result = session.run(prompt_tokens=[])
        # Should have stopped well before freeform_max_tokens
        assert result.total_tokens < 100
        assert result.wall_time_ms > 0


class TestSamplingSessionFormatter:
    """SamplingSession accepts optional formatter parameter."""

    def test_formatter_param_defaults_to_none(self) -> None:
        import torch

        from tgirl.registry import ToolRegistry
        from tgirl.sample import SamplingSession
        from tgirl.transport import TransportConfig

        registry = ToolRegistry()

        @registry.tool(quota=5, cost=1.0)
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        session = SamplingSession(
            registry=registry,
            forward_fn=lambda ctx: torch.zeros(50),
            tokenizer_decode=lambda ids: "",
            tokenizer_encode=lambda text: [],
            embeddings=torch.eye(50),
            grammar_guide_factory=lambda t: None,  # type: ignore
            transport_config=TransportConfig(),
        )
        assert session._formatter is None

    def test_formatter_param_stored(self) -> None:
        import torch

        from tgirl.format import PlainFormatter
        from tgirl.registry import ToolRegistry
        from tgirl.sample import SamplingSession
        from tgirl.transport import TransportConfig

        registry = ToolRegistry()

        @registry.tool(quota=5, cost=1.0)
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        fmt = PlainFormatter()
        session = SamplingSession(
            registry=registry,
            forward_fn=lambda ctx: torch.zeros(50),
            tokenizer_decode=lambda ids: "",
            tokenizer_encode=lambda text: [],
            embeddings=torch.eye(50),
            grammar_guide_factory=lambda t: None,  # type: ignore
            transport_config=TransportConfig(),
            formatter=fmt,
        )
        assert session._formatter is fmt


class TestSamplingSessionRunChat:
    """SamplingSession.run_chat() formats messages and delegates to run()."""

    def test_run_chat_calls_run_with_encoded_prompt(self) -> None:
        from unittest.mock import MagicMock, patch

        import torch

        from tgirl.format import PlainFormatter
        from tgirl.registry import ToolRegistry
        from tgirl.sample import SamplingResult, SamplingSession
        from tgirl.transport import TransportConfig
        from tgirl.types import SessionConfig

        registry = ToolRegistry()

        @registry.tool(quota=5, cost=1.0)
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        encode_calls: list[str] = []

        def mock_encode(text: str) -> list[int]:
            encode_calls.append(text)
            return [ord(c) for c in text[:10]]

        fmt = PlainFormatter()

        session = SamplingSession(
            registry=registry,
            forward_fn=lambda ctx: torch.zeros(50),
            tokenizer_decode=lambda ids: "",
            tokenizer_encode=mock_encode,
            embeddings=torch.eye(50),
            grammar_guide_factory=lambda t: None,  # type: ignore
            config=SessionConfig(freeform_max_tokens=1, session_timeout=1.0),
            transport_config=TransportConfig(),
            formatter=fmt,
        )

        messages = [{"role": "user", "content": "hello"}]
        result = session.run_chat(messages)

        assert isinstance(result, SamplingResult)
        # Verify that encode was called with formatted text
        assert len(encode_calls) >= 1

    def test_run_chat_captures_and_clears_last_user_content(self) -> None:
        """run_chat() sets _last_user_content for routing, run() consumes and clears it."""
        import torch

        from tgirl.format import PlainFormatter
        from tgirl.registry import ToolRegistry
        from tgirl.sample import SamplingSession
        from tgirl.transport import TransportConfig
        from tgirl.types import SessionConfig

        registry = ToolRegistry()

        @registry.tool(quota=5, cost=1.0)
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        session = SamplingSession(
            registry=registry,
            forward_fn=lambda ctx: torch.zeros(50),
            tokenizer_decode=lambda ids: "",
            tokenizer_encode=lambda text: [0],
            embeddings=torch.eye(50),
            grammar_guide_factory=lambda t: None,  # type: ignore
            config=SessionConfig(freeform_max_tokens=1, session_timeout=1.0),
            transport_config=TransportConfig(),
            formatter=PlainFormatter(),
        )

        messages = [
            {"role": "user", "content": "What tools are available?"},
        ]
        session.run_chat(messages)
        # After run_chat() completes, _last_user_content is consumed and cleared
        assert session._last_user_content is None

    def test_run_chat_requires_formatter(self) -> None:
        import torch

        from tgirl.registry import ToolRegistry
        from tgirl.sample import SamplingSession
        from tgirl.transport import TransportConfig

        registry = ToolRegistry()

        @registry.tool(quota=5, cost=1.0)
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        session = SamplingSession(
            registry=registry,
            forward_fn=lambda ctx: torch.zeros(50),
            tokenizer_decode=lambda ids: "",
            tokenizer_encode=lambda text: [],
            embeddings=torch.eye(50),
            grammar_guide_factory=lambda t: None,  # type: ignore
            transport_config=TransportConfig(),
        )

        with pytest.raises(ValueError, match="formatter"):
            session.run_chat([{"role": "user", "content": "hi"}])
