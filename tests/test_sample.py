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
