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
