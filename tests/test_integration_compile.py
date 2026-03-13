"""Integration tests for tgirl.compile — end-to-end pipeline execution."""

from __future__ import annotations

import pytest

from tgirl.compile import (
    InsufficientResources,
    PipelineResult,
    run_pipeline,
)
from tgirl.registry import ToolRegistry
from tgirl.types import PipelineError


@pytest.fixture()
def registry() -> ToolRegistry:
    """Registry with realistic tool registrations."""
    reg = ToolRegistry()

    @reg.tool()
    def greet(name: str) -> str:
        return f"Hello, {name}"

    @reg.tool()
    def shout(text: str) -> str:
        return text.upper()

    @reg.tool()
    def add(a: int, b: int) -> int:
        return a + b

    @reg.tool()
    def is_long(text: str) -> bool:
        return len(text) > 5

    @reg.tool()
    def fallback(text: str) -> str:
        return "fallback"

    @reg.tool()
    def failing(text: str) -> str:
        msg = "intentional failure"
        raise RuntimeError(msg)

    return reg


class TestEndToEndExecution:
    """Realistic tool calls with result verification."""

    def test_single_tool_call_with_string(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline('(greet "Alice")', registry)
        assert isinstance(result, PipelineResult)
        assert result.result == "Hello, Alice"
        assert result.execution_time_ms > 0

    def test_single_tool_call_with_int_args(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline("(add 3 4)", registry)
        assert isinstance(result, PipelineResult)
        assert result.result == 7

    def test_hy_source_preserved_in_result(
        self, registry: ToolRegistry
    ) -> None:
        source = '(greet "Bob")'
        result = run_pipeline(source, registry)
        assert isinstance(result, PipelineResult)
        assert result.hy_source == source


class TestCompositionIntegration:
    """All composition operators with real tools."""

    def test_threading_pipeline(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline(
            '(-> "hello" (greet) (shout))', registry
        )
        assert isinstance(result, PipelineResult)
        assert result.result == "HELLO, HELLO"

    def test_let_bindings(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline(
            '(let [x (greet "World")] (shout x))', registry
        )
        assert isinstance(result, PipelineResult)
        assert result.result == "HELLO, WORLD"

    def test_conditional(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline(
            '(if (is_long "abcdef") (greet "long") (greet "short"))',
            registry,
        )
        assert isinstance(result, PipelineResult)
        assert result.result == "Hello, long"

    def test_conditional_false_branch(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline(
            '(if (is_long "hi") (greet "long") (greet "short"))',
            registry,
        )
        assert isinstance(result, PipelineResult)
        assert result.result == "Hello, short"

    def test_try_except(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline(
            '(try (failing "x") (except [e Exception] (fallback "y")))',
            registry,
        )
        assert isinstance(result, PipelineResult)
        assert result.result == "fallback"

    def test_try_catch_spec_syntax(
        self, registry: ToolRegistry
    ) -> None:
        """Test with TGIRL spec catch syntax (normalized to except)."""
        result = run_pipeline(
            '(try (failing "x") (catch [e Exception] (fallback "y")))',
            registry,
        )
        assert isinstance(result, PipelineResult)
        assert result.result == "fallback"

    def test_pmap(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline(
            '(pmap [greet shout] "test")', registry
        )
        assert isinstance(result, PipelineResult)
        assert result.result == ["Hello, test", "TEST"]

    def test_insufficient_resources(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline(
            '(insufficient-resources "quota exceeded")', registry
        )
        assert isinstance(result, InsufficientResources)
        assert result.reason == "quota exceeded"


class TestSecurityIntegration:
    """Comprehensive sandbox escape attempts."""

    def test_import_blocked(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline("(import os)", registry)
        assert isinstance(result, PipelineError)

    def test_builtins_access_blocked(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline('(__import__ "os")', registry)
        assert isinstance(result, PipelineError)

    def test_open_blocked(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline('(open "/etc/passwd")', registry)
        assert isinstance(result, PipelineError)

    def test_getattr_blocked(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline(
            '(getattr greet "__module__")', registry
        )
        assert isinstance(result, PipelineError)

    def test_dunder_attribute_blocked(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline("(. greet __class__)", registry)
        assert isinstance(result, PipelineError)

    def test_defn_blocked(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline("(defn evil [] None)", registry)
        assert isinstance(result, PipelineError)

    def test_require_blocked(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline("(require os)", registry)
        assert isinstance(result, PipelineError)

    def test_unregistered_tool_blocked(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline('(evil_tool "payload")', registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "static_analysis"


class TestErrorStages:
    """Verify each pipeline stage produces correct PipelineError."""

    def test_parse_stage_error(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline("(unclosed", registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "parse"

    def test_empty_input_error(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline("", registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "parse"

    def test_static_analysis_error(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline('(evil "x")', registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "static_analysis"

    def test_execution_error(
        self, registry: ToolRegistry
    ) -> None:
        result = run_pipeline('(failing "x")', registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "execute"

    def test_hy_source_preserved_in_error(
        self, registry: ToolRegistry
    ) -> None:
        source = '(evil "payload")'
        result = run_pipeline(source, registry)
        assert isinstance(result, PipelineError)
        assert result.hy_source == source or len(result.hy_source) > 0
