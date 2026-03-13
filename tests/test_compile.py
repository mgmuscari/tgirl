"""Tests for tgirl.compile — Hy compilation and sandboxed execution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tgirl.registry import ToolRegistry
from tgirl.types import PipelineError


class TestCompileTypes:
    """Task 1: Verify PipelineResult, InsufficientResources, CompileConfig."""

    def test_pipeline_result_is_frozen(self) -> None:
        from tgirl.compile import PipelineResult

        r = PipelineResult(
            result="hello", hy_source="(greet)", execution_time_ms=1.0
        )
        with pytest.raises(ValidationError):
            r.result = "changed"  # type: ignore[misc]

    def test_pipeline_result_fields(self) -> None:
        from tgirl.compile import PipelineResult

        r = PipelineResult(
            result=42, hy_source="(compute)", execution_time_ms=5.5
        )
        assert r.result == 42
        assert r.hy_source == "(compute)"
        assert r.execution_time_ms == 5.5

    def test_insufficient_resources_is_frozen(self) -> None:
        from tgirl.compile import InsufficientResources

        ir = InsufficientResources(
            reason="no tools available",
            hy_source="(insufficient-resources ...)",
        )
        with pytest.raises(ValidationError):
            ir.reason = "changed"  # type: ignore[misc]

    def test_insufficient_resources_fields(self) -> None:
        from tgirl.compile import InsufficientResources

        ir = InsufficientResources(
            reason="quota exceeded",
            hy_source="(insufficient-resources ...)",
        )
        assert ir.reason == "quota exceeded"
        assert ir.hy_source == "(insufficient-resources ...)"

    def test_compile_config_is_frozen(self) -> None:
        from tgirl.compile import CompileConfig

        c = CompileConfig()
        with pytest.raises(ValidationError):
            c.pipeline_timeout = 999.0  # type: ignore[misc]

    def test_compile_config_defaults(self) -> None:
        from tgirl.compile import CompileConfig

        c = CompileConfig()
        assert c.pipeline_timeout == 60.0
        assert c.max_depth == 50

    def test_compile_config_custom_values(self) -> None:
        from tgirl.compile import CompileConfig

        c = CompileConfig(pipeline_timeout=30.0, max_depth=25)
        assert c.pipeline_timeout == 30.0
        assert c.max_depth == 25


class TestCompileStubs:
    """Task 1: Verify stub entry point exists."""

    def test_run_pipeline_stub_returns_pipeline_error(self) -> None:
        from tgirl.compile import run_pipeline

        registry = ToolRegistry()
        result = run_pipeline("(greet)", registry)
        assert isinstance(result, PipelineError)
        assert "NotImplementedError" in result.error_type

    def test_pipeline_stages_defined(self) -> None:
        from tgirl.compile import (
            STAGE_AST_ANALYSIS,
            STAGE_COMPILE,
            STAGE_EXECUTE,
            STAGE_PARSE,
            STAGE_STATIC_ANALYSIS,
        )

        assert STAGE_PARSE == "parse"
        assert STAGE_STATIC_ANALYSIS == "static_analysis"
        assert STAGE_COMPILE == "compile"
        assert STAGE_AST_ANALYSIS == "ast_analysis"
        assert STAGE_EXECUTE == "execute"
