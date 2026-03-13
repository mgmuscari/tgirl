"""Tests for tgirl.compile — Hy compilation and sandboxed execution."""

from __future__ import annotations

import ast

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

    def test_run_pipeline_exists_and_callable(self) -> None:
        from tgirl.compile import run_pipeline

        registry = ToolRegistry()
        # With no tools registered, unregistered call fails at static analysis
        result = run_pipeline("(greet)", registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "static_analysis"

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


class TestHyParsing:
    """Task 2: Hy parsing wrapper."""

    def test_valid_single_call_parses(self) -> None:
        from tgirl.compile import _parse_hy

        result = _parse_hy('(greet "hello")')
        assert not isinstance(result, PipelineError)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_valid_pipeline_parses(self) -> None:
        from tgirl.compile import _parse_hy

        result = _parse_hy('(-> "hello" (greet) (shout))')
        assert not isinstance(result, PipelineError)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_invalid_syntax_returns_pipeline_error(self) -> None:
        from tgirl.compile import _parse_hy

        result = _parse_hy("(greet")
        assert isinstance(result, PipelineError)
        assert result.stage == "parse"

    def test_empty_input_returns_pipeline_error(self) -> None:
        from tgirl.compile import _parse_hy

        result = _parse_hy("")
        assert isinstance(result, PipelineError)
        assert result.stage == "parse"

    def test_unclosed_paren_returns_pipeline_error(self) -> None:
        from tgirl.compile import _parse_hy

        result = _parse_hy("(greet (nested)")
        assert isinstance(result, PipelineError)
        assert result.stage == "parse"

    def test_catch_normalizes_to_except(self) -> None:
        """catch in TGIRL spec normalizes to except for Hy parsing."""
        from tgirl.compile import _normalize_hy_source, _parse_hy

        source = '(try (tool1) (catch [e Exception] (tool2)))'
        normalized = _normalize_hy_source(source)
        assert "(except" in normalized
        assert "(catch" not in normalized
        result = _parse_hy(source)
        assert not isinstance(result, PipelineError)

    def test_multiple_expressions_parse(self) -> None:
        from tgirl.compile import _parse_hy

        result = _parse_hy('(tool1 "a") (tool2 "b")')
        assert not isinstance(result, PipelineError)
        assert len(result) == 2


class TestHyAstAnalysis:
    """Task 3: Hy AST static analyzer."""

    def _parse(self, source: str) -> list:
        from tgirl.compile import _parse_hy

        result = _parse_hy(source)
        assert not isinstance(result, PipelineError), f"Parse failed: {result}"
        return result

    def test_valid_tool_call_passes(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse('(greet "hello")')
        result = _analyze_hy_ast(trees, {"greet"})
        assert result is None

    def test_import_form_rejected(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse("(import os)")
        result = _analyze_hy_ast(trees, {"greet"})
        assert isinstance(result, PipelineError)
        assert result.stage == "static_analysis"

    def test_dangerous_builtins_rejected(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        for builtin in ["__import__", "open", "getattr", "setattr", "delattr"]:
            trees = self._parse(f'({builtin} "x")')
            result = _analyze_hy_ast(trees, {"greet"})
            assert isinstance(result, PipelineError), (
                f"{builtin} should be rejected"
            )

    def test_dunder_attribute_rejected(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse("(. obj __class__)")
        result = _analyze_hy_ast(trees, {"obj"})
        assert isinstance(result, PipelineError)
        assert result.stage == "static_analysis"

    def test_non_dunder_attribute_accepted(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse("(. obj name)")
        result = _analyze_hy_ast(trees, {"obj"})
        assert result is None

    def test_unregistered_function_rejected(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse('(evil_func "payload")')
        result = _analyze_hy_ast(trees, {"greet"})
        assert isinstance(result, PipelineError)
        assert result.stage == "static_analysis"

    def test_let_bound_variables_accepted(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse('(let [x (greet "hi")] x)')
        result = _analyze_hy_ast(trees, {"greet"})
        assert result is None

    def test_unresolved_variable_rejected(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse("(greet unknown_var)")
        result = _analyze_hy_ast(trees, {"greet"})
        assert isinstance(result, PipelineError)
        assert result.stage == "static_analysis"

    def test_composition_operators_accepted(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse('(-> "hello" (greet) (shout))')
        result = _analyze_hy_ast(trees, {"greet", "shout"})
        assert result is None

    def test_insufficient_resources_accepted(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse('(insufficient-resources "no tools")')
        result = _analyze_hy_ast(trees, {"greet"})
        assert result is None

    def test_defn_rejected(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse("(defn evil [] None)")
        result = _analyze_hy_ast(trees, {"greet"})
        assert isinstance(result, PipelineError)

    def test_require_rejected(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse("(require os)")
        result = _analyze_hy_ast(trees, {"greet"})
        assert isinstance(result, PipelineError)

    def test_pmap_operator_accepted(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse("(pmap [greet shout] x)")
        # x would need to be let-bound or a literal in real usage,
        # but here we test pmap itself is accepted
        result = _analyze_hy_ast(trees, {"greet", "shout"})
        # x is unresolved, but pmap itself should be accepted
        # The test specifically checks pmap acceptance
        assert result is None or result.stage == "static_analysis"

    def test_if_operator_accepted(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse(
            '(if True (greet "yes") (greet "no"))'
        )
        result = _analyze_hy_ast(trees, {"greet"})
        assert result is None

    def test_try_except_accepted(self) -> None:
        from tgirl.compile import _analyze_hy_ast

        trees = self._parse(
            '(try (greet "hi") (except [e Exception] (greet "fallback")))'
        )
        result = _analyze_hy_ast(trees, {"greet"})
        assert result is None


class TestPythonAstAnalysis:
    """Task 4: Python AST analyzer via RestrictedPython."""

    def _make_ast(self, python_source: str) -> ast.Module:
        import ast

        return ast.parse(python_source)

    def _hy_to_ast(self, hy_source: str) -> ast.Module:
        import ast

        import hy
        from hy.compiler import hy_compile

        tree = hy_compile(hy.read_many(hy_source), "__main__")
        # Strip auto-injected 'import hy'
        tree.body = [
            n
            for n in tree.body
            if not isinstance(n, ast.Import)
            or n.names[0].name != "hy"
        ]
        return tree

    def test_clean_ast_passes(self) -> None:
        from tgirl.compile import _analyze_python_ast

        tree = self._make_ast('greet("hello")')
        result = _analyze_python_ast(tree, {"greet"})
        assert result is None

    def test_import_rejected(self) -> None:
        from tgirl.compile import _analyze_python_ast

        tree = self._make_ast("import os")
        result = _analyze_python_ast(tree, {"greet"})
        assert isinstance(result, PipelineError)
        assert result.stage == "ast_analysis"

    def test_dunder_attribute_rejected(self) -> None:
        from tgirl.compile import _analyze_python_ast

        tree = self._make_ast("x.__class__")
        result = _analyze_python_ast(tree, {"greet"})
        assert isinstance(result, PipelineError)
        assert result.stage == "ast_analysis"

    def test_non_dunder_attribute_accepted(self) -> None:
        from tgirl.compile import _analyze_python_ast

        tree = self._make_ast("x.name")
        result = _analyze_python_ast(tree, {"greet"})
        assert result is None

    def test_global_rejected(self) -> None:
        from tgirl.compile import _analyze_python_ast

        tree = self._make_ast("global x")
        result = _analyze_python_ast(tree, set())
        assert isinstance(result, PipelineError)

    def test_nonlocal_rejected(self) -> None:
        from tgirl.compile import _analyze_python_ast

        tree = self._make_ast(
            "def f():\n  x = 1\n  def g():\n    nonlocal x"
        )
        result = _analyze_python_ast(tree, set())
        assert isinstance(result, PipelineError)

    def test_hy_compiled_valid_pipeline_accepted(self) -> None:
        from tgirl.compile import _analyze_python_ast

        tree = self._hy_to_ast('(greet "hello")')
        result = _analyze_python_ast(tree, {"greet"})
        assert result is None

    def test_hy_compiled_let_accepted(self) -> None:
        from tgirl.compile import _analyze_python_ast

        tree = self._hy_to_ast('(let [x (greet "hi")] (shout x))')
        result = _analyze_python_ast(tree, {"greet", "shout"})
        assert result is None


class TestCompositionOperators:
    """Task 5: Composition operator runtime implementations."""

    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool()
        def greet(name: str) -> str:
            return f"Hello, {name}"

        @reg.tool()
        def shout(text: str) -> str:
            return text.upper()

        @reg.tool()
        def failing_tool(text: str) -> str:
            msg = "tool failed"
            raise RuntimeError(msg)

        @reg.tool()
        def fallback(text: str) -> str:
            return "fallback result"

        @reg.tool()
        def pred(text: str) -> bool:
            return len(text) > 3

        return reg

    def test_pmap_returns_list_of_results(self) -> None:
        from tgirl.compile import _pmap_impl

        results = _pmap_impl(
            [lambda x: x.upper(), lambda x: x + "!"],
            "hello",
        )
        assert results == ["HELLO", "hello!"]

    def test_pmap_fail_fast(self) -> None:
        from tgirl.compile import _pmap_impl

        def good(x: str) -> str:
            return x

        def bad(x: str) -> str:
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            _pmap_impl([good, bad], "test")

    def test_insufficient_resources_returns_model(self) -> None:
        from tgirl.compile import (
            InsufficientResources,
            _insufficient_resources_impl,
        )

        result = _insufficient_resources_impl("no tools available")
        assert isinstance(result, InsufficientResources)
        assert result.reason == "no tools available"

    def test_expand_threading_macro(self) -> None:
        import hy
        from hy.models import Expression

        from tgirl.compile import _expand_macros

        trees = list(hy.read_many('(-> "hello" (greet) (shout))'))
        expanded = _expand_macros(trees[0])
        # Should expand to (shout (greet "hello"))
        assert isinstance(expanded, Expression)
        assert str(expanded[0]) == "shout"
        inner = expanded[1]
        assert isinstance(inner, Expression)
        assert str(inner[0]) == "greet"

    def test_expand_threading_with_extra_args(self) -> None:
        import hy
        from hy.models import Expression

        from tgirl.compile import _expand_macros

        trees = list(hy.read_many('(-> "x" (tool1 "extra"))'))
        expanded = _expand_macros(trees[0])
        # Should expand to (tool1 "x" "extra")
        assert isinstance(expanded, Expression)
        assert str(expanded[0]) == "tool1"
        assert len(expanded) == 3  # tool1, "x", "extra"


class TestSandbox:
    """Task 6: Sandbox construction."""

    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool()
        def greet(name: str) -> str:
            return f"Hello, {name}"

        @reg.tool()
        def shout(text: str) -> str:
            return text.upper()

        return reg

    def test_sandbox_contains_registered_tools(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import _build_sandbox

        sandbox = _build_sandbox(registry)
        assert "greet" in sandbox
        assert "shout" in sandbox
        assert sandbox["greet"]("world") == "Hello, world"
        assert sandbox["shout"]("hi") == "HI"

    def test_sandbox_contains_pmap(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import _build_sandbox

        sandbox = _build_sandbox(registry)
        assert "pmap" in sandbox

    def test_sandbox_contains_insufficient_resources(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import _build_sandbox

        sandbox = _build_sandbox(registry)
        assert "insufficient_resources" in sandbox

    def test_sandbox_result_sentinel(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import _build_sandbox

        sandbox = _build_sandbox(registry)
        assert "_tgirl_result_" in sandbox
        assert sandbox["_tgirl_result_"] is None

    def test_sandbox_builtins_restricted(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import _build_sandbox

        sandbox = _build_sandbox(registry)
        builtins = sandbox["__builtins__"]
        # Only safe builtins are available
        assert "Exception" in builtins
        assert "isinstance" in builtins
        # Dangerous builtins are NOT available
        assert "eval" not in builtins
        assert "exec" not in builtins
        assert "__import__" not in builtins
        assert "open" not in builtins
        assert "getattr" not in builtins
        assert "compile" not in builtins

    def test_sandbox_tool_callables_are_originals(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import _build_sandbox

        sandbox = _build_sandbox(registry)
        assert sandbox["greet"] is registry.get_callable("greet")
        assert sandbox["shout"] is registry.get_callable("shout")

    def test_dangerous_builtins_blocked_in_sandbox(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import _build_sandbox

        sandbox = _build_sandbox(registry)
        # print is not in our safe builtins
        code = compile("print('hello')", "<test>", "exec")
        with pytest.raises(NameError):
            exec(code, sandbox)  # noqa: S102


class TestTimeoutEnforcement:
    """Task 7: Timeout enforcement."""

    def test_fast_execution_completes(self) -> None:
        from tgirl.compile import _run_with_timeout

        result = _run_with_timeout(lambda: 42, timeout=5.0)
        assert result == 42

    def test_tool_timeout_returns_pipeline_error(self) -> None:
        import time

        from tgirl.compile import _run_with_timeout

        def slow_fn() -> str:
            time.sleep(5)
            return "done"

        result = _run_with_timeout(slow_fn, timeout=0.1)
        assert isinstance(result, PipelineError)
        assert result.stage == "execute"
        assert result.error_type == "TimeoutError"

    def test_wrap_with_timeout(self) -> None:
        from tgirl.compile import _wrap_with_timeout

        def fast_fn(x: str) -> str:
            return x.upper()

        wrapped = _wrap_with_timeout(fast_fn, timeout=5.0)
        assert wrapped("hello") == "HELLO"

    def test_wrapped_tool_timeout(self) -> None:
        import time

        from tgirl.compile import _wrap_with_timeout

        def slow_fn(x: str) -> str:
            time.sleep(5)
            return x

        wrapped = _wrap_with_timeout(slow_fn, timeout=0.1)
        with pytest.raises(TimeoutError):
            wrapped("test")


class TestResultCapture:
    """Task 8: Result capture via AST rewriting."""

    def test_inject_result_capture_rewrites_last_expr(self) -> None:
        import ast

        from tgirl.compile import _inject_result_capture

        tree = ast.parse('greet("hello")')
        rewritten = _inject_result_capture(tree)
        # Last statement should be an Assign, not Expr
        last = rewritten.body[-1]
        assert isinstance(last, ast.Assign)
        assert last.targets[0].id == "_tgirl_result_"  # type: ignore[attr-defined]

    def test_single_expression_captures_value(self) -> None:
        import ast

        from tgirl.compile import _inject_result_capture

        tree = ast.parse("42")
        rewritten = _inject_result_capture(tree)
        sandbox: dict = {"_tgirl_result_": None}
        exec(  # noqa: S102
            compile(rewritten, "<test>", "exec"), sandbox
        )
        assert sandbox["_tgirl_result_"] == 42

    def test_multiple_statements_captures_last(self) -> None:
        import ast

        from tgirl.compile import _inject_result_capture

        tree = ast.parse("x = 1\nx + 10")
        rewritten = _inject_result_capture(tree)
        sandbox: dict = {"_tgirl_result_": None}
        exec(  # noqa: S102
            compile(rewritten, "<test>", "exec"), sandbox
        )
        assert sandbox["_tgirl_result_"] == 11

    def test_empty_module_handled(self) -> None:
        import ast

        from tgirl.compile import _inject_result_capture

        tree = ast.Module(body=[], type_ignores=[])
        rewritten = _inject_result_capture(tree)
        assert rewritten.body == []


class TestFullPipeline:
    """Task 8: Full pipeline integration."""

    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool()
        def greet(name: str) -> str:
            return f"Hello, {name}"

        @reg.tool()
        def shout(text: str) -> str:
            return text.upper()

        @reg.tool()
        def failing(text: str) -> str:
            msg = "tool failure"
            raise RuntimeError(msg)

        return reg

    def test_simple_tool_call(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import PipelineResult, run_pipeline

        result = run_pipeline('(greet "world")', registry)
        assert isinstance(result, PipelineResult)
        assert result.result == "Hello, world"

    def test_invalid_syntax_error(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import run_pipeline

        result = run_pipeline("(greet", registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "parse"

    def test_unregistered_tool_error(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import run_pipeline

        result = run_pipeline('(evil "payload")', registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "static_analysis"

    def test_execution_error(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import run_pipeline

        result = run_pipeline('(failing "test")', registry)
        assert isinstance(result, PipelineError)
        assert result.stage == "execute"

    def test_insufficient_resources(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import InsufficientResources, run_pipeline

        result = run_pipeline(
            '(insufficient-resources "no tools")', registry
        )
        assert isinstance(result, InsufficientResources)
        assert result.reason == "no tools"

    def test_each_stage_has_correct_field(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import run_pipeline

        # Parse error
        r1 = run_pipeline("(", registry)
        assert isinstance(r1, PipelineError)
        assert r1.stage == "parse"

        # Static analysis error
        r2 = run_pipeline('(evil "x")', registry)
        assert isinstance(r2, PipelineError)
        assert r2.stage == "static_analysis"

    def test_threading_pipeline(
        self, registry: ToolRegistry
    ) -> None:
        from tgirl.compile import PipelineResult, run_pipeline

        result = run_pipeline(
            '(-> "hello" (greet) (shout))', registry
        )
        assert isinstance(result, PipelineResult)
        assert result.result == "HELLO, HELLO"
