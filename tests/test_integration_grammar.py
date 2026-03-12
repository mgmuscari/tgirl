"""Integration tests for tgirl.grammar — end-to-end registry to grammar."""

from __future__ import annotations

import lark
import pytest

from tgirl.grammar import (
    GrammarOutput,
    diff,
    generate,
)
from tgirl.registry import ToolRegistry


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


class TestRegistryToGrammar:
    """End-to-end: register tools, take snapshot, generate grammar."""

    def test_single_tool_end_to_end(self, registry: ToolRegistry) -> None:
        @registry.tool(description="Search for items")
        def search(query: str, limit: int = 10) -> str:
            return query

        snap = registry.snapshot()
        output = generate(snap)
        assert "search" in output.text
        assert isinstance(output, GrammarOutput)
        # Must produce valid LALR(1) grammar
        parser = lark.Lark(output.text, parser="lalr")
        assert parser is not None

    def test_multiple_tools_end_to_end(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def fetch(url: str) -> str:
            return url

        @registry.tool()
        def store(key: str, value: str) -> bool:
            return True

        @registry.tool()
        def delete(key: str) -> bool:
            return True

        snap = registry.snapshot()
        output = generate(snap)
        for name in ["fetch", "store", "delete"]:
            assert name in output.text
        parser = lark.Lark(output.text, parser="lalr")
        assert parser is not None

    def test_empty_registry(self, registry: ToolRegistry) -> None:
        snap = registry.snapshot()
        output = generate(snap)
        assert output.text
        parser = lark.Lark(output.text, parser="lalr")
        assert parser is not None

    def test_diverse_parameter_types(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def process(
            name: str,
            count: int,
            ratio: float,
            active: bool,
        ) -> str:
            return name

        snap = registry.snapshot()
        output = generate(snap)
        parser = lark.Lark(output.text, parser="lalr")
        assert parser is not None


class TestScopeFilteringGrammar:
    """Verify scoped snapshots produce different grammars."""

    def test_scope_filters_tools(self, registry: ToolRegistry) -> None:
        @registry.tool(scope="admin")
        def admin_action(cmd: str) -> str:
            return cmd

        @registry.tool()
        def public_action(msg: str) -> str:
            return msg

        # Unscoped snapshot includes all
        snap_all = registry.snapshot()
        out_all = generate(snap_all)
        assert "admin_action" in out_all.text
        assert "public_action" in out_all.text

        # Scoped snapshot without admin excludes admin tool
        snap_user = registry.snapshot(scopes={"user"})
        out_user = generate(snap_user)
        assert "admin_action" not in out_user.text
        assert "public_action" in out_user.text

        # Diff should show admin tool removed
        result = diff(out_all, out_user)
        removed_names = {p.name for p in result.removed}
        assert "call_admin_action" in removed_names


class TestGrammarWithQuotas:
    """Verify quota information flows through to GrammarOutput."""

    def test_quotas_in_output(self, registry: ToolRegistry) -> None:
        @registry.tool(quota=5, cost=0.1)
        def expensive_op(data: str) -> str:
            return data

        snap = registry.snapshot(cost_budget=10.0)
        output = generate(snap)
        assert output.tool_quotas == {"expensive_op": 5}
        assert output.cost_remaining == 10.0


class TestLarkParseValidation:
    """Parse generated grammars with Lark and verify expressions."""

    def test_single_tool_call_parses(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def greet(name: str) -> str:
            return name

        snap = registry.snapshot()
        output = generate(snap)
        parser = lark.Lark(output.text, parser="lalr")
        tree = parser.parse('(greet "world")')
        assert tree is not None

    def test_threading_pipeline_parses(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def step(data: str) -> str:
            return data

        snap = registry.snapshot()
        output = generate(snap)
        parser = lark.Lark(output.text, parser="lalr")
        tree = parser.parse('(-> (step "a") (step "b"))')
        assert tree is not None

    def test_nested_composition_parses(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def op(x: str) -> str:
            return x

        snap = registry.snapshot()
        output = generate(snap)
        parser = lark.Lark(output.text, parser="lalr")
        tree = parser.parse(
            '(if (op "check") '
            '(-> (op "a") (op "b")) '
            '(op "fallback"))'
        )
        assert tree is not None

    def test_insufficient_resources_parses(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def noop(x: str) -> str:
            return x

        snap = registry.snapshot()
        output = generate(snap)
        parser = lark.Lark(output.text, parser="lalr")
        tree = parser.parse(
            '(insufficient-resources "quota-exhausted")'
        )
        assert tree is not None
