"""Tests for tgirl.grammar — Dynamic CFG generation."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from tgirl.types import (
    RegistrySnapshot,
)

# --- Task 1: Grammar output types and module skeleton ---


class TestGrammarTypes:
    """Verify GrammarOutput, Production, GrammarDiff, GrammarConfig models."""

    def test_production_is_frozen(self) -> None:
        from tgirl.grammar import Production

        p = Production(name="test_rule", rule='"hello"')
        with pytest.raises(ValidationError):
            p.name = "mutated"  # type: ignore[misc]

    def test_production_fields(self) -> None:
        from tgirl.grammar import Production

        p = Production(name="my_rule", rule='"a" | "b"')
        assert p.name == "my_rule"
        assert p.rule == '"a" | "b"'

    def test_grammar_output_is_frozen(self) -> None:
        from tgirl.grammar import GrammarOutput, Production

        go = GrammarOutput(
            text="start: expr",
            productions=(Production(name="start", rule="expr"),),
            snapshot_hash="abc123",
            tool_quotas={},
            cost_remaining=None,
        )
        with pytest.raises(ValidationError):
            go.text = "mutated"  # type: ignore[misc]

    def test_grammar_output_fields(self) -> None:
        from tgirl.grammar import GrammarOutput, Production

        p = Production(name="start", rule="expr")
        go = GrammarOutput(
            text="start: expr",
            productions=(p,),
            snapshot_hash="abc123",
            tool_quotas={"search": 5},
            cost_remaining=10.0,
        )
        assert go.text == "start: expr"
        assert go.productions == (p,)
        assert go.snapshot_hash == "abc123"
        assert go.tool_quotas == {"search": 5}
        assert go.cost_remaining == 10.0

    def test_grammar_diff_is_frozen(self) -> None:
        from tgirl.grammar import GrammarDiff

        gd = GrammarDiff(added=(), removed=(), changed=())
        with pytest.raises(ValidationError):
            gd.added = ()  # type: ignore[misc]

    def test_grammar_diff_fields(self) -> None:
        from tgirl.grammar import GrammarDiff, Production

        p1 = Production(name="a", rule="x")
        p2 = Production(name="b", rule="y")
        gd = GrammarDiff(
            added=(p1,),
            removed=(p2,),
            changed=((p1, p2),),
        )
        assert len(gd.added) == 1
        assert len(gd.removed) == 1
        assert len(gd.changed) == 1
        assert gd.changed[0] == (p1, p2)

    def test_grammar_config_defaults(self) -> None:
        from tgirl.grammar import GrammarConfig

        config = GrammarConfig()
        assert config.enumeration_threshold == 256

    def test_grammar_config_is_frozen(self) -> None:
        from tgirl.grammar import GrammarConfig

        config = GrammarConfig()
        with pytest.raises(ValidationError):
            config.enumeration_threshold = 10  # type: ignore[misc]


class TestGrammarStubs:
    """Verify generate() and diff() stubs exist."""

    def _make_empty_snapshot(self) -> RegistrySnapshot:
        return RegistrySnapshot(
            tools=(),
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )

    def test_generate_stub_exists(self) -> None:
        from tgirl.grammar import generate

        snap = self._make_empty_snapshot()
        with pytest.raises(NotImplementedError):
            generate(snap)

    def test_diff_stub_exists(self) -> None:
        from tgirl.grammar import GrammarOutput, diff

        go = GrammarOutput(
            text="",
            productions=(),
            snapshot_hash="",
            tool_quotas={},
            cost_remaining=None,
        )
        with pytest.raises(NotImplementedError):
            diff(go, go)
