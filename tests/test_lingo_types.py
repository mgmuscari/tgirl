"""Tests for tgirl.lingo.types — Type hierarchy with O(1) subsumption."""

from __future__ import annotations

from pathlib import Path

import pytest

from tgirl.lingo.tdl_parser import TdlDefinition, TdlFeatStruct, TdlType, tokenize_tdl, parse_tdl
from tgirl.lingo.types import TypeHierarchy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _def(name: str, supertypes: list[str], *, is_addendum: bool = False) -> TdlDefinition:
    """Shorthand for creating a TdlDefinition."""
    return TdlDefinition(
        name=name,
        supertypes=supertypes,
        body=None,
        docstring=None,
        section=None,
        is_instance=False,
        is_addendum=is_addendum,
        suffix=None,
    )


def _def_with_body(name: str, supertypes: list[str], body=None, *, is_addendum: bool = False) -> TdlDefinition:
    return TdlDefinition(
        name=name,
        supertypes=supertypes,
        body=body,
        docstring=None,
        section=None,
        is_instance=False,
        is_addendum=is_addendum,
        suffix=None,
    )


# ---------------------------------------------------------------------------
# Basic hierarchy tests
# ---------------------------------------------------------------------------

class TestTypeHierarchyBasic:

    def test_simple_subtype(self) -> None:
        h = TypeHierarchy([_def("a", ["b"])])
        assert h.is_subtype("a", "b")

    def test_reflexive(self) -> None:
        h = TypeHierarchy([_def("a", ["b"])])
        assert h.is_subtype("a", "a")
        assert h.is_subtype("b", "b")

    def test_not_reverse(self) -> None:
        h = TypeHierarchy([_def("a", ["b"])])
        assert not h.is_subtype("b", "a")

    def test_transitivity(self) -> None:
        h = TypeHierarchy([_def("a", ["b"]), _def("b", ["c"])])
        assert h.is_subtype("a", "c")
        assert h.is_subtype("a", "b")
        assert h.is_subtype("b", "c")
        assert not h.is_subtype("c", "a")

    def test_multiple_inheritance(self) -> None:
        h = TypeHierarchy([_def("a", ["b", "c"])])
        assert h.is_subtype("a", "b")
        assert h.is_subtype("a", "c")

    def test_diamond_inheritance(self) -> None:
        defs = [
            _def("d", ["b", "c"]),
            _def("b", ["a"]),
            _def("c", ["a"]),
        ]
        h = TypeHierarchy(defs)
        assert h.is_subtype("d", "a")
        assert h.is_subtype("d", "b")
        assert h.is_subtype("d", "c")

    def test_unknown_type_not_subtype(self) -> None:
        h = TypeHierarchy([_def("a", ["b"])])
        assert not h.is_subtype("a", "unknown")
        assert not h.is_subtype("unknown", "a")


class TestCommonSupertypes:

    def test_common_supertypes(self) -> None:
        defs = [
            _def("d", ["b", "c"]),
            _def("b", ["a"]),
            _def("c", ["a"]),
        ]
        h = TypeHierarchy(defs)
        common = h.common_supertypes("b", "c")
        assert "a" in common

    def test_no_common_supertypes(self) -> None:
        defs = [_def("a", ["x"]), _def("b", ["y"])]
        h = TypeHierarchy(defs)
        common = h.common_supertypes("a", "b")
        assert len(common) == 0


class TestGLB:

    def test_glb_exists(self) -> None:
        defs = [
            _def("d", ["b", "c"]),
            _def("b", ["a"]),
            _def("c", ["a"]),
        ]
        h = TypeHierarchy(defs)
        glb = h.greatest_lower_bound("b", "c")
        assert glb == "d"

    def test_glb_none_when_incompatible(self) -> None:
        defs = [_def("a", ["x"]), _def("b", ["y"])]
        h = TypeHierarchy(defs)
        assert h.greatest_lower_bound("x", "y") is None

    def test_glb_reflexive(self) -> None:
        h = TypeHierarchy([_def("a", ["b"])])
        assert h.greatest_lower_bound("a", "a") == "a"


class TestLeafAndSubtypes:

    def test_leaf_types(self) -> None:
        defs = [_def("a", ["b"]), _def("b", ["c"])]
        h = TypeHierarchy(defs)
        assert "a" in h.leaf_types
        assert "b" not in h.leaf_types
        assert "c" not in h.leaf_types

    def test_subtypes_of(self) -> None:
        defs = [_def("a", ["b"]), _def("b", ["c"])]
        h = TypeHierarchy(defs)
        subs = h.subtypes_of("c")
        assert "b" in subs
        assert "a" in subs

    def test_all_types(self) -> None:
        defs = [_def("a", ["b"]), _def("c", ["b"])]
        h = TypeHierarchy(defs)
        assert h.all_types == frozenset({"a", "b", "c"})


class TestAddenda:

    def test_addendum_recorded(self) -> None:
        defs = [
            _def("a", ["b"]),
            _def_with_body("a", [], body=TdlFeatStruct({"FEAT": TdlType("val")}), is_addendum=True),
        ]
        h = TypeHierarchy(defs)
        # Addendum doesn't change the supertype graph
        assert h.is_subtype("a", "b")
        # Type still exists
        assert "a" in h.all_types

    def test_addendum_unknown_type_logged(self) -> None:
        """Addendum for a type not defined via := should not crash."""
        defs = [
            _def_with_body("unknown_type", [], body=TdlFeatStruct({"F": TdlType("v")}), is_addendum=True),
        ]
        # Should not raise
        h = TypeHierarchy(defs)
        # The type should still appear (it was referenced)
        assert "unknown_type" in h.all_types


class TestFromParsedTDL:

    def test_parsed_definitions(self) -> None:
        """Build hierarchy from actual parsed TDL."""
        tdl = "a := b & c.\nb := d.\nc := d."
        definitions = parse_tdl(tokenize_tdl(tdl))
        h = TypeHierarchy(definitions)
        assert h.is_subtype("a", "b")
        assert h.is_subtype("a", "c")
        assert h.is_subtype("a", "d")
        assert h.is_subtype("b", "d")
        assert "a" in h.leaf_types


# ---------------------------------------------------------------------------
# Real ERG tests (slow)
# ---------------------------------------------------------------------------

ERG_DIR = Path.home() / "ontologi" / "tools" / "erg-2025"


@pytest.mark.slow
class TestRealERG:

    def test_partial_erg_hierarchy(self) -> None:
        """Build hierarchy from fundamentals + lextypes + letypes."""
        from tgirl.lingo.tdl_parser import parse_tdl_file

        defs = []
        for fname in ("fundamentals.tdl", "lextypes.tdl", "letypes.tdl"):
            path = ERG_DIR / fname
            if not path.exists():
                pytest.skip(f"ERG file not found: {path}")
            defs.extend(parse_tdl_file(path))

        # Filter to TdlDefinition only
        type_defs = [d for d in defs if isinstance(d, TdlDefinition)]
        h = TypeHierarchy(type_defs)
        assert len(h.all_types) > 3000

    def test_full_erg_hierarchy(self) -> None:
        """Build hierarchy from all TDL files via recursive include."""
        from tgirl.lingo.tdl_parser import parse_tdl_directory

        top = ERG_DIR / "english.tdl"
        if not top.exists():
            pytest.skip(f"ERG top file not found: {top}")

        all_nodes = parse_tdl_directory(top)
        type_defs = [d for d in all_nodes if isinstance(d, TdlDefinition)]
        h = TypeHierarchy(type_defs)
        # ERG 2025 has ~51k types
        assert len(h.all_types) > 40000

    def test_sign_hierarchy(self) -> None:
        """Verify known ERG hierarchy relationship."""
        from tgirl.lingo.tdl_parser import parse_tdl_directory

        top = ERG_DIR / "english.tdl"
        if not top.exists():
            pytest.skip(f"ERG top file not found: {top}")

        all_nodes = parse_tdl_directory(top)
        type_defs = [d for d in all_nodes if isinstance(d, TdlDefinition)]
        h = TypeHierarchy(type_defs)
        assert "sign" in h.all_types
