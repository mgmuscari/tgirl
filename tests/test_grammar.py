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


# --- Task 2: Type-to-production converter ---


class TestTypeProductions:
    """Verify _type_to_rule for each TypeRepr variant."""

    def test_primitive_str(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import PrimitiveType

        t = PrimitiveType(kind="str")
        prods = _type_to_rule(t, "str_val", GrammarConfig())
        assert len(prods) >= 1
        assert prods[0].name == "str_val"
        assert "ESCAPED_STRING" in prods[0].rule

    def test_primitive_int(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import PrimitiveType

        t = PrimitiveType(kind="int")
        prods = _type_to_rule(t, "int_val", GrammarConfig())
        assert len(prods) >= 1
        assert prods[0].name == "int_val"
        assert "SIGNED_INT" in prods[0].rule

    def test_primitive_float(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import PrimitiveType

        t = PrimitiveType(kind="float")
        prods = _type_to_rule(t, "float_val", GrammarConfig())
        assert len(prods) >= 1
        assert prods[0].name == "float_val"
        assert "SIGNED_FLOAT" in prods[0].rule

    def test_primitive_bool(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import PrimitiveType

        t = PrimitiveType(kind="bool")
        prods = _type_to_rule(t, "bool_val", GrammarConfig())
        assert len(prods) >= 1
        assert prods[0].name == "bool_val"
        assert '"true"' in prods[0].rule
        assert '"false"' in prods[0].rule

    def test_primitive_none(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import PrimitiveType

        t = PrimitiveType(kind="none")
        prods = _type_to_rule(t, "none_val", GrammarConfig())
        assert len(prods) >= 1
        assert prods[0].name == "none_val"
        assert '"nil"' in prods[0].rule

    def test_list_type(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import ListType, PrimitiveType

        t = ListType(element=PrimitiveType(kind="int"))
        prods = _type_to_rule(t, "list_int", GrammarConfig())
        assert len(prods) >= 1
        # Should have list rule with brackets
        list_rule = prods[0].rule
        assert '"["' in list_rule
        assert '"]"' in list_rule

    def test_dict_type(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import DictType, PrimitiveType

        t = DictType(key=PrimitiveType(kind="str"), value=PrimitiveType(kind="int"))
        prods = _type_to_rule(t, "dict_str_int", GrammarConfig())
        assert len(prods) >= 1
        dict_rule = prods[0].rule
        assert '"{"' in dict_rule
        assert '"}"' in dict_rule

    def test_literal_type_strings(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import LiteralType

        t = LiteralType(values=("hello", "world"))
        prods = _type_to_rule(t, "lit_val", GrammarConfig())
        assert len(prods) >= 1
        rule = prods[0].rule
        # String literals should be double-quoted in grammar
        assert "hello" in rule
        assert "world" in rule

    def test_literal_type_numbers(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import LiteralType

        t = LiteralType(values=(1, 2, 3))
        prods = _type_to_rule(t, "lit_num", GrammarConfig())
        assert len(prods) >= 1
        rule = prods[0].rule
        assert '"1"' in rule
        assert '"2"' in rule
        assert '"3"' in rule

    def test_enum_type(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import EnumType

        t = EnumType(name="Color", values=("red", "green", "blue"))
        prods = _type_to_rule(t, "color_val", GrammarConfig())
        assert len(prods) >= 1
        rule = prods[0].rule
        assert "red" in rule
        assert "green" in rule
        assert "blue" in rule

    def test_optional_type(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import OptionalType, PrimitiveType

        t = OptionalType(inner=PrimitiveType(kind="str"))
        prods = _type_to_rule(t, "opt_str", GrammarConfig())
        assert len(prods) >= 1
        rule = prods[0].rule
        assert '"nil"' in rule

    def test_union_type(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import PrimitiveType, UnionType

        t = UnionType(
            members=(PrimitiveType(kind="str"), PrimitiveType(kind="int"))
        )
        prods = _type_to_rule(t, "union_val", GrammarConfig())
        assert len(prods) >= 1
        # Union should reference member type rules
        rule = prods[0].rule
        assert "|" in rule

    def test_model_type(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import FieldDef, ModelType, PrimitiveType

        t = ModelType(
            name="Point",
            fields=(
                FieldDef(
                    name="x", type_repr=PrimitiveType(kind="float"), required=True
                ),
                FieldDef(
                    name="y", type_repr=PrimitiveType(kind="float"), required=True
                ),
            ),
        )
        prods = _type_to_rule(t, "point_val", GrammarConfig())
        assert len(prods) >= 1
        rule = prods[0].rule
        assert '"{"' in rule
        assert '"}"' in rule

    def test_annotated_type_no_constraints(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import AnnotatedType, PrimitiveType

        t = AnnotatedType(base=PrimitiveType(kind="int"), constraints=())
        prods = _type_to_rule(t, "ann_int", GrammarConfig())
        assert len(prods) >= 1
        # Should delegate to base type
        assert "SIGNED_INT" in prods[0].rule

    def test_annotated_type_enumerable_range(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import AnnotatedType, ConstraintRepr, PrimitiveType

        t = AnnotatedType(
            base=PrimitiveType(kind="int"),
            constraints=(
                ConstraintRepr(kind="ge", value=1),
                ConstraintRepr(kind="le", value=5),
            ),
        )
        prods = _type_to_rule(t, "ann_range", GrammarConfig())
        assert len(prods) >= 1
        rule = prods[0].rule
        # Should enumerate: "1" | "2" | "3" | "4" | "5"
        for i in range(1, 6):
            assert f'"{i}"' in rule

    def test_annotated_type_large_range_uses_base(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import AnnotatedType, ConstraintRepr, PrimitiveType

        t = AnnotatedType(
            base=PrimitiveType(kind="int"),
            constraints=(
                ConstraintRepr(kind="ge", value=1),
                ConstraintRepr(kind="le", value=1000),
            ),
        )
        config = GrammarConfig(enumeration_threshold=256)
        prods = _type_to_rule(t, "ann_big", config)
        assert len(prods) >= 1
        # Range > threshold, should fall back to base type
        assert "SIGNED_INT" in prods[0].rule

    def test_any_type(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import AnyType

        t = AnyType()
        prods = _type_to_rule(t, "any_val", GrammarConfig())
        assert len(prods) >= 1
        rule = prods[0].rule
        # Should include string, int, float, bool, nil
        assert "ESCAPED_STRING" in rule
        assert "SIGNED_INT" in rule
        assert "SIGNED_FLOAT" in rule
        assert '"true"' in rule
        assert '"false"' in rule
        assert '"nil"' in rule

    def test_recursive_list_of_lists(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import ListType, PrimitiveType

        inner = ListType(element=PrimitiveType(kind="int"))
        outer = ListType(element=inner)
        prods = _type_to_rule(outer, "list_of_list_int", GrammarConfig())
        # Should produce multiple productions (outer list + inner list element)
        assert len(prods) >= 2

    def test_deterministic_rule_naming(self) -> None:
        from tgirl.grammar import GrammarConfig, _type_to_rule
        from tgirl.types import ListType, PrimitiveType

        t = ListType(element=PrimitiveType(kind="int"))
        prods1 = _type_to_rule(t, "my_list", GrammarConfig())
        prods2 = _type_to_rule(t, "my_list", GrammarConfig())
        assert prods1 == prods2
