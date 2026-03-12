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


# --- Task 3: Tool-to-production converter ---


class TestToolProductions:
    """Verify _tool_to_rules for various tool signatures."""

    def test_tool_no_parameters(self) -> None:
        from tgirl.grammar import GrammarConfig, _tool_to_rules
        from tgirl.types import PrimitiveType, ToolDefinition

        tool = ToolDefinition(
            name="ping",
            parameters=(),
            return_type=PrimitiveType(kind="str"),
        )
        prods = _tool_to_rules(tool, GrammarConfig())
        # Should produce call_ping rule with no args
        call_rule = next(p for p in prods if p.name == "call_ping")
        assert '"(" "ping" ")"' in call_rule.rule

    def test_tool_single_required_param(self) -> None:
        from tgirl.grammar import GrammarConfig, _tool_to_rules
        from tgirl.types import ParameterDef, PrimitiveType, ToolDefinition

        tool = ToolDefinition(
            name="greet",
            parameters=(
                ParameterDef(
                    name="name",
                    type_repr=PrimitiveType(kind="str"),
                ),
            ),
            return_type=PrimitiveType(kind="str"),
        )
        prods = _tool_to_rules(tool, GrammarConfig())
        call_rule = next(p for p in prods if p.name == "call_greet")
        assert '"greet"' in call_rule.rule

    def test_tool_multiple_required_params(self) -> None:
        from tgirl.grammar import GrammarConfig, _tool_to_rules
        from tgirl.types import ParameterDef, PrimitiveType, ToolDefinition

        tool = ToolDefinition(
            name="add",
            parameters=(
                ParameterDef(
                    name="a", type_repr=PrimitiveType(kind="int")
                ),
                ParameterDef(
                    name="b", type_repr=PrimitiveType(kind="int")
                ),
            ),
            return_type=PrimitiveType(kind="int"),
        )
        prods = _tool_to_rules(tool, GrammarConfig())
        call_rule = next(p for p in prods if p.name == "call_add")
        assert '"add"' in call_rule.rule

    def test_tool_with_optional_params(self) -> None:
        from tgirl.grammar import GrammarConfig, _tool_to_rules
        from tgirl.types import ParameterDef, PrimitiveType, ToolDefinition

        tool = ToolDefinition(
            name="search",
            parameters=(
                ParameterDef(
                    name="query",
                    type_repr=PrimitiveType(kind="str"),
                ),
                ParameterDef(
                    name="limit",
                    type_repr=PrimitiveType(kind="int"),
                    has_default=True,
                    default=10,
                ),
                ParameterDef(
                    name="offset",
                    type_repr=PrimitiveType(kind="int"),
                    has_default=True,
                    default=0,
                ),
            ),
            return_type=PrimitiveType(kind="str"),
        )
        prods = _tool_to_rules(tool, GrammarConfig())
        call_rule = next(p for p in prods if p.name == "call_search")
        # Should have trailing optional chain
        assert '"search"' in call_rule.rule
        # The optional params should be wrapped in (...)?
        assert "?" in call_rule.rule

    def test_tool_all_optional_params(self) -> None:
        from tgirl.grammar import GrammarConfig, _tool_to_rules
        from tgirl.types import ParameterDef, PrimitiveType, ToolDefinition

        tool = ToolDefinition(
            name="configure",
            parameters=(
                ParameterDef(
                    name="verbose",
                    type_repr=PrimitiveType(kind="bool"),
                    has_default=True,
                    default=False,
                ),
            ),
            return_type=PrimitiveType(kind="none"),
        )
        prods = _tool_to_rules(tool, GrammarConfig())
        call_rule = next(p for p in prods if p.name == "call_configure")
        # Entire args section should be optional
        assert "?" in call_rule.rule

    def test_tool_with_complex_types(self) -> None:
        from tgirl.grammar import GrammarConfig, _tool_to_rules
        from tgirl.types import (
            ListType,
            ParameterDef,
            PrimitiveType,
            ToolDefinition,
        )

        tool = ToolDefinition(
            name="process",
            parameters=(
                ParameterDef(
                    name="items",
                    type_repr=ListType(element=PrimitiveType(kind="str")),
                ),
            ),
            return_type=PrimitiveType(kind="int"),
        )
        prods = _tool_to_rules(tool, GrammarConfig())
        # Should produce call rule + type productions for the list
        assert len(prods) >= 2
        rule_names = [p.name for p in prods]
        assert "call_process" in rule_names


# --- Task 4: Jinja2 template system ---


class TestTemplateLoading:
    """Verify templates load without error."""

    def test_load_templates(self) -> None:
        from tgirl.grammar import _load_templates

        env = _load_templates()
        assert env is not None
        # Should be able to get the base template
        tmpl = env.get_template("base.cfg.j2")
        assert tmpl is not None

    def test_all_templates_exist(self) -> None:
        from tgirl.grammar import _load_templates

        env = _load_templates()
        for name in [
            "base.cfg.j2",
            "tools.cfg.j2",
            "types.cfg.j2",
            "composition.cfg.j2",
        ]:
            tmpl = env.get_template(name)
            assert tmpl is not None


class TestTemplateRendering:
    """Verify rendered output is valid Lark EBNF."""

    def test_empty_snapshot_produces_valid_grammar(self) -> None:
        import time

        import lark

        from tgirl.grammar import GrammarConfig, _render_grammar
        from tgirl.types import RegistrySnapshot

        snap = RegistrySnapshot(
            tools=(),
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        grammar_text = _render_grammar(snap, GrammarConfig())
        assert grammar_text
        # Must parse as valid Lark LALR(1) grammar
        parser = lark.Lark(grammar_text, parser="lalr")
        assert parser is not None

    def test_single_tool_produces_valid_grammar(self) -> None:
        import time

        import lark

        from tgirl.grammar import GrammarConfig, _render_grammar
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        tool = ToolDefinition(
            name="greet",
            parameters=(
                ParameterDef(
                    name="name",
                    type_repr=PrimitiveType(kind="str"),
                ),
            ),
            return_type=PrimitiveType(kind="str"),
        )
        snap = RegistrySnapshot(
            tools=(tool,),
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        grammar_text = _render_grammar(snap, GrammarConfig())
        assert grammar_text
        assert "greet" in grammar_text
        # Must parse as valid Lark LALR(1) grammar
        parser = lark.Lark(grammar_text, parser="lalr")
        assert parser is not None

    def test_single_tool_call_alternatives(self) -> None:
        import time

        from tgirl.grammar import GrammarConfig, _render_grammar
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        tool = ToolDefinition(
            name="greet",
            parameters=(
                ParameterDef(
                    name="name",
                    type_repr=PrimitiveType(kind="str"),
                ),
            ),
            return_type=PrimitiveType(kind="str"),
        )
        snap = RegistrySnapshot(
            tools=(tool,),
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        grammar_text = _render_grammar(snap, GrammarConfig())
        assert "call_greet" in grammar_text
