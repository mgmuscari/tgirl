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

    def test_generate_exists(self) -> None:
        from tgirl.grammar import generate

        snap = self._make_empty_snapshot()
        output = generate(snap)
        assert output.text
        assert output.snapshot_hash

    def test_diff_exists(self) -> None:
        from tgirl.grammar import GrammarOutput, diff

        go = GrammarOutput(
            text="",
            productions=(),
            snapshot_hash="",
            tool_quotas={},
            cost_remaining=None,
        )
        result = diff(go, go)
        assert result.added == ()
        assert result.removed == ()
        assert result.changed == ()


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
        # No mandatory space between tool name and optional args
        # Rule should be: "(" "configure" (SPACE ...)? ")"
        # NOT: "(" "configure" SPACE (SPACE ...)? ")"
        assert '"configure" SPACE (' not in call_rule.rule

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


# --- Task 5: Composition operator productions ---


class TestCompositionProductions:
    """Verify composition operators produce valid grammar rules."""

    def _make_grammar_with_tool(self) -> str:
        import time

        from tgirl.grammar import GrammarConfig, _render_grammar
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        tool = ToolDefinition(
            name="fetch",
            parameters=(
                ParameterDef(
                    name="url",
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
        return _render_grammar(snap, GrammarConfig())

    def test_composition_rules_in_grammar(self) -> None:
        grammar_text = self._make_grammar_with_tool()
        assert "threading" in grammar_text
        assert "let_expr" in grammar_text
        assert "if_expr" in grammar_text
        assert "try_expr" in grammar_text
        assert "pmap_expr" in grammar_text

    def test_threading_parses(self) -> None:
        import lark

        grammar_text = self._make_grammar_with_tool()
        parser = lark.Lark(grammar_text, parser="lalr")
        # (-> (fetch "url1") (fetch "url2"))
        tree = parser.parse('(-> (fetch "url1") (fetch "url2"))')
        assert tree is not None

    def test_let_parses(self) -> None:
        import lark

        grammar_text = self._make_grammar_with_tool()
        parser = lark.Lark(grammar_text, parser="lalr")
        # (let [x (fetch "url")] (fetch "other"))
        tree = parser.parse('(let [x (fetch "url")] (fetch "other"))')
        assert tree is not None

    def test_if_parses(self) -> None:
        import lark

        grammar_text = self._make_grammar_with_tool()
        parser = lark.Lark(grammar_text, parser="lalr")
        # (if (fetch "check") (fetch "yes") (fetch "no"))
        tree = parser.parse(
            '(if (fetch "check") (fetch "yes") (fetch "no"))'
        )
        assert tree is not None

    def test_try_parses(self) -> None:
        import lark

        grammar_text = self._make_grammar_with_tool()
        parser = lark.Lark(grammar_text, parser="lalr")
        # (try (fetch "url") (catch e (fetch "fallback")))
        tree = parser.parse(
            '(try (fetch "url") (catch e (fetch "fallback")))'
        )
        assert tree is not None

    def test_pmap_parses(self) -> None:
        import lark

        grammar_text = self._make_grammar_with_tool()
        parser = lark.Lark(grammar_text, parser="lalr")
        # (pmap [(fetch "a") (fetch "b")] (fetch "combine"))
        tree = parser.parse(
            '(pmap [(fetch "a") (fetch "b")] (fetch "combine"))'
        )
        assert tree is not None

    def test_nested_threading_in_let(self) -> None:
        import lark

        grammar_text = self._make_grammar_with_tool()
        parser = lark.Lark(grammar_text, parser="lalr")
        # (let [x (fetch "url")] (-> (fetch "a") (fetch "b")))
        tree = parser.parse(
            '(let [x (fetch "url")] (-> (fetch "a") (fetch "b")))'
        )
        assert tree is not None

    def test_conditional_containing_pipeline(self) -> None:
        import lark

        grammar_text = self._make_grammar_with_tool()
        parser = lark.Lark(grammar_text, parser="lalr")
        tree = parser.parse(
            '(if (fetch "check") '
            '(-> (fetch "a") (fetch "b")) '
            '(fetch "fallback"))'
        )
        assert tree is not None


# --- Task 6: Grammar generation and determinism ---


class TestGenerate:
    """Verify full generate() from realistic snapshots."""

    def test_generate_empty_snapshot(self) -> None:
        import time

        from tgirl.grammar import generate
        from tgirl.types import RegistrySnapshot

        snap = RegistrySnapshot(
            tools=(),
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        output = generate(snap)
        assert output.text
        assert output.snapshot_hash
        assert output.tool_quotas == {}
        assert output.cost_remaining is None

    def test_generate_single_tool(self) -> None:
        import time

        import lark

        from tgirl.grammar import generate
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
            quota=5,
            cost=0.1,
        )
        snap = RegistrySnapshot(
            tools=(tool,),
            quotas={"greet": 5},
            cost_remaining=10.0,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        output = generate(snap)
        assert "greet" in output.text
        assert output.tool_quotas == {"greet": 5}
        assert output.cost_remaining == 10.0
        assert len(output.productions) > 0
        # Must parse as valid Lark grammar
        parser = lark.Lark(output.text, parser="lalr")
        assert parser is not None

    def test_generate_multiple_tools(self) -> None:
        import time

        import lark

        from tgirl.grammar import generate
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        tools = tuple(
            ToolDefinition(
                name=name,
                parameters=(
                    ParameterDef(
                        name="x",
                        type_repr=PrimitiveType(kind="str"),
                    ),
                ),
                return_type=PrimitiveType(kind="str"),
            )
            for name in ["alpha", "beta", "gamma"]
        )
        snap = RegistrySnapshot(
            tools=tools,
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        output = generate(snap)
        for name in ["alpha", "beta", "gamma"]:
            assert name in output.text
        parser = lark.Lark(output.text, parser="lalr")
        assert parser is not None


class TestDeterminism:
    """Verify same snapshot (different timestamps) produces identical grammar."""

    def test_same_snapshot_same_grammar(self) -> None:
        from tgirl.grammar import generate
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

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
            ),
            return_type=PrimitiveType(kind="str"),
            quota=3,
        )
        snap1 = RegistrySnapshot(
            tools=(tool,),
            quotas={"search": 3},
            cost_remaining=5.0,
            scopes=frozenset(),
            timestamp=1000.0,
        )
        snap2 = RegistrySnapshot(
            tools=(tool,),
            quotas={"search": 3},
            cost_remaining=5.0,
            scopes=frozenset(),
            timestamp=2000.0,
        )
        out1 = generate(snap1)
        out2 = generate(snap2)
        assert out1.text == out2.text
        assert out1.snapshot_hash == out2.snapshot_hash

    def test_shared_types_deduplicated(self) -> None:
        import time

        from tgirl.grammar import generate
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        # Two tools sharing the same parameter type
        tools = tuple(
            ToolDefinition(
                name=name,
                parameters=(
                    ParameterDef(
                        name="x",
                        type_repr=PrimitiveType(kind="str"),
                    ),
                ),
                return_type=PrimitiveType(kind="str"),
            )
            for name in ["foo", "bar"]
        )
        snap = RegistrySnapshot(
            tools=tools,
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        output = generate(snap)
        # Productions should not have duplicates
        names = [p.name for p in output.productions]
        assert len(names) == len(set(names))


# --- Task 7: Grammar diffing ---


class TestDiff:
    """Verify diff() between two grammars."""

    def _make_output(
        self, tools: tuple, quotas: dict | None = None
    ) -> GrammarOutput:  # noqa: F821
        import time

        from tgirl.grammar import generate
        from tgirl.types import RegistrySnapshot

        snap = RegistrySnapshot(
            tools=tools,
            quotas=quotas or {},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        return generate(snap)

    def test_diff_identical_grammars(self) -> None:
        from tgirl.grammar import diff
        from tgirl.types import (
            PrimitiveType,
            ToolDefinition,
        )

        tool = ToolDefinition(
            name="ping",
            parameters=(),
            return_type=PrimitiveType(kind="str"),
        )
        out1 = self._make_output((tool,))
        out2 = self._make_output((tool,))
        result = diff(out1, out2)
        assert result.added == ()
        assert result.removed == ()
        assert result.changed == ()

    def test_diff_added_tool(self) -> None:
        from tgirl.grammar import diff
        from tgirl.types import (
            PrimitiveType,
            ToolDefinition,
        )

        tool1 = ToolDefinition(
            name="ping",
            parameters=(),
            return_type=PrimitiveType(kind="str"),
        )
        tool2 = ToolDefinition(
            name="pong",
            parameters=(),
            return_type=PrimitiveType(kind="str"),
        )
        out1 = self._make_output((tool1,))
        out2 = self._make_output((tool1, tool2))
        result = diff(out1, out2)
        added_names = {p.name for p in result.added}
        assert "call_pong" in added_names

    def test_diff_removed_tool(self) -> None:
        from tgirl.grammar import diff
        from tgirl.types import (
            PrimitiveType,
            ToolDefinition,
        )

        tool1 = ToolDefinition(
            name="ping",
            parameters=(),
            return_type=PrimitiveType(kind="str"),
        )
        tool2 = ToolDefinition(
            name="pong",
            parameters=(),
            return_type=PrimitiveType(kind="str"),
        )
        out1 = self._make_output((tool1, tool2))
        out2 = self._make_output((tool1,))
        result = diff(out1, out2)
        removed_names = {p.name for p in result.removed}
        assert "call_pong" in removed_names

    def test_diff_changed_parameter(self) -> None:
        from tgirl.grammar import diff
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            ToolDefinition,
        )

        tool_v1 = ToolDefinition(
            name="fetch",
            parameters=(
                ParameterDef(
                    name="url",
                    type_repr=PrimitiveType(kind="str"),
                ),
            ),
            return_type=PrimitiveType(kind="str"),
        )
        tool_v2 = ToolDefinition(
            name="fetch",
            parameters=(
                ParameterDef(
                    name="url",
                    type_repr=PrimitiveType(kind="int"),
                ),
            ),
            return_type=PrimitiveType(kind="str"),
        )
        out1 = self._make_output((tool_v1,))
        out2 = self._make_output((tool_v2,))
        result = diff(out1, out2)
        # The param type production should differ
        assert len(result.changed) > 0


# --- Routing grammar ---


class TestRoutingGrammar:
    """Verify generate_routing_grammar() for tool re-ranking."""

    def test_routing_grammar_contains_all_tool_names(self) -> None:
        import time

        from tgirl.grammar import generate_routing_grammar
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        tools = tuple(
            ToolDefinition(
                name=name,
                parameters=(
                    ParameterDef(
                        name="x", type_repr=PrimitiveType(kind="str")
                    ),
                ),
                return_type=PrimitiveType(kind="str"),
                description=f"Tool {name}",
            )
            for name in ["get_field", "set_field", "delete_field"]
        )
        snap = RegistrySnapshot(
            tools=tools,
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        grammar_text = generate_routing_grammar(snap)
        assert '"get_field"' in grammar_text
        assert '"set_field"' in grammar_text
        assert '"delete_field"' in grammar_text

    def test_routing_grammar_tool_names_as_quoted_alternatives(self) -> None:
        import time

        from tgirl.grammar import generate_routing_grammar
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        tools = tuple(
            ToolDefinition(
                name=name,
                parameters=(
                    ParameterDef(
                        name="x", type_repr=PrimitiveType(kind="str")
                    ),
                ),
                return_type=PrimitiveType(kind="str"),
            )
            for name in ["alpha", "beta"]
        )
        snap = RegistrySnapshot(
            tools=tools,
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        grammar_text = generate_routing_grammar(snap)
        assert '"alpha" | "beta"' in grammar_text

    def test_routing_grammar_raises_for_empty_snapshot(self) -> None:
        import time

        from tgirl.grammar import generate_routing_grammar
        from tgirl.types import RegistrySnapshot

        snap = RegistrySnapshot(
            tools=(),
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        with pytest.raises(ValueError, match="empty snapshot"):
            generate_routing_grammar(snap)

    def test_routing_grammar_single_tool(self) -> None:
        import time

        from tgirl.grammar import generate_routing_grammar
        from tgirl.types import (
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        tool = ToolDefinition(
            name="only_tool",
            parameters=(),
            return_type=PrimitiveType(kind="str"),
        )
        snap = RegistrySnapshot(
            tools=(tool,),
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=time.time(),
        )
        grammar_text = generate_routing_grammar(snap)
        assert '"only_tool"' in grammar_text
        # No pipe since single alternative
        assert "|" not in grammar_text

    def test_routing_grammar_deterministic(self) -> None:
        import time

        from tgirl.grammar import generate_routing_grammar
        from tgirl.types import (
            ParameterDef,
            PrimitiveType,
            RegistrySnapshot,
            ToolDefinition,
        )

        tools = tuple(
            ToolDefinition(
                name=name,
                parameters=(
                    ParameterDef(
                        name="x", type_repr=PrimitiveType(kind="str")
                    ),
                ),
                return_type=PrimitiveType(kind="str"),
            )
            for name in ["alpha", "beta", "gamma"]
        )
        snap1 = RegistrySnapshot(
            tools=tools,
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=1000.0,
        )
        snap2 = RegistrySnapshot(
            tools=tools,
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=2000.0,
        )
        assert generate_routing_grammar(snap1) == generate_routing_grammar(snap2)
