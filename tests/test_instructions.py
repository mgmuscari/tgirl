"""Tests for the instruction generator module."""

from __future__ import annotations

import json

import pytest

from tgirl.registry import ToolRegistry
from tgirl.types import ParameterDef, PrimitiveType, ToolDefinition


# --- Fixtures ---


@pytest.fixture()
def simple_registry() -> ToolRegistry:
    """Registry with simple, non-overlapping tools."""
    registry = ToolRegistry()

    @registry.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @registry.tool()
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    return registry


@pytest.fixture()
def json_registry() -> ToolRegistry:
    """Registry with overlapping str-typed JSON tools."""
    registry = ToolRegistry()

    @registry.tool()
    def get_field(obj: str, key: str) -> str:
        """Extract a field value from a JSON object string."""
        return str(json.loads(obj)[key])

    @registry.tool()
    def merge_objects(a: str, b: str) -> str:
        """Merge two JSON objects. Keys in b override keys in a."""
        merged = {**json.loads(a), **json.loads(b)}
        return json.dumps(merged)

    @registry.tool()
    def count_keys(obj: str) -> int:
        """Count the number of keys in a JSON object."""
        return len(json.loads(obj))

    return registry


# --- Tests: generate_tool_doc ---


class TestGenerateToolDoc:
    """Tests for single-tool documentation generation."""

    def test_includes_tool_name(self, simple_registry: ToolRegistry) -> None:
        from tgirl.instructions import generate_tool_doc

        doc = generate_tool_doc(simple_registry.get("add"))
        assert "add" in doc

    def test_includes_parameter_names(self, simple_registry: ToolRegistry) -> None:
        from tgirl.instructions import generate_tool_doc

        doc = generate_tool_doc(simple_registry.get("add"))
        assert "a" in doc
        assert "b" in doc

    def test_includes_parameter_types(self, simple_registry: ToolRegistry) -> None:
        from tgirl.instructions import generate_tool_doc

        doc = generate_tool_doc(simple_registry.get("add"))
        assert "int" in doc

    def test_includes_return_type(self, simple_registry: ToolRegistry) -> None:
        from tgirl.instructions import generate_tool_doc

        doc = generate_tool_doc(simple_registry.get("add"))
        assert "-> int" in doc

    def test_includes_description(self, simple_registry: ToolRegistry) -> None:
        from tgirl.instructions import generate_tool_doc

        doc = generate_tool_doc(simple_registry.get("add"))
        assert "Add two integers" in doc

    def test_sexpr_signature_format(self, simple_registry: ToolRegistry) -> None:
        """Tool doc should show s-expression call syntax."""
        from tgirl.instructions import generate_tool_doc

        doc = generate_tool_doc(simple_registry.get("add"))
        # Should contain something like (add <a:int> <b:int>)
        assert "(add" in doc

    def test_str_params_show_parameter_names(
        self, json_registry: ToolRegistry
    ) -> None:
        """When multiple str params exist, parameter names must appear to disambiguate."""
        from tgirl.instructions import generate_tool_doc

        doc = generate_tool_doc(json_registry.get("get_field"))
        # Must show 'obj' and 'key' so the model can distinguish roles
        assert "obj" in doc
        assert "key" in doc

    def test_different_tools_produce_different_docs(
        self, json_registry: ToolRegistry
    ) -> None:
        from tgirl.instructions import generate_tool_doc

        doc_get = generate_tool_doc(json_registry.get("get_field"))
        doc_merge = generate_tool_doc(json_registry.get("merge_objects"))
        assert doc_get != doc_merge


# --- Tests: generate_system_prompt ---


class TestGenerateSystemPrompt:
    """Tests for full system prompt generation from a snapshot."""

    def test_returns_string(self, simple_registry: ToolRegistry) -> None:
        from tgirl.instructions import generate_system_prompt

        prompt = generate_system_prompt(simple_registry.snapshot())
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_all_tool_names(self, json_registry: ToolRegistry) -> None:
        from tgirl.instructions import generate_system_prompt

        prompt = generate_system_prompt(json_registry.snapshot())
        assert "get_field" in prompt
        assert "merge_objects" in prompt
        assert "count_keys" in prompt

    def test_contains_syntax_instructions(
        self, simple_registry: ToolRegistry
    ) -> None:
        from tgirl.instructions import generate_system_prompt

        prompt = generate_system_prompt(simple_registry.snapshot())
        # Should explain s-expression syntax
        assert "(" in prompt
        assert ")" in prompt

    def test_contains_type_information(
        self, simple_registry: ToolRegistry
    ) -> None:
        from tgirl.instructions import generate_system_prompt

        prompt = generate_system_prompt(simple_registry.snapshot())
        assert "int" in prompt
        assert "str" in prompt

    def test_deterministic(self, json_registry: ToolRegistry) -> None:
        """Same snapshot produces same prompt."""
        from tgirl.instructions import generate_system_prompt

        snap = json_registry.snapshot()
        p1 = generate_system_prompt(snap)
        p2 = generate_system_prompt(snap)
        assert p1 == p2

    def test_parameter_names_disambiguate_str_tools(
        self, json_registry: ToolRegistry
    ) -> None:
        """Prompt must show param names to help model distinguish str-typed tools."""
        from tgirl.instructions import generate_system_prompt

        prompt = generate_system_prompt(json_registry.snapshot())
        # get_field params
        assert "obj" in prompt
        assert "key" in prompt
        # merge_objects params — must be distinguishable from get_field
        # Both have str params but different names/roles


# --- Tests: enrichment ---


class TestEnrichment:
    """Tests for post-registration enrichment API."""

    def test_enrich_adds_param_tags(self) -> None:
        from tgirl.instructions import generate_tool_doc

        registry = ToolRegistry()

        @registry.tool()
        def get_field(obj: str, key: str) -> str:
            """Extract a field value from a JSON object string."""
            return ""

        registry.enrich("get_field", param_tags={"obj": "JsonObject", "key": "FieldName"})
        doc = generate_tool_doc(registry.get("get_field"))
        assert "JsonObject" in doc
        assert "FieldName" in doc

    def test_enrich_nonexistent_tool_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.enrich("nonexistent", param_tags={"x": "Foo"})

    def test_enrich_unknown_param_raises(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        with pytest.raises(ValueError):
            registry.enrich("add", param_tags={"unknown_param": "Foo"})

    def test_enrichment_survives_snapshot(self) -> None:
        """Enrichment metadata must be visible in snapshots."""
        from tgirl.instructions import generate_system_prompt

        registry = ToolRegistry()

        @registry.tool()
        def get_field(obj: str, key: str) -> str:
            """Extract a field."""
            return ""

        registry.enrich("get_field", param_tags={"obj": "JsonObject", "key": "FieldName"})
        prompt = generate_system_prompt(registry.snapshot())
        assert "JsonObject" in prompt
        assert "FieldName" in prompt

    def test_unenriched_tools_use_param_names(self) -> None:
        """Tools without enrichment fall back to parameter names."""
        from tgirl.instructions import generate_tool_doc

        registry = ToolRegistry()

        @registry.tool()
        def get_field(obj: str, key: str) -> str:
            """Extract a field."""
            return ""

        doc = generate_tool_doc(registry.get("get_field"))
        # Should still show param names even without enrichment
        assert "obj" in doc
        assert "key" in doc

    def test_enrich_adds_examples(self) -> None:
        from tgirl.instructions import generate_tool_doc

        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        registry.enrich("add", examples=['(add 3 5) -> 8'])
        doc = generate_tool_doc(registry.get("add"))
        assert "(add 3 5) -> 8" in doc

    def test_register_type_grammar(self) -> None:
        """Registered type grammars appear in snapshot."""
        registry = ToolRegistry()
        registry.register_type("JsonObject", "ESCAPED_STRING")

        @registry.tool()
        def get_field(obj: str, key: str) -> str:
            return ""

        registry.enrich("get_field", param_tags={"obj": "JsonObject"})
        snap = registry.snapshot()
        tg = dict(snap.type_grammars)
        assert "JsonObject" in tg
        assert tg["JsonObject"] == "ESCAPED_STRING"

    def test_register_type_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register_type("Foo", "ESCAPED_STRING")
        with pytest.raises(ValueError, match="already registered"):
            registry.register_type("Foo", "ESCAPED_STRING")

    def test_type_grammar_only_includes_referenced(self) -> None:
        """Snapshot only includes type grammars referenced by included tools."""
        registry = ToolRegistry()
        registry.register_type("JsonObject", "ESCAPED_STRING")
        registry.register_type("Unused", "SIGNED_INT")

        @registry.tool()
        def get_field(obj: str, key: str) -> str:
            return ""

        registry.enrich("get_field", param_tags={"obj": "JsonObject"})
        snap = registry.snapshot()
        tg = dict(snap.type_grammars)
        assert "JsonObject" in tg
        assert "Unused" not in tg

    def test_type_grammar_in_generated_grammar(self) -> None:
        """Semantic types produce distinct grammar rules."""
        from tgirl.grammar import generate as generate_grammar

        registry = ToolRegistry()
        registry.register_type("FieldName", '/\\"[a-zA-Z_]+\\"/')

        @registry.tool()
        def get_field(obj: str, key: str) -> str:
            return ""

        registry.enrich("get_field", param_tags={"key": "FieldName"})
        snap = registry.snapshot()
        g = generate_grammar(snap)
        assert "stype_fieldname" in g.text
        assert '/\\"[a-zA-Z_]+\\"/' in g.text

    def test_enrichment_overrides_not_replaces(self) -> None:
        """Enriching some params leaves others as default."""
        from tgirl.instructions import generate_tool_doc

        registry = ToolRegistry()

        @registry.tool()
        def get_field(obj: str, key: str) -> str:
            """Extract a field."""
            return ""

        registry.enrich("get_field", param_tags={"obj": "JsonObject"})
        doc = generate_tool_doc(registry.get("get_field"))
        assert "JsonObject" in doc
        # 'key' should still appear with its default representation
        assert "key" in doc


# --- Tests: generate_routing_prompt ---


class TestGenerateRoutingPrompt:
    """Tests for routing prompt generation."""

    def test_routing_prompt_contains_all_tool_names(
        self, simple_registry: ToolRegistry
    ) -> None:
        from tgirl.instructions import generate_routing_prompt

        prompt = generate_routing_prompt(simple_registry.snapshot())
        assert "add" in prompt
        assert "greet" in prompt

    def test_routing_prompt_contains_descriptions(
        self, simple_registry: ToolRegistry
    ) -> None:
        from tgirl.instructions import generate_routing_prompt

        prompt = generate_routing_prompt(simple_registry.snapshot())
        assert "Add two integers" in prompt
        assert "Greet someone by name" in prompt

    def test_routing_prompt_contains_directive(
        self, simple_registry: ToolRegistry
    ) -> None:
        from tgirl.instructions import generate_routing_prompt

        prompt = generate_routing_prompt(simple_registry.snapshot())
        assert "Reply with ONLY the tool name" in prompt

    def test_routing_prompt_deterministic(
        self, simple_registry: ToolRegistry
    ) -> None:
        from tgirl.instructions import generate_routing_prompt

        snap = simple_registry.snapshot()
        p1 = generate_routing_prompt(snap)
        p2 = generate_routing_prompt(snap)
        assert p1 == p2

    def test_routing_prompt_contains_parameter_names(
        self, simple_registry: ToolRegistry
    ) -> None:
        from tgirl.instructions import generate_routing_prompt

        prompt = generate_routing_prompt(simple_registry.snapshot())
        assert "a, b" in prompt
        assert "name" in prompt
