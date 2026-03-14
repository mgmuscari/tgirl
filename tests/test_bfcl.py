"""Tests for tgirl.bfcl — BFCL benchmark adapter module."""

from __future__ import annotations

from bfcl_eval.model_handler.utils import default_decode_ast_prompting

from tgirl.bfcl import (
    load_ground_truth,
    load_test_data,
    register_bfcl_tools,
    sexpr_to_bfcl,
)
from tgirl.registry import ToolRegistry
from tgirl.types import PrimitiveType


class TestLoadTestData:
    def test_load_test_data_simple_python(self) -> None:
        """Loads simple_python category, correct count and fields."""
        entries = load_test_data("simple_python")
        assert len(entries) > 0
        entry = entries[0]
        assert "id" in entry
        assert "question" in entry
        assert "function" in entry
        assert entry["id"] == "simple_python_0"


class TestLoadGroundTruth:
    def test_load_ground_truth_simple_python(self) -> None:
        """Loads ground truth for simple_python."""
        entries = load_ground_truth("simple_python")
        assert len(entries) > 0
        entry = entries[0]
        assert "id" in entry
        assert "ground_truth" in entry


class TestRegisterBfclTools:
    def test_register_bfcl_tools(self) -> None:
        """Registers BFCL function defs, name mapping correct."""
        functions = [
            {
                "name": "calculate_triangle_area",
                "description": "Calculate area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["base", "height"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        assert "calculate_triangle_area" in registry
        assert name_map == {
            "calculate_triangle_area": "calculate_triangle_area"
        }
        td = registry.get("calculate_triangle_area")
        assert td.parameters[0].type_repr == PrimitiveType(kind="int")

    def test_register_bfcl_tools_dotted_name(self) -> None:
        """Dotted names are sanitized to underscores."""
        functions = [
            {
                "name": "spotify.play",
                "description": "Play music",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "artist": {"type": "string"},
                    },
                    "required": ["artist"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        assert "spotify_play" in registry
        assert "spotify.play" not in registry
        assert name_map == {"spotify_play": "spotify.play"}


class TestSexprToBfcl:
    def test_sexpr_to_bfcl_simple(self) -> None:
        """Simple int args map to named params."""
        functions = [
            {
                "name": "calculate_triangle_area",
                "description": "Calculate area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["base", "height"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        result = sexpr_to_bfcl(
            "(calculate_triangle_area 10 5)", registry, name_map
        )
        assert (
            result
            == "[calculate_triangle_area(base=10, height=5)]"
        )

    def test_sexpr_to_bfcl_string_args(self) -> None:
        """(reverse "hello") -> [reverse(text="hello")]"""
        functions = [
            {
                "name": "reverse",
                "description": "Reverse text",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        result = sexpr_to_bfcl(
            '(reverse "hello")', registry, name_map
        )
        assert result == '[reverse(text="hello")]'

    def test_sexpr_to_bfcl_string_with_spaces(self) -> None:
        """(greet "hello world") -> [greet(msg="hello world")]"""
        functions = [
            {
                "name": "greet",
                "description": "Greet",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "msg": {"type": "string"},
                    },
                    "required": ["msg"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        result = sexpr_to_bfcl(
            '(greet "hello world")', registry, name_map
        )
        assert result == '[greet(msg="hello world")]'

    def test_sexpr_to_bfcl_nested_list(self) -> None:
        """(func [1 2 3] "hello") -> [func(nums=[1, 2, 3], label="hello")]"""
        functions = [
            {
                "name": "func",
                "description": "",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "nums": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "label": {"type": "string"},
                    },
                    "required": ["nums", "label"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        result = sexpr_to_bfcl(
            '(func [1 2 3] "hello")', registry, name_map
        )
        assert result == '[func(nums=[1, 2, 3], label="hello")]'

    def test_sexpr_to_bfcl_boolean_none(self) -> None:
        """(func True False None) -> [func(a=True, b=False, c=None)]"""
        functions = [
            {
                "name": "func",
                "description": "",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "a": {"type": "boolean"},
                        "b": {"type": "boolean"},
                        "c": {"type": "any"},
                    },
                    "required": ["a", "b", "c"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        result = sexpr_to_bfcl(
            "(func True False None)", registry, name_map
        )
        assert result == "[func(a=True, b=False, c=None)]"

    def test_sexpr_to_bfcl_dotted_name(self) -> None:
        """Sanitized name maps back to dotted original."""
        functions = [
            {
                "name": "spotify.play",
                "description": "Play music",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "artist": {"type": "string"},
                        "duration": {"type": "integer"},
                    },
                    "required": ["artist", "duration"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        result = sexpr_to_bfcl(
            '(spotify_play "Taylor Swift" 20)',
            registry,
            name_map,
        )
        assert (
            result
            == '[spotify.play(artist="Taylor Swift", duration=20)]'
        )

    def test_sexpr_to_bfcl_optional_params_omitted(self) -> None:
        """2 required + 2 optional, only 3 args -> maps to req0, req1, opt0."""
        functions = [
            {
                "name": "search",
                "description": "Search",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                        "sort": {"type": "string"},
                    },
                    "required": ["query", "limit"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        result = sexpr_to_bfcl(
            '(search "test" 10 5)', registry, name_map
        )
        assert (
            result
            == '[search(query="test", limit=10, offset=5)]'
        )

    def test_sexpr_to_bfcl_multiple_calls(self) -> None:
        """Multiple s-expressions -> [func1(...), func2(...)]."""
        functions = [
            {
                "name": "add",
                "description": "Add",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
            {
                "name": "multiply",
                "description": "Multiply",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                },
            },
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        result = sexpr_to_bfcl(
            "(add 1 2)(multiply 3 4)", registry, name_map
        )
        assert (
            result
            == "[add(a=1, b=2), multiply(x=3, y=4)]"
        )

    def test_bfcl_output_parseable(self) -> None:
        """Verify translated output parses through default_decode_ast_prompting."""
        functions = [
            {
                "name": "calculate_triangle_area",
                "description": "Calculate area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["base", "height"],
                },
            }
        ]
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, functions)

        bfcl_output = sexpr_to_bfcl(
            "(calculate_triangle_area 10 5)", registry, name_map
        )
        parsed = default_decode_ast_prompting(bfcl_output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "calculate_triangle_area" in parsed[0]
        assert parsed[0]["calculate_triangle_area"]["base"] == 10
        assert parsed[0]["calculate_triangle_area"]["height"] == 5
