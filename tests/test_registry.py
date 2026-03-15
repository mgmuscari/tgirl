"""Tests for tgirl.registry — ToolRegistry and decorator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tgirl.registry import ToolRegistry
from tgirl.types import (
    AnyType,
    DictType,
    ListType,
    LiteralType,
    ModelType,
    PrimitiveType,
    RegistrySnapshot,
    ToolDefinition,
)


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


class TestRegisterTool:
    def test_register_simple_tool(self, registry: ToolRegistry) -> None:
        @registry.tool()
        def greet(name: str) -> str:
            return f"Hello, {name}"

        assert "greet" in registry
        assert greet("world") == "Hello, world"

    def test_register_tool_extracts_metadata(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool(
            quota=10,
            cost=0.5,
            scope="admin",
            timeout=30.0,
            cacheable=True,
            description="A test tool",
        )
        def do_thing(x: int) -> str:
            return str(x)

        td = registry.get("do_thing")
        assert td.quota == 10
        assert td.cost == 0.5
        assert td.scope == "admin"
        assert td.timeout == 30.0
        assert td.cacheable is True
        assert td.description == "A test tool"

    def test_register_tool_extracts_types(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        td = registry.get("add")
        assert len(td.parameters) == 2
        assert td.parameters[0].name == "a"
        assert td.parameters[0].type_repr == PrimitiveType(kind="int")
        assert td.return_type == PrimitiveType(kind="int")

    def test_register_duplicate_name_raises(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def dup(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="already registered"):

            @registry.tool()
            def dup(x: int) -> int:  # noqa: F811
                return x

    def test_register_missing_annotation_raises(
        self, registry: ToolRegistry
    ) -> None:
        with pytest.raises(TypeError, match="missing type annotation"):

            @registry.tool()
            def bad(x) -> str:  # type: ignore[no-untyped-def]
                return str(x)

    def test_register_missing_return_type_raises(
        self, registry: ToolRegistry
    ) -> None:
        with pytest.raises(TypeError, match="missing return type"):

            @registry.tool()
            def bad(x: int):  # type: ignore[no-untyped-def]
                return x

    def test_decorator_returns_original_function(
        self, registry: ToolRegistry
    ) -> None:
        def orig(x: int) -> int:
            return x

        decorated = registry.tool()(orig)
        assert decorated is orig

    def test_registered_tool_is_callable(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        assert registry.get_callable("add")(1, 2) == 3


class TestRegistryAccessors:
    def test_registry_len(self, registry: ToolRegistry) -> None:
        assert len(registry) == 0

        @registry.tool()
        def a(x: int) -> int:
            return x

        assert len(registry) == 1

    def test_registry_contains(self, registry: ToolRegistry) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        assert "a" in registry
        assert "b" not in registry

    def test_registry_get(self, registry: ToolRegistry) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        td = registry.get("a")
        assert isinstance(td, ToolDefinition)
        assert td.name == "a"

    def test_registry_get_missing_raises(
        self, registry: ToolRegistry
    ) -> None:
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_registry_names(self, registry: ToolRegistry) -> None:
        @registry.tool()
        def beta(x: int) -> int:
            return x

        @registry.tool()
        def alpha(x: int) -> int:
            return x

        assert registry.names() == ["alpha", "beta"]


class TestSnapshot:
    def test_snapshot_basic(self, registry: ToolRegistry) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        snap = registry.snapshot()
        assert isinstance(snap, RegistrySnapshot)
        assert len(snap.tools) == 1
        assert snap.tools[0].name == "a"

    def test_snapshot_scope_filtering_includes_matching(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool(scope="admin")
        def a(x: int) -> int:
            return x

        snap = registry.snapshot(scopes={"admin"})
        assert len(snap.tools) == 1

    def test_snapshot_scope_filtering_excludes_nonmatching(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool(scope="admin")
        def a(x: int) -> int:
            return x

        snap = registry.snapshot(scopes={"user"})
        assert len(snap.tools) == 0

    def test_snapshot_scope_filtering_includes_unrestricted(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        snap = registry.snapshot(scopes={"admin"})
        assert len(snap.tools) == 1

    def test_snapshot_scope_none_includes_all(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool(scope="admin")
        def a(x: int) -> int:
            return x

        @registry.tool()
        def b(x: int) -> int:
            return x

        snap = registry.snapshot()
        assert len(snap.tools) == 2

    def test_snapshot_restrict_to(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        @registry.tool()
        def b(x: int) -> int:
            return x

        snap = registry.snapshot(restrict_to=["a"])
        assert len(snap.tools) == 1
        assert snap.tools[0].name == "a"

    def test_snapshot_restrict_to_unknown_ignored(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        snap = registry.snapshot(restrict_to=["a", "nonexistent"])
        assert len(snap.tools) == 1

    def test_snapshot_quotas(self, registry: ToolRegistry) -> None:
        @registry.tool(quota=5)
        def a(x: int) -> int:
            return x

        @registry.tool(quota=10)
        def b(x: int) -> int:
            return x

        snap = registry.snapshot()
        assert snap.quotas["a"] == 5
        assert snap.quotas["b"] == 10

    def test_snapshot_cost_budget(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        snap = registry.snapshot(cost_budget=100.0)
        assert snap.cost_remaining == 100.0

    def test_snapshot_deterministic(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool(quota=5)
        def a(x: int) -> int:
            return x

        @registry.tool(quota=10)
        def b(x: int) -> int:
            return x

        snap1 = registry.snapshot()
        snap2 = registry.snapshot()
        # Compare everything except timestamp
        assert snap1.tools == snap2.tools
        assert snap1.quotas == snap2.quotas
        assert snap1.scopes == snap2.scopes

    def test_snapshot_tools_sorted_by_name(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def charlie(x: int) -> int:
            return x

        @registry.tool()
        def alpha(x: int) -> int:
            return x

        @registry.tool()
        def bravo(x: int) -> int:
            return x

        snap = registry.snapshot()
        names = [t.name for t in snap.tools]
        assert names == ["alpha", "bravo", "charlie"]

    def test_snapshot_is_frozen(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        snap = registry.snapshot()
        with pytest.raises(ValidationError):
            snap.tools = ()  # type: ignore[misc]

    def test_snapshot_serializes_to_json(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool(quota=5)
        def a(x: int) -> int:
            return x

        snap = registry.snapshot()
        json_str = snap.model_dump_json()
        assert '"a"' in json_str

    def test_tool_definition_json_schema_from_snapshot(
        self, registry: ToolRegistry
    ) -> None:
        @registry.tool()
        def a(x: int) -> int:
            return x

        snap = registry.snapshot()
        schema = snap.model_json_schema()
        assert isinstance(schema, dict)


class TestRegisterFromSchema:
    def test_register_from_schema_simple(
        self, registry: ToolRegistry
    ) -> None:
        """Register a function with int and str params, verify ToolDefinition."""
        schema_params = {
            "type": "dict",
            "properties": {
                "base": {"type": "integer"},
                "height": {"type": "string"},
            },
            "required": ["base", "height"],
        }
        registry.register_from_schema(
            name="calculate_area",
            parameters=schema_params,
            description="Calculate area",
        )
        assert "calculate_area" in registry
        td = registry.get("calculate_area")
        assert td.name == "calculate_area"
        assert len(td.parameters) == 2
        assert td.parameters[0].name == "base"
        assert td.parameters[0].type_repr == PrimitiveType(kind="int")
        assert td.parameters[1].name == "height"
        assert td.parameters[1].type_repr == PrimitiveType(kind="str")
        assert td.description == "Calculate area"

    def test_register_from_schema_types(
        self, registry: ToolRegistry
    ) -> None:
        """All JSON schema types map to correct TypeRepr."""
        schema_params = {
            "type": "dict",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "f": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "any"},
            },
            "required": ["s", "i", "f", "b", "a"],
        }
        registry.register_from_schema(
            name="all_types",
            parameters=schema_params,
            description="",
        )
        td = registry.get("all_types")
        assert td.parameters[0].type_repr == PrimitiveType(kind="str")
        assert td.parameters[1].type_repr == PrimitiveType(kind="int")
        assert td.parameters[2].type_repr == PrimitiveType(kind="float")
        assert td.parameters[3].type_repr == PrimitiveType(kind="bool")
        assert td.parameters[4].type_repr == AnyType()

    def test_register_from_schema_optional_params(
        self, registry: ToolRegistry
    ) -> None:
        """Parameters not in required list get has_default=True."""
        schema_params = {
            "type": "dict",
            "properties": {
                "required_param": {"type": "string"},
                "optional_param": {"type": "integer"},
            },
            "required": ["required_param"],
        }
        registry.register_from_schema(
            name="opt_func",
            parameters=schema_params,
            description="",
        )
        td = registry.get("opt_func")
        req = td.parameters[0]
        opt = td.parameters[1]
        assert req.name == "required_param"
        assert req.has_default is False
        assert opt.name == "optional_param"
        assert opt.has_default is True
        assert opt.default is None

    def test_register_from_schema_accepts_any_name(
        self, registry: ToolRegistry
    ) -> None:
        """Registers with pre-sanitized name, no name transformation."""
        schema_params = {
            "type": "dict",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        registry.register_from_schema(
            name="spotify_play",
            parameters=schema_params,
            description="",
        )
        assert "spotify_play" in registry
        td = registry.get("spotify_play")
        assert td.name == "spotify_play"

    def test_register_from_schema_enum(
        self, registry: ToolRegistry
    ) -> None:
        """Enum field creates LiteralType."""
        schema_params = {
            "type": "dict",
            "properties": {
                "color": {
                    "type": "string",
                    "enum": ["red", "green", "blue"],
                },
            },
            "required": ["color"],
        }
        registry.register_from_schema(
            name="paint",
            parameters=schema_params,
            description="",
        )
        td = registry.get("paint")
        assert td.parameters[0].type_repr == LiteralType(
            values=("red", "green", "blue")
        )

    def test_register_from_schema_array(
        self, registry: ToolRegistry
    ) -> None:
        """Array with items type maps to ListType."""
        schema_params = {
            "type": "dict",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "tags": {"type": "array"},
            },
            "required": ["numbers", "tags"],
        }
        registry.register_from_schema(
            name="list_func",
            parameters=schema_params,
            description="",
        )
        td = registry.get("list_func")
        assert td.parameters[0].type_repr == ListType(
            element=PrimitiveType(kind="int")
        )
        assert td.parameters[1].type_repr == ListType(element=AnyType())

    def test_register_from_schema_snapshot(
        self, registry: ToolRegistry
    ) -> None:
        """Registered schema tools appear in snapshot."""
        schema_params = {
            "type": "dict",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        registry.register_from_schema(
            name="snap_tool",
            parameters=schema_params,
            description="A snap tool",
        )
        snap = registry.snapshot()
        assert len(snap.tools) == 1
        assert snap.tools[0].name == "snap_tool"

    def test_register_from_schema_duplicate_raises(
        self, registry: ToolRegistry
    ) -> None:
        """Duplicate registration raises ValueError."""
        schema_params = {
            "type": "dict",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        registry.register_from_schema(
            name="dup", parameters=schema_params, description=""
        )
        with pytest.raises(ValueError, match="already registered"):
            registry.register_from_schema(
                name="dup", parameters=schema_params, description=""
            )

    def test_register_from_schema_dict_type(
        self, registry: ToolRegistry
    ) -> None:
        """Dict type without properties maps to DictType."""
        schema_params = {
            "type": "dict",
            "properties": {
                "data": {"type": "dict"},
            },
            "required": ["data"],
        }
        registry.register_from_schema(
            name="dict_func", parameters=schema_params, description=""
        )
        td = registry.get("dict_func")
        assert td.parameters[0].type_repr == DictType(
            key=PrimitiveType(kind="str"), value=AnyType()
        )

    def test_register_from_schema_tuple_type(
        self, registry: ToolRegistry
    ) -> None:
        """Tuple type maps to ListType(element=AnyType)."""
        schema_params = {
            "type": "dict",
            "properties": {
                "coords": {"type": "tuple"},
            },
            "required": ["coords"],
        }
        registry.register_from_schema(
            name="tuple_func", parameters=schema_params, description=""
        )
        td = registry.get("tuple_func")
        assert td.parameters[0].type_repr == ListType(
            element=AnyType()
        )

    def test_register_from_schema_with_quota(
        self, registry: ToolRegistry
    ) -> None:
        """Register with quota: snapshot includes quota in quotas dict."""
        schema_params = {
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        registry.register_from_schema(
            name="quota_tool",
            parameters=schema_params,
            description="",
            quota=5,
        )
        td = registry.get("quota_tool")
        assert td.quota == 5
        snap = registry.snapshot()
        assert snap.quotas["quota_tool"] == 5

    def test_register_from_schema_with_scope(
        self, registry: ToolRegistry
    ) -> None:
        """Register with scope: snapshot(scopes=...) filtering works."""
        schema_params = {
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        registry.register_from_schema(
            name="scoped_tool",
            parameters=schema_params,
            description="",
            scope="admin",
        )
        # Included with matching scope
        snap = registry.snapshot(scopes={"admin"})
        assert len(snap.tools) == 1
        # Excluded with non-matching scope
        snap = registry.snapshot(scopes={"user"})
        assert len(snap.tools) == 0

    def test_register_from_schema_with_callable_fn(
        self, registry: ToolRegistry
    ) -> None:
        """Register with callable_fn: get_callable returns it, not no-op."""
        def my_impl(**kwargs: object) -> str:
            return "real"

        schema_params = {
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        registry.register_from_schema(
            name="callable_tool",
            parameters=schema_params,
            description="",
            callable_fn=my_impl,
        )
        assert registry.get_callable("callable_tool") is my_impl
        assert registry.get_callable("callable_tool")(x=1) == "real"

    def test_register_from_schema_without_new_params(
        self, registry: ToolRegistry
    ) -> None:
        """Without new params: behavior identical to current."""
        schema_params = {
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        registry.register_from_schema(
            name="default_tool",
            parameters=schema_params,
            description="",
        )
        td = registry.get("default_tool")
        assert td.quota is None
        assert td.scope is None
        assert td.cost == 0.0
        assert td.timeout is None
        # Default callable is a no-op
        result = registry.get_callable("default_tool")(x=1)
        assert result is None

    def test_register_from_schema_with_cost_and_timeout(
        self, registry: ToolRegistry
    ) -> None:
        """Register with cost and timeout."""
        schema_params = {
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        registry.register_from_schema(
            name="cost_tool",
            parameters=schema_params,
            description="",
            cost=1.5,
            timeout=30.0,
        )
        td = registry.get("cost_tool")
        assert td.cost == 1.5
        assert td.timeout == 30.0

    def test_schema_type_to_repr_object_with_properties(
        self, registry: ToolRegistry
    ) -> None:
        """JSON Schema 'object' with properties maps to ModelType."""
        schema_params = {
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["name"],
                },
            },
            "required": ["config"],
        }
        registry.register_from_schema(
            name="obj_tool", parameters=schema_params, description=""
        )
        td = registry.get("obj_tool")
        type_repr = td.parameters[0].type_repr
        assert isinstance(type_repr, ModelType)
        assert len(type_repr.fields) == 2
        # Check field names and types
        field_map = {f.name: f for f in type_repr.fields}
        assert field_map["name"].type_repr == PrimitiveType(kind="str")
        assert field_map["name"].required is True
        assert field_map["count"].type_repr == PrimitiveType(kind="int")
        assert field_map["count"].required is False

    def test_schema_type_to_repr_nested_object(
        self, registry: ToolRegistry
    ) -> None:
        """Nested object properties recurse correctly."""
        schema_params = {
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "boolean"},
                            },
                            "required": ["value"],
                        },
                    },
                    "required": ["inner"],
                },
            },
            "required": ["outer"],
        }
        registry.register_from_schema(
            name="nested_tool", parameters=schema_params, description=""
        )
        td = registry.get("nested_tool")
        outer = td.parameters[0].type_repr
        assert isinstance(outer, ModelType)
        inner_field = outer.fields[0]
        assert inner_field.name == "inner"
        assert isinstance(inner_field.type_repr, ModelType)
        value_field = inner_field.type_repr.fields[0]
        assert value_field.name == "value"
        assert value_field.type_repr == PrimitiveType(kind="bool")

    def test_schema_type_to_repr_object_without_properties(
        self, registry: ToolRegistry
    ) -> None:
        """JSON Schema 'object' without properties maps to DictType."""
        schema_params = {
            "properties": {
                "data": {"type": "object"},
            },
            "required": ["data"],
        }
        registry.register_from_schema(
            name="dict_obj_tool", parameters=schema_params, description=""
        )
        td = registry.get("dict_obj_tool")
        assert td.parameters[0].type_repr == DictType(
            key=PrimitiveType(kind="str"), value=AnyType()
        )

    def test_schema_type_to_repr_null(
        self, registry: ToolRegistry
    ) -> None:
        """JSON Schema 'null' maps to PrimitiveType(kind='none')."""
        schema_params = {
            "properties": {
                "nothing": {"type": "null"},
            },
            "required": ["nothing"],
        }
        registry.register_from_schema(
            name="null_tool", parameters=schema_params, description=""
        )
        td = registry.get("null_tool")
        assert td.parameters[0].type_repr == PrimitiveType(kind="none")
