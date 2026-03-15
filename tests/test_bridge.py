"""Tests for tgirl.bridge — MCP tool import/export."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tgirl.registry import ToolRegistry
from tgirl.types import (
    AnnotatedType,
    AnyType,
    ConstraintRepr,
    DictType,
    EnumType,
    FieldDef,
    ListType,
    LiteralType,
    ModelType,
    OptionalType,
    PrimitiveType,
    UnionType,
)


def _make_mock_tool(
    name: str,
    input_schema: dict[str, Any],
    description: str = "",
) -> MagicMock:
    """Create a mock MCP Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema
    return tool


def _make_mock_call_result(
    text: str = "ok", is_error: bool = False
) -> MagicMock:
    """Create a mock MCP CallToolResult."""
    content_item = MagicMock()
    content_item.type = "text"
    content_item.text = text
    result = MagicMock()
    result.content = [content_item]
    result.isError = is_error
    return result


class TestImportMcpTools:
    """Tests for import_mcp_tools function."""

    def test_registers_tools_with_correct_types(self) -> None:
        """Tools from MCP server are registered with correct TypeRepr."""
        from tgirl.bridge import import_mcp_tools

        registry = ToolRegistry()
        mock_tools = [
            _make_mock_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
                description="Get weather for a city",
            ),
        ]

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.tools = mock_tools
        mock_session.list_tools.return_value = mock_list_result
        mock_session.initialize.return_value = MagicMock()

        with patch(
            "tgirl.bridge._create_mcp_session",
            return_value=mock_session,
        ):
            conn = import_mcp_tools(
                registry, "echo test_server"
            )
            try:
                assert "get_weather" in registry
                td = registry.get("get_weather")
                assert td.description == "Get weather for a city"
                # city is str
                city_param = next(
                    p for p in td.parameters if p.name == "city"
                )
                assert city_param.type_repr == PrimitiveType(kind="str")
                # units is enum
                units_param = next(
                    p for p in td.parameters if p.name == "units"
                )
                assert units_param.type_repr == LiteralType(
                    values=("celsius", "fahrenheit")
                )
            finally:
                conn.close()

    def test_name_sanitization(self) -> None:
        """Dotted and special-char names are sanitized."""
        from tgirl.bridge import import_mcp_tools

        registry = ToolRegistry()
        mock_tools = [
            _make_mock_tool(
                "my.dotted.tool",
                {"type": "object", "properties": {}},
            ),
        ]

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.tools = mock_tools
        mock_session.list_tools.return_value = mock_list_result
        mock_session.initialize.return_value = MagicMock()

        with patch(
            "tgirl.bridge._create_mcp_session",
            return_value=mock_session,
        ):
            conn = import_mcp_tools(registry, "echo test")
            try:
                assert "my_dotted_tool" in registry
                assert conn.name_map["my_dotted_tool"] == "my.dotted.tool"
            finally:
                conn.close()

    def test_scope_prefix_applied(self) -> None:
        """Scope prefix is set on registered tools."""
        from tgirl.bridge import import_mcp_tools

        registry = ToolRegistry()
        mock_tools = [
            _make_mock_tool(
                "tool_a",
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            ),
        ]

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.tools = mock_tools
        mock_session.list_tools.return_value = mock_list_result
        mock_session.initialize.return_value = MagicMock()

        with patch(
            "tgirl.bridge._create_mcp_session",
            return_value=mock_session,
        ):
            conn = import_mcp_tools(
                registry, "echo test", scope_prefix="mcp"
            )
            try:
                td = registry.get("tool_a")
                assert td.scope == "mcp"
            finally:
                conn.close()

    def test_default_quota_applied(self) -> None:
        """Default quota is set on registered tools."""
        from tgirl.bridge import import_mcp_tools

        registry = ToolRegistry()
        mock_tools = [
            _make_mock_tool(
                "tool_b",
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            ),
        ]

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.tools = mock_tools
        mock_session.list_tools.return_value = mock_list_result
        mock_session.initialize.return_value = MagicMock()

        with patch(
            "tgirl.bridge._create_mcp_session",
            return_value=mock_session,
        ):
            conn = import_mcp_tools(
                registry, "echo test", default_quota=3
            )
            try:
                td = registry.get("tool_b")
                assert td.quota == 3
            finally:
                conn.close()

    def test_sync_wrapper_bridges_to_async(self) -> None:
        """Sync callable wrapper calls session.call_tool on background loop."""
        from tgirl.bridge import import_mcp_tools

        registry = ToolRegistry()
        mock_tools = [
            _make_mock_tool(
                "add",
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            ),
        ]

        call_result = _make_mock_call_result("42")
        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.tools = mock_tools
        mock_session.list_tools.return_value = mock_list_result
        mock_session.call_tool.return_value = call_result
        mock_session.initialize.return_value = MagicMock()

        with patch(
            "tgirl.bridge._create_mcp_session",
            return_value=mock_session,
        ):
            conn = import_mcp_tools(registry, "echo test")
            try:
                wrapper = registry.get_callable("add")
                result = wrapper(a=1, b=2)
                # Result should be the text content from CallToolResult
                assert result == "42"
                mock_session.call_tool.assert_awaited_once_with(
                    "add", {"a": 1, "b": 2}
                )
            finally:
                conn.close()


class TestMcpConnection:
    """Tests for McpConnection lifecycle."""

    def test_close_shuts_down_cleanly(self) -> None:
        """close() stops background thread without leaked threads."""
        from tgirl.bridge import import_mcp_tools

        registry = ToolRegistry()
        mock_tools = [
            _make_mock_tool(
                "noop",
                {"type": "object", "properties": {}},
            ),
        ]

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.tools = mock_tools
        mock_session.list_tools.return_value = mock_list_result
        mock_session.initialize.return_value = MagicMock()

        with patch(
            "tgirl.bridge._create_mcp_session",
            return_value=mock_session,
        ):
            conn = import_mcp_tools(registry, "echo test")
            thread = conn._thread
            assert thread.is_alive()
            conn.close()
            # Give thread time to stop
            thread.join(timeout=2.0)
            assert not thread.is_alive()

    def test_context_manager_protocol(self) -> None:
        """McpConnection works as a context manager."""
        from tgirl.bridge import import_mcp_tools

        registry = ToolRegistry()
        mock_tools = [
            _make_mock_tool(
                "ctx_tool",
                {"type": "object", "properties": {}},
            ),
        ]

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.tools = mock_tools
        mock_session.list_tools.return_value = mock_list_result
        mock_session.initialize.return_value = MagicMock()

        with patch(
            "tgirl.bridge._create_mcp_session",
            return_value=mock_session,
        ):
            with import_mcp_tools(registry, "echo test") as conn:
                assert "ctx_tool" in registry
                thread = conn._thread
                assert thread.is_alive()
            # After exiting context, thread should be stopped
            thread.join(timeout=2.0)
            assert not thread.is_alive()

    def test_call_after_close_raises(self) -> None:
        """Calling a tool after close() raises a clear error."""
        from tgirl.bridge import import_mcp_tools

        registry = ToolRegistry()
        mock_tools = [
            _make_mock_tool(
                "post_close",
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            ),
        ]

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.tools = mock_tools
        mock_session.list_tools.return_value = mock_list_result
        mock_session.initialize.return_value = MagicMock()

        with patch(
            "tgirl.bridge._create_mcp_session",
            return_value=mock_session,
        ):
            conn = import_mcp_tools(registry, "echo test")
            conn.close()
            wrapper = registry.get_callable("post_close")
            with pytest.raises(RuntimeError, match="closed"):
                wrapper(x=1)


class TestTypeReprToSchema:
    """Tests for _type_repr_to_schema reverse mapping."""

    def test_primitive_str(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        assert _type_repr_to_schema(PrimitiveType(kind="str")) == {
            "type": "string"
        }

    def test_primitive_int(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        assert _type_repr_to_schema(PrimitiveType(kind="int")) == {
            "type": "integer"
        }

    def test_primitive_float(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        assert _type_repr_to_schema(PrimitiveType(kind="float")) == {
            "type": "number"
        }

    def test_primitive_bool(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        assert _type_repr_to_schema(PrimitiveType(kind="bool")) == {
            "type": "boolean"
        }

    def test_primitive_none(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        assert _type_repr_to_schema(PrimitiveType(kind="none")) == {
            "type": "null"
        }

    def test_list_type(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            ListType(element=PrimitiveType(kind="int"))
        )
        assert result == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_dict_type(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            DictType(
                key=PrimitiveType(kind="str"),
                value=AnyType(),
            )
        )
        assert result == {"type": "object"}

    def test_literal_type(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            LiteralType(values=("a", "b", "c"))
        )
        assert result == {"enum": ["a", "b", "c"]}

    def test_model_type(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        model = ModelType(
            name="TestModel",
            fields=(
                FieldDef(
                    name="x",
                    type_repr=PrimitiveType(kind="int"),
                    required=True,
                ),
                FieldDef(
                    name="y",
                    type_repr=PrimitiveType(kind="str"),
                    required=False,
                ),
            ),
        )
        result = _type_repr_to_schema(model)
        assert result == {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "string"},
            },
            "required": ["x"],
        }

    def test_any_type(self) -> None:
        from tgirl.bridge import _type_repr_to_schema

        assert _type_repr_to_schema(AnyType()) == {}

    def test_round_trip_basic_types(self) -> None:
        """Round-trip: schema -> TypeRepr -> schema produces equivalent JSON Schema."""
        from tgirl.bridge import _type_repr_to_schema
        from tgirl.registry import _schema_type_to_repr

        cases = [
            {"type": "string"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
            {"type": "array", "items": {"type": "integer"}},
            {"enum": ["a", "b"]},
        ]
        for schema in cases:
            type_repr = _schema_type_to_repr(schema)
            roundtripped = _type_repr_to_schema(type_repr)
            assert roundtripped == schema, (
                f"Round-trip failed for {schema}: got {roundtripped}"
            )


class TestExposeMcp:
    """Tests for expose_as_mcp and create_mcp_server."""

    def test_create_mcp_server_has_tools(self) -> None:
        """create_mcp_server creates a server with all registry tools."""
        from tgirl.bridge import create_mcp_server

        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        @registry.tool(description="Subtract two numbers")
        def sub(a: int, b: int) -> int:
            return a - b

        server = create_mcp_server(registry, name="test")
        # FastMCP stores tools internally; list them
        import asyncio

        async def _list():
            return await server.list_tools()

        tools = asyncio.run(_list())
        tool_names = {t.name for t in tools}
        assert "add" in tool_names
        assert "sub" in tool_names

    def test_create_mcp_server_tool_schemas(self) -> None:
        """Tool schemas are correctly converted from ToolDefinition."""
        from tgirl.bridge import create_mcp_server

        registry = ToolRegistry()

        @registry.tool()
        def greet(name: str, times: int) -> str:
            return name * times

        server = create_mcp_server(registry)
        import asyncio

        async def _list():
            return await server.list_tools()

        tools = asyncio.run(_list())
        tool = next(t for t in tools if t.name == "greet")
        props = tool.inputSchema["properties"]
        # FastMCP may add extra fields (e.g., title); check type is correct
        assert props["name"]["type"] == "string"
        assert props["times"]["type"] == "integer"
        assert set(tool.inputSchema["required"]) == {"name", "times"}

    def test_expose_as_mcp_adds_tool(self) -> None:
        """expose_as_mcp adds a tool to an existing FastMCP server."""
        from mcp.server import FastMCP

        from tgirl.bridge import expose_as_mcp

        registry = ToolRegistry()

        @registry.tool()
        def calc(x: int) -> int:
            return x * 2

        server = FastMCP("test")
        expose_as_mcp(
            registry=registry,
            pipeline_name="calculator",
            description="A calculator pipeline",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
            mcp_server=server,
        )

        import asyncio

        async def _list():
            return await server.list_tools()

        tools = asyncio.run(_list())
        tool_names = {t.name for t in tools}
        assert "calculator" in tool_names

    def test_expose_as_mcp_runs_pipeline_via_session_factory(self) -> None:
        """expose_as_mcp handler uses session_factory to run pipeline."""
        import asyncio

        from mcp.server import FastMCP

        from tgirl.bridge import expose_as_mcp

        registry = ToolRegistry()

        @registry.tool()
        def calc(x: int) -> int:
            return x * 2

        # Mock session with run_chat that returns a result
        mock_result = MagicMock()
        mock_result.text = "Pipeline result: 42"
        mock_session = MagicMock()
        mock_session.run_chat.return_value = mock_result

        session_factory = MagicMock(return_value=mock_session)

        server = FastMCP("test")
        expose_as_mcp(
            registry=registry,
            pipeline_name="calculator",
            description="A calculator pipeline",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
            mcp_server=server,
            session_factory=session_factory,
        )

        # Call the tool through the server
        async def _call():
            return await server.call_tool(
                "calculator", {"expression": "1+1"}
            )

        result = asyncio.run(_call())
        # session_factory should have been called
        session_factory.assert_called_once()
        # run_chat should have been called with a user message containing kwargs
        mock_session.run_chat.assert_called_once()
        call_args = mock_session.run_chat.call_args
        messages = call_args[0][0]
        assert any("expression" in str(m) for m in messages)
        # Result should contain the session's return value
        if isinstance(result, tuple):
            content_list = result[0]
        else:
            content_list = result
        texts = [
            item.text for item in content_list if hasattr(item, "text")
        ]
        assert any("Pipeline result: 42" in t for t in texts)

    def test_expose_as_mcp_without_session_factory_returns_stub(self) -> None:
        """expose_as_mcp without session_factory returns stub message."""
        import asyncio

        from mcp.server import FastMCP

        from tgirl.bridge import expose_as_mcp

        registry = ToolRegistry()

        server = FastMCP("test")
        expose_as_mcp(
            registry=registry,
            pipeline_name="stub_pipe",
            description="A stub pipeline",
            input_schema={
                "type": "object",
                "properties": {},
            },
            mcp_server=server,
        )

        async def _call():
            return await server.call_tool("stub_pipe", {})

        result = asyncio.run(_call())
        # call_tool returns (content_list, metadata) tuple or just content_list
        # Extract text from whichever format
        if isinstance(result, tuple):
            content_list = result[0]
        else:
            content_list = result
        texts = [
            item.text
            for item in content_list
            if hasattr(item, "text")
        ]
        full_text = " ".join(texts).lower()
        assert "no session_factory" in full_text


class TestTypeReprToSchemaGap1:
    """Tests for _type_repr_to_schema with previously-missing TypeRepr variants."""

    def test_enum_type(self) -> None:
        """EnumType produces {"enum": [values]}."""
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            EnumType(name="Color", values=("red", "green", "blue"))
        )
        assert result == {"enum": ["red", "green", "blue"]}

    def test_optional_type(self) -> None:
        """OptionalType produces anyOf with inner type and null."""
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            OptionalType(inner=PrimitiveType(kind="str"))
        )
        assert result == {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        }

    def test_optional_type_nested(self) -> None:
        """OptionalType with a complex inner type."""
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            OptionalType(
                inner=ListType(element=PrimitiveType(kind="int"))
            )
        )
        assert result == {
            "anyOf": [
                {"type": "array", "items": {"type": "integer"}},
                {"type": "null"},
            ],
        }

    def test_union_type(self) -> None:
        """UnionType produces anyOf with all member schemas."""
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            UnionType(
                members=(
                    PrimitiveType(kind="str"),
                    PrimitiveType(kind="int"),
                    PrimitiveType(kind="bool"),
                )
            )
        )
        assert result == {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "boolean"},
            ],
        }

    def test_union_type_single_member(self) -> None:
        """UnionType with one member still uses anyOf."""
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            UnionType(members=(PrimitiveType(kind="float"),))
        )
        assert result == {
            "anyOf": [{"type": "number"}],
        }

    def test_annotated_type_delegates_to_base(self) -> None:
        """AnnotatedType delegates to base type, ignoring constraints."""
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            AnnotatedType(
                base=PrimitiveType(kind="int"),
                constraints=(
                    ConstraintRepr(kind="gt", value=0),
                    ConstraintRepr(kind="lt", value=100),
                ),
            )
        )
        # Constraints are runtime-only; JSON Schema gets just the base type
        assert result == {"type": "integer"}

    def test_annotated_type_with_complex_base(self) -> None:
        """AnnotatedType with a list base type."""
        from tgirl.bridge import _type_repr_to_schema

        result = _type_repr_to_schema(
            AnnotatedType(
                base=ListType(element=PrimitiveType(kind="str")),
                constraints=(ConstraintRepr(kind="ge", value=1),),
            )
        )
        assert result == {
            "type": "array",
            "items": {"type": "string"},
        }
