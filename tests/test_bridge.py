"""Tests for tgirl.bridge — MCP tool import/export."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tgirl.registry import ToolRegistry
from tgirl.types import (
    LiteralType,
    PrimitiveType,
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
