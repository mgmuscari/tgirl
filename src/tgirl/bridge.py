"""MCP compatibility layer — tool import/export.

Provides bidirectional bridging between tgirl's ToolRegistry and the
Model Context Protocol (MCP):

- **Import**: ``import_mcp_tools`` connects to an MCP server and
  registers its tools into a ToolRegistry.
- **Export**: ``expose_as_mcp`` / ``create_mcp_server`` expose
  registry tools as MCP server tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
import warnings
from collections.abc import Callable
from typing import Any

import structlog

from tgirl.registry import ToolRegistry

logger = structlog.get_logger()

try:
    from mcp import StdioServerParameters  # noqa: F401 (used in type checks)

    _HAS_MCP_CLIENT = True
except ImportError:
    _HAS_MCP_CLIENT = False

try:
    from mcp.server import FastMCP  # noqa: F401 (used in expose_as_mcp)

    _HAS_MCP_SERVER = True
except ImportError:
    _HAS_MCP_SERVER = False


def _sanitize_tool_name(name: str) -> str:
    """Sanitize an MCP tool name for use as a Python identifier.

    Replaces dots, hyphens, and other non-alphanumeric characters
    with underscores (same pattern as bfcl.py:register_bfcl_tools).
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _extract_call_result_text(result: Any) -> str | None:
    """Extract text content from an MCP CallToolResult."""
    if result.isError:
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        error_msg = "; ".join(texts) if texts else "MCP tool call failed"
        msg = f"MCP tool error: {error_msg}"
        raise RuntimeError(msg)

    for item in result.content:
        if hasattr(item, "text"):
            return item.text
    return None


class McpConnection:
    """Holds an MCP session on a background event loop thread.

    The connection stays alive until close() is called or the
    McpConnection is used as a context manager and exits.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
        session: Any,
        name_map: dict[str, str],
        transport_cm: Any = None,
    ) -> None:
        self._loop = loop
        self._thread = thread
        self._session = session
        self.name_map = name_map
        self._closed = False
        self._transport_cm = transport_cm

    def close(self) -> None:
        """Shut down the MCP session and background thread."""
        if self._closed:
            return
        self._closed = True
        # Clean up transport context manager if present
        if self._transport_cm is not None:

            async def _cleanup() -> None:
                with contextlib.suppress(Exception):
                    await self._transport_cm.__aexit__(None, None, None)

            try:
                future = asyncio.run_coroutine_threadsafe(
                    _cleanup(), self._loop
                )
                future.result(timeout=5.0)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    @property
    def closed(self) -> bool:
        return self._closed

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                "McpConnection was not closed",
                ResourceWarning,
                stacklevel=1,
            )
            self.close()

    def __enter__(self) -> McpConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


async def _create_mcp_session(
    server_params: Any,
) -> tuple[Any, Any]:
    """Create and initialize an MCP ClientSession.

    Opens the stdio_client transport and creates a ClientSession on it.
    Returns (session, transport_cm) — the caller must keep transport_cm
    alive and call its __aexit__ on cleanup.

    This is extracted as a separate function to allow mocking
    in tests without mocking the entire stdio_client machinery.
    """
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    transport_cm = stdio_client(server_params)
    read, write = await transport_cm.__aenter__()
    session = ClientSession(read, write)
    await session.initialize()
    return session, transport_cm


def _make_sync_wrapper(
    connection: McpConnection,
    original_name: str,
    timeout: float = 30.0,
) -> Any:
    """Build a sync callable that bridges to async session.call_tool."""

    def wrapper(**kwargs: Any) -> Any:
        if connection.closed:
            msg = "MCP connection is closed"
            raise RuntimeError(msg)
        coro = connection._session.call_tool(original_name, kwargs)
        future = asyncio.run_coroutine_threadsafe(
            coro, connection._loop
        )
        result = future.result(timeout=timeout)
        return _extract_call_result_text(result)

    return wrapper


def import_mcp_tools(
    registry: ToolRegistry,
    server_params: Any,
    scope_prefix: str = "",
    default_quota: int | None = None,
) -> McpConnection:
    """Import tools from an MCP server into a tgirl registry.

    Starts a background thread with an event loop that owns the MCP
    session. Returns an McpConnection that must be kept alive (or used
    as a context manager) for tool calls to work.

    The function itself is synchronous -- it blocks until tools are
    registered, then returns.

    Args:
        registry: ToolRegistry to register imported tools into.
        server_params: StdioServerParameters or a command string.
        scope_prefix: If set, all imported tools get this scope.
        default_quota: If set, all imported tools get this quota.

    Returns:
        McpConnection that must be kept alive for tool calls.
    """
    if not _HAS_MCP_CLIENT:
        msg = "mcp package is required for import_mcp_tools"
        raise ImportError(msg)

    # Parse command string to StdioServerParameters if needed
    if isinstance(server_params, str):
        parts = server_params.split()
        server_params = StdioServerParameters(
            command=parts[0], args=parts[1:] if len(parts) > 1 else []
        )

    # Set up background event loop
    loop = asyncio.new_event_loop()
    ready_event = threading.Event()
    session_holder: list[Any] = [None]
    transport_holder: list[Any] = [None]
    error_holder: list[Exception | None] = [None]

    async def _run_session() -> None:
        try:
            result = await _create_mcp_session(server_params)
            # _create_mcp_session returns (session, transport_cm) in
            # production, or just a mock session in tests
            if isinstance(result, tuple):
                session_holder[0], transport_holder[0] = result
            else:
                session_holder[0] = result
            ready_event.set()
            # Keep the loop running until stopped
            stop_event = asyncio.Event()
            await stop_event.wait()
        except Exception as e:
            error_holder[0] = e
            ready_event.set()

    def _thread_target() -> None:
        asyncio.set_event_loop(loop)
        with contextlib.suppress(RuntimeError):
            loop.run_until_complete(_run_session())

    thread = threading.Thread(
        target=_thread_target, daemon=True, name="mcp-bridge"
    )
    thread.start()

    # Wait for session to be ready
    ready_event.wait(timeout=30.0)

    if error_holder[0] is not None:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        raise error_holder[0]

    session = session_holder[0]
    if session is None:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        msg = "Failed to create MCP session"
        raise RuntimeError(msg)

    # List tools from the MCP server
    future = asyncio.run_coroutine_threadsafe(
        session.list_tools(), loop
    )
    list_result = future.result(timeout=30.0)

    name_map: dict[str, str] = {}
    conn = McpConnection(
        loop=loop,
        thread=thread,
        session=session,
        name_map=name_map,
        transport_cm=transport_holder[0],
    )

    for tool in list_result.tools:
        original_name = tool.name
        sanitized_name = _sanitize_tool_name(original_name)
        name_map[sanitized_name] = original_name

        wrapper = _make_sync_wrapper(conn, original_name)

        registry.register_from_schema(
            name=sanitized_name,
            parameters=tool.inputSchema,
            description=tool.description or "",
            scope=scope_prefix or None,
            quota=default_quota,
            callable_fn=wrapper,
        )

        logger.debug(
            "mcp_tool_imported",
            original=original_name,
            sanitized=sanitized_name,
            scope=scope_prefix or None,
            quota=default_quota,
        )

    logger.info(
        "mcp_tools_imported",
        count=len(list_result.tools),
        server=str(server_params),
    )

    return conn


# --- Export: TypeRepr -> JSON Schema ---


def _type_repr_to_schema(type_repr: Any) -> dict[str, Any]:
    """Convert a TypeRepr to a JSON Schema property definition."""
    from tgirl.types import (
        AnnotatedType,
        AnyType,
        DictType,
        EnumType,
        ListType,
        LiteralType,
        ModelType,
        OptionalType,
        PrimitiveType,
        UnionType,
    )

    primitive_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "none": "null",
    }

    match type_repr:
        case PrimitiveType(kind=kind):
            return {"type": primitive_map[kind]}
        case ListType(element=elem):
            return {
                "type": "array",
                "items": _type_repr_to_schema(elem),
            }
        case DictType():
            return {"type": "object"}
        case LiteralType(values=vals):
            return {"enum": list(vals)}
        case ModelType(fields=fields):
            props = {
                f.name: _type_repr_to_schema(f.type_repr)
                for f in fields
            }
            req = [f.name for f in fields if f.required]
            schema: dict[str, Any] = {
                "type": "object",
                "properties": props,
            }
            if req:
                schema["required"] = req
            return schema
        case EnumType(values=vals):
            return {"enum": list(vals)}
        case OptionalType(inner=inner):
            return {
                "anyOf": [_type_repr_to_schema(inner), {"type": "null"}],
            }
        case UnionType(members=members):
            return {
                "anyOf": [_type_repr_to_schema(m) for m in members],
            }
        case AnnotatedType(base=base):
            return _type_repr_to_schema(base)
        case AnyType():
            return {}
        case _:
            return {}


def _type_repr_to_python_type(type_repr: Any) -> type:
    """Map TypeRepr to a Python type for function annotations."""
    from tgirl.types import (
        ListType,
        LiteralType,
        PrimitiveType,
    )

    type_map: dict[str, type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "none": type(None),
    }

    match type_repr:
        case PrimitiveType(kind=kind):
            return type_map.get(kind, str)
        case ListType():
            return list
        case LiteralType():
            return str
        case _:
            return str


def _build_typed_handler(tool_def: Any, callable_fn: Any) -> Any:
    """Build an async handler with correct type annotations for FastMCP.

    FastMCP infers JSON Schema from the handler's type annotations,
    so we must create a function with properly typed parameters.
    """
    import inspect

    params = []
    annotations: dict[str, type] = {}

    for p in tool_def.parameters:
        python_type = _type_repr_to_python_type(p.type_repr)
        if p.has_default:
            params.append(
                inspect.Parameter(
                    p.name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=p.default,
                    annotation=python_type,
                )
            )
        else:
            params.append(
                inspect.Parameter(
                    p.name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=python_type,
                )
            )
        annotations[p.name] = python_type

    annotations["return"] = str
    sig = inspect.Signature(params, return_annotation=str)

    fn = callable_fn

    async def handler(**kwargs: Any) -> Any:
        return fn(**kwargs)

    handler.__signature__ = sig  # type: ignore[attr-defined]
    handler.__annotations__ = annotations

    return handler


# --- Export: MCP server creation ---


def expose_as_mcp(
    registry: ToolRegistry,
    pipeline_name: str,
    description: str,
    input_schema: dict[str, Any],
    mcp_server: Any,
    session_factory: Callable[[], Any] | None = None,
) -> None:
    """Wrap a tgirl pipeline as a single MCP tool on an existing server.

    The MCP tool, when called, runs the tgirl sampling engine with the
    pipeline's registered tools and returns the pipeline result. This
    exposes a *composed pipeline* as one MCP tool -- not individual tools.

    Args:
        registry: ToolRegistry with the pipeline's tools registered.
        pipeline_name: Name for the MCP tool.
        description: Human-readable description for the MCP tool.
        input_schema: JSON Schema for the tool's input parameters.
        mcp_server: An existing FastMCP server to add the tool to.
        session_factory: Callable that returns a SamplingSession.
            When provided, the handler creates a session and runs
            run_chat() with the tool call kwargs as a user message.
            When None, the handler returns an informative stub message.
    """
    if not _HAS_MCP_SERVER:
        msg = "mcp.server is required for expose_as_mcp"
        raise ImportError(msg)

    import inspect

    # Build typed parameters from input_schema for FastMCP compatibility
    params = []
    annotations: dict[str, type] = {}
    schema_props = input_schema.get("properties", {})
    schema_required = set(input_schema.get("required", []))

    json_type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    for prop_name, prop_schema in schema_props.items():
        python_type = json_type_map.get(prop_schema.get("type", ""), str)
        if prop_name in schema_required:
            params.append(
                inspect.Parameter(
                    prop_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=python_type,
                )
            )
        else:
            params.append(
                inspect.Parameter(
                    prop_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=python_type,
                )
            )
        annotations[prop_name] = python_type

    annotations["return"] = str
    sig = inspect.Signature(params, return_annotation=str)

    # Capture factory in closure for the handler
    factory = session_factory
    name = pipeline_name

    async def pipeline_handler(**kwargs: Any) -> str:
        if factory is None:
            return (
                f"No session_factory configured for pipeline "
                f"'{name}'. Cannot run inference without "
                f"a session factory."
            )
        session = factory()
        user_content = (
            f"Call the '{name}' pipeline with arguments: {kwargs}"
        )
        messages = [{"role": "user", "content": user_content}]
        result = session.run_chat(messages)
        return result.text

    pipeline_handler.__signature__ = sig  # type: ignore[attr-defined]
    pipeline_handler.__annotations__ = annotations

    # Use add_tool for programmatic registration
    mcp_server.add_tool(
        pipeline_handler,
        name=pipeline_name,
        description=description,
    )

    logger.info(
        "mcp_pipeline_exposed",
        name=pipeline_name,
        description=description,
        has_session_factory=session_factory is not None,
    )


def create_mcp_server(
    registry: ToolRegistry,
    name: str = "tgirl",
    description: str = "tgirl grammar-constrained inference",
) -> Any:
    """Create an MCP server exposing each registered tool individually.

    Uses _type_repr_to_schema for ToolDefinition -> JSON Schema conversion.

    Args:
        registry: ToolRegistry with tools to expose.
        name: Server name.
        description: Server description.

    Returns:
        A FastMCP server instance.
    """
    if not _HAS_MCP_SERVER:
        msg = "mcp.server is required for create_mcp_server"
        raise ImportError(msg)

    server = FastMCP(name)

    snapshot = registry.snapshot()
    for tool_def in snapshot.tools:
        callable_fn = registry.get_callable(tool_def.name)

        handler = _build_typed_handler(tool_def, callable_fn)

        server.add_tool(
            handler,
            name=tool_def.name,
            description=tool_def.description or None,
        )

        logger.debug(
            "mcp_tool_exported",
            name=tool_def.name,
            params=len(tool_def.parameters),
        )

    logger.info(
        "mcp_server_created",
        name=name,
        tools=len(snapshot.tools),
    )

    return server
