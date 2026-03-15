# PRP: IOC Completion — MCP Bridge + Local Inference Server

## Source PRD: docs/PRDs/ioc-completion.md
## Date: 2026-03-14

## 1. Context Summary

The IOC (I/O Controller) block of the VCogPU architecture needs two modules to enable external integration: `bridge.py` for MCP tool import/export, and `serve.py` for a FastAPI local inference server. Both are greenfield implementations with dependencies already declared in `pyproject.toml`. The existing `ToolRegistry.register_from_schema()` handles JSON Schema → tgirl type conversion, so bridge.py can leverage it directly.

## 2. Codebase Analysis

### Existing patterns to reuse

- **`registry.py:register_from_schema()`** — Converts JSON Schema properties to `ParameterDef` with `TypeRepr`. Bridge.py should use this for MCP tool import rather than duplicating type mapping. (`src/tgirl/registry.py:150`)
- **`registry.py:_schema_type_to_repr()`** — Internal function that maps JSON Schema types to `TypeRepr`. Handles string, integer, number, boolean, array, object, enum. (`src/tgirl/registry.py:115`)
- **`SamplingSession`** — The inference entry point. `run_chat(messages)` accepts `list[dict]` and returns `SamplingResult`. (`src/tgirl/sample.py:531`)
- **`SamplingResult`** — Has `output_text`, `tool_calls`, `total_tokens`, `wall_time_ms`, `telemetry`. (`src/tgirl/sample.py:519`)
- **`ToolCallRecord`** — Has `pipeline`, `result`, `error`. (`src/tgirl/sample.py:510`)
- **`ChatTemplateFormatter`** — Formats messages for model input. (`src/tgirl/format.py`)
- **`generate` in `grammar.py`** — Returns `GrammarOutput` with `.text` for grammar preview endpoint.
- **`bfcl.py:register_bfcl_tools()`** — Example of registering tools from external schema, with name sanitization and name mapping. Useful reference pattern.

### MCP package API (v1.x)

- **Client**: `mcp.ClientSession` — async, requires anyio memory streams. `list_tools()` → `ListToolsResult`, `call_tool(name, arguments)` → `CallToolResult`
- **Server**: `mcp.server.FastMCP` — decorator-based tool registration
- **Transport**: `mcp.stdio_client()` for subprocess-based MCP servers, SSE/HTTP for network servers
- **Tool schema**: `mcp.Tool` has `name`, `description`, `inputSchema` (JSON Schema dict)

### Conventions

- All models use `ConfigDict(frozen=True)` (Pydantic)
- Structlog for logging
- Optional dependencies: modules must be importable without their optional deps installed (use try/except or lazy imports)
- Both torch and MLX backends: serve.py must detect available backend

## 3. Implementation Plan

**Test Command:** `pytest tests/test_bridge.py tests/test_serve.py -v`

### Task 1: bridge.py — MCP type mapping and tool import

**Files:** `src/tgirl/bridge.py` (create), `tests/test_bridge.py` (create)

**Approach:**

```python
# bridge.py

async def import_mcp_tools(
    registry: ToolRegistry,
    server_params: StdioServerParameters | str,
    scope_prefix: str = "",
    default_quota: int | None = None,
) -> dict[str, str]:
    """Import tools from an MCP server into a tgirl registry.

    Returns name_map: {sanitized_name: original_name}
    """
    # 1. Connect to MCP server via stdio_client or SSE
    # 2. session.list_tools() → ListToolsResult
    # 3. For each tool: sanitize name, register_from_schema()
    # 4. Wrap execution: registry registers a callable that calls
    #    session.call_tool(name, arguments) and returns the result
```

The key insight: `register_from_schema()` already handles JSON Schema → TypeRepr. We just need to:
1. Extract `inputSchema` from each MCP `Tool`
2. Sanitize the tool name (same pattern as `bfcl.py:register_bfcl_tools`)
3. Register with `register_from_schema(name, parameters=inputSchema, description=tool.description)`
4. Wrap execution as an async-to-sync callable that invokes `session.call_tool()`

**Tests:**
- Type mapping: mock MCP Tool objects with various inputSchema types, verify register_from_schema produces correct TypeRepr
- Name sanitization: dotted names, special characters
- Scope prefix applied correctly
- Default quota applied
- Execution wrapper calls session.call_tool with correct arguments

**Validation:** `pytest tests/test_bridge.py -v`

### Task 2: bridge.py — MCP tool export

**Files:** `src/tgirl/bridge.py`, `tests/test_bridge.py`

**Approach:**

```python
def create_mcp_server(
    registry: ToolRegistry,
    name: str = "tgirl",
    description: str = "tgirl grammar-constrained inference",
) -> FastMCP:
    """Create an MCP server exposing tgirl tools.

    Each registered tool becomes an MCP tool. The MCP server
    can be run via stdio or SSE transport.
    """
    server = FastMCP(name, description=description)

    for tool_name in registry.names():
        tool_def = registry.get(tool_name)
        callable_fn = registry.get_callable(tool_name)
        # Register as MCP tool with JSON Schema from tool_def
        # FastMCP's @server.tool() decorator or server.add_tool()
```

**Tests:**
- Server creation with multiple tools
- Tool schemas correctly converted from ToolDefinition to JSON Schema
- Tool invocation through MCP protocol (mock client)

**Validation:** `pytest tests/test_bridge.py -v`

### Task 3: serve.py — Core server with /generate and /health

**Files:** `src/tgirl/serve.py` (create), `tests/test_serve.py` (create)

**Approach:**

```python
# Pydantic request/response models
class GenerateRequest(BaseModel):
    intent: str
    scopes: list[str] | None = None
    max_cost: float | None = None
    restrict_tools: list[str] | None = None

class GenerateResponse(BaseModel):
    output: str
    tool_calls: list[ToolCallResponse]
    error: str | None = None

class ToolCallResponse(BaseModel):
    pipeline: str
    result: Any | None
    error: str | None

# Server factory
def create_app(
    model_id: str,
    tool_modules: list[str],
    backend: str = "auto",
    **session_kwargs,
) -> FastAPI:
    app = FastAPI(title="tgirl")
    # Load model, create registry, session factory

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> GenerateResponse:
        session = make_session(scopes=request.scopes, max_cost=request.max_cost)
        result = session.run_chat([{"role": "user", "content": request.intent}])
        return GenerateResponse(...)

    @app.get("/health")
    async def health() -> dict:
        return {"model": model_id, "tools": len(registry.names()), ...}
```

Use FastAPI's TestClient for testing (no real model needed — mock forward_fn).

**Tests:**
- `/generate` returns correct response structure
- `/health` returns model info and tool count
- Request validation (missing intent, invalid scopes)
- Error handling (model failure, timeout)

**Validation:** `pytest tests/test_serve.py -v`

### Task 4: serve.py — /tools, /grammar, /grammar/preview, /telemetry

**Files:** `src/tgirl/serve.py`, `tests/test_serve.py`

**Approach:**

```python
@app.get("/tools")
async def list_tools() -> list[dict]:
    snapshot = registry.snapshot()
    return [tool_to_dict(t) for t in snapshot.tools]

@app.get("/grammar")
async def get_grammar() -> dict:
    snapshot = registry.snapshot()
    output = generate(snapshot)
    return {"text": output.text, "hash": output.snapshot_hash}

@app.post("/grammar/preview")
async def preview_grammar(request: GrammarPreviewRequest) -> dict:
    # Filter snapshot by requested scopes/tools
    # Generate grammar from filtered snapshot

@app.get("/telemetry")
async def get_telemetry(limit: int = 100) -> list[dict]:
    # Return recent telemetry records from in-memory buffer
```

**Tests:**
- `/tools` lists all registered tools with correct schema
- `/grammar` returns valid Lark EBNF
- `/grammar/preview` with tool restriction produces filtered grammar
- `/telemetry` returns recent records, respects limit

**Validation:** `pytest tests/test_serve.py -v`

### Task 5: serve.py — WebSocket streaming

**Files:** `src/tgirl/serve.py`, `tests/test_serve.py`

**Approach:**

```python
@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()
    # Create session with streaming hooks
    # Hook emits per-token events to websocket
    # Final message includes complete result
```

This requires a streaming-aware inference hook that sends per-token data to the WebSocket as tokens are generated.

**Tests:**
- WebSocket connection and message exchange
- Per-token events have correct structure
- Final message includes complete result
- Client disconnect handling

**Validation:** `pytest tests/test_serve.py -v`

### Task 6: serve.py — Hot reload and CLI

**Files:** `src/tgirl/serve.py`, `src/tgirl/cli.py` (create or extend)

**Approach:**

Hot reload: use `watchfiles` (or `watchdog`) to monitor tool module files. On change, rebuild registry and regenerate grammar atomically.

CLI:
```python
# cli.py
@click.command()
@click.option("--model", required=True)
@click.option("--tools", required=True, multiple=True)
@click.option("--port", default=8420)
@click.option("--hot-reload", is_flag=True)
def serve(model, tools, port, hot_reload):
    app = create_app(model, tools, hot_reload=hot_reload)
    uvicorn.run(app, host="0.0.0.0", port=port)
```

**Tests:**
- Hot reload: modify tool file, verify registry rebuilds
- CLI argument parsing
- Default port/options

**Validation:** `pytest tests/test_serve.py -v && python -m tgirl.cli serve --help`

## 4. Validation Gates

```bash
# Lint
ruff check src/tgirl/bridge.py src/tgirl/serve.py src/tgirl/cli.py

# Unit tests
pytest tests/test_bridge.py tests/test_serve.py -v

# Full test suite (no regressions)
pytest tests/ -v --ignore=tests/test_cache.py --ignore=tests/test_compile.py \
    --ignore=tests/test_transport.py --ignore=tests/test_transport_mlx.py

# Integration: start server, hit endpoints
python -m tgirl.cli serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit --tools examples/showcase_unified_api.py &
curl http://localhost:8420/health
curl http://localhost:8420/tools
curl -X POST http://localhost:8420/generate -d '{"intent": "Add 2 and 3"}'
```

## 5. Rollback Plan

Both modules are new files with no changes to existing code. Rollback: delete `bridge.py`, `serve.py`, `cli.py` and their test files. Remove from `__init__.py` exports.

## 6. Uncertainty Log

- **MCP client lifecycle**: The `mcp.ClientSession` requires anyio streams as constructor args. Need to investigate `mcp.stdio_client()` context manager for the actual connection pattern. May need to hold the session open for the lifetime of the registry.
- **FastMCP tool registration**: The `FastMCP` class uses decorators (`@server.tool()`). Need to verify if programmatic tool registration is supported (adding tools after construction).
- **WebSocket streaming integration**: The sampling loop is synchronous. Streaming to WebSocket requires either running inference in a thread pool or adding async yield points to the sampling loop. Thread pool executor is the safer approach for v1.
- **Hot reload dependency**: `watchfiles` vs `watchdog` — need to pick one and add to optional deps. `watchfiles` is lighter but less feature-rich.
