# PRP: IOC Completion — MCP Bridge + Local Inference Server

## Source PRD: docs/PRDs/ioc-completion.md
## Date: 2026-03-14

## 1. Context Summary

The IOC (I/O Controller) block of the VCogPU architecture needs two modules to enable external integration: `bridge.py` for MCP tool import/export, and `serve.py` for a FastAPI local inference server. Both are greenfield implementations with dependencies already declared in `pyproject.toml`. The existing `ToolRegistry.register_from_schema()` handles JSON Schema → tgirl type conversion, so bridge.py can leverage it directly.

## 2. Codebase Analysis

### Existing patterns to reuse

- **`registry.py:register_from_schema()`** — Converts JSON Schema properties to `ParameterDef` with `TypeRepr`. Bridge.py should use this for MCP tool import rather than duplicating type mapping. (`src/tgirl/registry.py:150`)
- **`registry.py:_schema_type_to_repr()`** — Internal function that maps JSON Schema types to `TypeRepr`. Handles string, integer, number, boolean, array, object, enum. (`src/tgirl/registry.py:36`)
- **`SamplingSession`** — The inference entry point. `run_chat(messages)` accepts `list[dict]` and returns `SamplingResult`. (`src/tgirl/sample.py:677`)
- **`SamplingResult`** — Has `output_text`, `tool_calls`, `telemetry`, `total_tokens`, `total_cycles`, `wall_time_ms`, `quotas_consumed`. (`src/tgirl/sample.py:664`)
- **`ToolCallRecord`** — Has `pipeline`, `result`, `error`, `cycle_number`, `tool_invocations`. (`src/tgirl/sample.py:653`)
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

**Test Command:** `pytest tests/test_registry.py tests/test_bridge.py tests/test_serve.py -v`

### Task 0: Extend `register_from_schema` for external tool integration

**Files:** `src/tgirl/registry.py`, `tests/test_registry.py`

**Rationale:** Two gaps block bridge.py:

1. `register_from_schema()` only accepts `name`, `parameters`, `description`, and `return_type`. It hardcodes `quota=None`, `scope=None`, and `callable=lambda **kwargs: None`. Bridge.py needs to set quota, scope, and a real execution callable for each imported MCP tool.

2. `_schema_type_to_repr()` does not handle JSON Schema `"object"` type (it only handles `"dict"`, which is not a JSON Schema type). Objects with `properties` should map to `ModelType` (which exists in `types.py:100`); objects without `properties` should map to `DictType(key=str, value=AnyType)`. `"null"` is also unhandled.

**Approach:**

**Part A: Extend `register_from_schema` signature.**

Add keyword-only parameters:

```python
def register_from_schema(
    self,
    name: str,
    parameters: dict[str, Any],
    description: str = "",
    *,
    return_type: TypeRepr | None = None,
    quota: int | None = None,
    cost: float = 0.0,
    scope: str | None = None,
    timeout: float | None = None,
    callable_fn: Callable[..., Any] | None = None,
) -> None:
```

- New params are keyword-only and default to current behavior (None/0.0/no-op lambda), so existing callers (`bfcl.py:register_bfcl_tools`) are unaffected.
- `callable_fn` replaces the hardcoded no-op lambda when provided.
- The `ToolDefinition` construction passes through quota, cost, scope, timeout.

**Part B: Fix `_schema_type_to_repr` for JSON Schema `"object"` and `"null"`.**

```python
# In _schema_type_to_repr:

if schema_type == "object":
    properties = prop_schema.get("properties")
    if properties:
        # Structured object -> ModelType
        required_names = set(prop_schema.get("required", []))
        fields = tuple(
            FieldDef(
                name=fname,
                type_repr=_schema_type_to_repr(fschema),  # recursive
                required=fname in required_names,
                default=fschema.get("default"),
            )
            for fname, fschema in properties.items()
        )
        # Generate a name from property names for grammar uniqueness
        name_hash = "_".join(sorted(properties.keys()))[:32]
        return ModelType(name=f"Object_{name_hash}", fields=fields)
    else:
        # Unstructured object -> DictType
        return DictType(key=PrimitiveType(kind="str"), value=AnyType())

if schema_type == "null":
    return PrimitiveType(kind="none")
```

Also fix the existing `"dict"` case: JSON Schema never uses `"dict"` — change the check to `"object"` and remove the dead `"dict"` branch (or keep it as a non-standard alias if backward compatibility is needed for internal usage).

**Tests:**
- Existing tests still pass (backward compatibility — no args change)
- Register with quota: verify `snapshot()` includes quota in `quotas` dict
- Register with scope: verify `snapshot(scopes={"x"})` filtering works
- Register with callable_fn: verify `get_callable()` returns the provided callable, not the no-op
- Register without new params: verify behavior identical to current (no-op callable, no quota/scope)
- `_schema_type_to_repr({"type": "object", "properties": {...}})` returns `ModelType` with correct `FieldDef` entries
- Nested objects: `{"type": "object", "properties": {"inner": {"type": "object", "properties": {...}}}}` recurses correctly
- `_schema_type_to_repr({"type": "object"})` (no properties) returns `DictType(key=str, value=AnyType)`
- `_schema_type_to_repr({"type": "null"})` returns `PrimitiveType(kind="none")`

**Validation:** `pytest tests/test_registry.py -v`

### Task 1: bridge.py — MCP type mapping and tool import

**Files:** `src/tgirl/bridge.py` (create), `tests/test_bridge.py` (create)

**Depends on:** Task 0 (extended `register_from_schema`)

#### Session Lifecycle Design

**Problem:** MCP `stdio_client()` is an async context manager — exiting it kills the subprocess. But `import_mcp_tools` must return control to the caller while keeping the connection alive for future tool calls. Additionally, `compile.py:run_pipeline` calls tool callables synchronously via the sandbox (line 799), so MCP's async `session.call_tool()` cannot be called directly from tool wrappers.

**Solution: Background event loop thread.** A dedicated daemon thread runs an asyncio event loop that owns the MCP session lifetime. The sync callable wrappers use `asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)` to bridge sync-to-async. This avoids:
- Deadlocks from nested `asyncio.run()` (which fails if a loop is already running)
- Session death from exiting the context manager
- Conflict with FastAPI's event loop in `serve.py` (separate loop, separate thread)

```python
# bridge.py

class McpConnection:
    """Holds an MCP session on a background event loop thread.

    The connection stays alive until close() is called or the
    McpConnection is used as a context manager and exits.
    """
    _loop: asyncio.AbstractEventLoop
    _thread: threading.Thread
    _session: ClientSession  # lives on _loop
    name_map: dict[str, str]  # sanitized -> original

    def close(self) -> None:
        """Shut down the MCP session and background thread."""

    def __enter__(self) -> McpConnection: ...
    def __exit__(self, *exc: object) -> None: self.close()

def import_mcp_tools(
    registry: ToolRegistry,
    server_params: StdioServerParameters | str,
    scope_prefix: str = "",
    default_quota: int | None = None,
) -> McpConnection:
    """Import tools from an MCP server into a tgirl registry.

    Starts a background thread with an event loop that owns the MCP
    session. Returns an McpConnection that must be kept alive (or used
    as a context manager) for tool calls to work. Calling close()
    or exiting the context manager shuts down the session.

    The function itself is synchronous — it blocks until tools are
    registered, then returns.
    """
    # 1. Start background thread with new event loop
    # 2. On that loop: open stdio_client, create ClientSession, list_tools()
    # 3. For each tool:
    #    a. Sanitize name (same pattern as bfcl.py:register_bfcl_tools)
    #    b. Build sync wrapper:
    #       def wrapper(**kw):
    #           future = asyncio.run_coroutine_threadsafe(
    #               session.call_tool(original_name, kw), loop)
    #           return future.result(timeout)
    #    c. register_from_schema(name, parameters=inputSchema,
    #         description=tool.description, scope=scope_prefix or None,
    #         quota=default_quota, callable_fn=wrapper)
    # 4. Return McpConnection (caller keeps it alive)
```

**Key design decisions:**
- `import_mcp_tools` is **sync**, not async — it blocks until registration completes. This matches the sync nature of `ToolRegistry` and `compile.py`.
- Returns `McpConnection` (not `dict[str, str]`) — caller must manage lifecycle. Name map is available as `McpConnection.name_map`.
- Background thread is a daemon thread — if the process exits, it dies automatically.
- `McpConnection` supports context manager protocol for clean resource management.

**Approach:**

`register_from_schema()` (after Task 0 extension) handles JSON Schema to TypeRepr and accepts quota, scope, and callable. We:
1. Extract `inputSchema` from each MCP `Tool`
2. Sanitize the tool name (same pattern as `bfcl.py:register_bfcl_tools`)
3. Build a sync wrapper that bridges to async `session.call_tool()` via `run_coroutine_threadsafe`
4. Register with `register_from_schema(name, parameters=inputSchema, description=tool.description, scope=scope_prefix, quota=default_quota, callable_fn=wrapper)`

**Tests:**
- Type mapping: mock MCP Tool objects with various inputSchema types, verify register_from_schema produces correct TypeRepr
- Name sanitization: dotted names, special characters
- Scope prefix applied correctly
- Default quota applied
- Sync wrapper correctly bridges to async session.call_tool (use mock session on a test event loop)
- McpConnection.close() shuts down cleanly (no leaked threads)
- McpConnection context manager protocol works
- Calling a tool after close() raises a clear error

**Validation:** `pytest tests/test_bridge.py -v`

### Task 2: bridge.py — MCP tool export (`expose_as_mcp` + `_type_repr_to_schema`)

**Files:** `src/tgirl/bridge.py`, `tests/test_bridge.py`

#### TypeRepr -> JSON Schema reverse mapping

MCP export requires converting `TypeRepr` back to JSON Schema. No reverse mapping exists in the codebase. This task must implement `_type_repr_to_schema(type_repr: TypeRepr) -> dict[str, Any]`:

```python
def _type_repr_to_schema(type_repr: TypeRepr) -> dict[str, Any]:
    """Convert a TypeRepr to a JSON Schema property definition."""
    match type_repr:
        case PrimitiveType(kind="str"):   return {"type": "string"}
        case PrimitiveType(kind="int"):   return {"type": "integer"}
        case PrimitiveType(kind="float"): return {"type": "number"}
        case PrimitiveType(kind="bool"):  return {"type": "boolean"}
        case PrimitiveType(kind="none"):  return {"type": "null"}
        case ListType(element=elem):
            return {"type": "array", "items": _type_repr_to_schema(elem)}
        case DictType():
            return {"type": "object"}
        case LiteralType(values=vals):
            return {"enum": list(vals)}
        case ModelType(fields=fields):
            props = {f.name: _type_repr_to_schema(f.type_repr) for f in fields}
            req = [f.name for f in fields if f.required]
            schema: dict[str, Any] = {"type": "object", "properties": props}
            if req:
                schema["required"] = req
            return schema
        case AnyType():
            return {}
```

#### API: `expose_as_mcp` (per TGIRL.md section 8.2 and PRD acceptance criterion #3)

```python
def expose_as_mcp(
    registry: ToolRegistry,
    pipeline_name: str,
    description: str,
    input_schema: dict[str, Any],
    mcp_server: FastMCP,
) -> None:
    """Wrap a tgirl pipeline as a single MCP tool on an existing server.

    The MCP tool, when called, runs the tgirl sampling engine with the
    pipeline's registered tools and returns the pipeline result. This
    exposes a *composed pipeline* as one MCP tool — not individual tools.

    Args:
        registry: ToolRegistry with the pipeline's tools registered.
        pipeline_name: Name for the MCP tool (e.g., "enrich_and_store").
        description: Human-readable description for the MCP tool.
        input_schema: JSON Schema for the tool's input parameters.
        mcp_server: An existing FastMCP server to add the tool to.
    """
```

#### Optional: `create_mcp_server` (convenience for exposing all tools individually)

```python
def create_mcp_server(
    registry: ToolRegistry,
    name: str = "tgirl",
    description: str = "tgirl grammar-constrained inference",
) -> FastMCP:
    """Create an MCP server exposing each registered tool as an individual MCP tool.

    Uses _type_repr_to_schema for ToolDefinition -> JSON Schema conversion.
    """
```

Both functions are useful but `expose_as_mcp` is the spec-required one.

**Tests:**
- `_type_repr_to_schema` round-trips with `_schema_type_to_repr`: for each JSON Schema type, `_type_repr_to_schema(_schema_type_to_repr(schema))` produces equivalent JSON Schema
- `_type_repr_to_schema` handles all TypeRepr variants (PrimitiveType, ListType, DictType, LiteralType, ModelType, AnyType)
- `expose_as_mcp` adds a tool to an existing FastMCP server with correct name and schema
- `create_mcp_server` creates a server with all registry tools
- Tool schemas correctly converted from ToolDefinition to JSON Schema

**Validation:** `pytest tests/test_bridge.py -v`

### Task 3: serve.py — Model loading and session bootstrap

**Files:** `src/tgirl/serve.py` (create), `tests/test_serve.py` (create)

**Rationale:** `SamplingSession.__init__` (`sample.py:685-701`) requires 6 mandatory arguments beyond the registry: `forward_fn`, `tokenizer_decode`, `tokenizer_encode`, `embeddings`, `grammar_guide_factory`, and `formatter` (required for `run_chat()`, enforced at line 767). These must be correctly constructed from a model ID string for both torch (HuggingFace) and MLX backends. This is the hardest engineering in serve.py and must be specified explicitly.

**Approach:**

```python
# serve.py — model bootstrap

@dataclass(frozen=True)
class InferenceContext:
    """Everything needed to create SamplingSession instances."""
    registry: ToolRegistry
    forward_fn: Callable[[list[int]], Any]
    tokenizer_decode: Callable[[list[int]], str]
    tokenizer_encode: Callable[[str], list[int]]
    embeddings: Any  # torch.Tensor or mx.array
    grammar_guide_factory: Callable[[str], GrammarState]
    mlx_grammar_guide_factory: Callable | None
    formatter: PromptFormatter
    backend: Literal["torch", "mlx"]
    model_id: str
    stop_token_ids: list[int]

def load_inference_context(
    model_id: str,
    backend: str = "auto",
) -> InferenceContext:
    """Load model and construct all SamplingSession dependencies.

    Backend detection:
    - "auto": try MLX first (Apple Silicon), fall back to torch
    - "mlx": require mlx-lm, fail if unavailable
    - "torch": require transformers, fail if unavailable

    MLX path:
    - mlx_lm.load(model_id) -> (model, tokenizer)
    - cache.make_mlx_forward_fn(model) -> forward_fn
    - model.model.embed_tokens.as_mx() -> embeddings
    - outlines_adapter.make_outlines_grammar_factory_mlx(tokenizer) -> grammar_guide_factory
    - tokenizer.eos_token_id -> stop_token_ids

    Torch/HF path:
    - AutoModelForCausalLM.from_pretrained(model_id) -> model
    - AutoTokenizer.from_pretrained(model_id) -> tokenizer
    - cache.make_hf_forward_fn(model) -> forward_fn
    - model.get_input_embeddings().weight -> embeddings
    - outlines_adapter.make_outlines_grammar_factory(tokenizer) -> grammar_guide_factory
    - tokenizer.eos_token_id -> stop_token_ids

    Both paths:
    - tokenizer.decode -> tokenizer_decode
    - tokenizer.encode -> tokenizer_encode
    - format.ChatTemplateFormatter(tokenizer) -> formatter
    """
```

**Tests:**
- `load_inference_context` with mocked MLX imports returns correct InferenceContext
- `load_inference_context` with mocked HF imports returns correct InferenceContext
- Backend "auto" prefers MLX when available
- Backend "mlx" fails with clear error when mlx unavailable
- Backend "torch" fails with clear error when transformers unavailable
- `InferenceContext` fields match `SamplingSession.__init__` parameter requirements

**Validation:** `pytest tests/test_serve.py -v`

### Task 4: serve.py — Core server with /generate and /health

**Files:** `src/tgirl/serve.py`, `tests/test_serve.py`

**Depends on:** Task 3 (`InferenceContext`)

**Approach:**

```python
# Pydantic request/response models
class GenerateRequest(BaseModel):
    intent: str
    scopes: list[str] | None = None
    max_cost: float | None = None
    restrict_tools: list[str] | None = None
    ot_epsilon: float | None = None       # per TGIRL.md 9.3
    base_temperature: float | None = None  # per TGIRL.md 9.3

class GenerateResponse(BaseModel):
    output: str
    tool_calls: list[ToolCallResponse]
    total_tokens: int
    total_cycles: int
    wall_time_ms: float
    quotas_consumed: dict[str, int]
    error: str | None = None

class ToolCallResponse(BaseModel):
    pipeline: str
    result: Any | None
    error: str | None
    cycle_number: int
    tool_invocations: dict[str, int]

# Server factory
def create_app(
    ctx: InferenceContext,
    session_config: SessionConfig | None = None,
    transport_config: TransportConfig | None = None,
    hooks: list[InferenceHook] | None = None,
) -> FastAPI:
    """Create FastAPI app from a pre-loaded InferenceContext.

    Each /generate request creates a new SamplingSession from
    the shared InferenceContext. Model loading happens before
    this function is called (in load_inference_context or CLI).
    """
    app = FastAPI(title="tgirl")

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> GenerateResponse:
        session = SamplingSession(
            registry=ctx.registry,
            forward_fn=ctx.forward_fn,
            tokenizer_decode=ctx.tokenizer_decode,
            tokenizer_encode=ctx.tokenizer_encode,
            embeddings=ctx.embeddings,
            grammar_guide_factory=ctx.grammar_guide_factory,
            mlx_grammar_guide_factory=ctx.mlx_grammar_guide_factory,
            formatter=ctx.formatter,
            backend=ctx.backend,
            config=session_config,
            transport_config=transport_config,
            hooks=hooks,
            stop_token_ids=ctx.stop_token_ids,
        )
        result = await asyncio.to_thread(
            session.run_chat,
            [{"role": "user", "content": request.intent}],
        )
        return GenerateResponse(...)

    @app.get("/health")
    async def health() -> dict:
        return {"model": ctx.model_id, "tools": len(ctx.registry.names()), ...}
```

**Key design decisions:**
- `create_app` takes `InferenceContext`, not raw model_id — separation of concerns. Model loading is done once at startup, not inside the factory.
- `run_chat` is sync and blocking — wrap in `asyncio.to_thread()` to avoid blocking the event loop.
- No `**session_kwargs` — all SamplingSession parameters are explicit.

Use FastAPI's TestClient for testing (no real model needed — mock InferenceContext).

**Tests:**
- `/generate` returns correct response structure with all fields
- `/health` returns model info and tool count
- Request validation (missing intent, invalid scopes)
- Error handling (model failure, timeout)
- `run_chat` is called in a thread (not blocking event loop)

**Validation:** `pytest tests/test_serve.py -v`

### Task 5: serve.py — /tools, /grammar, /grammar/preview, /telemetry

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

### Task 6: serve.py — WebSocket streaming

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

### Task 7: serve.py — Hot reload and CLI

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
@click.option("--backend", default="auto")
@click.option("--hot-reload", is_flag=True)
def serve(model, tools, port, backend, hot_reload):
    ctx = load_inference_context(model, backend=backend)
    # Register tools from tool_modules into ctx.registry
    app = create_app(ctx, hot_reload=hot_reload)
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
pytest tests/test_registry.py tests/test_bridge.py tests/test_serve.py -v

# Full test suite (no regressions)
# Note: test_cache, test_transport, test_transport_mlx require hardware-specific
# deps (torch GPU tensors, MLX on Apple Silicon). These are excluded from the
# default gate but must pass on their respective hardware before merge.
# test_compile is included — it has no hardware deps.
pytest tests/ -v --ignore=tests/test_cache.py \
    --ignore=tests/test_transport.py --ignore=tests/test_transport_mlx.py

# Integration: start server, hit endpoints
python -m tgirl.cli serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit --tools examples/showcase_unified_api.py &
curl http://localhost:8420/health
curl http://localhost:8420/tools
curl -X POST http://localhost:8420/generate -d '{"intent": "Add 2 and 3"}'
```

## 5. Rollback Plan

Task 0 modifies `registry.py` (backward-compatible extension — new keyword-only params with defaults). Tasks 1-6 are new files. Rollback: revert `register_from_schema` changes in `registry.py`, delete `bridge.py`, `serve.py`, `cli.py` and their test files. Remove from `__init__.py` exports.

## 6. Uncertainty Log

- **MCP client lifecycle**: ~~Need to investigate connection pattern.~~ **RESOLVED**: `McpConnection` class owns a background daemon thread with its own event loop. The MCP session lives on that loop inside a never-exiting async task. Sync callable wrappers use `asyncio.run_coroutine_threadsafe()` to bridge sync→async. Caller manages lifecycle via `McpConnection.close()` or context manager protocol. See Task 1 for full design.
- **FastMCP tool registration**: The `FastMCP` class uses decorators (`@server.tool()`). Need to verify if programmatic tool registration is supported (adding tools after construction).
- **WebSocket streaming integration**: The sampling loop is synchronous. Streaming to WebSocket requires either running inference in a thread pool or adding async yield points to the sampling loop. Thread pool executor is the safer approach for v1.
- **Hot reload dependency**: `watchfiles` vs `watchdog` — need to pick one and add to optional deps. `watchfiles` is lighter but less feature-rich.
