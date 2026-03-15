# PRD: IOC Completion — MCP Bridge + Local Inference Server

## Status: DRAFT
## Author: Claude (Proposer stance)
## Date: 2026-03-14
## Branch: feature/ioc-completion

## 1. Problem Statement

The tgirl CPU architecture has a functional core — CLU, RSF, GSU, SCU, STB, CSG all operational — but the IOC (I/O Controller) is incomplete. The system can only be exercised programmatically via Python imports. This limits:

- **Integration**: Other tools and systems (Claude Desktop, agent frameworks, MCP ecosystems) cannot call tgirl's inference engine
- **Testing**: Benchmarking and demos require writing Python scripts; no HTTP API for external tooling
- **Adoption**: Users must understand tgirl's Python API to use it; no standalone inference endpoint

The IOC block needs two modules specified in TGIRL.md sections 9-10:

1. **bridge.py** — MCP compatibility layer (import external tools, export tgirl pipelines)
2. **serve.py** — FastAPI local inference server wrapping SamplingSession

## 2. Proposed Solution

### bridge.py — MCP Compatibility Layer

Two-directional bridge between tgirl's tool registry and the MCP protocol:

- **Import**: Connect to an MCP server, enumerate its tools, convert JSON Schema definitions to tgirl `ParameterDef`/`ToolDefinition`, register them with scope/quota, wrap execution to serialize/deserialize through MCP
- **Export**: Wrap a tgirl pipeline as a single MCP tool that external systems can call

### serve.py — FastAPI Local Inference Server

HTTP/WebSocket API wrapping `SamplingSession`:

- `/generate` — POST: natural language → tool pipeline execution → result
- `/tools` — GET: list registered tools with types and quotas
- `/grammar` — GET: current generated grammar
- `/grammar/preview` — POST: preview grammar for a scope restriction
- `/telemetry` — GET: recent telemetry records
- `/health` — GET: server health, model info, uptime
- `/stream` — WebSocket: token-by-token streaming with per-token telemetry

CLI entrypoint: `tgirl serve --model X --tools Y --port N`

## 3. Architecture Impact

### New files
- `src/tgirl/bridge.py` — MCP import/export
- `src/tgirl/serve.py` — FastAPI server
- `src/tgirl/cli.py` — CLI entrypoint (if not already present)
- `tests/test_bridge.py` — bridge tests
- `tests/test_serve.py` — serve tests

### Existing files affected
- `src/tgirl/__init__.py` — export new modules
- `src/tgirl/registry.py` — `register_from_schema` already handles JSON Schema; bridge.py will use it

### Dependencies (already declared in pyproject.toml)
- `bridge = ["mcp>=1.0"]`
- `serve = ["fastapi>=0.110", "uvicorn>=0.28", "transformers>=5.0"]`

### Data model
- No new types needed — bridge.py uses existing `ToolRegistry`, `ParameterDef`, `ToolDefinition`
- serve.py request/response models are Pydantic models local to the module

## 4. Acceptance Criteria

1. `import_mcp_tools(registry, server_url, ...)` connects to an MCP server, registers tools, and wraps execution
2. MCP JSON Schema types map correctly to tgirl TypeRepr (string→str, integer→int, number→float, boolean→bool, array→list[T], object→ModelType, enum→LiteralType)
3. `expose_as_mcp(registry, pipeline_name, ...)` wraps a tgirl pipeline as an MCP tool callable by external systems
4. `POST /generate` accepts intent + optional scopes/cost/params, returns output + tool_calls + telemetry
5. `GET /tools` returns registered tools with parameter types and quotas
6. `GET /grammar` returns current generated grammar text
7. `GET /health` returns model info, tool count, uptime
8. `WebSocket /stream` sends token-by-token events with per-token telemetry
9. `--hot-reload` watches tool files and rebuilds registry on change
10. Both modules are independently importable — serve.py does not require bridge.py and vice versa
11. All existing tests continue to pass
12. New tests cover both modules with >90% line coverage

## 5. Risk Assessment

- **MCP protocol stability**: The `mcp` Python package is relatively new. API changes could break bridge.py. Mitigate: pin version, wrap MCP-specific calls behind an adapter.
- **Model loading in serve.py**: Loading a model on startup blocks the event loop. Mitigate: load model before starting uvicorn, not inside the request handler.
- **Hot reload race conditions**: Rebuilding the registry while a request is in-flight could cause inconsistency. Mitigate: swap the registry atomically (build new, then replace reference).
- **WebSocket backpressure**: Slow clients could cause memory buildup during streaming. Mitigate: bounded buffer with drop-oldest policy.

## 6. Open Questions

1. Should `import_mcp_tools` support async MCP connections, or is sync sufficient for v1?
2. Does the `mcp` Python package expose a client API for tool enumeration, or do we need raw HTTP?
3. Should serve.py support both MLX and torch backends, or MLX-only for v1?
4. What authentication (if any) should the server support? API keys? None for v1?

## 7. Out of Scope

- Authentication/authorization on the server (v1 is localhost-only)
- Multi-model serving (one model per server instance)
- GPU/device management (user provides --device flag)
- Production deployment (this is a local development/testing server)
- MCP server-side implementation (tgirl acts as MCP client for import, MCP tool provider for export)
