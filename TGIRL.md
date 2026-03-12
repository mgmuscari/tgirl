# tgirl — Technical Requirements Document

**Transformational Grammar for Inference-Restricting Languages**

Version 1.0 · March 2026 · Ontologi LLC

---

## 1. Project Definition

tgirl is a Python library for local LLM inference with grammar-constrained compositional tool calling. It supports two generation modes: freeform natural language output and grammar-constrained Hy s-expression tool pipelines. The model generates freely until it needs to invoke tools, at which point the grammar dynamically constrains the output distribution to only permit well-formed Hy s-expressions composing registered tools within their type, quota, and scope constraints. Once the tool expression is complete and executed, the model returns to freeform generation to present results, reason, or converse.

This dual-mode design means tgirl is a complete local inference solution, not just a tool-calling layer. The model can think, explain, converse, and act — with mathematical safety guarantees only during the acting.

tgirl is not an application or a framework. It is a library that other systems (inference servers, agent frameworks, companion apps) import and use. An optional CLI/server entrypoint is provided for standalone local inference, but the library must be usable without it.

### 1.1 Scope

**In scope:**

- Dual-mode inference: freeform natural language generation and grammar-constrained tool call generation within a single session
- Mode switching: detection of tool call intent (via special tokens, model output patterns, or explicit delimiters) to transition between freeform and constrained modes
- Tool registry with type extraction, quotas, cost budgets, and scopes
- Dynamic CFG generation from registry state via Jinja templates
- Hy s-expression compilation to Python AST and sandboxed execution
- Optimal transport–based logit redistribution for constrained generation (applied only during constrained mode)
- Grammar-implied temperature scheduling and per-token sampling control (applied only during constrained mode)
- Inference hook protocol for per-token interventions (active in both modes, but grammar-dependent hooks only fire during constrained mode)
- Constrained generation integration (Outlines as primary backend)
- MCP bridge layer (import MCP tools, expose tgirl pipelines as MCP tools)
- Telemetry and observability for every generated pipeline
- Benchmark suite with reproducible methodology

**Out of scope for v1.0:**

- Activation steering (ESTRADIOL) — tgirl defines the hook interface only
- Companion app (BLAHAJ) — tgirl provides tool registry infrastructure only
- Bluesky integration (SPIRO)
- Fine-tuning or training of any kind
- Cloud inference or API proxy

### 1.2 Design Principles

- **Safety by construction, not validation.** Invalid tool calls are inexpressible at the token level, not caught at runtime.
- **Model-agnostic.** Any model that produces a probability distribution over a vocabulary can be grammar-constrained. No instruct tuning or tool call fine-tuning should be required.
- **Composable over sequential.** The model emits a single compositional expression representing an entire pipeline, not a sequence of individual calls requiring multiple inference passes.
- **Auditable.** The generated grammar can be inspected, diffed between calls, and logged. The OT transport cost at every token position is recorded.
- **Incrementally adoptable.** The MCP bridge allows tgirl to be introduced into existing MCP architectures without rewriting tool definitions.

### 1.3 Core Claim

Grammar-constrained compositional generation should produce valid tool call pipelines at lower token cost, fewer inference round-trips, and zero structural error rate compared to sequential JSON/XML tool calling — including on base (non-instruct) models that have never been trained for tool use. The benchmark suite exists to validate or refute this claim.

---

## 2. Module Architecture

The library consists of six modules. Each must be independently importable and testable.

```
tgirl/
├── __init__.py
├── registry.py       # Tool registration, type extraction, snapshots
├── grammar.py        # Dynamic CFG generation from registry state
├── compile.py        # Hy parsing, AST compilation, sandboxed execution
├── transport.py      # Optimal transport logit redistribution
├── sample.py         # Constrained sampling engine, hooks, telemetry
├── bridge.py         # MCP compatibility layer
├── serve.py          # Optional: FastAPI local inference server
├── cli.py            # Optional: CLI entrypoint
├── templates/        # Jinja grammar templates
│   ├── base.cfg.j2
│   ├── tools.cfg.j2
│   ├── types.cfg.j2
│   └── composition.cfg.j2
├── telemetry.py      # Telemetry data structures and logging
└── types.py          # Shared type definitions
```

### 2.1 Dependency Graph

```
registry  (pydantic, annotated-types)
    ↓
grammar   (jinja2, depends on registry)
    ↓
compile   (hy, RestrictedPython, depends on registry)
    ↓
transport (torch, POT or geomloss — no other tgirl deps)
    ↓
sample    (outlines, torch — depends on grammar, transport)
    ↓
bridge    (mcp — depends on registry)
    ↓
serve     (fastapi, transformers — depends on all above)
```

`transport` must have zero coupling to the grammar or registry modules. It operates on raw logit tensors and valid token sets. This allows it to be used independently or swapped out.

All modules use `structlog` for structured logging. Telemetry data structures use Pydantic models.

---

## 3. Dual-Mode Generation

tgirl operates in two modes within a single inference session:

- **Freeform mode**: the model generates unconstrained natural language. No grammar masking, no OT redistribution. Standard sampling parameters apply. The model can reason, explain, converse, summarize tool results, or do anything a normal LLM does.
- **Constrained mode**: the grammar mask is active. OT redistribution and grammar-implied temperature scheduling are applied. The model can only emit valid Hy s-expressions over registered tools. This mode is entered when the model needs to invoke tools and exited when the tool expression is complete.

### 3.1 Mode Switching Mechanism

The transition between modes must be handled via delimiter tokens. The model emits a special delimiter to signal "I want to call tools now," at which point the sampling loop activates grammar constraints. When the grammar reaches an accepting state (the s-expression is syntactically complete), the constrained mode ends, the expression is compiled and executed, and the model returns to freeform mode with the tool results injected into its context.

Delimiter options (decide during implementation):

- **Special tokens**: define `<tool>` and `</tool>` tokens (or reuse the model's existing tool call tokens if present). The sampling loop watches for the opening delimiter in freeform mode and activates constraints. The closing delimiter is implicit — the grammar's accepting state signals completion.
- **Prompt-structured**: the system prompt instructs the model to emit tool calls inside a known delimiter pair. The sampling loop pattern-matches on the opening delimiter. This works with any model without tokenizer modification.
- **Hybrid**: use the model's native tool call delimiters if they exist (Qwen3.5 has built-in tool calling tokens), but override the content between delimiters with grammar-constrained generation instead of the model's default JSON output.

The hybrid approach is preferred for Qwen3.5 since it already has tool-calling token conventions. The model's existing instruct tuning for tool invocation provides the intent signal; tgirl replaces the output format with grammar-constrained Hy.

### 3.2 Context Management

After a tool pipeline executes, the results must be injected back into the model's context for continued freeform generation. The context format:

```
[freeform model output before tool call]
<tool>
(-> (fetch-weather "Berkeley, CA") (summarize-weather))
</tool>
<tool_result>
{"summary": "Rain expected tomorrow.", "temperature": 58, ...}
</tool_result>
[model continues freeform generation with results in context]
```

The tool result serialization format should be configurable (JSON, plain text summary, or structured Hy). The model sees the result in its context and can reference it in subsequent freeform output.

### 3.3 Multi-Turn Tool Use

A single session may involve multiple freeform→constrained→freeform cycles. Each cycle:

1. Model generates freeform until it emits the tool call delimiter
2. Grammar constraints activate; model emits a valid Hy pipeline
3. Pipeline compiles, executes, results injected into context
4. Model returns to freeform, can reference results
5. Model may emit another tool call delimiter for a subsequent pipeline
6. Quota state persists across cycles within a session (quotas are per-session, not per-call)

### 3.4 Freeform Sampling Configuration

Freeform mode uses its own sampling configuration, independent of the grammar-implied settings used in constrained mode:

```python
@dataclass
class SessionConfig:
    # Freeform mode sampling
    freeform_temperature: float = 0.7
    freeform_top_p: float = 0.9
    freeform_top_k: int | None = None
    freeform_repetition_penalty: float = 1.0
    freeform_max_tokens: int = 4096

    # Constrained mode sampling (base values, overridden by grammar-implied scheduling)
    constrained_base_temperature: float = 0.3
    constrained_ot_epsilon: float = 0.1
    constrained_max_tokens: int = 512

    # Session-level
    max_tool_cycles: int = 10          # max freeform→constrained round-trips
    session_cost_budget: float | None = None
    session_timeout: float = 300.0     # seconds
```

---

## 4. tgirl.registry — Tool Registration

### 3.1 Decorator API

```python
from tgirl.registry import tool, ToolRegistry

registry = ToolRegistry()

@registry.tool(
    quota=10,                # max calls per pipeline execution
    cost=0.5,                # abstract cost units per call
    cost_budget=5.0,         # max total cost per pipeline
    scope="db:write",        # authorization scope string
    timeout=30.0,            # execution timeout in seconds
    cacheable=True,          # results can be memoized within pipeline
    description="Write a record to the database",
)
def db_write(table: str, record: dict[str, Any]) -> WriteResult:
    ...
```

The decorator must:

- Extract all parameter type hints via `inspect` and `typing.get_type_hints`
- Extract the return type hint
- Validate that all parameters have type annotations (raise at registration time if not)
- Store the tool definition in the registry with all metadata
- Return the original function unmodified (the decorator is metadata-only, not a wrapper)

### 3.2 Type System

The registry must extract and represent the following Python types as grammar-producible constructs:

| Python Type | Grammar Representation |
|---|---|
| `str` | Quoted string literal with escape handling |
| `int` | Integer literal (optional range via `Annotated`) |
| `float` | Float literal (optional range via `Annotated`) |
| `bool` | `true` / `false` |
| `None` / `NoneType` | `nil` |
| `list[T]` | Hy list literal `[item ...]`, recursive on T |
| `dict[K, V]` | Hy dict literal `{:key val ...}`, recursive on K and V |
| `Literal["a", "b", "c"]` | Enumerated alternatives |
| `Enum` subclass | Enumerated alternatives from enum values |
| `Optional[T]` | `T | nil` |
| `Union[A, B]` | `A | B` as grammar alternatives |
| `Pydantic BaseModel` | Hy dict with required/optional keys matching field definitions |
| `Annotated[int, Gt(0), Lt(100)]` | Enumerated integer range or constrained production |

For `Annotated` types with numeric range constraints, the grammar generator must decide whether to enumerate (small ranges) or use a constrained numeric production (large ranges). The threshold for this decision should be configurable, defaulting to enumeration for ranges ≤ 256 values.

### 3.3 Registry Snapshot

```python
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    parameters: tuple[ParameterDef, ...]
    return_type: TypeRepresentation
    quota: int | None
    cost: float
    cost_budget: float | None
    scope: str | None
    timeout: float | None
    cacheable: bool
    description: str

@dataclass(frozen=True)
class RegistrySnapshot:
    tools: tuple[ToolDefinition, ...]
    quotas: dict[str, int]             # remaining calls per tool
    cost_remaining: float | None       # remaining cost budget
    scopes: frozenset[str]             # authorized scopes for this call
    timestamp: float
```

The snapshot is immutable. It is produced at the start of each generation request and passed to the grammar generator. Quota state is tracked per-pipeline-execution, not globally (the registry maintains global state; the snapshot is a point-in-time copy with call-specific scope restrictions applied).

### 3.4 Scope Filtering

When a snapshot is created, it must accept an optional set of authorized scopes. Only tools whose scope is `None` (unrestricted) or matches one of the authorized scopes are included in the snapshot. This is the authorization boundary.

### 3.5 Tool Restriction

In addition to scope filtering, snapshot creation must accept an optional explicit tool allowlist (`restrict_to: list[str] | None`). When provided, only named tools are included. This enables per-call tool restriction for the grammar.

---

## 5. tgirl.grammar — Dynamic CFG Generation

### 4.1 Grammar Template Architecture

Grammars are generated via Jinja2 templates. The template system takes a `RegistrySnapshot` and produces a CFG specification compatible with Outlines' grammar format.

The grammar must be split across composable Jinja templates:

- `base.cfg.j2`: top-level start rule, pipeline/single-call/insufficient-resources alternatives, whitespace rules
- `tools.cfg.j2`: per-tool productions generated from snapshot (tool names, per-tool argument patterns)
- `types.cfg.j2`: type productions generated from the union of all parameter types across all tools
- `composition.cfg.j2`: composition operator productions (threading, let, conditional, error handling, parallel)

The main template includes the others. This separation allows templates to be tested and modified independently.

### 4.2 Core Grammar Structure

The generated grammar must support at minimum:

```
start           ::= pipeline | single_call | insufficient
pipeline        ::= "(-> " call (" " call)+ ")"
single_call     ::= call
call            ::= "(" tool_name " " args ")"
insufficient    ::= "(insufficient-resources " reason ")"

tool_name       ::= <dynamically generated from snapshot>
args            ::= <per-tool argument production>
reason          ::= "quota-exhausted" | "cost-exceeded" | "scope-denied"
                   | "type-mismatch" | "no-valid-plan"
```

### 4.3 Composition Operators

The grammar must support these Hy composition forms:

| Operator | Form | Semantics |
|---|---|---|
| Threading | `(-> expr (f1 ...) (f2 ...))` | Result of each step piped as first arg to next |
| Named binding | `(let [x (f1 ...)] (f2 x ...))` | Bind intermediate results to names |
| Conditional | `(if (pred ...) (then ...) (else ...))` | Branch on predicate result |
| Error handling | `(try (expr ...) (catch e (fallback ...)))` | Structured error recovery |
| Parallel | `(pmap [f1 f2 f3] arg)` | Execute tools in parallel, collect results |

Each operator is a grammar production. They compose: a threading pipeline can contain let bindings, conditionals can contain pipelines, etc. The grammar must handle this nesting correctly.

### 4.4 Quota Enforcement in Grammar

This is the critical innovation. Quotas must be enforced at the grammar level, not at runtime.

The grammar generator must produce a *stateful* grammar (or the equivalent mechanism in Outlines) where:

1. Each tool's quota counter is initialized from the snapshot
2. The `tool_name` production only includes tools with remaining quota > 0
3. After a tool name token is emitted, the grammar state updates to decrement that tool's quota
4. If quota reaches 0, subsequent `tool_name` productions exclude that tool
5. The `insufficient-resources` terminal is always available as an alternative

Implementation options:

- **Preferred:** Use Outlines' guide/state mechanism if it supports stateful transitions. The grammar state carries quota counters and the valid production set changes per-step.
- **Fallback:** Generate a grammar with explicit states (e.g., `tool_name_0_remaining_3`, `tool_name_0_remaining_2`, etc.). This expands the grammar but keeps it context-free. Feasible for small quotas, may be impractical for large ones.
- **Fallback 2:** Apply quota enforcement as a post-grammar logit mask in the sampling loop. Less elegant but guaranteed to work. The grammar handles type safety; the sampler handles quotas.

The requirements document does not mandate the implementation approach. It mandates the outcome: the model cannot exceed quotas.

### 4.5 Grammar Diffing

The grammar module must support deterministic grammar generation (same snapshot → same grammar) and provide a diff utility:

```python
grammar_a = tgirl.grammar.generate(snapshot_a)
grammar_b = tgirl.grammar.generate(snapshot_b)
diff = tgirl.grammar.diff(grammar_a, grammar_b)
# Returns structured diff showing which productions changed
```

This is essential for debugging and auditing.

### 4.6 Output Formats

The grammar generator must support multiple output formats via additional Jinja templates:

| Format | Target | Priority |
|---|---|---|
| Outlines CFG | Outlines constrained generation | v1.0 (required) |
| GBNF | llama.cpp | v1.1 (planned) |
| Regex | vLLM guided decoding | v1.1 (planned) |
| JSON Schema | Diagnostic/documentation | v1.0 (nice to have) |

---

## 6. tgirl.compile — Hy Compilation and Execution

### 5.1 Why Hy

- Homoiconicity: the grammar constraining generation and the AST being executed are the same data structure
- Python AST compilation: zero FFI overhead, full Python ecosystem available at execution
- S-expressions are naturally compositional and token-efficient
- Well-maintained with stable compilation semantics in 1.x

Pin to `hy>=1.0,<2.0`. Test against Python 3.11, 3.12, and 3.13.

### 5.2 Compilation Pipeline

```
model output (str)
    → hy.read_many()           # parse to Hy AST
    → static_analysis()        # verify safety constraints
    → hy.compile()             # compile to Python AST
    → ast_analysis()           # verify no unauthorized ops in Python AST
    → exec(code, sandbox)      # execute in restricted namespace
```

### 5.3 Static Analysis Requirements

Before compilation, the Hy AST must be analyzed to verify:

- All function call targets are names registered in the current tool registry
- No `import` forms
- No `eval`, `exec`, `__import__`, `compile`, `open`, `getattr`, `setattr`, `delattr`
- No attribute access (`foo.bar`) except on dict-like results via Hy's `.` syntax for known safe patterns
- All variable references resolve to either tool names, let-bound names, or threading-bound names
- No recursive definitions

### 5.4 Python AST Analysis

After Hy compilation to Python AST, walk the AST to verify:

- No `ast.Import` or `ast.ImportFrom` nodes
- No `ast.Call` to anything not in the sandbox namespace
- No `ast.Attribute` access to dunder methods
- No `ast.Global` or `ast.Nonlocal`

This is defense-in-depth. The grammar should prevent all of these. Static analysis catches template bugs.

### 5.5 Execution Sandbox

The compiled code executes in a namespace containing only:

- Registered tool functions (by their registered names)
- Composition operator implementations (threading, let, if, try, pmap)
- A result accumulator

No builtins, no modules, no global state. The sandbox namespace is constructed fresh per execution.

### 5.6 Execution Timeout

Each pipeline execution must respect the per-tool timeouts defined in the registry. Additionally, an overall pipeline timeout must be configurable (default: 60 seconds). The execution engine must cancel and return a structured error if either timeout is exceeded.

### 5.7 Error Handling

Execution errors must be structured:

```python
@dataclass
class PipelineError:
    stage: str              # "parse", "static_analysis", "compile", "ast_analysis", "execute"
    tool_name: str | None   # which tool failed, if applicable
    error_type: str         # exception class name
    message: str
    hy_source: str          # the original model output
    position: int | None    # token position where the issue originated, if known
```

---

## 7. tgirl.transport — Optimal Transport for Constrained Generation

### 6.1 Problem Statement

Standard constrained generation masks invalid tokens (sets logits to -inf) and renormalizes. This loses information about the model's intent by redistributing probability mass uniformly across valid tokens.

tgirl.transport provides an alternative: use optimal transport to redistribute the model's full output distribution onto the grammar-valid subset, with transport cost defined by semantic distance in embedding space. The model's intent is preserved as much as the grammar allows.

### 6.2 Interface

```python
def redistribute_logits(
    logits: torch.Tensor,           # shape: (vocab_size,)
    valid_mask: torch.Tensor,       # shape: (vocab_size,), boolean
    embeddings: torch.Tensor,       # shape: (vocab_size, embed_dim)
    epsilon: float = 0.1,           # entropic regularization
    max_iterations: int = 50,       # Sinkhorn iterations
    convergence_threshold: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """
    Returns:
        - redistributed logits (shape: vocab_size, -inf for invalid tokens)
        - wasserstein distance (scalar, transport cost)
    """
```

This function must have **zero dependency** on any other tgirl module. It operates on raw tensors. It must be usable independently of the rest of the library.

### 6.3 Cost Matrix

The transport cost between tokens i and j is:

```
c(i, j) = 1 - cosine_similarity(embeddings[i], embeddings[j])
```

The cost matrix only needs to be computed between invalid source tokens and valid target tokens (not the full V×V matrix). For efficiency:

- Precompute and cache the embedding matrix for the model's vocabulary (once at model load)
- At each token position, extract the submatrix for invalid→valid pairs only
- If the valid set is large (>1000 tokens), fall back to standard masking (OT adds negligible value when most tokens are valid)

### 6.4 Sinkhorn Algorithm

Use the Sinkhorn-Knopp algorithm with entropic regularization:

1. Compute cost submatrix C between invalid sources and valid targets
2. Initialize: K = exp(-C / epsilon)
3. Iterate: alternating row and column normalization
4. Converge when marginal error < threshold or max iterations reached
5. Extract the transport plan and compute redistributed probability mass

The implementation must be GPU-compatible (all operations via torch tensors). CPU fallback must also work.

### 6.5 Bypass Conditions

OT computation should be skipped (fall back to standard masking) when:

- Only 1 valid token exists (forced decoding, no redistribution needed)
- Valid set contains > 50% of vocabulary (OT adds minimal value)
- The probability mass on invalid tokens is < 1% (model already wants valid tokens)

These thresholds should be configurable.

### 6.6 Diagnostics

The Wasserstein distance returned at each token position is a diagnostic signal:

- Low distance: grammar aligns with model intent
- High distance: grammar is fighting the model
- Sustained high distance: model cannot find a good plan within constraints

This signal must be included in telemetry records for every generation.

---

## 8. tgirl.sample — Constrained Sampling Engine

### 7.1 Integration with Outlines

tgirl.sample wraps Outlines' constrained generation to add:

- Dynamic grammar loading per-call (Outlines typically uses static grammars)
- OT-based logit redistribution via tgirl.transport (replaces Outlines' default masking)
- Per-token intervention hooks
- Grammar-implied temperature scheduling
- Telemetry recording at every token position

The module must work with Outlines' `transformers` integration for HuggingFace models. If Outlines' API does not expose per-token hooks, the sampling loop must be reimplemented using Outlines' grammar state tracking but with custom logit processing.

### 7.2 Inference Hook Protocol

```python
from typing import Protocol

class InferenceHook(Protocol):
    def pre_forward(
        self,
        position: int,
        grammar_state: GrammarState,
        token_history: list[int],
        logits: torch.Tensor,
    ) -> ModelIntervention:
        ...

@dataclass
class ModelIntervention:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    # Reserved for ESTRADIOL (v0.2). Not implemented in v1.0.
    activation_steering: Any | None = None
```

The sampling loop must call all registered hooks at each token position, merge their interventions (last writer wins per field, or configurable merge strategy), and apply the merged intervention before sampling.

### 7.3 Grammar-Implied Temperature Scheduling

The default `InferenceHook` implementation. At each token position:

1. Query the grammar state for the number of valid continuations
2. Derive temperature from the valid count and vocabulary size
3. Return a `ModelIntervention` with the computed temperature

```python
def grammar_implied_temperature(
    base_temp: float,
    valid_count: int,
    vocab_size: int,
) -> float:
    if valid_count <= 1:
        return 0.0
    freedom = valid_count / vocab_size
    return base_temp * (freedom ** 0.5)
```

The scaling function (sqrt here) should be configurable. Other candidates: linear, log, sigmoid. The benchmark suite should evaluate which scaling produces the best pipelines.

### 7.4 Per-Token Sampling Parameter Shaping

Beyond temperature, the grammar state implies adjustments to other sampling parameters:

- **top_p**: narrow at structural positions (low valid count), wider at value positions
- **repetition_penalty**: higher for tool name tokens (encourage diverse composition), baseline for argument tokens
- **presence_penalty**: increase for tools approaching quota limits (soft quota pressure before the hard grammar cutoff)

These are implemented as additional `InferenceHook` instances that can be composed with the temperature scheduler.

### 8.5 Sampling Loop

The sampling loop handles both freeform and constrained modes:

```
load model, tokenizer
load registry, precompute embedding cost matrix for transport
initialize session state (quotas, cost budget, cycle count)

FREEFORM MODE:
for each token position:
    1. run model forward pass → logits
    2. apply freeform sampling parameters (temperature, top_p, etc.)
    3. sample token
    4. if token matches tool call delimiter → switch to CONSTRAINED MODE
    5. if stop condition → end generation
    6. record token

CONSTRAINED MODE:
    1. generate grammar from current registry snapshot
    2. initialize grammar state
    for each token position:
        a. run model forward pass → logits
        b. get valid token mask from grammar state
        c. call all InferenceHooks → merged ModelIntervention
        d. apply OT redistribution (tgirl.transport) using valid mask
        e. apply temperature, top_p, top_k from intervention
        f. apply repetition/presence/frequency penalties
        g. sample token
        h. update grammar state with sampled token
        i. record telemetry for this position
        j. if grammar reaches accepting state → compile and execute pipeline
    3. inject tool results into context
    4. update quota state
    5. return to FREEFORM MODE
```

### 8.6 Telemetry

```python
@dataclass
class TelemetryRecord:
    pipeline_id: str
    # Constrained mode telemetry
    tokens: list[int]
    grammar_valid_counts: list[int]
    temperatures_applied: list[float]
    wasserstein_distances: list[float]
    top_p_applied: list[float]
    token_log_probs: list[float]
    grammar_generation_ms: float
    ot_computation_total_ms: float
    ot_bypassed_count: int
    hy_source: str
    execution_result: Any | None
    execution_error: PipelineError | None
    # Session context
    cycle_number: int                   # which freeform→constrained cycle
    freeform_tokens_before: int         # tokens generated in freeform before this call
    wall_time_ms: float
    total_tokens: int
    model_id: str
    registry_snapshot_hash: str
```

Telemetry records must be emittable as JSON. An optional file-based telemetry sink should write JSONL to a configurable directory.

---

## 9. tgirl.bridge — MCP Compatibility

### 8.1 Import MCP Tools

```python
from tgirl.bridge import import_mcp_tools

import_mcp_tools(
    registry,
    server_url="http://localhost:3000/mcp",
    scope_prefix="mcp:external",
    default_quota=5,
)
```

This must:

- Connect to an MCP server and enumerate its tools
- Convert MCP JSON Schema tool definitions to tgirl `ToolDefinition` with Python type hints
- Register each tool in the provided registry with the given scope prefix and default quota
- Wrap each tool's execution to serialize arguments to JSON, call the MCP server, and deserialize the response

### 8.2 Export tgirl Pipelines as MCP Tools

```python
from tgirl.bridge import expose_as_mcp

expose_as_mcp(
    registry,
    pipeline_name="enrich_and_store",
    description="Fetch, transform, and store a record",
    input_schema={...},        # JSON Schema for the pipeline's input
    mcp_server=server,
)
```

This wraps a tgirl pipeline (defined as a named Hy expression template or a Python function that invokes the tgirl sampling engine) as a single MCP tool that other systems can call.

### 8.3 MCP Type Mapping

| MCP JSON Schema Type | Python Type |
|---|---|
| `string` | `str` |
| `integer` | `int` |
| `number` | `float` |
| `boolean` | `bool` |
| `array` (items: T) | `list[T]` |
| `object` (properties: ...) | Generated `TypedDict` or Pydantic model |
| `enum` | `Literal[...]` |
| `null` | `None` |

---

## 10. tgirl.serve — Local Inference Server (Optional)

This module is optional. The library must be fully usable without it. The server is a convenience for standalone local inference.

### 9.1 CLI

```bash
# Minimal
tgirl serve --model Qwen/Qwen2.5-7B --tools ./my_tools.py

# Full options
tgirl serve \
    --model meta-llama/Llama-3-8B \
    --quantize int4 \
    --device mps \
    --port 8420 \
    --tools ./tools/ \
    --hot-reload \
    --telemetry-dir ./logs/ \
    --ot-epsilon 0.1 \
    --base-temperature 0.7
```

### 9.2 Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | POST | Generate and execute a tool pipeline from natural language |
| `/tools` | GET | List registered tools with types, quotas, scopes |
| `/grammar` | GET | Return current generated grammar |
| `/grammar/preview` | POST | Preview grammar for given scope/restriction |
| `/telemetry` | GET | Recent telemetry records |
| `/health` | GET | Server health, model info |
| `/stream` | WebSocket | Token-by-token streaming with per-token telemetry |

### 9.3 Request Format

```json
{
    "intent": "Check if I need an umbrella tomorrow in Berkeley",
    "scopes": ["api:weather", "calendar:read"],
    "max_cost": 2.0,
    "restrict_tools": null,
    "ot_epsilon": null,
    "base_temperature": null
}
```

### 10.4 Response Format

```json
{
    "output": "Let me check the weather for you.\n\n<tool>(-> (fetch-weather \"Berkeley, CA\") (summarize-weather))</tool>\n\nLooks like rain tomorrow — you'll want an umbrella.",
    "tool_calls": [
        {
            "pipeline": "(-> (fetch-weather \"Berkeley, CA\") (summarize-weather))",
            "result": { "summary": "Rain expected tomorrow.", "temperature": 58 },
            "telemetry": { ... }
        }
    ],
    "error": null
}
```

### 9.5 Hot Reload

The server must watch tool files for changes (via filesystem events or polling) and rebuild the registry on change. The grammar regenerates on the next request. No server restart required.

---

## 11. Benchmark Suite

The benchmark suite validates or refutes the core claims. It is not a marketing artifact. It is a scientific instrument.

### 10.1 Benchmark Workflows

Five workflows, chosen for real-world representativeness:

| ID | Workflow | Tools Required | Pipeline Shape |
|---|---|---|---|
| B1 | File processing | file_read, grep, transform, file_write | Linear 4-step |
| B2 | Multi-API orchestration | fetch (×3), merge, transform, db_write | Fan-in + linear |
| B3 | RAG pipeline | retrieve, rerank, generate | Linear 3-step, large payloads |
| B4 | Conditional workflow | check_status, branch_true, branch_false | Branching |
| B5 | Quota-constrained | db_read (quota:5), db_write (quota:2), notify | Must plan within limits |

Each workflow must have a reference implementation as a Python function (the "correct" execution path) so that generated pipelines can be validated for correctness, not just structural validity.

### 10.2 Comparison Methods

| Method | Implementation | What It Measures |
|---|---|---|
| MCP sequential | Call a tool-calling-capable model API (e.g., OpenAI, Anthropic) using standard MCP/function calling protocol | Baseline cost and error rate for the industry standard |
| Shell piping | Prompt a model to emit shell commands, pipe via subprocess | The "cheap but dangerous" alternative people actually use |
| tgirl (instruct, medium) | Grammar-constrained generation on Qwen3.5-9B (instruct) | tgirl on a current-gen instruct model at consumer hardware scale |
| tgirl (instruct, small) | Grammar-constrained generation on Qwen3.5-4B (instruct) | Lower bound of viable model size with grammar constraints |
| tgirl (base model) | Grammar-constrained generation on Qwen3.5-9B-Base or Qwen3.5-35B-A3B-Base | Validates the model-agnostic claim — no instruct tuning |
| tgirl (MoE, large) | Grammar-constrained generation on Qwen3.5-35B-A3B (3B active params, instruct) | MoE efficiency — frontier reasoning at 3B active compute |

### 10.3 Metrics

For every (workflow × method) combination, measure:

| Metric | Unit | Description |
|---|---|---|
| Total tokens | count | Sum of all tokens across all inference calls for the workflow |
| Inference round-trips | count | Number of separate model forward passes / API calls |
| Wall time | ms | End-to-end from intent to result |
| Structural validity | % | Pipeline parses and type-checks (for tgirl: should be 100% by construction) |
| Execution correctness | % | Pipeline produces the expected result (validated against reference implementation) |
| Error rate | % | Percentage of runs with any failure (structural or execution) |
| Grammar generation time | ms | Time to produce CFG from snapshot (tgirl only) |
| OT overhead | ms | Total OT computation time across all tokens (tgirl only) |
| Mean Wasserstein distance | float | Average transport cost per token (tgirl only) |

### 10.4 Statistical Requirements

- Minimum 100 runs per (workflow × method) combination
- Report: mean, median, p5, p95, standard deviation
- Pairwise significance testing: Mann-Whitney U test, α = 0.05
- All raw data committed to repo as JSONL
- Reproducible via single command: `tgirl benchmark --all`
- Random seed control for reproducibility: `tgirl benchmark --all --seed 42`

### 10.5 OT Ablation

Run the full benchmark suite with OT enabled and disabled (standard masking) to measure:

- Does OT improve execution correctness? (model intent preservation)
- What is the latency cost of OT per token?
- At what grammar constraint tightness does OT provide the most benefit?

This is an honest question. If standard masking produces equivalent results, OT should be documented as available but not default.

### 10.6 Temperature Scheduling Ablation

Run the benchmark suite with:

- Fixed temperature (0.0, 0.3, 0.7, 1.0)
- Grammar-implied temperature with sqrt scaling
- Grammar-implied temperature with linear scaling
- Grammar-implied temperature with log scaling

Measure execution correctness and pipeline diversity across all settings. Identify the best default.

---

## 12. Interface Contracts for Downstream Projects

tgirl v1.0 defines but does not implement interfaces for downstream AUTISM framework components. These interfaces must be stable.

### 11.1 ESTRADIOL Hook Interface

The `InferenceHook` protocol (Section 8.2) and the `activation_steering` field on `ModelIntervention` are the integration points. The `activation_steering` field must be typed as `Any` in v1.0 and documented as reserved.

### 11.2 BLAHAJ/VIC Tool Registration

VIC boundary tools are registered via the standard `@registry.tool` decorator with a `scope="vic:boundary"` convention. tgirl provides no special handling for VIC tools in v1.0 — they are ordinary tools that the companion app's prompt engineering invokes at appropriate intervals.

The `invisible` and `priority` parameters on the tool decorator (mentioned in design discussions) are deferred to v1.1 pending design review of how invisible tools interact with grammar generation.

---

## 13. Implementation Priorities

### Phase 1: Core (Days 1–2)

- `tgirl.registry`: full decorator API, type extraction, snapshot generation, scope/restriction filtering
- `tgirl.compile`: Hy parsing, static analysis, AST analysis, sandbox execution
- `tgirl.grammar`: Jinja template architecture, CFG generation for all supported types and composition operators
- Unit tests for each module independently
- **Exit criterion:** can register tools, generate a grammar, compile and execute a hand-written Hy pipeline

### Phase 2: Sampling (Day 3)

- `tgirl.transport`: Sinkhorn OT implementation, bypass conditions, Wasserstein diagnostic
- `tgirl.sample`: Outlines integration (or custom sampling loop), grammar-implied temperature, hook protocol, telemetry
- Integration test: generate and execute a tool pipeline from a natural language prompt against a local model
- **Exit criterion:** end-to-end generation and execution works on at least one model

### Phase 3: Benchmarks (Day 4)

- Implement all five benchmark workflows with reference implementations
- Implement all comparison methods
- Run full benchmark suite
- Statistical analysis
- **Exit criterion:** benchmark results exist and are reproducible

### Phase 4: Ship (Day 5)

- `tgirl.bridge`: MCP import/export
- `tgirl.serve`: FastAPI server, hot reload, CLI
- Packaging: pyproject.toml, PyPI publication
- README with benchmark methodology and results
- **Exit criterion:** `pip install tgirl` works, `tgirl serve` starts

---

## 14. Dependencies

| Package | Version Constraint | Required For | Install Group |
|---|---|---|---|
| pydantic | >=2.0 | Data models, type introspection, validation, JSON Schema | core |
| annotated-types | >=0.6 | Constraint types (Gt, Lt, Ge, Le, MultipleOf) — via pydantic | core |
| structlog | >=24.0 | Structured logging, telemetry | core |
| hy | >=1.0,<2.0 | Compilation | core |
| RestrictedPython | >=7.0 | Sandboxed execution (AST-level restrictions) | core |
| jinja2 | >=3.0 | Grammar templates | core |
| outlines | >=0.1 | Constrained generation | core |
| torch | >=2.0 | Tensors, model inference | core |
| POT | >=0.9 | Optimal transport (Sinkhorn, Wasserstein) | core |
| transformers | >=4.48 | Model/tokenizer loading (Qwen3.5 support) | core |
| fastapi | >=0.110 | Inference server | `[serve]` |
| uvicorn | >=0.28 | ASGI server | `[serve]` |
| mcp | >=1.0 | MCP bridge | `[bridge]` |
| hypothesis | >=6.0 | Property-based testing, fuzzing | `[dev]` |

Python >=3.11 required.

### Primary Model Targets

Qwen3.5 is the primary model family. All models use a hybrid Gated DeltaNet + MoE architecture with 262K native context (extendable to ~1M). All are Apache 2.0. The small and medium series were released in February–March 2026.

| Model | Total / Active Params | Min RAM/VRAM | Architecture | Use Case |
|---|---|---|---|---|
| Qwen3.5-4B | 4B dense | ~4GB | Hybrid GatedDeltaNet | Minimum viable. Grammar constraints compensate for size. |
| Qwen3.5-9B | 9B dense | ~8GB | Hybrid GatedDeltaNet | Recommended default for consumer hardware. |
| Qwen3.5-27B | 27B dense | ~20GB | Hybrid GatedDeltaNet | Higher quality dense model. Single GPU. |
| Qwen3.5-35B-A3B | 35B / 3B active | ~6GB | MoE + GatedDeltaNet | Sweet spot: frontier reasoning at 3B active compute. |
| Qwen3.5-122B-A10B | 122B / 10B active | ~24GB | MoE + GatedDeltaNet | High-end. Scores 72.2 on BFCL-V4 tool use benchmark. |
| Qwen3.5-397B-A17B | 397B / 17B active | ~80GB+ | MoE + GatedDeltaNet | Flagship. Research and benchmarking on Yorick (256GB M3 Ultra). |

The Qwen3.5-35B-A3B is particularly interesting: it surpasses the previous-gen Qwen3-235B-A22B on core benchmarks with only 3B active parameters. For grammar-constrained generation where the grammar handles structural correctness, this efficiency is ideal.

**Key architectural note:** Qwen3.5 uses Gated Delta Networks for linear attention in a 3:1 ratio with full softmax attention. This hybrid architecture may interact non-trivially with constrained generation — the linear attention layers compress context into fixed-size state, which could affect how the model responds to grammar constraints at long sequence positions. This should be investigated during benchmarking.

**Thinking mode:** Qwen3.5 defaults to thinking mode (emitting `<think>...</think>` before responses). With tgirl's dual-mode architecture, thinking mode is naturally compatible — the `<think>` block is freeform generation (unconstrained), and grammar constraints only activate when the model emits the tool call delimiter. Thinking mode may actually *improve* tool composition quality by letting the model reason about which tools to use before entering constrained mode. This should be benchmarked both ways (thinking enabled vs. disabled) to determine the best default.

**Base model availability:** Alibaba has released Qwen3.5-35B-A3B-Base alongside instruct variants. This is the primary target for validating the model-agnostic claim.

Install groups via extras: `pip install tgirl` (core), `pip install tgirl[serve]` (with server), `pip install tgirl[bridge]` (with MCP), `pip install tgirl[all]`.

---

## 15. Open Questions

These are genuine unknowns that the implementation and benchmarks must resolve:

1. **Quota enforcement mechanism.** Can Outlines' grammar state carry mutable quota counters, or must quotas be enforced via post-grammar logit masking in the sampling loop? Investigate Outlines' API before committing to an approach.

2. **Grammar complexity scaling.** What is the grammar compilation time and per-token masking overhead as a function of number of tools and type complexity? At what scale does it become a bottleneck?

3. **OT value proposition.** Does optimal transport measurably improve execution correctness over standard masking? The ablation benchmark will answer this. If not, OT should be available but not default.

4. **Base model viability.** Qwen3.5-35B-A3B-Base is released. At what model size does grammar-constrained base model tool calling match instruct model tool calling? Test across 4B, 9B, and 35B-A3B-Base.

5. **Gated DeltaNet interaction with constrained generation.** Qwen3.5's hybrid linear/full attention architecture compresses context into fixed-size state in the linear layers. Does this affect grammar-constrained generation quality at long sequence positions? Compare token-level Wasserstein distances early vs. late in generation.

6. **Thinking mode impact on tool composition.** Qwen3.5's thinking mode lets the model reason in freeform before entering constrained tool generation. Benchmark thinking-enabled vs. thinking-disabled to determine if chain-of-thought reasoning before tool calls improves pipeline correctness or diversity. If thinking helps, it comes for free in the dual-mode architecture.

7. **Hy compilation edge cases.** What Hy constructs, if any, produce Python AST that escapes the sandbox? Fuzzing the compilation pipeline is necessary.

8. **Streaming execution.** Can pipeline results stream as intermediate tools complete? This requires rethinking threading macro semantics and is likely v1.1.

9. **Grammar-implied temperature scaling function.** Sqrt, linear, log, or sigmoid? The ablation benchmark will identify the best default.

---

*"We don't catch quota violations. We make them inexpressible."*