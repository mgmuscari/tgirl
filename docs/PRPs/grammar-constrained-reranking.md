# PRP: Grammar-Constrained Tool Re-Ranking

## Source PRD: docs/PRDs/grammar-constrained-reranking.md
## Date: 2026-03-13

## 1. Context Summary

Grammar-constrained generation guarantees syntactically valid tool calls but cannot influence **semantic** tool selection — at the tool-name token position, all tools are equally valid grammar alternatives. Small models frequently pick the wrong tool.

Experimental results (examples/test_rerank.py) show that a **grammar-constrained re-ranking pass** — using a tiny grammar that only accepts tool names — achieves 7/7 tool selection accuracy vs 5/7 baseline on Qwen3.5-0.8B, with sub-second latency and 3-8x speedup on the subsequent generation pass.

We are building a first-class `ToolRouter` that integrates this re-ranking pass into the `SamplingSession` dual-mode loop.

**Branch dependency:** `sample.py` and `outlines_adapter.py` exist on `feature/constrained-sampling-engine` (not yet merged to `main`). This branch carries uncommitted changes from that work (registry enrichment, instructions.py, type grammars). Implementation will need to either rebase onto that branch or merge it first.

## 2. Codebase Analysis

### Relevant existing patterns

- **Grammar generation:** `tgirl.grammar.generate(snapshot)` → `GrammarOutput` with Lark EBNF text. Uses Jinja2 templates (`base.cfg.j2`). The routing grammar is much simpler — just string alternatives, no templates needed. (`src/tgirl/grammar.py:399-446`)
- **Registry restriction:** `registry.snapshot(restrict_to=["tool_a"])` already filters tools and their type_grammars. This is the mechanism for narrowing the grammar after re-ranking. (`src/tgirl/registry.py:106-165`)
- **Instruction generation:** `tgirl.instructions.generate_system_prompt(snapshot)` generates structured prompts from registry metadata. The routing prompt needs a classification-optimized variant. (`src/tgirl/instructions.py:105-135`)
- **Constrained generation:** `run_constrained_generation()` is the core loop — takes `GrammarState`, `forward_fn`, hooks, transport config. The re-ranking pass reuses this with a tiny grammar and short `max_tokens`. (`sample.py:232-354` on constrained-sampling-engine branch)
- **SamplingSession integration point:** The constrained mode section at `sample.py:480-500` (on constrained-sampling-engine) is where `snapshot` is created and `grammar_output` is generated. Re-ranking inserts between these two steps.
- **Frozen Pydantic models:** All types use `ConfigDict(frozen=True)`. New config/result types follow this pattern.

### Conventions to follow

- All types in `types.py`, all frozen Pydantic models
- Module-level `structlog.get_logger()` for logging
- `from __future__ import annotations` in every module
- Protocol classes for pluggable interfaces
- `Callable[[str], GrammarState]` is the grammar factory type (same one used for routing and main grammars)

### Integration points

1. `SamplingSession.__init__()` — add optional `rerank_config` parameter
2. `SamplingSession.run()` — insert re-ranking pass before grammar generation in constrained mode
3. `grammar.py` — add `generate_routing_grammar()` as a standalone function
4. `types.py` — add `RerankConfig` and `RerankResult` models
5. `TelemetryRecord` — extend with re-ranking metadata (optional fields)

## 3. Implementation Plan

**Test Command:** `pytest tests/ -v`

### Task 1: Add RerankConfig and RerankResult types

**Files:** `src/tgirl/types.py`, `tests/test_types.py`
**Approach:**
Add two new frozen Pydantic models:
```python
class RerankConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_tokens: int = 16           # Routing pass is short
    temperature: float = 0.3       # Low temperature for classification
    top_k: int = 1                 # Number of tools to select
    enabled: bool = True           # Allow disabling without removing config

class RerankResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    selected_tools: tuple[str, ...]  # Ranked tool names
    routing_tokens: int              # Tokens used in routing pass
    routing_latency_ms: float        # Time for routing pass
    routing_grammar_text: str        # The grammar used (for debugging)
```
**Tests:**
- `RerankConfig` constructs with defaults
- `RerankConfig` is frozen
- `RerankResult` constructs with required fields
- `RerankResult` is frozen
**Validation:** `pytest tests/test_types.py -v -k rerank`

### Task 2: Add generate_routing_grammar() to grammar module

**Files:** `src/tgirl/grammar.py`, `tests/test_grammar.py`
**Approach:**
Add a standalone function that produces a minimal Lark grammar from a snapshot:
```python
def generate_routing_grammar(snapshot: RegistrySnapshot) -> str:
    """Generate a minimal grammar that accepts only tool names.

    The grammar is a simple alternation of string literals:
        start: tool_choice
        tool_choice: "get_field" | "set_field" | ...

    Returns Lark EBNF text (not a GrammarOutput — no productions or hash needed).
    """
    if not snapshot.tools:
        msg = "Cannot generate routing grammar for empty snapshot"
        raise ValueError(msg)
    alternatives = " | ".join(f'"{tool.name}"' for tool in snapshot.tools)
    return f"start: tool_choice\ntool_choice: {alternatives}\n"
```
**Tests:**
- Produces valid Lark grammar text containing all tool names from snapshot
- Each tool name appears as a quoted alternative
- Raises ValueError for empty snapshot
- Single-tool snapshot produces single-alternative grammar
- Deterministic — same snapshot produces same grammar
**Validation:** `pytest tests/test_grammar.py -v -k routing`

### Task 3: Add generate_routing_prompt() to instructions module

**Files:** `src/tgirl/instructions.py`, `tests/test_instructions.py`
**Approach:**
Add a function that generates a classification-optimized prompt from a snapshot:
```python
def generate_routing_prompt(snapshot: RegistrySnapshot) -> str:
    """Generate a routing prompt for tool selection.

    Optimized for classification: lists tools with descriptions,
    instructs the model to output ONLY the tool name.
    """
    lines = [
        "You are a tool routing assistant. "
        "Given a user request, pick the best tool.",
        "",
        "Available tools:",
    ]
    for tool in snapshot.tools:
        params = ", ".join(p.name for p in tool.parameters)
        lines.append(f"- {tool.name}({params}): {tool.description}")
    lines.append("")
    lines.append("Reply with ONLY the tool name, nothing else.")
    return "\n".join(lines)
```
**Tests:**
- Returns a string containing all tool names
- Contains tool descriptions
- Contains "Reply with ONLY the tool name"
- Deterministic
- Parameter names appear in the listing
**Validation:** `pytest tests/test_instructions.py -v -k routing`

### Task 4: Create ToolRouter class in rerank module

**Files:** `src/tgirl/rerank.py`, `tests/test_rerank.py`
**Approach:**
Create the core `ToolRouter` class:
```python
class ToolRouter:
    """Routes user requests to the best tool via grammar-constrained re-ranking."""

    def __init__(
        self,
        grammar_guide_factory: Callable[[str], GrammarState],
        forward_fn: Callable[[list[int]], torch.Tensor],
        tokenizer_decode: Callable[[list[int]], str],
        embeddings: torch.Tensor,
        config: RerankConfig | None = None,
    ) -> None: ...

    def route(
        self,
        snapshot: RegistrySnapshot,
        context_tokens: list[int],
        transport_config: TransportConfig | None = None,
    ) -> RerankResult:
        """Run grammar-constrained re-ranking to select the best tool.

        1. Generate routing grammar from snapshot
        2. Run short constrained generation pass
        3. Parse the selected tool name from output
        4. Return RerankResult with selected tools

        If snapshot has 0 or 1 tools, returns immediately (no routing needed).
        """
```

The `route()` method:
1. Short-circuits if `len(snapshot.tools) <= 1`
2. Calls `generate_routing_grammar(snapshot)` for the grammar
3. Calls `run_constrained_generation()` with the routing grammar, short max_tokens, and the context
4. Parses the output to extract the tool name
5. Returns `RerankResult`

**Tests:**
- Mock `GrammarState`, `forward_fn`, `tokenizer_decode` to test routing logic
- Single-tool snapshot returns that tool immediately (no generation call)
- Empty snapshot raises ValueError (from generate_routing_grammar)
- `route()` returns `RerankResult` with `selected_tools` containing the generated tool name
- `top_k=2` returns up to 2 tools (run routing grammar twice? Or use top-K from a single pass — design decision noted in uncertainty log)
- Config `enabled=False` returns all tools (no restriction)
- Respects max_tokens from config
**Validation:** `pytest tests/test_rerank.py -v`

### Task 5: Integrate ToolRouter into SamplingSession

**Files:** `src/tgirl/sample.py`, `tests/test_sample_rerank.py`
**Approach:**

**Note:** This task depends on `sample.py` being available. If working from `main`, cherry-pick the relevant commits from `feature/constrained-sampling-engine` first. If working from that branch, modify in place.

Modify `SamplingSession.__init__()`:
```python
def __init__(self, ..., rerank_config: RerankConfig | None = None) -> None:
    ...
    self._rerank_config = rerank_config
    self._router = (
        ToolRouter(
            grammar_guide_factory=grammar_guide_factory,
            forward_fn=forward_fn,
            tokenizer_decode=tokenizer_decode,
            embeddings=embeddings,
            config=rerank_config,
        )
        if rerank_config is not None
        else None
    )
```

Modify `SamplingSession.run()` constrained mode section — insert between snapshot creation and grammar generation:
```python
# --- Constrained mode ---
snapshot = self._snapshot_with_remaining_quotas()

# Optional re-ranking pass
if self._router is not None and self._rerank_config and self._rerank_config.enabled:
    rerank_result = self._router.route(
        snapshot=snapshot,
        context_tokens=token_history,
        transport_config=self._transport_config,
    )
    total_tokens += rerank_result.routing_tokens
    # Restrict snapshot to selected tools
    snapshot = self._registry.snapshot(
        restrict_to=list(rerank_result.selected_tools),
        cost_budget=self._config.session_cost_budget,
    )

grammar_output = generate_grammar(snapshot)
...
```

**Tests:**
- `SamplingSession` with `rerank_config=None` works identically to current behavior
- `SamplingSession` with `rerank_config` runs routing before constrained generation
- Routing tokens are counted in total_tokens
- Restricted snapshot is used for grammar generation (verify via mock inspection)
- Multiple cycles each get their own re-ranking pass
**Validation:** `pytest tests/test_sample_rerank.py -v`

### Task 6: Add rerank telemetry to TelemetryRecord

**Files:** `src/tgirl/types.py`, `src/tgirl/sample.py`
**Approach:**
Add optional fields to `TelemetryRecord`:
```python
class TelemetryRecord(BaseModel):
    ...
    rerank_selected_tool: str | None = None
    rerank_routing_tokens: int | None = None
    rerank_latency_ms: float | None = None
```
Update `SamplingSession.run()` to populate these fields when re-ranking is active.

**Tests:**
- Existing TelemetryRecord tests still pass (fields are optional)
- When re-ranking is active, telemetry records include rerank fields
- When re-ranking is inactive, rerank fields are None
**Validation:** `pytest tests/ -v -k telemetry`

### Task 7: Wire up __init__.py exports

**Files:** `src/tgirl/__init__.py`
**Approach:**
Add public exports:
```python
from tgirl.rerank import ToolRouter
from tgirl.types import RerankConfig, RerankResult
```
**Tests:**
- `from tgirl import ToolRouter, RerankConfig, RerankResult` works
**Validation:** `python -c "from tgirl import ToolRouter, RerankConfig, RerankResult; print('OK')"`

## 4. Validation Gates

```bash
# Syntax/Style
ruff check src/ --fix && ruff format src/

# Type checking
mypy src/tgirl/rerank.py src/tgirl/types.py

# Unit Tests
pytest tests/ -v

# Focused tests
pytest tests/test_rerank.py tests/test_instructions.py tests/test_grammar.py -v
```

## 5. Rollback Plan

The feature is entirely additive:
- New module `rerank.py` can be deleted
- New types `RerankConfig`, `RerankResult` can be removed from `types.py`
- `SamplingSession` changes are guarded by `if self._router is not None` — removing the config restores original behavior
- No existing API signatures change

## 6. Uncertainty Log

1. **Branch dependency is the biggest risk.** `sample.py` doesn't exist on `main`. Tasks 4-6 cannot be implemented without it. Recommend: merge `feature/constrained-sampling-engine` first, then rebase this branch, OR implement Tasks 1-3 first (they don't depend on sample.py) and defer Tasks 4-6.

2. **Top-K re-ranking strategy.** The experiment only tested top-1 (grammar produces exactly one tool name). For top-K, options include: (a) run routing K times with different temperatures, (b) use a grammar that allows comma-separated tool names, (c) run once and also include the runner-up from the constrained generation's logit distribution. Defaulting to top-1 for now with `top_k` config field reserved.

3. **Routing prompt vs tool-call prompt.** The routing prompt in `generate_routing_prompt()` is a separate function from `generate_system_prompt()`. This is intentional — the routing prompt is optimized for classification (short, directive), while the tool-call prompt needs full signatures and examples. But there's an argument for reusing the same prompt. Leaving as separate for now.

4. **Context tokens for routing.** The `route()` method receives the same `context_tokens` (token_history) as the main generation. This means the routing grammar sees the full conversation context. Whether the routing prompt should be prepended to this context or replace it is a design question — the experiment prepended it (via chat template). Keeping that approach.

5. **Re-ranking with composition.** This PRP routes to a single tool per cycle. If the user's request requires composition (e.g., "get the name field and convert to uppercase"), the router will pick one tool and the restricted grammar won't allow the other. This is explicitly out of scope per the PRD, but worth noting as a future enhancement.
