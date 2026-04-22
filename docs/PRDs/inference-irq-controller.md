# PRD: Inference Interrupt Controller

## Status: DRAFT
## Author: Maddy Muscari (drafted with Claude Opus 4.7)
## Date: 2026-04-22
## Branch: feature/inference-irq-controller

## 1. Problem Statement

`SamplingSession` today has grown ad-hoc mode-switching. Mode transitions are driven by `TransitionPolicy` objects (`DelimiterTransitionPolicy`, `ImmediateTransitionPolicy`, `BudgetTransitionPolicy` — `sample.py:761,786,799` and `benchmarks/run_bfcl.py:121–162`) that only see raw token signals; they cannot express "cancel on disconnect," "inject a tool result," or "external caller wants the grammar swapped now." The API server (`serve.py:835–897`) reaches directly into session internals for per-request overrides instead of signaling intent. Tool execution is fully out-of-loop and synchronous (`sample.py:1243, run_pipeline(...)`); the loop does not cooperate with tool completion beyond appending result tokens afterward. The autotuner runs at turn boundaries via bespoke hooks (`serve.py:1158–1209, 1347, 1366`). ESTRADIOL probe capture happens via a monkey-patched layer hook that is invisible to the sampling loop (`cache.py:28–222`).

Three gaps follow directly from this:

1. **Client disconnect does not preempt generation.** The OpenAI SSE streaming handler (`serve.py:1616–1779`) only notices a closed connection when the next `await websocket.send_json` raises. Forward passes between the disconnect and the next send continue unabated — a production gap also present in OpenAI and Anthropic's hosted APIs, where aborted HTTP requests routinely continue generating tokens the user still pays for. **This is the load-bearing differentiator for the v0.2 public launch: "aborted requests actually stop."** Claim must be substantiated by a provider-comparison test before any public writeup references it.
2. **Every new cross-cutting concern becomes another bespoke hook.** The autotuner, ESTRADIOL calibration, transition policies, and per-request overrides are variations on the same pattern — "something external to the core sampling loop wants to observe or mutate session state at well-defined points." Each has invented its own integration. The next concern will invent its own too.
3. **No principled dispatch point for out-of-band events.** Tool completion, user intent changes mid-generation, scheduled timer ticks, cancellations — these all need a uniform way to enter the loop. Today there is no such way, so they either don't exist (cancellation, tool completion feedback) or sit in their own modules with no shared abstraction (autotuner, calibration).

**Why now:** v0.2 launch wants cancel-on-disconnect in the narrative. More importantly, the cost of unifying these patterns is lowest now, before a sixth ad-hoc hook joins the five that already exist. The per-token-fire-control-autotuner branch is the most recent example — it added another cross-cutting path because no shared substrate existed.

## 2. Proposed Solution

Build `InterruptController` — a per-session, internal-API component that sits between `SamplingSession` and its generation loop. Drivers (API server, CLI, BFCL runner, autotuner, future orchestrators) raise interrupts instead of reaching into session internals. The sampling loop dispatches pending interrupts at natural yield points (between tokens in both freeform and constrained loops — `sample.py:1024–1025, 679–680`; `sample_mlx.py:700–701`).

**Interrupt sources (Phase 1):**

| Source | Priority | Today's origin | Handler responsibility |
|---|---|---|---|
| `CANCEL` | 0 (highest) | FastAPI disconnect, Ctrl-C | Break loop, finalize telemetry, close streams |
| `GRAMMAR_EXIT` | 1 | Constrained pipeline close (`sample.py:683`) | Return session to freeform, release grammar state |
| `GRAMMAR_TRIGGER` | 2 | Delimiter / latch policy (`sample.py:1017`) | Enter constrained mode with grammar snapshot |
| `BUDGET_EXHAUSTED` | 3 | Per-token budget counter (`BudgetTransitionPolicy`) | Force mode transition or terminate |
| `TOOL_COMPLETE` | 4 | Out-of-band tool dispatch return | Inject result tokens into history, resume |
| `MODE_SWITCH_REQUEST` | 5 | External driver (server override) | Apply grammar / temperature / scope change |
| `TURN_COMPLETE` | 6 | Turn boundary | Autotuner / calibration / telemetry finalize |
| `TIMER` | 7 (lowest) | Scheduled check (heartbeat, periodic calibration) | Periodic housekeeping |

**Handler contract:**

- Synchronous. Run between tokens on the sampling thread.
- May mutate `token_history`, swap `grammar_state`, change sampling params, finalize telemetry.
- **May NOT** roll back KV cache in Phase 1 (Phase 2 concern — requires `cache.fork()`).
- Return an `InterruptAction`: `CONTINUE` (resume loop), `BREAK` (exit cleanly), `RESTART_TURN` (return to top of turn).

**Priority + masking model:** `IntEnum` with fixed per-source priority (table above). Masking via context manager: `with session.irq.mask(IrqSource.TIMER): ...` suppresses dispatch until exit; raised-while-masked interrupts queue and fire on unmask in priority order. Fixed priorities are simpler than per-handler configurable priority; revisit in Phase 2 if drivers need finer control.

**Public API commitment:** Internal only. `tgirl.interrupts` module exports the types, but no stability promise in v0.2. Drivers inside the library use it; external consumers should not rely on the signature. Stability gate is v1.0.

**Relationship to v0.2 launch:** CANCEL is the load-bearing interrupt for launch narrative. Other sources migrate existing ad-hoc behavior behind the substrate. Migration is mostly behavior-preserving (e.g., `DelimiterTransitionPolicy` becomes a thin wrapper that raises `GRAMMAR_TRIGGER`). A runway decision lives in Open Questions: ship the full substrate into v0.2, or ship CANCEL-only + migrate the rest post-launch.

**Autotuner coordination:** The in-flight `feature/per-token-fire-control-autotuner` branch adds turn-level hooks. Rather than double-implement, the autotuner becomes a `TURN_COMPLETE` handler. Branch strategy is an Open Question.

## 3. Architecture Impact

### New module

- **`src/tgirl/interrupts.py`** — `IrqSource` enum, `InterruptAction` enum, `InterruptPayload` dataclass, `HandlerFn` protocol, `InterruptController` class. No tgirl-module dependencies (sits below `sample.py`). Pure-Python, stdlib only.

### Modified modules

- **`src/tgirl/sample.py`** — Inject `controller.dispatch_pending(session)` at per-token boundaries:
  - Freeform loop: after line 1024 (after token acceptance, before next iteration).
  - Constrained loop: after line 680 (after grammar advance, before accept check).
  - Replace existing `TransitionPolicy.evaluate` call at line 1017 with `controller.raise_interrupt(GRAMMAR_TRIGGER, ...)` when policy signals; the policy object becomes an internal adapter.
  - Existing confidence-monitor backtrack at lines 642–677 remains — it is orthogonal (operates within a single constrained decode).
- **`src/tgirl/sample_mlx.py`** — Symmetric changes at lines 485–702 (dispatch point around 700–701). MLX variant keeps zero-torch, zero-numpy invariant: controller dispatch is pure Python, does not touch tensors.
- **`src/tgirl/serve.py`** —
  - OpenAI SSE handler (lines 1616–1779): wire `request.is_disconnected()` polling (or a `disconnect_task = asyncio.create_task(request.is_disconnected())` race) and `controller.raise_interrupt(IrqSource.CANCEL)` on trigger.
  - `/generate` endpoint (lines 835–897): per-request overrides (`restrict_tools`, `scopes`, `ot_epsilon`, `base_temperature`, `max_cost`) become `MODE_SWITCH_REQUEST` interrupts raised before session run, or pre-run config applied once — both paths preserved but unified.
  - `/stream` WebSocket handler (lines 980–1028): add explicit disconnect → CANCEL, not the current implicit-exception path.
- **`src/tgirl/cache.py`** — No changes. Cache is keyed on token-list identity (lines 234–306) and is transparent to handlers that mutate `token_history`. Grammar swap does not invalidate cache.
- **`src/tgirl/cli.py`** — Thread SIGINT handler → raise `CANCEL` on active session. Minor.
- **`benchmarks/run_bfcl.py`** — Transition policies (lines 121–162) become thin adapters: each policy's `evaluate` call is replaced by raising the corresponding interrupt source. Behavior preserved bit-identically.
- **`src/tgirl/autotune.py`** — No structural changes. Hook-up in `serve.py` is rewired from bespoke post-turn callback (lines 1158–1209, 1347, 1366) to a registered `TURN_COMPLETE` handler.

### Tests

- **NEW: `tests/test_interrupts.py`** — Per-source firing, priority ordering, masking semantics, handler action dispatch, cache consistency under handler mutation, two-session isolation.
- **Modified: `tests/test_sample.py`, `tests/test_sample_mlx.py`** — Integration tests for loop-with-controller, including empty-queue overhead micro-benchmark.
- **Modified: `tests/test_serve.py`** — SSE cancel-on-disconnect integration test using the official `openai` Python client with pre-emptive HTTP disconnect.
- **NEW: `benchmarks/launch/cancel-behavior/`** — Script comparing local tgirl vs OpenAI vs Anthropic cancel behavior (requires API credits). Output committed as JSON + methodology notes.

### Data model (excerpt)

```python
class IrqSource(IntEnum):
    CANCEL = 0
    GRAMMAR_EXIT = 1
    GRAMMAR_TRIGGER = 2
    BUDGET_EXHAUSTED = 3
    TOOL_COMPLETE = 4
    MODE_SWITCH_REQUEST = 5
    TURN_COMPLETE = 6
    TIMER = 7

class InterruptAction(Enum):
    CONTINUE = "continue"
    BREAK = "break"
    RESTART_TURN = "restart_turn"

@dataclass
class InterruptPayload:
    source: IrqSource
    data: dict
    timestamp: float

HandlerFn = Callable[["SamplingSession", InterruptPayload], InterruptAction]
```

### Dependencies

No new external dependencies. Uses stdlib `enum`, `dataclasses`, `collections.deque`, `threading.Lock`. CANCEL cross-thread signaling uses `threading.Event` (the sampling thread runs under `asyncio.to_thread` per `serve.py:989`).

## 4. Acceptance Criteria

1. **CANCEL latency:** Given a running `SamplingSession`, `controller.raise_interrupt(IrqSource.CANCEL)` from another thread causes the loop to exit at the next token boundary within one forward-pass of wall-clock time. Verified by a test that measures tokens generated between raise and actual break (expected: ≤ 1).
2. **SSE cancel-on-disconnect end-to-end:** An integration test using the official `openai` Python client initiates a streaming `/v1/chat/completions` request, then forces HTTP disconnect mid-stream. Within one forward-pass of the disconnect, the sampling thread exits, telemetry is finalized, and no further forward passes execute. Verified by instrumenting `forward_fn` call counts.
3. **Provider-comparison evidence (precondition for public cancel claim):** A script at `benchmarks/launch/cancel-behavior/compare.py` runs identical cancel scenarios against local tgirl, OpenAI, and Anthropic. Output (token counts, billed usage, exact timings) committed to `benchmarks/launch/cancel-behavior/results.json`. No public writeup references cancel behavior without citing this file.
4. **Priority ordering:** Given simultaneously-pending `CANCEL` and `TIMER` interrupts, the `CANCEL` handler fires first regardless of registration order. Verified by a test that raises both within a single dispatch window.
5. **Masking semantics:** `TIMER` masked via `with session.irq.mask(IrqSource.TIMER):` suppresses dispatch; raised-while-masked interrupts queue. On context exit, queued interrupts fire in priority order. Verified.
6. **Handler grammar swap safety:** A handler that swaps `grammar_state` mid-generation does not corrupt KV cache. Subsequent forward passes use extended history correctly. Verified by comparing token sequences under swap-vs-no-swap with a fixed seed.
7. **Tool result injection via `TOOL_COMPLETE`:** Handler appends result tokens to `token_history`; next forward pass reuses cached prefix; new suffix generated. Cache hit rate verified via existing cache telemetry.
8. **Existing TransitionPolicy parity:** `tests/test_sample.py` and any existing transition-policy tests pass unchanged. Policies become adapters; externally observable behavior is bit-identical with a fixed seed.
9. **BFCL runner parity:** `python benchmarks/run_bfcl.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit --category simple_python --limit 20 --seed 0` produces bit-identical output before and after the refactor. Committed as a regression fixture.
10. **Per-token overhead:** Empty-queue `dispatch_pending()` adds ≤ 50µs per token on M1-class hardware. Measured via micro-benchmark over 10,000 tokens with no registered handlers. Exceeding this budget blocks merge.
11. **Autotuner migration:** The existing per-turn autotuner behavior is preserved when rewired as a `TURN_COMPLETE` handler. Autotune test suite passes with no changes to expected trajectories.
12. **Per-session isolation:** Raising an interrupt on one `SamplingSession`'s controller has no effect on another session running concurrently. Verified via a two-session test with independent event streams.
13. **Hypothesis property tests:** For the priority+masking model, property tests verify (a) masked sources never dispatch while masked, (b) priority order is a total order respected under all enqueue sequences, (c) unmasking releases queued interrupts in priority order.
14. **No public API surface leaks:** `tgirl.__init__` does not export anything from `tgirl.interrupts`; the module is reachable only via full path. Documented as internal, not-for-use-by-consumers.

## 5. Risk Assessment

- **Hot-path overhead (P0).** Dispatch runs between every token. Per-token latency is currently in the ms-per-token regime (MLX timing breakdown at `sample.py:970–979`). A 50µs budget is tight; Python dispatch of an empty deque + priority check must fit. Mitigation: inline fast-path for empty queue (`if not self._pending: return`), bench before merge (AC#10), profile if over budget.
- **Security — sample.py contact.** `CLAUDE.md` marks `sample.py` as strongly recommended for security audit (sampling loop integrity). Cancel semantics that *actually stop forward passes* are a correctness-critical change to that loop. Run `/security-audit` before PR. **Recommend escalating to full tier if any audit finding lands at HIGH.**
- **Grammar swap mid-advance undefined behavior.** If a handler swaps grammar while `grammar_state` holds partial-advance state (mid-token inside a constrained sub-expression), behavior is undefined. Mitigation: handler contract forbids grammar swap during `GRAMMAR_TRIGGER`-to-`GRAMMAR_EXIT` window; assertion in `InterruptController.dispatch_pending` checks invariant.
- **KV cache consistency under injection.** Cache keys on token-list identity (`cache.py:266–297`). Injecting result tokens extends the list → cache hits prefix, re-forwards suffix — safe. Cache is NOT forked in Phase 1; a handler that wants to speculatively append and roll back must not; this is enforced by contract (no rollback primitive).
- **Cross-thread signaling correctness.** `CANCEL` raised from FastAPI event-loop thread lands on sampling thread (running under `asyncio.to_thread`). Use `threading.Event` or a lock-protected deque; avoid `queue.Queue` overhead in the hot path.
- **Autotuner branch coordination.** Two active branches touch similar surface. Risk: merge conflicts, or autotuner assumptions break under the refactor. Mitigation: decide merge order (see Open Question #2) before implementation begins.
- **Provider-comparison test (AC#3) may behave unpredictably.** OpenAI and Anthropic client SDKs differ in how they signal disconnect; hosted API billing may not expose the exact data needed. Mitigation: methodology document explains exactly what was measured and what was inferred. If inferences must be made, state them as inferences in the writeup.
- **Test suite changes propagate.** Transition policies and server integration tests assume today's control flow. Even with behavior preservation, test plumbing touches will be broad. Mitigation: migration task runs after behavior-preserving refactor, test updates in same commit as corresponding code change.
- **Scope creep: "let's add another source."** Phase 1 source list is load-bearing; adding a ninth source is tempting but inflates the audit surface. Defer additions to Phase 2 unless absolutely required for v0.2 narrative.

## 6. Open Questions

1. **Relationship to v0.2 launch.** Option A: full substrate lands in v0.2, launch narrative is "grammar + cancel + principled inference substrate." Option B: CANCEL-only ships for v0.2 launch; remaining sources migrate post-launch. Recommend A if the tight-spec agentic estimate (~1 hour of implementation) holds; B is the conservative fallback. Needs decision before `/execute-team` kicks off.
2. **Autotuner branch strategy.** (a) Merge `feature/per-token-fire-control-autotuner` to `main` first, then build IRQ controller and migrate the autotuner to `TURN_COMPLETE` within this branch; or (b) rebase autotuner branch onto `feature/inference-irq-controller` after substrate lands. (a) is lower-risk; (b) is cleaner history. Pick one.
3. **Priority model rigidity.** Fixed `IntEnum` priorities are simpler; per-handler configurable priority is more flexible. Recommend starting with `IntEnum`. If Phase 2 drivers genuinely need per-handler priority, revisit.
4. **TIMER source mechanism.** Driven by real `asyncio.Task`, or by an internal clock checked inside `dispatch_pending()`? Latter keeps sampling thread self-contained and avoids event-loop involvement on the inference thread. Recommend internal clock.
5. **Observability surface.** Expose an interrupt event stream on the `/telemetry` endpoint so a launch demo can show interrupts firing in real time? Useful for the "see it cancel" demo in the launch writeup; adds complexity. Recommend yes (read-only listener, no mutation).
6. **Cache fork abstraction.** Declare `cache.fork()` in Phase 1 (even if not implemented) so Phase 2 can land it cleanly, or strictly defer? Recommend defer — keep Phase 1 contract narrow.
7. **Handler exceptions.** If a handler raises, what happens? Fail the turn? Log and continue? Propagate to the driver? Recommend: log + convert to `BREAK` + set error state, so the sampling thread exits cleanly and the driver sees a failed turn. Document explicitly.
8. **Interaction with existing confidence-monitor backtrack** (`sample.py:642–677`). Backtrack is within-constrained-decode and does not rollback KV cache; it predates the interrupt controller. Leave untouched in Phase 1; document as orthogonal.

## 7. Out of Scope

- **Cross-session multiplexing / scheduling.** Different problem (scheduler, batching). InterruptController is per-session.
- **Cache forking / speculative rollback.** Phase 2.
- **Public API stability guarantee.** No stability promise in v0.2; external consumers should not import `tgirl.interrupts`. Gate for public stability is v1.0.
- **Preemption within a single forward pass.** Not possible without model-internal changes. All dispatch is between tokens.
- **Model-originated interrupts** (e.g., model self-cancel on low-confidence). Could be added as an `INTERNAL` source later, but not in Phase 1.
- **Multi-handler chaining / composition.** A single priority-ordered dispatch is the Phase 1 surface. Handler pipelines are a Phase 2 concern if they prove useful.
- **Interrupt replay / recording for deterministic debugging.** Nice-to-have, Phase 2.
- **Reworking the existing confidence-monitor backtrack** (`sample.py:642–677`). Orthogonal; it operates within a single constrained decode, not at turn boundaries.
- **Removing `TransitionPolicy` classes.** They become adapters, not deletions. Removal can happen in a later cleanup pass once all call sites are migrated and the interrupt-raising API is proven.
