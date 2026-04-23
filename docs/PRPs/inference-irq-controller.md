# PRP: Inference Interrupt Controller

## Source PRD: docs/PRDs/inference-irq-controller.md
## Date: 2026-04-22

## 1. Context Summary

Build `InterruptController` — a per-session, internal-API component sitting between `SamplingSession` and its generation loop. Drivers (API server, CLI, BFCL runner, autotuner) raise interrupts rather than reaching into session internals. Dispatch happens between tokens (natural yield point). Primary motivation: ship cancel-on-disconnect as load-bearing differentiator for v0.2 launch narrative ("aborted requests actually stop; hosted providers routinely don't"). Secondary: unify five existing ad-hoc hook patterns before a sixth accumulates.

Phase 1 sources: CANCEL, GRAMMAR_EXIT, GRAMMAR_TRIGGER, BUDGET_EXHAUSTED, TOOL_COMPLETE, MODE_SWITCH_REQUEST, TURN_COMPLETE, TIMER. Internal API only; no public stability promise. Handlers synchronous, run between tokens, may mutate `token_history` / `grammar_state` / sampling params; MAY NOT roll back KV cache in Phase 1.

## 2. Codebase Analysis

### Dispatch points (where `dispatch_pending()` is inserted)

- `src/tgirl/sample.py:1024` — freeform per-token loop (after token acceptance, before next iteration)
- `src/tgirl/sample.py:680` — torch constrained per-token loop (after `grammar_state.advance`, before accept check)
- `src/tgirl/sample_mlx.py:700–701` — MLX constrained per-token loop (mirror location)

### Today's ad-hoc integrations that migrate

- `src/tgirl/sample.py:1017` — `transition_policy.evaluate()` → raise `GRAMMAR_TRIGGER` (Task 6)
- `src/tgirl/sample.py:761,786,799–802` — `DelimiterTransitionPolicy` / `LatchedTransitionPolicy` / composition (Task 6)
- `src/tgirl/sample.py:1243` — `run_pipeline(...)` tool dispatch → raise `TOOL_COMPLETE` (Task 8)
- `src/tgirl/sample.py:1304–1311` — result-token injection path stays; `TOOL_COMPLETE` default handler orchestrates
- `src/tgirl/serve.py:835–897` — `/generate` per-request overrides → raise `MODE_SWITCH_REQUEST` (Task 8)
- `src/tgirl/serve.py:1616–1779` — OpenAI SSE handler → raise `CANCEL` on disconnect (Task 4)
- `src/tgirl/serve.py:980–1028` — `/stream` WebSocket → explicit disconnect → `CANCEL` (Task 4)
- `src/tgirl/serve.py:1158–1209, 1347, 1366` — autotuner post-turn hook → `TURN_COMPLETE` handler (Task 7)
- `benchmarks/run_bfcl.py:121–162` — transition-policy factory → adapters raising the corresponding source (Task 6)

### KV cache (transparent to handlers — no code changes)

- `src/tgirl/cache.py:234–306` — `make_mlx_forward_fn` prefix-continuation
- `src/tgirl/cache.py:266` — cache-key is token-list identity; handler-driven `token_history.extend(...)` re-uses cached prefix automatically
- Confirmed safe for: grammar swap, tool-result injection, parameter change (PRD §5)

### Test fixtures to reuse / extend

- `tests/test_sample.py:1–150` — `SessionConfig` fixtures, intervention-merge pattern, mock `forward_fn` shape
- `tests/test_sample_mlx.py` — MLX-native mock forward_fn and session fixtures
- `tests/test_integration_sample.py` — end-to-end session tests
- `tests/test_serve.py` — FastAPI `TestClient` + mocked inference-context patterns
- `tests/test_autotune.py` — per-turn observables fixtures

### Conventions (CLAUDE.md)

- TDD RED → GREEN → REFACTOR → COMMIT per task
- `structlog` for structured logging
- Hypothesis for state-machine property tests
- Zero torch / zero numpy in MLX hot loop
- No cross-framework conversions; no Python-fu on tensor data
- Each PRP task = one atomic commit; test + implementation together
- If test fails, fix the real issue — never weaken tests or mock to pass

### Representative integration pseudocode (`sample.py` freeform loop)

```python
for position in range(max_tokens):
    logits = self._forward_fn(token_history)
    # ... existing sample / accept / telemetry ...
    token_history.append(token_id)

    # NEW: dispatch pending interrupts at natural yield point
    action = self._irq.dispatch_pending(session=self)
    if action == InterruptAction.BREAK:
        break
    if action == InterruptAction.RESTART_TURN:
        continue
    # CONTINUE falls through to next iteration
```

## 3. Implementation Plan

**Test Command:** `pytest tests/ -v`

### Task 1: InterruptController core + Hypothesis property tests

**Files:**
- NEW `src/tgirl/interrupts.py`
- NEW `tests/test_interrupts.py`

**Approach:**
- `IrqSource(IntEnum)` with priorities 0–7 per PRD §2.
- `InterruptAction(Enum)`: `CONTINUE`, `BREAK`, `RESTART_TURN`.
- `@dataclass InterruptPayload(source, data, timestamp)`.
- `HandlerFn` Protocol: `(session, payload) -> InterruptAction`. The `session` arg is supplied by the controller, not the caller of `dispatch_pending` — see binding below.
- `InterruptController`:
  - **Session-bound at construction.** `__init__(self, session: "SamplingSession")` stores `self._session = session`. Handlers receive this session, so `dispatch_pending()` takes no session argument. This resolves the constrained-gen call-site gap (Y3): the module-level `run_constrained_generation[_mlx]` functions only need to receive the `InterruptController` instance, not the session.
  - **Naming:** the session field is `self._irq`. ESTRADIOL's existing `self._controller` is unchanged. All PRP references to "the controller" mean the IRQ controller unless stated otherwise.
  - `collections.deque` for pending queue; `threading.Lock` (plain, not RLock) guarding `_pending`, `_masked`, `_scheduled_timers`, `_next_timer_fire`, and the Task 10 event buffer.
  - `register(source, handler)` / `unregister(source, handler)` — list per source; registration-order is tiebreak within priority.
  - `mask(source)` → context manager; suppresses dispatch while active; raised-while-masked interrupts queue. Mutations to `_masked` happen under `self._lock`. The mask set is snapshotted at dispatch start (under lock); changes during handler execution take effect on the *next* dispatch, by design — prevents re-entrant mask surprises.
  - `raise_interrupt(source, data=None)` — thread-safe; respects current mask set; acquires `self._lock` briefly to append to `_pending`.
  - `dispatch_pending() -> InterruptAction` — two-phase:
    1. **Fast-path (lockless, common case):** `now = time.monotonic(); if not self._pending and (self._next_timer_fire is None or now < self._next_timer_fire): return CONTINUE`. This preserves AC#10 budget while making TIMER (Task 9) semantically sound (Y4).
    2. **Slow-path (under lock):** acquire `self._lock`; fire any due scheduled timers into `_pending` (recompute `_next_timer_fire`); drain `_pending`; filter by snapshotted mask set; sort by `(priority, registration_order)`; build `work: list[tuple[InterruptPayload, list[HandlerFn]]]`; append to event buffer; **release the lock**. Then invoke handlers outside the lock (Y5):
       ```python
       action = CONTINUE
       for payload, handlers in work:
           for h in handlers:
               try:
                   a = h(self._session, payload)
               except Exception as e:
                   log.exception("interrupt.handler_error", source=payload.source, error=str(e))
                   self._session.telemetry.record_handler_error(payload.source, e)
                   a = InterruptAction.BREAK
               action = self._fold(action, a)  # BREAK > RESTART_TURN > CONTINUE
       return action
       ```
    Handlers MAY call `self.raise_interrupt` / `self.mask` during their execution — the lock is released, so these calls are safe. Handlers MUST NOT block indefinitely; no explicit handler-timeout in Phase 1 (tracked as uncertainty).
  - Handler exception policy: log via structlog + annotate session telemetry + return `BREAK` (per PRD OQ#7). Exception inside a handler does NOT prevent dispatch of remaining same-priority handlers in the current batch — the fold continues, but the terminal action is BREAK.
  - `_scheduled_timers` and `_next_timer_fire` are declared here (used by Task 9) so the Y4 fix is structural, not retrofitted. Task 9 populates `schedule()`/reschedule logic; Task 1 wires the fast-path and slow-path to honor an empty `_scheduled_timers` correctly (`_next_timer_fire = None`).
- `__init__.py` does NOT export from `tgirl.interrupts` (AC#14: no public API surface).

**Tests (RED first):**
- `test_raise_and_dispatch_single_source` — raise CANCEL, dispatch, handler fires, action observed.
- `test_priority_ordering` — raise TIMER before CANCEL, CANCEL handler fires first regardless of registration order (AC#4).
- `test_masking_suppresses_dispatch` — mask TIMER, raise TIMER, dispatch → no fire (AC#5).
- `test_unmask_releases_queued_in_priority_order` — mask TIMER, raise TIMER + MODE_SWITCH_REQUEST, unmask, dispatch → MODE_SWITCH_REQUEST fires first (priority 5 < TIMER 7).
- `test_handler_exception_returns_break_and_logs` — handler raises; dispatch returns BREAK; structlog event captured.
- `test_two_controllers_isolated` — raise on A, dispatch on B, no effect (AC#12).
- `test_handler_may_raise_interrupt_during_dispatch` — handler calls `self_irq.raise_interrupt(...)` inside its body; assert no deadlock (proves lock-released-before-invoke).
- `test_handler_may_call_mask_during_dispatch` — handler enters `mask(X)` context; assert no deadlock and that X's mask change takes effect on next dispatch, not the current one (snapshot semantics).
- `test_fast_path_with_unused_timer_scheduled` — schedule TIMER 10s from now; empty queue; dispatch → returns CONTINUE within AC#10 budget (covers Y4 fast-path branch cost).
- **Hypothesis stateful (concurrent):** N raiser threads + 1 dispatch thread (e.g., 4 raisers raising 1000 sources each). Invariants: (a) every raised non-masked source is eventually dispatched exactly once; (b) no dispatch fires a masked source; (c) priority ordering holds *within each dispatch batch* (global priority ordering is not required — late-landing raises may fire in a subsequent dispatch). (Y5)
- Hypothesis: raising N arbitrary sources then dispatching → handler-fire order is non-decreasing by priority (AC#13b).
- Hypothesis: `mask S; raise S × n; unmask S` → exactly n dispatches in FIFO order (AC#13a).

**Validation:** `pytest tests/test_interrupts.py -v`

**Commit:** `feat(interrupts): add InterruptController core`

---

### Task 2: Wire controller into torch sample.py loops

**Files:**
- `src/tgirl/sample.py`
- `tests/test_sample.py`

**Approach:**
- Add `self._irq: InterruptController` to `SamplingSession` (default-constructed in `__init__` as `InterruptController(session=self)` — note Y3 resolution: session is bound to the controller, not passed through dispatch).
- **API asymmetry (documented):** the freeform loop lives inside `SamplingSession.run()` and calls `self._irq.dispatch_pending()` directly. The constrained loops live in module-level `run_constrained_generation[_mlx]` functions (sample.py:485, sample_mlx.py:420), which have no `self`. These functions gain a new keyword argument `irq: InterruptController | None = None`, passed down from `SamplingSession` at sample.py:1214 and sample.py:1228. Naming: the parameter is `irq`, NOT `controller` — the `controller` kwarg already exists for ESTRADIOL steering (sample.py:499, sample_mlx.py:434) and must not be overloaded.
- Freeform loop dispatch at `sample.py:1024` (after token append + telemetry update): call `action = self._irq.dispatch_pending()`, honor `BREAK` / `RESTART_TURN`.
- Constrained loop dispatch at `sample.py:680` (after `grammar_state.advance`, before accept check): `action = irq.dispatch_pending() if irq is not None else InterruptAction.CONTINUE`. Same action-honoring pattern.
- **Do NOT remove `TransitionPolicy.evaluate` call at line 1017 yet** — that is Task 6 (behavior-preserving migration). This task only adds the dispatch point.

**Tests (RED first):**
- `test_cancel_breaks_torch_freeform_within_one_forward` — register handler returning BREAK; start run in a thread; raise CANCEL from test thread; assert `tokens_generated <= tokens_at_raise + 1` (AC#1).
- `test_cancel_breaks_torch_constrained_within_one_forward` — same for constrained loop.
- `test_restart_turn_action_restarts` — handler returns `RESTART_TURN`; loop re-enters at top.
- Regression: all existing `test_sample.py` tests pass unchanged.

**Validation:** `pytest tests/test_sample.py tests/test_interrupts.py -v`

**Commit:** `feat(sample): dispatch interrupts in torch generation loops`

---

### Task 3: Wire controller into MLX sample_mlx.py loops

**Files:**
- `src/tgirl/sample_mlx.py`
- `tests/test_sample_mlx.py`

**Approach:**
- Symmetric to Task 2 at `sample_mlx.py:700–701`.
- Signature addition: `run_constrained_generation_mlx(..., controller: object | None = None, irq: "InterruptController | None" = None)`. The `controller` kwarg stays untouched (ESTRADIOL at sample_mlx.py:471-496).
- Dispatch point mirrors the torch constrained loop exactly: `action = irq.dispatch_pending() if irq is not None else InterruptAction.CONTINUE` after grammar-state advance.
- **Invariant:** zero torch, zero numpy in hot loop (CLAUDE.md). Controller dispatch is pure Python; does not touch `mx.array`.

**Tests (RED first):**
- `test_cancel_breaks_mlx_constrained_within_one_forward` — MLX-native version of AC#1.
- Regression: existing `test_sample_mlx.py` passes.

**Validation:** `pytest tests/test_sample_mlx.py tests/test_interrupts.py -v`

**Commit:** `feat(sample_mlx): dispatch interrupts in MLX generation loop`

---

### Task 4a: Route `/v1/chat/completions` (streaming + non-streaming) through `SamplingSession`

**Why this exists (Y1 + Y2 root cause):** the current `chat_completions` handler has its own ad-hoc MLX loop at `serve.py:1619–1779` (streaming, `stream_gen`) and `serve.py:1254–1366` (non-streaming, `_generate_tokens`). Neither path instantiates `SamplingSession`. Task 4b's "install CANCEL handler on the session" and Task 7's "TURN_COMPLETE handler at session creation" both have no session to attach to. Rather than decoupling `InterruptController` from `SamplingSession` (which breaks AC#12 isolation), we collapse both ad-hoc loops into the unified sampling engine.

**Files:**
- `src/tgirl/sample.py` (may gain a streaming token-iterator API on `SamplingSession` — see step 1)
- `src/tgirl/serve.py`
- `tests/test_serve.py`
- NEW `tests/regression/fixtures/openai_stream_fixed_seed.json` (golden fixture for bit-parity)

**Approach:**
1. **Verify whether `SamplingSession` already exposes a streaming token iterator.** If yes, reuse. If no (likely — `.run()` returns `SamplingResult`), add a minimal `SamplingSession.stream(prompt_tokens) -> Iterator[StreamEvent]` that yields per-token events with (token_id, probe, certainty, correction_norm, finish_reason_or_None). Test-first, behavior-identical to `.run()` for equivalent inputs (the non-streaming path can be rewritten as `list(session.stream(...))` post-refactor; if that rewrite fails parity, the bug is in `stream`).
2. **Extract `_run_session_chat_stream(ctx, messages, session_config, transport_config, hooks, request_overrides) -> AsyncIterator`** in `serve.py`. Internally: build SamplingSession, resolve per-request overrides (alpha, temp, beta, skew, seed, stop_ids) into config, iterate `session.stream()`, yield serializable per-token events. Extracted for mockability (mirrors existing `_run_session_chat` at L589).
3. **Refactor `stream_gen` (L1619-1774)** into a thin translator: consume `_run_session_chat_stream`, translate each event into `ChatCompletionChunk`, preserve emit-reasoning-vs-content branching, preserve `<think>` skip logic, preserve `[DONE]` trailer.
4. **Refactor `_generate_tokens` (L1254-1366)** into a thin wrapper: `async def _generate_tokens(request, prompt_tokens): tokens = []; async for ev in _run_session_chat_stream(...): tokens.append(ev.token_id); ...; return tokens, finish_reason`. Preserve finish_reason semantics, probe_cache writes, steering_stats writes, certainty/correction aggregation.
5. **Preserve ordering invariants (Y2):**
   - `_probe_cache["v_probe"]` write happens at turn end, before return (today: L1335/1353 for non-streaming, L1691/1755 for streaming). In the refactor, this write lives in the `finally` of the stream generator OR in a `TURN_COMPLETE` default handler registered on the session in Task 7. Task 4a keeps it inline (near the current call sites) to preserve bit-parity; Task 7 later migrates it into the handler with explicit ordering guarantees.
   - `_steering_stats["last_*"]` and `_run_autotune_after_turn(...)` calls stay at the same logical moment (turn end) — Task 4a does NOT migrate them yet. Autotuner migration is Task 7.
6. **Commit-size escape hatch:** if net diff exceeds ~400 LoC, split into 4a-i (SamplingSession streaming API) and 4a-ii (serve.py route through it). Decide at implementation time.

**Tests (RED first):**
- `test_openai_stream_bit_parity` — fixed-seed deterministic prompt, capture full token stream + finish_reason + `_steering_stats` dict pre-refactor (committed as `openai_stream_fixed_seed.json` fixture with token list, reasonings, chunk count, final `_steering_stats` state hash, finish_reason). Post-refactor: identical. Fixture captured BEFORE any code changes — commit it as the first substep of Task 4a.
- `test_generate_tokens_nonstream_bit_parity` — same prompt, non-streaming path, same token list + finish_reason.
- `test_per_request_override_alpha_still_applied` — set alpha=0.2 on request, assert steering_stats reflects it.
- `test_per_request_seed_reproducible` — seed=42 on two sequential requests, identical token streams.
- Regression: all existing `test_serve.py` tests pass unchanged.

**Uncertainty note:** the OpenAI SSE handler currently does NOT support grammar constraints, tool calling, transport (OT), or hooks — it's a probe-steered raw-sampling path. Routing it through `SamplingSession` *could* silently activate those paths if the session defaults enable them. Mitigation: confirm the shared `InferenceContext` initialized in `create_app` has null defaults for those paths when used for OpenAI endpoints (or: `_run_session_chat_stream` builds a session with explicit grammar/tool disabled). Document in uncertainty log.

**Validation:** `pytest tests/test_serve.py tests/regression/ -v`

**Commit:** `refactor(serve): route /v1/chat/completions through SamplingSession (prep for IRQ)`

---

### Task 4b: FastAPI SSE + WebSocket cancel-on-disconnect

**Files:**
- `src/tgirl/serve.py`
- `tests/test_serve.py`

**Prerequisite:** Task 4a landed — both OpenAI paths now instantiate `SamplingSession`, so `session.irq` exists at the disconnect-watcher boundary.

**Approach:**
- SSE handler (`serve.py:1616–1779`, post-4a):
  - Install a default CANCEL handler on the session that finalizes telemetry and returns `BREAK`.
  - Start `disconnect_watcher = asyncio.create_task(_watch_disconnect(request, session))`. Watcher polls `await request.is_disconnected()` every 50ms; on disconnect calls `session.irq.raise_interrupt(IrqSource.CANCEL)` from the event-loop thread into the sampling thread. This is the Y5-guarded cross-thread path — `raise_interrupt` is lock-safe.
  - Cancel the watcher in `finally` on normal completion.
- WebSocket handler (`serve.py:980–1028`): replace implicit exception-swallow with explicit `try: await ws.receive() except WebSocketDisconnect: session.irq.raise_interrupt(IrqSource.CANCEL)` loop. The `_stream` endpoint already creates a SamplingSession via `_run_session_chat`, but `.irq` needs to be surfaced — adjust `_run_session_chat` (or add a `_run_session_chat_with_session_handle` variant) so the caller can raise interrupts on the in-flight session.
- Thread-safety: `raise_interrupt` is lock-guarded (Task 1, Y5). AC#1 expected latency: ≤1 forward pass in the common case; ≤2 forward passes under adversarial cross-thread raise/dispatch contention (see AC#1 revision note).

**Tests (RED first):**
- Integration test using `httpx.AsyncClient` (or the `openai` Python client): open streaming `/v1/chat/completions`, close client mid-stream, assert the server's mock `forward_fn` was called at most `N+2` times where `N` was the token count at disconnect (AC#1 revised bound). Use a mocked inference-context to count.
- Manual smoke (documented as validation step): `tgirl serve &`; curl with `timeout 1`; verify structlog shows CANCEL dispatch within ≤2 forward passes.

**Validation:** `pytest tests/test_serve.py -v` + manual smoke.

**Commit:** `feat(serve): raise CANCEL interrupt on client disconnect`

---

### Task 5: Provider-comparison cancel-behavior benchmark (launch evidence)

**Files:**
- NEW `benchmarks/launch/cancel-behavior/compare.py`
- NEW `benchmarks/launch/cancel-behavior/README.md`
- NEW `benchmarks/launch/cancel-behavior/results.json` (populated by run)

**Approach:**
- Three backends: `local-tgirl` (localhost:8420), `openai` (official SDK), `anthropic` (official SDK).
- For each: issue a deterministic streaming request that generates ≥ 500 tokens with `temperature=0`; force client disconnect at token 10 (or ≈200ms); record
  - `tokens_received_by_client` (observed)
  - `tokens_billed` if provider exposes it via usage endpoint; else mark `inferred` and document the inference
  - `wall_clock_disconnect_to_stop` (how long the local model continued; for remote providers, inferred from billing or from subsequent usage delta)
- Methodology writeup: exactly what was measured vs inferred. Credits required; script exits cleanly if env vars missing.
- Output committed as canonical launch evidence (AC#3). No public writeup references cancel-behavior without citing this file.

**Tests:**
- Unit: parsing of each provider's usage response; computing deltas.
- Live run: the `results.json` file IS the test output; committed to repo.

**Validation:** `python benchmarks/launch/cancel-behavior/compare.py --providers local,openai,anthropic` produces `results.json`.

**Commit:** `bench: add cancel-behavior provider comparison (launch evidence)`

---

### Task 6: Migrate TransitionPolicy → GRAMMAR_TRIGGER / BUDGET_EXHAUSTED adapters

**Files:**
- `src/tgirl/sample.py`
- `src/tgirl/state_machine.py` (verify class location first)
- `tests/test_sample.py`
- `benchmarks/run_bfcl.py`
- NEW `tests/regression/bfcl_fixtures/qwen08b_simple_python_20_seed0.json` (golden fixture)

**Approach:**
- **Capture golden fixture FIRST:** before any refactor, run `benchmarks/run_bfcl.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit --category simple_python --limit 20 --seed 0` and commit the token-level output as fixture. This is the bit-identity target (AC#9).
- `DelimiterTransitionPolicy`: when match triggers, raise `GRAMMAR_TRIGGER` with `{'delimiter': matched}`. Session installs a default handler that performs existing mode-switch logic.
- `ImmediateTransitionPolicy`: raise `GRAMMAR_TRIGGER` on first dispatch; same handler.
- `BudgetTransitionPolicy`: raise `BUDGET_EXHAUSTED` with `{'budget': N, 'consumed': k}`. Default handler forces transition.
- At `sample.py:1017`: `evaluate()` now raises (or returns signal that the loop raises); loop no longer branches on return value.

**Tests (RED first):**
- All existing transition-policy tests pass unchanged (AC#8).
- BFCL bit-parity test: run the same command → diff against golden fixture → must be identical (AC#9).

**Validation:** `pytest tests/test_sample.py tests/test_state_machine.py tests/regression/ -v`

**Commit:** `refactor(sample): migrate TransitionPolicy to interrupt-raising adapters`

---

### Task 7: Migrate autotuner → TURN_COMPLETE handler

**Prerequisite:** Task 4a landed — the OpenAI paths now route through `SamplingSession`, so `TURN_COMPLETE` can be raised inside the session loop and all four current call sites disappear.

**Files:**
- `src/tgirl/sample.py` (or wherever `SamplingSession.run()` / `SamplingSession.stream()` end-of-turn fires)
- `src/tgirl/serve.py`
- `tests/test_serve.py`
- `tests/test_autotune.py`
- NEW `tests/regression/fixtures/autotune_3turn_trajectory.jsonl` (pre-refactor trajectory)

**Approach:**
- **Four current call sites to eliminate** (Y2 — the PRP previously cited only two):
  - `serve.py:1347` — `_run_autotune_after_turn("stop")` at `_generate_tokens` stop-token branch
  - `serve.py:1365` — `_run_autotune_after_turn("length")` at `_generate_tokens` length exit (NB: previous PRP cited L1366 — off by one)
  - `serve.py:1702` — `_run_autotune_after_turn("stop")` at `stream_gen` stop branch
  - `serve.py:1766` — `_run_autotune_after_turn("length")` at `stream_gen` length exit
  Post-4a, all four collapse into a single `TURN_COMPLETE` raise inside `SamplingSession` at turn-end, before control returns to the FastAPI handler.
- **Handler capture of closure state.** `_run_autotune_after_turn` (serve.py:1158) is a closure capturing `_autotune_state`, `_steering_stats`, `_steering_config`, `_probe_cache` in `create_app`'s lexical scope. The handler registration uses explicit partial application to preserve this:
  ```python
  from functools import partial
  def _autotune_handler(session, payload, *, state, stats, config, cache):
      finish_reason = payload.data.get("finish_reason", "unknown")
      _run_autotune(finish_reason, state=state, stats=stats, config=config, cache=cache)
      return InterruptAction.CONTINUE

  session.irq.register(
      IrqSource.TURN_COMPLETE,
      partial(_autotune_handler, state=_autotune_state,
              stats=_steering_stats, config=_steering_config, cache=_probe_cache),
  )
  ```
  `_run_autotune_after_turn` is renamed `_run_autotune` and made a module-level function taking explicit `state`, `stats`, `config`, `cache` params. No global state introduced.
- **Mutation-ordering invariant (Y2 AC#11 addendum):** `_autotune_state["next_alpha"]`, `next_beta`, `next_temperature`, `last_regime`, `last_rationale` are read by the *next* request's `_resolve_alpha` / `_resolve_temperature` (serve.py:1152 pattern). Today, these writes happen inline *before* `return generated, "stop"` (L1347→L1348). Post-refactor: `TURN_COMPLETE` is dispatched synchronously at the SamplingSession turn-end, before control returns to the async handler — preserving write-before-next-read.
- Handler returns `CONTINUE`; autotuner state progression is idempotent w.r.t. the fold order.
- The `_probe_cache["v_probe"]` and `_steering_stats["last_*"]` writes from Task 4a (inline at turn end) may optionally be migrated into the same TURN_COMPLETE handler for consistency, but this is not required — they are single-reader / single-writer under session ownership.

**Tests (RED first):**
- **Capture golden trajectory FIRST (before any Task 7 edit):** run a scripted 3-turn conversation pre-refactor with `log_path` set, capture the autotune JSONL output + final `_autotune_state` snapshot. Commit as fixture.
- Autotuner regression suite passes unchanged (AC#11).
- `test_autotune_trajectory_bit_identical_after_refactor` — run the same 3-turn conversation post-refactor; JSONL lines + final `_autotune_state` must match byte-for-byte.
- `test_write_before_next_read_invariant` — two sequential SSE requests with autotune enabled; assert the second request's `_resolve_alpha`/`_resolve_temperature` reads the `next_*` values written by the first request's TURN_COMPLETE dispatch.
- Integration test: multi-turn request, autotune observables match pre-refactor values for a fixed input.

**Validation:** `pytest tests/test_autotune.py tests/test_serve.py tests/regression/ -v`

**Commit:** `refactor(serve): migrate autotuner to TURN_COMPLETE handler`

---

### Task 8: TOOL_COMPLETE + MODE_SWITCH_REQUEST sources

**Files:**
- `src/tgirl/sample.py`
- `src/tgirl/serve.py`
- `tests/test_sample.py`
- `tests/test_serve.py`

**Approach:**
- Tool dispatch at `sample.py:1243`: after `run_pipeline(...)`, raise `TOOL_COMPLETE` with `{'result': ..., 'error': ...}`. Default handler performs existing result-token injection (lines 1304–1311) and returns `CONTINUE`.
- `/generate` per-request overrides at `serve.py:835–897`: before `session.run()`, raise `MODE_SWITCH_REQUEST` with the overrides dict. Default handler applies `restrict_tools`, `scopes`, `ot_epsilon`, `base_temperature`, `max_cost` to session state. Behavior-preserving.

**Tests (RED first):**
- Tool injection bit-parity: fixed-seed tool pipeline produces identical result tokens.
- Mode-switch parity: existing `/generate` override tests pass unchanged.

**Validation:** `pytest tests/test_sample.py tests/test_serve.py -v`

**Commit:** `refactor: migrate tool dispatch and /generate overrides to interrupts`

---

### Task 9: TIMER source via internal clock

**Files:**
- `src/tgirl/interrupts.py`
- `tests/test_interrupts.py`

**Approach:**
- Task 1 already wired the fast-path to honor `_next_timer_fire` (Y4). This task populates the scheduling surface.
- Add `schedule(source, interval_ms, data=None)` to `InterruptController`:
  - Appends `(source, interval_ms, data, next_fire_monotonic)` to `self._scheduled_timers` under `self._lock`.
  - Recomputes `self._next_timer_fire = min(t.next_fire_monotonic for t in self._scheduled_timers)` under the same lock.
- In the slow-path of `dispatch_pending`, at the start of the locked section: for each scheduled timer where `now >= t.next_fire_monotonic`, synthesize an `InterruptPayload(source=t.source, data=t.data, timestamp=now)` and append to `_pending`; set `t.next_fire_monotonic = now + t.interval_ms / 1000.0` (re-arm). Recompute `_next_timer_fire` afterwards. Then proceed with drain/fold.
- Self-contained: no asyncio.Task, no event-loop coupling on sampling thread (PRD OQ#4).

**Tests (RED first):**
- `test_timer_fires_after_interval` — schedule 10ms, monkeypatch `time.monotonic`, dispatch → fires.
- `test_timer_does_not_fire_early` — dispatch before interval → no fire.
- `test_timer_reschedules_after_fire` — fire, advance clock, dispatch again → fires again.
- **`test_timer_fires_on_empty_queue_after_interval` (Y4 critical case):** schedule 10ms, monkeypatch clock forward 15ms, dispatch with empty `_pending` → fires. This is the exact silent-bug case the Y4 revision guards against.
- `test_multiple_timers_use_earliest_next_fire` — schedule two timers with 5ms and 50ms intervals; `_next_timer_fire` tracks the 5ms one; after 5ms only that timer fires.

**Validation:** `pytest tests/test_interrupts.py -v`

**Commit:** `feat(interrupts): add TIMER source with internal clock`

---

### Task 10: Observability — interrupt event stream on /telemetry

**Files:**
- `src/tgirl/interrupts.py`
- `src/tgirl/serve.py`
- `tests/test_serve.py`

**Approach:**
- `InterruptController` records a `collections.deque(maxlen=1024)` of `(timestamp, source, action)` tuples on every dispatch.
- Extend existing `/telemetry` endpoint with `interrupt_events` field returning the most recent events.
- Read-only; no mutation path.

**Tests (RED first):**
- Raise and dispatch an interrupt; GET /telemetry shows the event.
- Circular-buffer invariant: after 1025 dispatches, only most recent 1024 visible.

**Validation:** `pytest tests/test_serve.py -v`

**Commit:** `feat(serve): expose interrupt event stream on /telemetry`

---

### Task 11: Per-token overhead micro-benchmark + final regression guard

**Files:**
- NEW `benchmarks/perf/irq_overhead.py`
- NEW `tests/regression/test_bfcl_bit_identical.py`
- NEW `tests/regression/test_irq_overhead.py`
- NEW `benchmarks/perf/irq_overhead_results.json` (committed with latest run)

**Approach (Y6 — three-variant bench + autotuner probe):**
- **`bench_empty_no_handlers`** — AC#10 baseline: 10,000 dispatches, no handlers registered. Budget: **50µs/call** (500ms total). Catches regressions in the lockless fast-path.
- **`bench_empty_with_handlers`** — 10,000 dispatches with a realistic production handler table registered (CANCEL, GRAMMAR_TRIGGER, BUDGET_EXHAUSTED, TOOL_COMPLETE, MODE_SWITCH_REQUEST, TURN_COMPLETE — 6+ handlers, none raised). Budget: **50µs/call**. Guarantees the fast-path ignores handler-table size.
- **`bench_one_raise_per_100_tokens`** — 10,000 dispatches with 100 synthetic `GRAMMAR_TRIGGER` raises interleaved (≈1 per 100), trivial handler returning `CONTINUE`. Budget: **500µs/call amortized** (absorbs lock acquisition + handler invocation + fold). Catches regressions in slow-path cost.
- **`bench_turn_complete_handler_per_invocation`** — single-shot cost of the real autotuner `TURN_COMPLETE` handler end-to-end (Observables build + `autotune()` + JSONL append). Observed, NOT budget-gated; emits `warnings.warn` if > 1ms.
- BFCL bit-parity regression: compare `benchmarks/run_bfcl.py --limit 20 --seed 0` output against golden fixture captured in Task 6 (AC#9).
- Bench results (all four variants + system metadata — macOS version, chip model, Python version, git sha) committed to `benchmarks/perf/irq_overhead_results.json` as launch evidence and regression floor.

**Tests:**
- `tests/regression/test_irq_overhead.py::test_empty_no_handlers_budget` — asserts < 50µs/call.
- `tests/regression/test_irq_overhead.py::test_empty_with_handlers_budget` — asserts < 50µs/call.
- `tests/regression/test_irq_overhead.py::test_one_raise_per_100_tokens_budget` — asserts < 500µs/call amortized.
- `tests/regression/test_irq_overhead.py::test_turn_complete_handler_observed` — measures and warns; does not fail.
- BFCL parity test is a fixture-comparison pytest.

**AC#10 tightening (PRP ≥ PRD):** PRD AC#10 currently specifies only the empty-no-handlers budget. This PRP guards loaded-dispatch as well. PRD revision recommended in follow-up (see Uncertainty Log).

**Validation:** `pytest tests/regression/ -v` + `python benchmarks/perf/irq_overhead.py`

**Commit:** `test(perf): three-variant dispatch overhead bench + BFCL bit-parity regression`

## 4. Validation Gates

```bash
# Syntax / style
ruff check src/ tests/ --fix && mypy src/

# Unit + Hypothesis tests
pytest tests/ -v --cov=src/tgirl

# Regression: BFCL bit-identity pre/post refactor
pytest tests/regression/ -v

# Perf budget: per-token dispatch overhead < 50µs on M1
python benchmarks/perf/irq_overhead.py

# Manual SSE cancel smoke test
tgirl serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit &
SERVER_PID=$!
sleep 2
curl -N http://localhost:8420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"...", "messages":[{"role":"user","content":"write a poem"}], "stream":true}' &
CURL_PID=$!
sleep 0.5 && kill $CURL_PID
# Verify structlog output shows `interrupt.dispatched source=CANCEL` within ≤1 forward pass
kill $SERVER_PID

# Launch evidence (requires API keys in env)
python benchmarks/launch/cancel-behavior/compare.py --providers local,openai,anthropic
cat benchmarks/launch/cancel-behavior/results.json
```

## 5. Rollback Plan

Cancel semantics are correctness-critical, and `sample.py` is on the CLAUDE.md "strongly recommended for security audit" list. Rollback granularity:

- Each task is one commit → revert-range granularity is per-task.
- Per-task failure → `git revert <sha>` for that task alone; substrate (Task 1) and CANCEL integration (Tasks 2–4) stay if the failure is downstream.
- Full-feature revert → revert in reverse order (Task 11 → Task 1); the feature branch can be abandoned.
- Perf budget failure at Task 11 → do not merge; investigate hot path and optimize, or isolate the offending handler.
- If `/security-audit` produces a HIGH+ finding, tier escalates to full; address finding before merge.

## 6. Uncertainty Log

- **TransitionPolicy class locations.** PRD anchors `DelimiterTransitionPolicy` / `LatchedTransitionPolicy` imports at `sample.py:761, 786` but implementations may live in `src/tgirl/state_machine.py`. Verify at start of Task 6; update file list accordingly.
- **`request.is_disconnected()` polling interval.** 50ms is a guess; OpenAI / Anthropic may detect client disconnect faster. Task 5 methodology should capture the actual median-detection time.
- **Provider usage-endpoint granularity.** OpenAI's `/v1/usage` and Anthropic's Admin API may not expose per-request billed-token counts in real time. If not, Task 5 methodology must state that billing-side evidence is inferential (delta of account-level usage across the test) and make the inference explicit in results.
- **TIMER internal clock vs asyncio.** Recommended internal clock in PRD OQ#4. If Task 9 reveals that forward passes on large MLX models can exceed TIMER interval for extended stretches (e.g., 500ms interval but 1s forward pass), revisit: either coarser TIMER semantics documented, or move TIMER to its own thread.
- **Autotuner branch merge order.** PRP assumes `feature/per-token-fire-control-autotuner` merges to `main` before Task 7 runs. If the other order is chosen, Task 7 becomes a cross-branch rebase — coordinate before starting.
- **Security audit.** Strongly recommend running `/security-audit` before the PR given sample.py contact. Not strictly blocking for standard tier, but flagging explicitly.
- **BFCL bit-parity tolerance.** "Bit-identical" assumes deterministic sampling with a fixed seed. If any non-deterministic ordering exists in the current loop (e.g., dict-iteration order affecting registry serialization), Task 6 will surface it as the parity test fails — in that case, document the non-determinism and relax to sequence-equivalence with justification.

### Uncertainties added during training-partner review (2026-04-22)

- **Task 4a scope creep risk (Y1/Y2).** The OpenAI SSE handler currently does not support grammar constraints, tool calling, transport (OT), or hooks — it's a probe-steered raw-sampling path. Routing it through `SamplingSession` *could* silently activate those paths if session defaults differ. Mitigation in Task 4a: explicitly disable grammar/tool paths when building the session for OpenAI endpoints, and prove equivalence with the bit-parity golden fixture. If 4a's net diff exceeds ~400 LoC, split into 4a-i (add `SamplingSession.stream()` API) + 4a-ii (route serve.py through it).
- **SamplingSession streaming API existence.** Task 4a assumes `SamplingSession` either has or can gain an `Iterator[StreamEvent]` API compatible with both `.run()` semantics and per-token event emission. If the current architecture can't be extended without cross-layer changes (e.g., constrained loops can't be made generator-friendly), Task 4a expands to add that API behavior-preservingly.
- **Handler timeout / deadline.** Task 1 does not specify a timeout on handler execution. A pathological handler could block the sampling thread indefinitely. Phase 1 accepts this risk (handlers are internal and audited); Phase 2 or post-merge hardening may add per-source handler deadline with structlog warning. Flagged for /review-code attention.
- **PRD AC#10 is looser than PRP AC#10.** Y6 revision tightens the PRP to guard three bench variants (empty-no-handlers, empty-with-handlers, loaded-dispatch). PRD currently specifies only the first. Recommended follow-up: update PRD AC#10 to match PRP — OR document in the PRP commit message that PRP intentionally tightens PRD.
- **AC#1 bound revision (Y5).** The PRD specifies CANCEL latency ≤ 1 forward pass. Under adversarial cross-thread raise/dispatch contention, a late-landing raise can fire on the *next* dispatch (one additional forward pass). PRP revises AC#1 to "≤ 1 forward pass in the common case; ≤ 2 under adversarial contention" and tests assert the ≤ 2 bound. PRD revision recommended to match.
- **Autotuner closure state migration (Y2).** `_autotune_handler` uses `functools.partial` to capture `_autotune_state` / `_steering_stats` / `_steering_config` / `_probe_cache` dicts by reference. If future work wants to make these per-session (rather than per-app-instance), the handler registration must be updated accordingly. Document for Task 7 reviewer.
- **`_next_timer_fire` fast-path cost (Y4).** Adding one `time.monotonic()` call per `dispatch_pending` invocation on the hot path. Task 11 bench variant `bench_empty_no_handlers` includes this cost implicitly; `bench_empty_with_handlers` includes it too. If either exceeds 50µs, investigate: consider caching `now` per token inside the sampling loop instead of re-computing per dispatch.

### Yield points accepted from training-partner review

- **Y1 — SSE bypasses SamplingSession (STRUCTURAL):** Task 4 split into Task 4a (refactor) + Task 4b (cancel-on-disconnect).
- **Y2 — Autotuner migration lacks session + misses call sites (STRUCTURAL):** Task 7 revised with all four call sites, `functools.partial` closure capture, write-before-return ordering invariant + regression fixture.
- **Y3 — `controller` param collision (STRUCTURAL):** InterruptController binds session at construction; `HandlerFn` receives session via controller, not via dispatch arg; module-level gen functions take new `irq=` kwarg; `self._irq` naming disambiguates from ESTRADIOL's `self._controller`.
- **Y4 — Fast-path breaks TIMER (MODERATE):** Fast-path conditioned on `_next_timer_fire`; Task 9 test `test_timer_fires_on_empty_queue_after_interval` added.
- **Y5 — Drain-side thread safety (MODERATE):** Snapshot-under-lock / invoke-handlers-outside-lock pattern; mask set snapshotted; Task 10 event buffer reads snapshot-copy under lock. AC#1 bound revised to "≤ 2 forward passes under adversarial contention".
- **Y6 — Perf budget measures the wrong thing (MODERATE):** Task 11 replaced with three-variant bench + autotuner-per-invocation observer.
