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
- `HandlerFn` Protocol: `(session, payload) -> InterruptAction`.
- `InterruptController`:
  - Per-session. `collections.deque` for pending queue, `threading.Lock` for `raise_interrupt` (cross-thread: FastAPI event-loop → sampling thread).
  - `register(source, handler)` / `unregister(source, handler)` — list per source; registration-order is tiebreak within priority.
  - `mask(source)` → context manager; suppresses dispatch while active; raised-while-masked interrupts queue.
  - `raise_interrupt(source, data=None)` — thread-safe; respects current mask set.
  - `dispatch_pending(session) -> InterruptAction` — fast-path `if not self._pending: return CONTINUE`; otherwise drain in priority order, call each handler, fold actions via `BREAK > RESTART_TURN > CONTINUE`.
  - Handler exception policy: log via structlog + annotate session telemetry + return `BREAK` (per PRD OQ#7).
- `__init__.py` does NOT export from `tgirl.interrupts` (AC#14: no public API surface).

**Tests (RED first):**
- `test_raise_and_dispatch_single_source` — raise CANCEL, dispatch, handler fires, action observed.
- `test_priority_ordering` — raise TIMER before CANCEL, CANCEL handler fires first regardless of registration order (AC#4).
- `test_masking_suppresses_dispatch` — mask TIMER, raise TIMER, dispatch → no fire (AC#5).
- `test_unmask_releases_queued_in_priority_order` — mask TIMER, raise TIMER + MODE_SWITCH_REQUEST, unmask, dispatch → MODE_SWITCH_REQUEST fires first (priority 5 < TIMER 7).
- `test_handler_exception_returns_break_and_logs` — handler raises; dispatch returns BREAK; structlog event captured.
- `test_two_controllers_isolated` — raise on A, dispatch on B, no effect (AC#12).
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
- Add `irq: InterruptController` to `SamplingSession` (default-factory-constructed per session).
- Freeform loop dispatch at `sample.py:1024` (after token append + telemetry update): call `action = self._irq.dispatch_pending(self)`, honor `BREAK` / `RESTART_TURN`.
- Constrained loop dispatch at `sample.py:680` (after `grammar_state.advance`, before accept check): same pattern.
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
- **Invariant:** zero torch, zero numpy in hot loop (CLAUDE.md). Controller dispatch is pure Python; does not touch `mx.array`.

**Tests (RED first):**
- `test_cancel_breaks_mlx_constrained_within_one_forward` — MLX-native version of AC#1.
- Regression: existing `test_sample_mlx.py` passes.

**Validation:** `pytest tests/test_sample_mlx.py tests/test_interrupts.py -v`

**Commit:** `feat(sample_mlx): dispatch interrupts in MLX generation loop`

---

### Task 4: FastAPI SSE + WebSocket cancel-on-disconnect

**Files:**
- `src/tgirl/serve.py`
- `tests/test_serve.py`

**Approach:**
- SSE handler (`serve.py:1616–1779`):
  - Install a default CANCEL handler on the session that finalizes telemetry and returns `BREAK`.
  - Start `disconnect_watcher = asyncio.create_task(_watch_disconnect(request, session))`. Watcher polls `await request.is_disconnected()` every 50ms; on disconnect calls `session.irq.raise_interrupt(IrqSource.CANCEL)`.
  - Cancel the watcher in `finally` on normal completion.
- WebSocket handler (`serve.py:980–1028`): replace implicit exception-swallow with explicit `try: await ws.receive() except WebSocketDisconnect: session.irq.raise_interrupt(IrqSource.CANCEL)` loop.
- Thread-safety: `raise_interrupt` is lock-guarded (Task 1).

**Tests (RED first):**
- Integration test using `httpx.AsyncClient` (or the `openai` Python client): open streaming `/v1/chat/completions`, close client mid-stream, assert the server's mock `forward_fn` was called at most `N+1` times where `N` was the token count at disconnect (AC#2). Use a mocked inference-context to count.
- Manual smoke (documented as validation step): `tgirl serve &`; curl with `timeout 1`; verify structlog shows CANCEL dispatch within one forward pass.

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

**Files:**
- `src/tgirl/serve.py`
- `tests/test_serve.py`
- `tests/test_autotune.py`

**Approach:**
- Replace direct `_run_autotune_after_turn` calls at `serve.py:1347, 1366` with a registered `TURN_COMPLETE` handler installed at session creation.
- Handler body unchanged; wrap to return `CONTINUE`.
- Autotuner observables and action state flow untouched.

**Tests (RED first):**
- Autotuner regression suite passes unchanged (AC#11).
- Integration test: multi-turn request, autotune observables match pre-refactor values for a fixed input.

**Validation:** `pytest tests/test_autotune.py tests/test_serve.py -v`

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
- Add `schedule(source, interval_ms, data=None)` to `InterruptController`.
- `dispatch_pending` checks `time.monotonic()`; fires any scheduled sources whose next-fire has passed; re-schedules.
- Self-contained: no asyncio.Task, no event-loop coupling on sampling thread (PRD OQ#4).

**Tests (RED first):**
- `test_timer_fires_after_interval` — schedule 10ms, monkeypatch clock, dispatch → fires.
- `test_timer_does_not_fire_early` — dispatch before interval → no fire.
- `test_timer_reschedules_after_fire` — fire, advance clock, dispatch again → fires again.

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
- NEW `benchmarks/perf/irq_overhead_results.json` (committed with latest run)

**Approach:**
- Micro-benchmark: 10,000 empty-queue `dispatch_pending()` calls; measured wall-clock; assert < 500ms (50µs/call budget, AC#10).
- BFCL bit-parity regression: compare `benchmarks/run_bfcl.py --limit 20 --seed 0` output against golden fixture captured in Task 6 (AC#9).
- Bench result committed as evidence, regression test committed as guard.

**Tests:**
- Micro-bench is a pytest test that fails on budget overage.
- BFCL parity test is a fixture-comparison pytest.

**Validation:** `pytest tests/regression/ -v` + `python benchmarks/perf/irq_overhead.py`

**Commit:** `test(perf): per-token overhead budget and BFCL bit-parity regression`

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
- **Hot-path overhead under real handlers.** 50µs budget is for empty queue. If any single handler — especially `TURN_COMPLETE` autotuner — exceeds a reasonable per-dispatch budget, measure separately in Task 7 and Task 11. If problematic, consider a deferred / async-dispatch path for handlers flagged as slow.
- **Security audit.** Strongly recommend running `/security-audit` before the PR given sample.py contact. Not strictly blocking for standard tier, but flagging explicitly.
- **BFCL bit-parity tolerance.** "Bit-identical" assumes deterministic sampling with a fixed seed. If any non-deterministic ordering exists in the current loop (e.g., dict-iteration order affecting registry serialization), Task 6 will surface it as the parity test fails — in that case, document the non-determinism and relax to sequence-equivalence with justification.
