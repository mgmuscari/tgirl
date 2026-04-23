# PRP: mypy Strict-Mode Cleanup

## Source PRD: docs/PRDs/mypy-strict-cleanup.md
## Date: 2026-04-22

## 1. Context Summary

Drive `mypy src/ --ignore-missing-imports` to a clean run under the existing `strict = true` config. Baseline is 100 errors across 17 `src/tgirl/*` files. No runtime behavior changes; protocols added are internal documentation of already-existing duck-typed contracts. Latent bugs surfaced in stage 4 (below) must ship with failing-then-passing regression tests.

## 2. Codebase Analysis

### Baseline error inventory

Captured via `mypy src/ --ignore-missing-imports` on the branch base. Distribution:

**By rule (100 total):**
- 20 `attr-defined` — `object` typing of fields that have a real protocol (ESTRADIOL controller, hooks, tokenizer)
- 19 `arg-type` — call-site type mismatches; includes MLX `DeviceType` vs `Stream | Device | None`, `Tensor | None` where `array` expected, pydantic init kwargs as `dict[str, object]`
- 13 `type-arg` — missing generic parameters: `Callable[..., Any]`, `list[X]`, `Pattern[str]`
- 12 `no-any-return` — function returns `Any` from MLX/MCP APIs but declares a concrete return type
- 6 `union-attr` — attribute access on unions mypy can't narrow
- 6 `no-untyped-call` — calls to untyped helpers (`_mx` in `estradiol.py`, `safe_open`)
- 5 `unused-ignore` — stale `# type: ignore` comments
- 4 `no-untyped-def` — functions without annotations
- 3 `index` — tuple-as-dict-key narrowing in `rerank.py`
- 3 `assignment` / 2 `comparison-overlap` — `lingo/types.py` set/str confusion (latent bug cluster)
- 2 `misc` — `serve.py:89` tuple unpack mismatch (latent bug), dynamic base class in `compile.py:438`
- 2 `call-overload` — `kwargs.get("token_id")` narrowing in `state_machine.py`
- 2 `call-arg` — `steering=` kwarg on `forward_fn`; `enable_thinking=` on `PromptFormatter`
- 1 `operator` — `>` on `list[list_or_scalar]` in `calibrate.py:39`

**By file (top offenders):**
- `src/tgirl/sample_mlx.py` — 18 (ESTRADIOL controller, hooks, grammar state all typed as `object`)
- `src/tgirl/calibrate.py` — 15 (`no-any-return` on MLX ops, `stream=` kwarg, union narrowing)
- `src/tgirl/sample.py` — 14 (generics, InferenceHook attrs, ModelIntervention kwargs)
- `src/tgirl/state_machine.py` — 8
- `src/tgirl/serve.py` — 8 (generics + 1 tuple unpack latent bug + InferenceContext signature)

### Existing patterns to reuse

- **Protocol definitions:** `src/tgirl/sample.py:GrammarState` and `src/tgirl/state_machine.py:52-61` (`TransitionPolicy`) both use `@runtime_checkable` Protocol with method-only surface. Follow this pattern for `EstradiolControllerProto` and `TokenizerProto`.
- **Concrete `V_basis` + controller surface:** `src/tgirl/estradiol.py:49-94` — `EstradiolController` class with `.V_basis`, `.alpha_current`, `.step()`, `.make_steering_state()`, `.reset()`. The Protocol is literally the public method surface of this class.
- **Forward-fn steered-result shape:** `src/tgirl/cache.py` produces forward functions; when `steering=` kwarg is passed the return has `.logits` and `.probe_alpha`. Currently typed as `mx.array` which is wrong in the steered path.
- **Pydantic model init patterns:** `src/tgirl/types.py:ModelIntervention` and other pydantic models use strict init. `sample.py:143` passes `**dict[str, object]` kwargs — fix via TypedDict or by typing the source dict correctly.
- **`# type: ignore[rule]` convention:** Existing code uses `[rule]` specifier (e.g. `sample.py:138: # type: ignore[misc]`). Keep this — bare `# type: ignore` silences all rules and is a smell.
- **Existing mypy overrides infrastructure:** `pyproject.toml` currently has `[tool.mypy]` with `strict = true`, `plugins = ["pydantic.mypy"]`, `mypy_path = "src"`. No per-module overrides yet — Task 1 adds the first.

### Conventions (CLAUDE.md + session memory)

- TDD mandatory: RED → GREEN → REFACTOR → COMMIT per task. For annotation-only tasks, the "test" is `mypy src/` going from failing to passing on the subset; plus the full pytest suite staying green.
- Each task = one atomic commit (test + implementation together). Commit messages in Conventional Commits format, first line ≤72 chars.
- `structlog` for logging. MLX hot loops are zero-torch, zero-numpy.
- No cross-framework tensor conversions.
- No "fix later" shims: if a type fix requires a Protocol, add the Protocol in the same task.
- `# type: ignore` budget: ≤10 new comments across the whole change, each with a `[rule]` specifier and a reason.

### Tests that validate each stage

- `tests/test_sample.py` — torch sampling path regression for any signature change in `sample.py`.
- `tests/test_sample_mlx.py` — MLX sampling path, including ESTRADIOL-steered calls where `controller` is passed.
- `tests/test_estradiol.py`, `tests/test_estradiol_integration.py` — ESTRADIOL controller behavior.
- `tests/test_format.py` — tokenizer / chat-template behavior.
- `tests/test_state_machine.py` — transition policy protocols and kwargs.
- `tests/test_serve.py` — FastAPI endpoints including the `_generate` tuple-unpack site.
- `tests/test_lingo_*` — lingo/types, lingo/lexicon, TDL parser.
- `tests/test_calibrate.py` — SVD-based calibration math (if exists; else add one in the latent-bug stage).
- Full suite baseline: **1118 passed in ~60s**.

## 3. Implementation Plan

**Test Command:** `pytest tests/` and `mypy src/ --ignore-missing-imports`

Each task below is scoped to a single logical commit. Tasks are ordered by risk (low to high) and dependency; later tasks may build on protocols introduced earlier.

---

### Task 1: Baseline capture + third-party mypy overrides

**Files:**
- `pyproject.toml` (modify)
- NEW `docs/reviews/mypy-baseline.txt` (committed for diff reference)

**Approach:**
- Snapshot current mypy output: `mypy src/ --ignore-missing-imports > docs/reviews/mypy-baseline.txt 2>&1`. Commit this so subsequent tasks can diff against a known starting point. Delete at the end of the cleanup.
- Add per-module overrides to `[tool.mypy]` in `pyproject.toml` for third-party packages without good stubs: `mlx`, `mlx.core`, `mlx_lm`, `mlx_lm.*`, `llguidance`, `llguidance.*`, `transformers`, `transformers.*`, `outlines`, `outlines.*`, `mcp`, `mcp.*`. Set `ignore_missing_imports = true` on each.
- This does NOT silence errors in our code — it only tells mypy it's OK that those modules don't have stubs, so it doesn't surface them as our problem.

**Tests (RED first):**
- RED: `mypy src/ --ignore-missing-imports 2>&1 | grep -c "error:"` → 100
- GREEN: no change in our-code error count (still 100) because the overrides only affect import-level errors we were already ignoring via `--ignore-missing-imports`. The value is that now `pyproject.toml` captures the intent, and future `mypy src/` invocations without the CLI flag behave consistently.

**Validation:** `mypy src/` (without `--ignore-missing-imports`) produces the same 100 errors as `mypy src/ --ignore-missing-imports`.

**Commit:** `chore(mypy): baseline snapshot + third-party import overrides`

---

### Task 2: Add missing generic type parameters (type-arg cluster)

**Files:**
- `src/tgirl/state_machine.py` — lines 215–217 (`Callable`), 606 (`list`)
- `src/tgirl/sample.py` — lines 754, 906 (`Callable`), 813 (`list`)
- `src/tgirl/sample_mlx.py` — line 430 (`Callable`)
- `src/tgirl/compile.py` — line 122 (`list`)
- `src/tgirl/grammar.py` — line 517 (`Pattern`)
- `src/tgirl/serve.py` — lines 49, 100, 123 (`Callable`)

**Approach:**
- Read each site. Infer the actual call signature from usage and callers; fill in the parameter types.
- Where the exact signature is genuinely variadic/heterogeneous, use `Callable[..., Any]` rather than inventing a signature. Prefer concrete `Callable[[list[int]], mx.array]` when inferable (e.g. forward_fn patterns already used elsewhere in the file).
- For `list` → `list[X]`, find one usage site and read the type; if truly heterogeneous, `list[Any]` — but this should be rare.
- `Pattern` → `Pattern[str]`.

**Tests (RED first):**
- RED: `mypy src/ 2>&1 | grep -c "type-arg"` → 13
- GREEN: `mypy src/ 2>&1 | grep -c "type-arg"` → 0
- Full suite still 1118 passed (annotation-only changes — zero runtime effect).

**Validation:** `mypy src/ 2>&1 | grep "type-arg"` returns nothing; `pytest tests/` → 1118 passed.

**Commit:** `fix(types): add missing generic parameters to Callable/list/Pattern`

---

### Task 3: Delete stale `# type: ignore` comments

**Files:**
- `src/tgirl/format.py` — lines 44, 52
- `src/tgirl/bfcl.py` — lines 28, 43
- `src/tgirl/cli.py` — line 60

**Approach:**
- For each unused-ignore flag, delete the comment. Run mypy to confirm the error it was suppressing genuinely no longer exists. If a new error appears in its place (unlikely but possible), the comment was actually needed — replace it with a properly-scoped `# type: ignore[rule]` and log in Uncertainty.
- `format.py:44, 52`: these are `# type: ignore[union-attr]` comments. The union doesn't exist anymore (the tokenizer field type has since narrowed). Delete the ignores. Note: this task will leave the underlying `attr-defined`/`no-any-return` errors on `format.py:44,52` visible — those are addressed in Task 7 (`TokenizerProto`).

**Tests (RED first):**
- RED: `mypy src/ 2>&1 | grep -c "unused-ignore"` → 5
- GREEN: `mypy src/ 2>&1 | grep -c "unused-ignore"` → 0
- Full suite still 1118 passed.

**Validation:** `pytest tests/test_format.py tests/test_bfcl.py tests/test_cli.py -v` → all pass; no new mypy errors introduced (the attr-defined on format.py:44,52 are already in baseline).

**Commit:** `fix(types): delete stale # type: ignore comments`

---

### Task 4: Add missing return/argument annotations (no-untyped-def + no-untyped-call)

**Files:**
- `src/tgirl/estradiol.py` — line 176 (missing return annotation); lines 108, 109, 112, 142 (`_mx` helper untyped); line 148 (`safe_open` untyped)
- `src/tgirl/lingo/lexicon.py` — line 71 (missing arg annotation)

**Approach:**
- `estradiol.py:176` — add return type annotation to the flagged function (likely `-> None` or `-> mx.array` depending on body).
- `estradiol.py:108-142` — the `_mx` helper is called 4 times in typed contexts. Read `_mx` at its definition site; add `-> mx.array` (or whatever it returns) and parameter annotations.
- `estradiol.py:148` — `safe_open` is from `safetensors`. Either annotate the call via `cast(Any, safe_open)(path)` (cheap) or add a `# type: ignore[no-untyped-call]` with a safetensors-stubs reason. Prefer cast for locality.
- `lingo/lexicon.py:71` — one flagged arg. Annotate.

**Tests (RED first):**
- RED: 6 `no-untyped-call` + 4 `no-untyped-def` errors; `mypy src/ 2>&1 | grep -cE "no-untyped-(call|def)"` → 10
- GREEN: 0
- Full suite still 1118 passed. Specifically `tests/test_estradiol.py` still passes — no behavior change.

**Validation:** `pytest tests/test_estradiol.py tests/test_lingo_lexicon.py -v`; `mypy src/ 2>&1 | grep -E "no-untyped"` returns nothing.

**Commit:** `fix(types): annotate estradiol._mx, safe_open call, and lingo helper`

---

### Task 5: EstradiolControllerProto + apply to sample_mlx.py

**Files:**
- `src/tgirl/estradiol.py` (add Protocol near the top of the file, after imports; reference the concrete class below)
- `src/tgirl/sample_mlx.py` (change `controller: object | None = None` → `controller: EstradiolControllerProto | None = None`, import the protocol)
- `src/tgirl/sample.py` (if the torch counterpart has the same pattern — check; else skip)
- `tests/test_sample_mlx.py` (verify the MLX constrained-gen test still accepts the concrete `EstradiolController` via the Protocol)

**Approach:**
- In `estradiol.py`, add a `@runtime_checkable` `EstradiolControllerProto(Protocol)` that exposes exactly what `sample_mlx.py` uses today:
  - `V_basis: Any  # mx.array`
  - `alpha_current: Any  # mx.array`
  - `def step(self, probe_alpha: Any) -> Any: ...`
  - `def make_steering_state(self, delta_alpha: Any) -> Any: ...`
  - `def reset(self) -> None: ...`
- Verify the concrete `EstradiolController` class implicitly satisfies this by running `isinstance(controller, EstradiolControllerProto)` in a one-off test before removing it.
- In `sample_mlx.py`: import the Protocol, change the parameter annotation at line 434 from `object | None` to `EstradiolControllerProto | None`. Remove the `hasattr(controller, "reset")` guard at line 470 (the Protocol now guarantees it) IF the existing runtime behavior doesn't depend on it; keep the `hasattr` if other callers pass controllers without `.reset()`.

**Tests (RED first):**
- RED: `mypy src/tgirl/sample_mlx.py 2>&1 | grep -cE "controller.*object|V_basis|make_steering_state|alpha_current|step.*attr-defined"` → ≥8 (the ESTRADIOL-on-`object` cluster)
- Add `tests/test_estradiol.py::test_controller_satisfies_proto` — `isinstance(EstradiolController(...), EstradiolControllerProto)` is True. This is the RED → GREEN check for the Protocol surface.
- GREEN: the 8 attr-defined errors in `sample_mlx.py:472-495` resolve; `test_controller_satisfies_proto` passes.
- Full suite: `tests/test_sample_mlx.py`, `tests/test_estradiol.py`, `tests/test_estradiol_integration.py` all still pass.

**Validation:** `pytest tests/test_sample_mlx.py tests/test_estradiol.py tests/test_estradiol_integration.py -v`; `mypy src/tgirl/sample_mlx.py` error count drops by ≥8.

**Commit:** `feat(types): define EstradiolControllerProto and type sample_mlx controller parameter`

---

### Task 6: Type the steered forward-fn return (`ForwardFnResult`)

**Files:**
- `src/tgirl/cache.py` or `src/tgirl/sample_mlx.py` (add a `ForwardFnResult` dataclass / Protocol with `.logits` and `.probe_alpha`)
- `src/tgirl/sample_mlx.py` (update the `forward_fn` call site's expected return type at lines 490–496)

**Approach:**
- The current flow: `forward_fn(token_history, steering=_steering)` returns an object with `.logits` and `.probe_alpha` when steering is active, but returns a raw `mx.array` when steering is inactive. This polymorphism is the root of 3 `attr-defined` errors on `_fwd_result.logits` / `.probe_alpha`.
- Two options:
  - (a) Define `ForwardFnResult` dataclass `(logits: mx.array, probe_alpha: mx.array | None)` and have the cache's forward factory ALWAYS return this shape (even in non-steered path where `probe_alpha=None`). Update callers. This is cleaner.
  - (b) Make the steered return a Union type: `mx.array | SteeredForwardResult`, and narrow at the call site.
- Recommend (a). Ripple: `cache.py` forward factories need updating; other callers (sample.py torch path, tests) may need adjustment.
- If (a) proves too invasive, fall back to (b) with `assert isinstance(_fwd_result, SteeredForwardResult)` right after the steered call, which narrows the type for mypy.

**Tests (RED first):**
- RED: `mypy src/tgirl/sample_mlx.py 2>&1 | grep -cE '"array" has no attribute.*(logits|probe_alpha)'` → 3
- Add `tests/test_sample_mlx.py::test_steered_forward_returns_forwardfnresult` — calls the steered forward function, asserts `hasattr(result, "logits") and hasattr(result, "probe_alpha")`.
- GREEN: the 3 attr-defined errors resolve.
- Full suite still 1118 passed.

**Validation:** `pytest tests/test_sample_mlx.py tests/test_cache.py -v`.

**Commit:** `feat(cache): define ForwardFnResult for steered forward passes`

---

### Task 7: TokenizerProto + apply to format.py and outlines_adapter.py

**Files:**
- `src/tgirl/format.py` (define `TokenizerProto` locally or in types.py; annotate `self._tokenizer`)
- `src/tgirl/outlines_adapter.py` (use the Protocol for the `tokenizer` param at lines 105–129 and 181–200)

**Approach:**
- `format.py` uses `self._tokenizer.apply_chat_template(...)`. Minimum Protocol:
  ```python
  class _TokenizerChatTemplateProto(Protocol):
      def apply_chat_template(
          self, messages: list[dict[str, str]], /, *,
          tokenize: bool = ..., add_generation_prompt: bool = ..., **kwargs: Any,
      ) -> str: ...
  ```
- `outlines_adapter.py:105, 181` take `tokenizer: object` and pass through to `llguidance.hf.from_tokenizer(...)`. The llguidance function expects a `TokenizersBackend`-compatible object. We can narrow to `TokenizerProto` with the subset of attrs `from_tokenizer` needs — usually `vocab`, `vocab_size`, and `encode`/`decode`. Read the llguidance docs or stub the minimum.
- For `outlines_adapter.py:142,146` (`llguidance.mlx` module has no explicit `allocate_token_bitmask` / `fill_next_token_bitmask` exports per mypy): add a module-level `# type: ignore[attr-defined]` with a comment citing the upstream llguidance-stubs gap. Budget: 2 of our 10.

**Tests (RED first):**
- RED: `mypy src/tgirl/format.py src/tgirl/outlines_adapter.py 2>&1 | grep -c "error:"` → ~7
- GREEN: all resolved.
- Full suite still 1118 passed. Specifically `tests/test_format.py` and `tests/test_outlines_adapter.py` (if exists) pass.

**Validation:** `pytest tests/test_format.py tests/test_outlines_adapter.py -v`.

**Commit:** `feat(types): define TokenizerProto for format and outlines_adapter`

---

### Task 8: InferenceHook attribute typing in sample.py

**Files:**
- `src/tgirl/sample.py` (lines 1187, 1190 use `hook.config` and `hook.max_tokens`)
- `src/tgirl/types.py` or `src/tgirl/sample.py` (wherever `InferenceHook` protocol/base is defined)

**Approach:**
- Find `InferenceHook` definition. It's likely a Protocol. Two paths:
  - (a) If only one concrete subclass uses `.config` and `.max_tokens` (e.g. `GrammarTemperatureHook`), `isinstance(hook, GrammarTemperatureHook)` narrow at the call site.
  - (b) If multiple subclasses share these, widen the protocol to include `config: Any` and `max_tokens: int`.
- Read sample.py:1187–1195 to understand the intent.

**Tests (RED first):**
- RED: `mypy src/tgirl/sample.py 2>&1 | grep -c 'InferenceHook.*has no attribute'` → 2
- GREEN: resolved.

**Validation:** `pytest tests/test_sample.py -v`.

**Commit:** `fix(types): narrow InferenceHook attribute access in sample.py`

---

### Task 9: no-any-return cluster (calibrate.py + bridge.py + format.py)

**Files:**
- `src/tgirl/calibrate.py` — 6 errors at lines 250, 255, 265, 269, 274, 278 (functions returning `Any` from MLX ops where `str` is declared)
- `src/tgirl/bridge.py` — 2 errors at lines 65, 538 (MCP API returning `Any` where `str` declared)
- `src/tgirl/format.py` — 2 errors at lines 44, 52 (`apply_chat_template` returning `Any`)
- `src/tgirl/state_machine.py` — 2 errors at lines 632, 649 (returning `Any` where `TransitionDecision` declared)

**Approach:**
- For each `no-any-return`, cast the return through a typed local:
  ```python
  result: str = some_untyped_call()  # type narrowed here
  return result
  ```
- This is cleaner than `cast(str, ...)` because it doubles as runtime check documentation.
- In `state_machine.py:632, 649`, the `Any` is likely coming from a polymorphic dispatch — read and pick the narrowing approach (could be `assert isinstance(...)` or `cast`).

**Tests (RED first):**
- RED: `mypy src/ 2>&1 | grep -c "no-any-return"` → 12
- GREEN: 0.

**Validation:** `pytest tests/test_calibrate.py tests/test_bridge.py tests/test_format.py tests/test_state_machine.py -v`.

**Commit:** `fix(types): narrow no-any-return sites through typed locals`

---

### Task 10: Union narrowing (calibrate.py, modulation.py, rerank.py)

**Files:**
- `src/tgirl/calibrate.py` — lines 36, 39, 467, 562, 575, 628 (union-attr and arg-type from MLX `svd` return narrowing; `stream=` kwarg type)
- `src/tgirl/modulation.py` — 4 errors around line 298 (conditioner tuple indexing, `SourceConditionerConfig` vs `float`)
- `src/tgirl/rerank.py` — 3 errors at lines 116, 121, 143 (tuple-as-dict-key narrowing, `GrammarState` vs `GrammarStateMlx`)

**Approach:**
- `calibrate.py:36, 467, 562, 628`: `stream=mx.cpu` has mypy type `DeviceType` but `svd` expects `Stream | Device | None`. This is an MLX stub gap — the real `mx.cpu` IS a `Device`. Use `cast(Any, mx.cpu)` at call sites, or (preferred) add `# type: ignore[arg-type]  # mlx stub gap: mx.cpu is Device` at each site. Budget: 4.
- `calibrate.py:39, 575`: `mx.linalg.svd(...)[1]` returns `int | float | list[list_or_scalar]` per mypy — but at runtime it's always `mx.array`. `cast(mx.array, result)` or `assert isinstance(result, mx.array)` narrowing.
- `modulation.py:298-305`: `list(self.conditioners)` is typed as `list[float]` but the member type is `SourceConditionerConfig`. Likely the caller's expected type is wrong; fix by annotating the `new: list[SourceConditionerConfig]` properly.
- `rerank.py:116, 121`: `tuple[object, ...]` as dict key — add explicit type annotation on the tuple construction: `key: tuple[str | int, ...] = tuple(...)`.
- `rerank.py:143`: `GrammarState` is passed to `run_constrained_generation_mlx` which expects `GrammarStateMlx`. This is a real API mismatch — either the function signature accepts a union protocol, or the caller is wrong. Investigate.

**Tests (RED first):**
- RED: `mypy src/ 2>&1 | grep -cE "union-attr|index"` in these files → 9
- GREEN: resolved.

**Validation:** `pytest tests/test_calibrate.py tests/test_modulation.py tests/test_rerank.py -v`.

**Commit:** `fix(types): narrow unions in calibrate, modulation, rerank`

---

### Task 11: state_machine.py kwargs narrowing

**Files:**
- `src/tgirl/state_machine.py` — lines 110, 546 (`int(kwargs.get("token_id"))` fails because mypy types `kwargs.get("token_id")` as `object`)

**Approach:**
- Add an explicit narrowing:
  ```python
  token_id_raw = kwargs.get("token_id")
  if token_id_raw is None:
      return TransitionDecision(...)
  if not isinstance(token_id_raw, (int, str)):
      return TransitionDecision(...)
  token_id = int(token_id_raw)
  ```
- Preserves existing runtime behavior; makes mypy happy.

**Tests (RED first):**
- RED: `mypy src/tgirl/state_machine.py 2>&1 | grep -c "call-overload"` → 2
- GREEN: 0.
- `tests/test_state_machine.py` still passes — behavior preserved.

**Validation:** `pytest tests/test_state_machine.py -v`.

**Commit:** `fix(types): narrow kwargs.get('token_id') before int() in state_machine`

---

### Task 12: Latent bug investigation (lingo/types.py, serve.py:89, sample_mlx.py:495)

**Files:**
- `src/tgirl/lingo/types.py` — lines 128–130 (set/str confusion — assignment/comparison-overlap/index cluster)
- `src/tgirl/serve.py` — line 89 (`Too many values to unpack (2 expected, 3 provided)`)
- `src/tgirl/sample_mlx.py` — line 495 (`Item "None" of "list[list[float]] | None" has no attribute "append"`)
- `src/tgirl/sample.py` — line 1090 (`ToolRouter | None` has no attribute "route") — similar None-dereference
- Tests: `tests/test_lingo_types.py`, `tests/test_serve.py`, `tests/test_sample_mlx.py`, `tests/test_sample.py`

**Approach — ONE investigation per bug:**

**12a. `lingo/types.py:128-130`:** Read lines 120-135 in context. The errors are consistent with an assignment where a string is assigned to a `set[str]`-typed variable, then compared/indexed as a set. This is either:
- Real bug: the variable should be named `values` (set) but got a single `value` assigned. Add test reproducing the broken behavior. Fix by using the correct variable / type.
- Type annotation wrong: if the runtime actually stores single strings, change the declared type.

Write a test that triggers this code path. If it fails on `main`, this is a real bug — commit test + fix together. If it passes, the code is somehow correct; document why (likely an `Any` bleeding through) and fix annotations.

**12b. `serve.py:89`:** `(batch=1, seq_len, d_model) → (seq_len, d_model)` comment hints the unpacking is a 3-tuple reshape. Read lines 82–95. If the code genuinely unpacks 3 into 2, it's a bug. Likely intention: `batch, seq_len, d_model = tensor.shape` (3-unpack into 3 names). Write a test that exercises this path. Fix the unpacking site.

**12c. `sample_mlx.py:495`:** `estradiol_alphas` is typed as `list[list[float]] | None`, initialized to `None` at line 461, re-assigned to `[]` at line 474 inside `if controller is not None`, then `.append()` at line 495 inside the same `if controller is not None and _steering is not None` guard. The guard narrows to `not None` for `_steering` but mypy can't see the same for `estradiol_alphas`. Fix: move `estradiol_alphas` / `estradiol_deltas` initialization outside the conditional (`= None` always), or narrow at the append site with `assert estradiol_alphas is not None`. The latter preserves the current "only allocated when needed" pattern.

**12d. `sample.py:1090`:** `self._tool_router: ToolRouter | None` is accessed as `self._tool_router.route(...)` without narrowing. Either narrow with `assert self._tool_router is not None` at the call site, or hoist the check.

**Tests (RED first):**
- For each latent bug 12a–12d: write a test that demonstrates the issue (even if it's type-only). Failing test → fix → passing test + type resolved.
- Aggregate: `mypy src/` error count drops by 6–8 for this task.

**Validation:** `pytest tests/test_lingo_types.py tests/test_serve.py tests/test_sample_mlx.py tests/test_sample.py -v -k "the new test names"`.

**Commit:** `fix(lang, serve, sample): resolve latent None-dereference and unpack bugs found by mypy`

---

### Task 13: Remaining scatter-gather (arg-type, call-arg, misc)

**Files:**
- `src/tgirl/sample.py` — lines 143 (`ModelIntervention(**dict)`), 823 (`ToolRouter(backend=str)`), 942, 987, 1002
- `src/tgirl/sample_mlx.py` — lines 490 (`steering=` kwarg), 506 (`get_valid_mask` vs `get_valid_mask_mx`)
- `src/tgirl/serve.py` — 1581 (`enable_thinking` kwarg on `PromptFormatter`), 745, 1623 (missing return annotations), 1781 (no-untyped-call on stream_gen)
- `src/tgirl/compile.py` — 438 (`class ... subclass "Any"`) — `RestrictedPython.RestrictingNodeTransformer` has no stubs. `# type: ignore[misc]` with reason.
- `src/tgirl/lingo/tdl_parser.py` — 461 (None-union-attr; similar to 12c, narrow at use site)

**Approach (each is small and independent):**
- **sample.py:143:** the `ModelIntervention(**source)` where `source: dict[str, object]` fails pydantic strict init. Define a `ModelInterventionDict` TypedDict with the exact fields, narrow `source` to it, then `**source` type-checks.
- **sample.py:823:** `backend=some_str` where `Literal["torch", "mlx"]` expected. Narrow: `backend_lit: Literal["torch", "mlx"] = "torch" if backend == "torch" else "mlx"` (or use an assert).
- **sample.py:942, 987, 1002:** `Tensor | None` vs `array` vs `Tensor` — framework-mismatch in a conditional branch. Narrow with `assert isinstance(x, torch.Tensor)` or split into framework-specific paths.
- **sample_mlx.py:490:** `steering=_steering` kwarg not in forward_fn signature. Either the cache's forward factory needs its signature widened (add `steering: Any = None`), or the call is guarded by `if controller is not None` and we `cast` the forward_fn to a `SteeredForwardFn` protocol. Task 6 may already have resolved this if it updated the forward factory contract.
- **sample_mlx.py:506:** `GrammarStateMlx.get_valid_mask` vs `get_valid_mask_mx` — the wrong method name. If `get_valid_mask` IS wrong, this is a real bug (Task 12 territory); if `get_valid_mask_mx` is correct, fix to that.
- **serve.py:1581:** `enable_thinking=...` kwarg not in `PromptFormatter.format_messages`. Either the formatter supports `**kwargs: Any` (extend), or the caller has a stale kwarg. Read both; fix accordingly.
- **compile.py:438:** `# type: ignore[misc]  # RestrictedPython has no stubs`. Budget: 1.

**Tests (RED first):**
- Full `mypy src/` → 0 errors at the end of this task.
- Full `pytest tests/` → 1118 passed.

**Validation:**
```bash
mypy src/                   # must produce: "Success: no issues found in 33 source files"
pytest tests/               # must produce: 1118 passed
```

**Commit:** `fix(types): resolve remaining arg-type and call-arg mismatches`

---

### Task 14: Cleanup — delete baseline snapshot + final verification

**Files:**
- `docs/reviews/mypy-baseline.txt` (delete)

**Approach:**
- Run `mypy src/` — must succeed.
- Run `pytest tests/` — must produce 1118 passed.
- Run `ruff check src/` — must succeed.
- Delete `docs/reviews/mypy-baseline.txt`.
- Manual smoke test per PRD AC#5: `tgirl serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit` + one `/v1/chat/completions` request with `estradiol_alpha=0.5`; verify output is coherent and `/v1/steering/status` reflects steering being active.

**Tests:**
- All gates green: mypy, ruff, pytest.
- Smoke test manual.

**Validation:**
```bash
ruff check src/ && mypy src/ && pytest tests/ --tb=short -q
```

**Commit:** `chore(mypy): remove baseline snapshot — cleanup complete`

---

## 4. Validation Gates

```bash
# Syntax / Style
ruff check src/

# Type-check
mypy src/                                      # must pass with 0 errors

# Unit tests (baseline 1118 passed)
pytest tests/ --tb=short -q

# Smoke test (manual, after Task 14)
tgirl serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit &
SERVER_PID=$!
sleep 3
curl -N http://localhost:8420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"...","messages":[{"role":"user","content":"write a short poem"}],"stream":false,"estradiol_alpha":0.5}'
curl http://localhost:8420/v1/steering/status
kill $SERVER_PID
# Expect: coherent poem; steering_status shows alpha=0.5 active.
```

## 5. Rollback Plan

Each task is one commit → revert-range granularity is per-task.

- Per-task failure → `git revert <sha>` for that task alone. Earlier task commits stay if they don't depend on it (Tasks 1–4 have no inter-dependencies; Tasks 5–8 depend on Task 1's mypy baseline capture; Task 10+ depends on earlier narrowing for some call sites).
- Protocol-addition tasks (5, 6, 7) are structural and non-revertible trivially if later tasks use them. If a Protocol turns out to be wrong-shaped, prefer fixing forward rather than reverting.
- Latent-bug fixes in Task 12 each include a regression test — if the fix regresses something else, revert the task and re-investigate (may indicate an unidentified dependency).
- Full-feature abandon: revert in reverse order (Task 14 → Task 1). The feature branch can be abandoned; main is untouched since this is a branch-only change until merge.

## 6. Uncertainty Log

- **MLX stub completeness.** MLX has partial `*.pyi` stubs but `mx.cpu` / `mx.gpu` / `mx.linalg.svd` have known gaps. Task 10 assumes 4 `# type: ignore[arg-type]` for these; if there are more, the budget creeps toward the limit. If we exceed 10 total `# type: ignore` comments, add a `[[tool.mypy.overrides]] module = "tgirl.calibrate"` override in pyproject.toml with `disable_error_code = ["arg-type"]` and document the scope.
- **`ForwardFnResult` scope of change.** Task 6 may require touching `cache.py` forward-fn factories more broadly than anticipated. If updating the factory signature breaks `sample.py` (torch path) or tests, split Task 6 into 6a (define) + 6b (migrate MLX) + 6c (migrate torch). Add tasks on the fly during execution.
- **Pydantic strict-init impact.** Task 13's `ModelInterventionDict` TypedDict might require cascade typing changes in callers. If the cascade is big (>2 files), defer and use `cast(dict[str, Any], source)` with a rationale comment.
- **`compile.py:438` dynamic base.** If the `# type: ignore[misc]` on `class ... RestrictingNodeTransformer` turns out to hide subclass-signature bugs, the right fix is contributing stubs upstream. Not in scope — document the deferral.
- **`rerank.py:143` GrammarState mismatch.** This may be a real bug (caller passes torch GrammarState to MLX function). Task 10 surfaces it; if it's a bug, escalate to Task 12 (latent-bug treatment with regression test).
- **`state_machine.py:632, 649` TransitionDecision narrowing.** Unknown what's returning `Any` there without reading. May require a larger Protocol-narrowing refactor. If the fix exceeds 20 LoC, split into a dedicated task.
- **Branch base.** This PRD is branched from `main`, not from `chore/dialectic-framework-sync`. If the sync branch doesn't merge first, the pre-commit hook here is the old one (not mypy-running), which means local commits won't fail and we won't notice if mypy regresses mid-work. Mitigation: Task 1 commits include a manual `mypy src/` check in the commit message; the interlocutor enforces discipline.
- **Test coverage for latent bugs.** If `tests/test_calibrate.py` doesn't exist and needs to be created for Task 12, add that as a prerequisite. Scanning `tests/` at PRP-generation time showed no dedicated `test_calibrate.py` — the `test_estradiol.py` tests may exercise calibrate transitively, or a new file is needed. Handle at execution time.
- **Mypy version drift.** CI may run a different mypy version than local. Pinning `mypy==1.x` in `pyproject.toml[project.optional-dependencies.dev]` is out of scope but worth noting if we hit drift.
