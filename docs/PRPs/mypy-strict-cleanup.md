# PRP: mypy Strict-Mode Cleanup

## Source PRD: docs/PRDs/mypy-strict-cleanup.md
## Date: 2026-04-22

## 1. Context Summary

Drive `mypy src/ --ignore-missing-imports` to a clean run under the existing `strict = true` config. Baseline is **100 errors across 18 `src/tgirl/*` files** (verified 2026-04-23 via `mypy src/ --ignore-missing-imports 2>&1 | grep "error:" | sed 's/:[0-9]*:.*//' | sort | uniq -c`; note the `grep "error:"` filter — a naive `grep "^src/"` inflates counts by including mypy `note:` lines). No runtime behavior changes; protocols added are internal documentation of already-existing duck-typed contracts. Latent bugs surfaced in stage 4 (below) must ship with failing-then-passing regression tests.

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
- 3 `assignment` / 2 `comparison-overlap` — `lingo/types.py` scope-shadowing cluster (NOT a bug — see Task 10 for rename fix; originally misdiagnosed as set/str confusion in earlier drafts)
- 2 `misc` — `serve.py:88` unpack mismatch (mlx_lm stub gap — see Task 13), dynamic base class in `compile.py:438`
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

Each task below is scoped to a single logical commit. Tasks are ordered by risk (low to high) and dependency; later tasks build on protocols introduced earlier.

### Task dependency DAG (must-precede relations)

Most tasks are independent. The non-trivial dependencies are:

- **Task 6 must precede Task 13's `sample_mlx.py:490`** (`steering=` kwarg). Task 13 fixes the `Unexpected keyword argument "steering"` call-arg error, but the fix is valid only after Task 6 has widened the `forward_fn` parameter to the `SteerableForwardFn` protocol with the `steering=` kwarg.
- **Task 5 must precede Task 12a** (`estradiol_alphas` append narrowing, sample_mlx.py:495). Note: the narrowing fix itself (`assert estradiol_alphas is not None`) is independent of the controller Protocol, but having `controller: EstradiolControllerProto | None` makes the surrounding code easier to read and removes adjacent `hasattr` guards. This is a *preferred* ordering, not a hard dependency.
- **Task 7 (TokenizerProto) must precede Task 9's `format.py:44, 52`** no-any-return fixes. Once `apply_chat_template` has a concrete `-> str` return type via the Protocol, the `no-any-return` errors vanish naturally. Doing Task 9 first would mean writing `cast(str, ...)` only to delete it in Task 7.
- **Task 12c (rerank cross-framework investigation)** subsumes what was originally scattered: `rerank.py:143` arg-type and `sample_mlx.py:506` attr-defined. Both must be fixed together because the fix is a single architectural decision (backend-aware factory), not two independent narrowings.
- **Tasks 1–4** have no inter-dependencies and can be done in any order.

Recommended execution order: 1 → 2 → 3 → 4 → **5 → 6 → 7 → 8** (Protocol cluster) → **9 → 10 → 11** (narrowing cluster) → **12** (latent bugs) → **13** (scatter-gather) → **14** (cleanup).

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

### Task 5: EstradiolControllerProto + ConfidenceMonitorProto + apply to sample_mlx.py

**Files:**
- `src/tgirl/estradiol.py` (add `EstradiolControllerProto` near the top of the file, after imports; reference the concrete class below)
- `src/tgirl/state_machine.py` (add `ConfidenceMonitorProto` co-located with its concrete consumer — the `should_checkpoint`/`record_log_prob`/`should_backtrack`/`record_backtrack`/`backtracks_remaining` methods are defined at lines 367–388 in state_machine.py, so the Protocol lives there)
- `src/tgirl/sample_mlx.py` (change `controller: object | None = None` at line 434 → `controller: EstradiolControllerProto | None = None`, and `confidence_monitor: object | None = None` at line 429 → `confidence_monitor: ConfidenceMonitorProto | None = None`; import both protocols)
- `src/tgirl/sample.py` (if the torch counterpart has the same pattern — check; else skip)
- `tests/test_sample_mlx.py` (verify the MLX constrained-gen test still accepts the concrete `EstradiolController` via the Protocol)

**Approach:**
- In `estradiol.py`, add a `@runtime_checkable` `EstradiolControllerProto(Protocol)` that exposes exactly what `sample_mlx.py` uses today:
  - `V_basis: Any  # mx.array`
  - `alpha_current: Any  # mx.array`
  - `def step(self, probe_alpha: Any) -> Any: ...`
  - `def make_steering_state(self, delta_alpha: Any) -> Any: ...`
  - `def reset(self) -> None: ...`
- In `state_machine.py`, add a `@runtime_checkable` `ConfidenceMonitorProto(Protocol)` matching the methods accessed in sample_mlx.py at lines 535, 640, 642, 643, 676:
  - `def should_checkpoint(self, grammar_valid_count: int) -> bool: ...`
  - `def record_log_prob(self, log_prob: float) -> None: ...`
  - `def should_backtrack(self) -> bool: ...`
  - `def record_backtrack(self) -> None: ...`
  - `backtracks_remaining: int  # property or attribute`
  Find the concrete monitor class via `grep -rn "def should_checkpoint" src/` before finalizing the Protocol surface — the five methods above are the *known* uses, but the concrete class may expose additional state that future call sites will want.
- Verify the concrete classes implicitly satisfy both Protocols by running `isinstance(controller, EstradiolControllerProto)` and `isinstance(monitor, ConfidenceMonitorProto)` in one-off tests.
- In `sample_mlx.py`: import both Protocols, change the parameter annotations at lines 429 and 434 as described above. Remove the `hasattr(controller, "reset")` guard at line 470 (the Protocol now guarantees it) IF the existing runtime behavior doesn't depend on it; keep the `hasattr` if other callers pass controllers without `.reset()`.

**Tests (RED first):**
- RED: `mypy src/tgirl/sample_mlx.py 2>&1 | grep -cE "object.*has no attribute"` → 13 (8 for controller/V_basis/etc. on lines 472–495, 5 for confidence_monitor on lines 535/640/642/643/676)
- Add `tests/test_estradiol.py::test_controller_satisfies_proto` — `isinstance(EstradiolController(...), EstradiolControllerProto)` is True.
- Add `tests/test_state_machine.py::test_monitor_satisfies_proto` — `isinstance(ConfidenceMonitor(...), ConfidenceMonitorProto)` is True (or whatever the concrete class is named — find via grep).
- GREEN: all 13 attr-defined errors resolve; both proto tests pass.
- Full suite: `tests/test_sample_mlx.py`, `tests/test_estradiol.py`, `tests/test_estradiol_integration.py`, `tests/test_state_machine.py` all still pass.

**Validation:** `pytest tests/test_sample_mlx.py tests/test_estradiol.py tests/test_estradiol_integration.py tests/test_state_machine.py -v`; `mypy src/tgirl/sample_mlx.py` error count drops by ≥13.

**Commit:** `feat(types): define Estradiol+ConfidenceMonitor protos and type sample_mlx params`

---

### Task 6: Type the steered forward-fn via existing `ForwardResult` + `SteerableForwardFn` protocol

**Files:**
- `src/tgirl/cache.py` (the `ForwardResult` NamedTuple already exists at `cache.py:21` with `logits: Any, probe_alpha: Any | None = None` — NO NEW TYPE NEEDED. `make_steerable_mlx_forward_fn` (lines 309–408) already returns `ForwardResult`. The non-steerable `make_mlx_forward_fn` still returns raw `mx.array` — this task unifies them.)
- `src/tgirl/sample_mlx.py` (update `forward_fn` parameter type at line 422 from `Callable[[list[int]], mx.array]` to a Protocol that models the optional `steering=` kwarg and returns `ForwardResult`; update callers at lines 490 and 498)
- `src/tgirl/cache.py` (update `make_mlx_forward_fn` — the non-steerable factory — to also return `ForwardResult(logits, probe_alpha=None)` for a unified return type)
- Callers of the non-steerable factory (grep for `make_mlx_forward_fn` and any torch equivalent) — may need one-line `.logits` accessor updates.

**Approach:**
- DO NOT define `ForwardFnResult`; the existing `cache.ForwardResult` NamedTuple is the canonical type. Training-partner reviewer flagged the original plan was duplicating an existing symbol.
- Define a new `SteerableForwardFn(Protocol)` inside `sample_mlx.py` (co-located with `run_constrained_generation_mlx`):
  ```python
  class SteerableForwardFn(Protocol):
      def __call__(
          self, token_history: list[int], *, steering: Any | None = None,
      ) -> ForwardResult: ...
  ```
  Import `ForwardResult` from `tgirl.cache`.
- Unify the two mlx forward factories in `cache.py` so both return `ForwardResult`. The non-steerable path builds `ForwardResult(logits, probe_alpha=None)`. This removes the return-type polymorphism that blocks mypy narrowing.
- At `sample_mlx.py:498` (the non-steered call), simply read `.logits` off the ForwardResult instead of the raw array. This is the rippled caller change.
- Resolves three errors directly (attr-defined at 491, 492, 493) AND the call-arg error at 490 (`Unexpected keyword argument "steering"`) — four errors total, not three as originally stated.

**Tests (RED first):**
- RED: `mypy src/tgirl/sample_mlx.py 2>&1 | grep -cE '"array" has no attribute.*(logits|probe_alpha)|Unexpected keyword argument "steering"'` → 4
- Add `tests/test_cache.py::test_make_mlx_forward_fn_returns_forwardresult` — asserts the non-steerable factory now returns `ForwardResult` with `probe_alpha=None`.
- Add `tests/test_sample_mlx.py::test_forward_fn_protocol_accepts_steering_kwarg` — asserts the steerable factory accepts the kwarg and returns a `ForwardResult` typed object.
- GREEN: 4 errors resolve.
- Full suite still 1118 passed. `tests/test_cache.py` remains green (non-steerable callers still work because `.logits` attribute exists).

**Validation:** `pytest tests/test_sample_mlx.py tests/test_cache.py -v`.

**Commit:** `refactor(cache): unify mlx forward factories to return ForwardResult`

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

### Task 10: Union narrowing & scope-shadowing cleanup (calibrate.py, modulation.py, rerank.py, lingo/types.py)

**Files:**
- `src/tgirl/calibrate.py` — lines 36, 39, 467, 562, 575, 628 (union-attr and arg-type from MLX `svd` return narrowing; `stream=` kwarg type)
- `src/tgirl/modulation.py` — 4 errors around line 298 (conditioner tuple indexing, `SourceConditionerConfig` vs `float`)
- `src/tgirl/rerank.py` — 2 errors at lines 116, 121 (tuple-as-dict-key narrowing). Line 143 (`GrammarState` vs `GrammarStateMlx`) was originally listed here but moved to Task 12d — see below.
- `src/tgirl/lingo/types.py` — 4 errors at lines 128–130: these are **variable-shadowing cleanup, not a latent bug**. Line 117 declares `anc = {name}` as `set[str]`; line 128's for-loop `for anc in ancs:` shadows the outer variable, iterating `frozenset[str]` so the loop variable is `str`. mypy carries the outer `set[str]` annotation into the inner scope. Runtime is correct. Reclassified from Task 12 to Task 10 after training-partner review established this is pure shadowing, not a set/str-confusion bug.

**Approach:**
- `calibrate.py:36, 467, 562, 628`: `stream=mx.cpu` has mypy type `DeviceType` but `svd` expects `Stream | Device | None`. This is an MLX stub gap — the real `mx.cpu` IS a `Device`. Use `cast(Any, mx.cpu)` at call sites, or (preferred) add `# type: ignore[arg-type]  # mlx stub gap: mx.cpu is Device` at each site. Budget: 4.
- `calibrate.py:39, 575`: `mx.linalg.svd(...)[1]` returns `int | float | list[list_or_scalar]` per mypy — but at runtime it's always `mx.array`. `cast(mx.array, result)` or `assert isinstance(result, mx.array)` narrowing.
- `modulation.py:298-305`: `list(self.conditioners)` is typed as `list[float]` but the member type is `SourceConditionerConfig`. Likely the caller's expected type is wrong; fix by annotating the `new: list[SourceConditionerConfig]` properly.
- `rerank.py:116, 121`: `tuple[object, ...]` as dict key — add explicit type annotation on the tuple construction: `key: tuple[str | int, ...] = tuple(...)`.
- `lingo/types.py:128`: rename the inner for-loop variable: `for anc in ancs:` → `for ancestor in ancs:` (and update `ancestor != name` and `ancestor in self._descendants` and `self._descendants[ancestor].add(name)`). One-line semantic-preserving rename. No test needed — runtime is already correct.

**Tests (RED first):**
- RED: `mypy src/ 2>&1 | grep -cE "union-attr|index|assignment|comparison-overlap"` in these files → 10 (4 calibrate + 4 modulation + 2 rerank.116/121 + 4 lingo/types.128-130; minus rerank.143 which moved to Task 12d; confirm with fresh mypy output)
- GREEN: resolved.
- Full suite still 1118 passed. Specifically `tests/test_lingo_types.py` passes unchanged (the shadowing rename is a no-op semantically).

**Validation:** `pytest tests/test_calibrate.py tests/test_modulation.py tests/test_rerank.py tests/test_lingo_types.py -v`.

**Commit:** `fix(types): narrow unions and rename shadowed loop variable in lingo/types`

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

### Task 12: Latent bug investigation (sample_mlx.py:495, sample.py:1090, rerank.py:143 cross-framework)

**Files:**
- `src/tgirl/sample_mlx.py` — line 495 (`Item "None" of "list[list[float]] | None" has no attribute "append"`)
- `src/tgirl/sample.py` — line 1090 (`ToolRouter | None` has no attribute "route") — similar None-dereference
- `src/tgirl/rerank.py` — line 143 (`GrammarState` passed where `GrammarStateMlx` expected) paired with `src/tgirl/sample_mlx.py:506-507` fallback (cross-framework conversion)
- Tests: `tests/test_sample_mlx.py`, `tests/test_sample.py`, `tests/test_rerank.py`, `tests/test_cache.py`

**Reclassifications established during training-partner plan review:**
- `serve.py:88` (originally 12b) — stub gap, not a bug. Moved to Task 13 as budgeted `# type: ignore[misc]`.
- `lingo/types.py:128-130` (originally 12a) — variable shadowing, not a set/str bug. Moved to Task 10 as a loop-variable rename.
- `rerank.py:143 + sample_mlx.py:506-507` (originally scattered across Task 10) — promoted to Task 12d because the runtime fallback performs a cross-framework `torch → numpy → mx` conversion, which violates CLAUDE.md's "No cross-framework conversions" invariant.

**Approach — ONE investigation per bug:**

**12a. `sample_mlx.py:495`:** `estradiol_alphas` is typed as `list[list[float]] | None`, initialized to `None` at line 461, re-assigned to `[]` at line 474 inside `if controller is not None`, then `.append()` at line 495 inside the same `if controller is not None and _steering is not None` guard. The guard narrows to `not None` for `_steering` but mypy can't see the same for `estradiol_alphas`. Fix: move `estradiol_alphas` / `estradiol_deltas` initialization outside the conditional (`= None` always), or narrow at the append site with `assert estradiol_alphas is not None`. The latter preserves the current "only allocated when needed" pattern.

**12b. `sample.py:1090`:** `self._tool_router: ToolRouter | None` is accessed as `self._tool_router.route(...)` without narrowing. Either narrow with `assert self._tool_router is not None` at the call site, or hoist the check.

**12c. `rerank.py:143 + sample_mlx.py:506-507` (cross-framework conversion investigation):** At `rerank.py:125` `self._grammar_guide_factory(routing_grammar_text)` returns a `GrammarState` (torch-protocol); at line 142 it's passed to `run_constrained_generation_mlx(grammar_state=...)` which expects `GrammarStateMlx`. Runtime only "works" because of the fallback at `sample_mlx.py:502-507`:
```python
if has_mlx_mask:
    valid_mask = grammar_state.get_valid_mask_mx(vocab_size)
else:
    # Fallback for torch-based grammar states
    valid_mask_torch = grammar_state.get_valid_mask(vocab_size)
    valid_mask = mx.array(valid_mask_torch.numpy())  # <-- cross-framework conversion
```
This violates CLAUDE.md: "No cross-framework conversions. If a function needs both MLX and torch, implement two variants with matching interfaces."

Two possibilities, one test to disambiguate:
1. **Factory is backend-aware** — when `_backend == "mlx"`, `_grammar_guide_factory` actually returns a `GrammarStateMlx`; the type is just widened. Fix: type the factory as `Callable[[str], GrammarState | GrammarStateMlx]`, `assert isinstance(grammar_state, GrammarStateMlx)` in the MLX branch of rerank.py. Delete the fallback path in sample_mlx.py:506-507 (no callers hit it). This eliminates the cross-framework conversion.
2. **Factory always returns torch `GrammarState`** — the MLX path silently pays the torch→numpy→mx cost every call. That's a real performance bug AND a CLAUDE.md violation. Fix: add a backend-aware factory dispatch in rerank (serve.py already provides `mlx_grammar_guide_factory` separately at line 48); rerank should choose the MLX factory when `_backend == "mlx"`. Delete the fallback at sample_mlx.py:506-507.

Regression test (RED): `tests/test_rerank.py::test_mlx_backend_uses_mlx_grammar_state` — assert `isinstance(grammar_state, GrammarStateMlx)` inside `run_constrained_generation_mlx` (hook via debug callback) when rerank is configured with `backend="mlx"`. If this test fails on `main`, it's hitting possibility (2) and needs the factory-dispatch fix. If it passes, the type annotations just need tightening (possibility 1).

Either way, the `valid_mask = mx.array(valid_mask_torch.numpy())` fallback at `sample_mlx.py:506-507` is deleted.

**Tests (RED first):**
- For each latent bug 12a–12c: write a test that demonstrates the issue. Failing test → fix → passing test + type resolved.
- For 12c: the regression test stays even after the fix, to catch re-introduction of the cross-framework fallback.
- Aggregate: `mypy src/` error count drops by 4 (12a: 2 union-attr at 495/496; 12b: 1 union-attr at 1090; 12c: 1 arg-type at rerank.py:143 + 1 attr-defined at sample_mlx.py:506 = 2, with possible additional errors if factory typing tightens).

**Validation:** `pytest tests/test_sample_mlx.py tests/test_sample.py tests/test_rerank.py tests/test_cache.py -v -k "the new test names"`.

**Commit:** `fix(rerank, sample): resolve None-dereference and cross-framework conversion bugs`

---

### Task 13: Remaining scatter-gather (arg-type, call-arg, misc)

**Files:**
- `src/tgirl/sample.py` — lines 143 (`ModelIntervention(**dict)`), 823 (`ToolRouter(backend=str)`), 942, 987, 1002
- `src/tgirl/sample_mlx.py` — lines 490 (`steering=` kwarg), 506 (`get_valid_mask` vs `get_valid_mask_mx`)
- `src/tgirl/serve.py` — 88 (`mlx_lm.load` stub declares 3-tuple but runtime returns 2-tuple — `# type: ignore[misc]  # mlx_lm.load stub gap: runtime returns (model, tokenizer)`; budget: 1), 1577 (`enable_thinking` kwarg on `PromptFormatter`), 739, 1619 (missing return annotations), 1777 (no-untyped-call on stream_gen)
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

### Per-task mypy gate (team-mode)

Because this branch is based on `main` (NOT on `chore/dialectic-framework-sync`), the local pre-commit hook does NOT run mypy. Without an explicit gate, a mid-work regression — e.g. Task 10 introducing a new `arg-type` while resolving union-narrowing — would go unnoticed until the final Task 14 check.

**Mitigation: the training partner enforces a per-commit mypy gate during `/execute-team`.**

Between every proposer commit, the training partner runs:
```bash
mypy src/ --ignore-missing-imports 2>&1 | grep -c "error:"
```
and compares the count to the baseline (100 at task start, decreasing monotonically per task's "GREEN" target). If the count rises relative to the previous commit, the training partner blocks the next task and messages the proposer to fix the regression before proceeding. This is message-gated, not hook-gated — the gate lives in the team protocol, not the git hook.

Secondary fallback: the proposer appends the pre/post mypy error counts to each commit footer (e.g. `mypy: 100 → 87 (-13)`) so reviewers can audit monotonic decrease without re-running mypy.

This is explicit in the PRP so the team lead can hold the training partner accountable to it.

## 5. Rollback Plan

Each task is one commit → revert-range granularity is per-task.

- Per-task failure → `git revert <sha>` for that task alone. Earlier task commits stay if they don't depend on it (Tasks 1–4 have no inter-dependencies; Tasks 5–8 depend on Task 1's mypy baseline capture; Task 10+ depends on earlier narrowing for some call sites).
- Protocol-addition tasks (5, 6, 7) are structural and non-revertible trivially if later tasks use them. If a Protocol turns out to be wrong-shaped, prefer fixing forward rather than reverting.
- Latent-bug fixes in Task 12 each include a regression test — if the fix regresses something else, revert the task and re-investigate (may indicate an unidentified dependency).
- Full-feature abandon: revert in reverse order (Task 14 → Task 1). The feature branch can be abandoned; main is untouched since this is a branch-only change until merge.

## 6. Uncertainty Log

- **MLX stub completeness.** MLX has partial `*.pyi` stubs but `mx.cpu` / `mx.gpu` / `mx.linalg.svd` have known gaps. Task 10 assumes 4 `# type: ignore[arg-type]` for these; if there are more, the budget creeps toward the limit. If we exceed 10 total `# type: ignore` comments, add a `[[tool.mypy.overrides]] module = "tgirl.calibrate"` override in pyproject.toml with `disable_error_code = ["arg-type"]` and document the scope.
- **Running `# type: ignore` budget accounting (post-review).** Task 7: 2 (llguidance.mlx re-export). Task 10: 4 (mx stream/svd stubs). Task 13: 1 (serve.py:88 mlx_lm.load stub) + 1 (compile.py:438 RestrictedPython). **Running total: 8 of 10.** Leaves 2 slack for unforeseen stub gaps. Tasks that were originally candidates for ignores and resolved WITHOUT them: Task 5 (ConfidenceMonitor + Estradiol Protocols — 13 errors fixed via Protocol, zero ignores); Task 6 (ForwardResult — unified types, zero ignores); Task 12c (rerank cross-framework — architectural fix, zero ignores). If the count threatens to exceed 10 during execution, escalate to a `[[tool.mypy.overrides]]` block (see MLX bullet above) rather than adding more scattered ignores.
- **Task 6 scope of change.** Unifying the two MLX forward factories (`make_mlx_forward_fn` + `make_steerable_mlx_forward_fn`) to both return `ForwardResult` may ripple to `sample.py` (torch path) and tests that consume the non-steerable factory. If updating the factory signature breaks more than 3 call sites, split Task 6 into 6a (unify factories) + 6b (migrate torch callers). Add tasks on the fly during execution.
- **Pydantic strict-init impact.** Task 13's `ModelInterventionDict` TypedDict might require cascade typing changes in callers. If the cascade is big (>2 files), defer and use `cast(dict[str, Any], source)` with a rationale comment.
- **`compile.py:438` dynamic base.** If the `# type: ignore[misc]` on `class ... RestrictingNodeTransformer` turns out to hide subclass-signature bugs, the right fix is contributing stubs upstream. Not in scope — document the deferral.
- **Task 12c disambiguation.** rerank.py's `_grammar_guide_factory` may be backend-aware at runtime (possibility 1 in Task 12c) or always-torch with a silent cross-framework conversion fallback (possibility 2). The Task 12c regression test disambiguates them on the first RED run. If it's possibility 2, the fix is larger (wire `mlx_grammar_guide_factory` through rerank); budget up to 30 LoC of rerank.py changes.
- **`state_machine.py:632, 649` TransitionDecision narrowing.** Unknown what's returning `Any` there without reading. May require a larger Protocol-narrowing refactor. If the fix exceeds 20 LoC, split into a dedicated task.
- **Branch base & mypy gate enforcement.** This branch is based on `main`, not `chore/dialectic-framework-sync`. The local pre-commit hook does NOT run mypy. Mitigation (see §4 "Per-task mypy gate (team-mode)"): the training partner runs `mypy src/ 2>&1 | grep -c "error:"` between every proposer commit and blocks the next task if the count rises. Commit footers also carry `mypy: X → Y` deltas for audit.
- **ConfidenceMonitor Protocol surface.** Task 5 infers the Protocol from five observed call sites (should_checkpoint, record_log_prob, should_backtrack, record_backtrack, backtracks_remaining). If the concrete monitor class exposes state that future tgirl call sites will want (e.g. `log_prob_history`), the Protocol may need expansion. Start minimal; expand if a future task trips attr-defined on a new method.
- **Mypy version drift.** CI may run a different mypy version than local. Pinning `mypy==1.x` in `pyproject.toml[project.optional-dependencies.dev]` is out of scope but worth noting if we hit drift.
