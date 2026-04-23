# PRD: mypy Strict-Mode Cleanup

## Status: DRAFT
## Author: Maddy Muscari (drafted with Claude Opus 4.7)
## Date: 2026-04-22
## Branch: feature/mypy-strict-cleanup

## 1. Problem Statement

The Dialectic framework sync (`chore/dialectic-framework-sync`) replaced tgirl's pre-commit hook with the upstream template's hook, which runs both `ruff check src/` and `mypy src/` under `strict = true`. Ruff is now clean; mypy reports **100 pre-existing type errors in 17 `src/tgirl/*` files**. Until these are addressed, every future `src/` commit requires `--no-verify`, which defeats the point of the gate.

The errors predate this session — they accumulated because the prior pre-commit hook did not run mypy exhaustively. Many are mechanical (missing generic type parameters, stale `# type: ignore` comments, missing return annotations). Others are substantive:

- Several fields typed as `object` when the code actually uses them through a well-defined protocol surface (ESTRADIOL controller, inference hooks, tokenizer — in `sample_mlx.py`, `format.py`).
- Union-type narrowing gaps (`calibrate.py`, `rerank.py`, `modulation.py`) where the code branches on types but mypy can't prove it.
- A handful of latent bugs where mypy is catching real problems: e.g. `sample_mlx.py:495` dereferences a list that's typed as possibly-None, `sample.py:1090` dereferences `ToolRouter | None`, `rerank.py:143` violates CLAUDE.md's "no cross-framework conversions" invariant by passing a torch `GrammarState` into an MLX-only function path. (Note: initial drafting also listed `serve.py:88` tuple unpack and `lingo/types.py` set/str confusion as bugs — plan review identified these as a mlx-lm stub gap and a variable-shadowing rename respectively, not bugs. PRP is the authoritative classification.)

**Why now:** This is the first time mypy is actually enforced on this codebase. Every week that passes accumulates more drift, and the `--no-verify` muscle memory erodes the gate's value. Fixing it once, now, re-establishes mypy as a genuine quality signal.

## 2. Proposed Solution

Work through the error set in four stages of decreasing mechanicalness, committing incrementally so each stage is independently reviewable and revertible:

1. **Mechanical annotations** — add missing generic parameters (`Callable[..., Any]`, `list[X]`, `Pattern[str]`), delete stale `# type: ignore` comments, annotate missing return types, annotate missing argument types. ~35 errors.
2. **Protocol definitions for `object`-typed fields** — define Protocols for the ESTRADIOL controller, inference hook extensions (config, max_tokens), and tokenizer surface; replace `object` annotations at call sites. ~20 errors. This is the biggest design chunk.
3. **Union narrowing** — add `isinstance` guards or `cast()` where mypy can't prove a union member; fix `no-any-return` by casting return values through typed locals. ~25 errors.
4. **Latent bugs** — each gets individual investigation. If a bug is real, add a failing test first (RED), then fix (GREEN), commit together. If the code is correct and mypy is confused, narrow the type or add a justified `# type: ignore[rule]`. ~5–10 errors.

A further **5 errors** fall into small buckets (call-arg mismatches, `no-untyped-call` on `_mx`/`safe_open` helpers) — handled in stage 1 or 3 depending on shape.

No runtime behavior changes. No new dependencies. No public API surface additions — the Protocols added in stage 2 are internal documentation of already-existing duck-typed contracts.

## 3. Architecture Impact

### Files affected

All changes are in `src/tgirl/`. Error counts by file (from `mypy src/ --ignore-missing-imports`):

| File | Count | Notes |
|---|---|---|
| `sample_mlx.py` | 18 | Most `object` annotations; ESTRADIOL controller + hooks + grammar state |
| `calibrate.py` | 15 | SVD return narrowing, `no-any-return` from MLX APIs, `stream=` arg type |
| `sample.py` | 14 | Missing generics, `ModelIntervention` kwargs type, InferenceHook attributes |
| `state_machine.py` | 8 | Missing generics, overload narrowing for `kwargs.get("token_id")` |
| `serve.py` | 8 | Missing generics, tuple-unpack mismatch (latent bug), call-arg |
| `format.py` | 6 | Unused ignores + tokenizer typing |
| `estradiol.py` | 6 | Untyped `_mx`/`safe_open` helpers, missing return annotation |
| `outlines_adapter.py` | 4 | Tokenizer typing, llguidance.mlx re-export issues |
| `modulation.py` | 4 | Union narrowing for conditioner tuple indexing |
| `lingo/types.py` | 4 | Latent bug cluster (set/str confusion) |
| `rerank.py` | 3 | Tuple-as-dict-key narrowing, GrammarState vs GrammarStateMlx mismatch |
| `compile.py` | 2 | Generic `list`, dynamic base class |
| `bridge.py` | 2 | `no-any-return` from MCP API |
| `bfcl.py` | 2 | Stale `# type: ignore` |
| `__init__.py`, `cli.py`, `grammar.py`, `lexicon.py`, `tdl_parser.py` | 1 each | Miscellaneous |

### New types / protocols

- **`EstradiolControllerProto`** (likely in `estradiol.py` or a new `protocols.py`): defines the `.V_basis`, `.alpha_current`, `.step()`, `.make_steering_state()`, `.reset()` surface that `sample_mlx.py` uses. Replaces `controller: object`.
- **`ForwardFnResult`** protocol or concrete dataclass in `cache.py` / `sample_mlx.py`: the steered forward pass returns an object with `.logits` and `.probe_alpha`. Currently typed as `mx.array` which is wrong when the forward is steered.
- **`TokenizerProto`** (in `format.py`): the chat-template-callable tokenizer surface (`apply_chat_template` at minimum). May also apply to `outlines_adapter.py`.
- Possibly **`InferenceHookConfig`** or direct attribute additions on `InferenceHook`: `config`, `max_tokens` (used in `sample.py:1187,1190`).

### No API changes

- No public entrypoint signature changes.
- No new runtime dependencies.
- Protocols added are `runtime_checkable` where duck-typing is already used; `@runtime_checkable` gives us `isinstance` checks for free.
- No changes to test infrastructure; existing 1118 tests continue to exercise the same code paths.

### Config changes

- `pyproject.toml`: may add `[[tool.mypy.overrides]]` blocks for third-party modules without stubs (`mlx`, `mlx_lm`, `llguidance`, `transformers`) to silence missing-import noise without relaxing strictness on our code.

## 4. Acceptance Criteria

1. `ruff check src/` passes cleanly (already true — baseline preserved).
2. `mypy src/` passes cleanly with no errors under the current `strict = true` config (or an explicitly documented relaxation, with justification, in `pyproject.toml`).
3. Full test suite passes: `pytest tests/` returns 1118 passed (the current count). No new tests required *for the annotation work itself*, but latent-bug fixes in stage 4 MUST ship with a failing-then-passing test.
4. `git commit` on this branch succeeds without `--no-verify` for all commits from stage 2 onward (stage 1 commits may need `--no-verify` because intermediate states leave mypy still failing — that's expected and documented in commit messages).
5. No behavioral regression on the ESTRADIOL calibration or steering paths: a manual smoke test (`tgirl serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit` + one `/v1/chat/completions` request with `estradiol_alpha=0.5`) still produces coherent output and the `/v1/steering/status` telemetry reflects steering being active.
6. The `ToolRouter` backend type-narrowing (`sample.py:823` — `Literal['torch', 'mlx']`) resolves via proper typing, not `# type: ignore`.
7. Latent bugs surfaced in stage 4 are documented in the Uncertainty Log with: bug location, explanation, whether it was a real bug (reproducible test added) or a type-correctness issue only (cast/narrow/ignore with rationale).

## 5. Risk Assessment

- **Protocol definitions could subtly change duck-typed contracts.** If a Protocol is too narrow (missing a method the real type has), mypy will pass but the code breaks at runtime on callers using the missing method. Mitigation: Protocols derived from the *actual* concrete class's public surface, with runtime_checkable for `isinstance` safety; existing test suite provides regression coverage.
- **Latent bugs may be load-bearing** — if `lingo/types.py` set/str confusion has been silently "working" due to duck-typing tolerances, fixing it properly could expose a missing test case. Mitigation: for each stage-4 latent bug, write a reproducing test FIRST and verify it fails on current `main`, then fix. If the bug doesn't reproduce, it's a type-correctness-only fix and should be documented.
- **MLX / llguidance stubs may be incomplete.** `mlx.core` has partial stubs; `llguidance.mlx` has exports mypy thinks aren't declared. We may need targeted `# type: ignore[attr-defined]` with explanation for third-party limitations. Don't conflate this with our own `object` typing problems.
- **`# type: ignore` proliferation.** Every `# type: ignore` is a future mypy-cost deferment. Budget: no more than 10 new `# type: ignore` comments total across the change, each with a `[rule]` specifier and a one-line reason. If we're tempted to add more, relax the mypy rule in `pyproject.toml` instead — visible policy beats scattered noise.
- **Error count may drift.** New errors could surface as we fix old ones (fixing a union collapse to a narrower type may expose a downstream mismatch). Plan for 1–2 additional commits of "new errors surfaced by fixes."
- **Cross-branch coordination.** The framework sync branch (`chore/dialectic-framework-sync`) is what made mypy fire in the first place. If this branch merges before that one, local mypy runs won't mirror CI. If `chore/dialectic-framework-sync` changes its pre-commit further, rebase may require re-verification. See Open Questions.
- **Pydantic plugin behavior.** `pyproject.toml` enables `pydantic.mypy` with `init_forbid_extra = true`. Some current `ModelIntervention(**dict)` call sites (e.g. `sample.py:143`) fail strict-init because the dict is typed as `dict[str, object]`. Fixing this may require TypedDict declarations.

## 6. Open Questions

1. **Relax strict mode instead?** Dropping `strict = true` from `pyproject.toml` would clear most errors but erodes the signal. Alternative: keep strict, add narrow per-module overrides for high-churn MLX files via `[[tool.mypy.overrides]]` — isolates the noise without global relaxation. Recommend: overrides-only, no global relaxation.
2. **Which branch does this base on?** This branch is created from `main`, which does NOT include the framework sync. The mypy cleanup is only meaningful with the new pre-commit hook in place. If the sync branch merges first, this branch rebases cleanly. If this branch merges first, the sync branch's pre-commit changes become redundant pre-commit invocations that the user is already in habit of bypassing. Recommend: merge `chore/dialectic-framework-sync` first.
3. **Protocol location convention.** Do Protocols live next to their implementations (`estradiol.py`) or in a separate `protocols.py` file? Consistent with the rest of the codebase: `GrammarState` protocol lives in `sample.py` (co-located). Follow that pattern.
4. **Is the `ruff` cleanup (from `chore/dialectic-framework-sync`) expected to merge before this?** If yes, this branch should rebase onto that branch, not off `main`, because the `per-file-ignores` in pyproject.toml will conflict. Recommend: rebase before merging.
5. **Pydantic strict init — preserve or loosen?** `init_forbid_extra = true` is load-bearing for registry snapshots (catches typos in kwargs). Don't relax; fix the call sites to use typed kwargs or TypedDict.
6. **Mypy on tests too?** Current `pyproject.toml` config scopes mypy to `src/`. Tests are not checked. This PRD does not expand scope; tests remain unchecked. Future work may extend.

## 7. Out of Scope

- **Runtime behavior changes.** This is a type-correctness pass. Any change that alters control flow or output must be accompanied by a test, and should be a separate commit from the annotation work.
- **Refactoring for ergonomics beyond what type-correctness requires.** If a function has bad ergonomics but mypy doesn't complain, leave it.
- **Expanding mypy scope to `tests/` or `benchmarks/`.** Out of scope; a future feature if desired.
- **Adding new tests beyond regressions for latent bugs.** Existing test suite is the coverage baseline.
- **Third-party stub contribution.** If `mlx-lm` or `llguidance.mlx` have incomplete stubs upstream, we do not patch them; we use targeted `# type: ignore` with explanations pointing to the upstream gap.
- **Removing `ruff` per-file-ignores.** Those are in the framework-sync branch's scope, not this one.
- **Fixing type errors in pre-existing `# type: ignore` comments that mypy now flags as unused.** Wait — that IS in scope (5 such errors). Clarify: deleting unused `# type: ignore` is in scope; adding new ones is strictly budgeted (see Risk #4).
- **Restructuring `sample_mlx.py` to collapse the `controller: object` parameter into a properly-typed dataclass.** Within scope to add a Protocol; out of scope to restructure the call signature.
