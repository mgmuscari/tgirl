# Code Review: probe-vector-persistence

## Verdict: APPROVED
## Reviewer Stance: Team — Code Reviewer + Implementation Defender
## Date: 2026-04-14
## Mode: Agent Team (message-gated code review)

## Context

This feature was built on **iterative tier**, so there is no PRP for the reviewer
to compare against. The review was conducted against three sources of truth:

1. CLAUDE.md operating principles and gotchas
2. Commit message intent across the feature commits
3. The test suite as behavioral contract

Branch: `feature/probe-vector-persistence`
Feature-branch commits at review start: `cdeb7b0`, `fbba2da`, `933820b`, `dcdc689`
Tests at review start: 42 passing in `tests/test_serve.py` (8 in `TestProbePersistence`)
Tests at review end: 55 passing (21 in `TestProbePersistence`)

## Feature Summary

CLI-flag-driven probe vector persistence for the tgirl inference server:

- `--probe-load PATH` — populate the self-steering probe cache at FastAPI lifespan
  startup, so the first generated token inherits behavioral state from a prior session.
- `--probe-save-on-shutdown PATH` — write the cache to disk on lifespan shutdown.
- `--probe-autosave-interval SECONDS` — periodic saves via an asyncio background
  task during server lifetime. Requires `--probe-save-on-shutdown`.

Plus a latent-bug fix to `/v1/steering/status` (the `**_steering_stats` unpack was
masking the live-computed `probe_cached`).

## Issues Found

### 1. Autosave task death silently blocks shutdown save

**Category:** Reliability / exception handling
**Severity:** Significant → Significant
**Location:** `src/tgirl/serve.py` — `_autosave_loop` and lifespan `finally` block

**Details:** Any `np.save` failure inside `_autosave_loop` (disk full, permission
denied, transient I/O) would kill the task. At lifespan exit, `await autosave_task`
re-raised the original OSError, which the `except asyncio.CancelledError` clause
did not catch — the `finally` exited before the shutdown save ran. This nullified
`--probe-save-on-shutdown` on exactly the failure mode the flag exists to protect.

**Defender Response:** Fixed in `7a9dabc`. Per-tick `try/except Exception` with
`logger.exception` keeps the task alive across transient write failures.
Regression test: `test_autosave_write_failure_does_not_block_shutdown_save`
patches `numpy.save` to raise on autosave ticks and asserts the lifespan exits
cleanly without propagating.

**Resolution:** Resolved.

### 2. Non-atomic writes corrupt probe on crash

**Category:** Data integrity / durability
**Severity:** Significant → Significant
**Location:** `src/tgirl/serve.py` — `_write_probe`

**Details:** `np.save(path, arr)` streams bytes directly to the canonical path.
A crash mid-write leaves a truncated file, which then crashes the next startup's
`np.load(path)` — on the exact recovery path persistence promises. The autosave
loop widens the exposure window (more writes, more crash opportunities).

**Defender Response:** Fixed in `3a2f662`. Write to sibling `.tmp` then
`os.replace` (atomic on POSIX and Windows for same-filesystem renames). Used
`open(path, "wb")` + `np.save(f, arr)` — the file-object form bypasses `np.save`'s
string-path `.npy` suffix auto-append, keeping save/load paths symmetric for users
who didn't include the `.npy` suffix. Additionally wrapped the shutdown save
call site in `try/except` for the same fragility class as the autosave loop.
Regression tests: atomic write under `os.replace` failure + no-`.npy`-suffix
round-trip.

**Resolution:** Resolved.

### 3. Missing lower-bound validation on autosave interval

**Category:** Input validation / DoS
**Severity:** Significant → Significant
**Location:** `src/tgirl/cli.py` (CLI guard) + `src/tgirl/serve.py` (`create_app` guard)

**Details:** `--probe-autosave-interval 0` (or negative) produced a tight
disk-saturating write loop, because `asyncio.sleep(<=0)` returns immediately
in Python 3.11+. A single-character fat-finger (`30` → `0`) would DoS the
server via event-loop starvation and disk saturation from `os.replace` per tick.

**Defender Response:** Fixed in `42ff2ea`. Added `<= 0` guards in both the
click layer (`UsageError` before model load) and in `create_app` (`ValueError`),
symmetric with the pre-existing `requires save_path` validation. 8 new
parametrized tests covering `0`, `0.0`, `-1.0`, `-0.001` (and CLI string
variants).

**Resolution:** Resolved.

### 4. Dead writes to `_steering_stats["probe_cached"]`

**Category:** Dead code / confusing state
**Severity:** Minor → Minor
**Location:** `src/tgirl/serve.py` — stats dict init, `_generate_tokens`,
`_generate_tokens_streaming`

**Details:** The `dcdc689` fix to `/v1/steering/status` flipped the unpack
order so the live-read `probe_cached` override always wins over the stat
entry. That made all three writes to `_steering_stats["probe_cached"]` dead
stores with no reader. Risk: a future reader restoring the original unpack
order assumes the override is redundant and reintroduces the masking bug.

**Defender Response:** Fixed in `ce16db9`. Removed the stat init entry and
both in-flight writes. Rewrote the override comment forward-looking — the
invariant is now stated as "live reads from source of truth, never mirror
into stats" rather than a historical note. No new tests (existing behavioral
test covers the live-read path).

**Resolution:** Resolved.

### 5. Missing shape/dtype validation on probe load

**Category:** Input validation / silent misuse
**Severity:** Minor-leaning-Significant → Minor
**Location:** `src/tgirl/serve.py` — lifespan startup load

**Details:** Startup `np.load` accepted any `.npy` file unconditionally.
A wrong-model probe with incompatible hidden dim would produce cryptic MLX
shape errors at first request; a compatible-dim mismatch (same width, foreign
basis) would produce silently-wrong steering with no error at all. Dtype
mismatches (float64 save vs float32 hook) would shift the `alpha * v_probe`
magnitude calibration.

**Defender Response:** Fixed in `a94a2c6`. Added pre-load path log for
operator diagnosability; shape check against `ctx.embeddings.shape[-1]`
(both MLX and torch backends already populate this, no new coupling) with
an actionable error message naming the shape, expected dim, path, and
remediation; cast loaded ndarray to float32 to match `_BottleneckHook`'s
native capture dtype. 2 new tests: shape mismatch rejection + float32 cast
verification. `_make_mock_ctx` extended with a backward-compatible
`hidden_dim` kwarg.

**Resolution:** Resolved.

## What's Done Well

- **Every finding was fixed rather than defended-only.** No back-and-forth on
  severity, no test weakening, no "will fix later" deferrals.
- **Each fix commit addressed the root cause, not the symptom.** The atomic-
  write fix also caught the `.npy` suffix asymmetry. The stats cleanup rewrote
  the override comment forward-looking to prevent regression. The shape check
  also added a pre-load log for operator diagnosability.
- **Commit messages name the failure mode by hazard, not by symptom.** A future
  reader can scan `git log --oneline` and immediately understand what class of
  bug each fix addressed ("autosave task death", "atomic probe writes", etc.).
- **No CLAUDE.md violations.** Cross-framework conversions (`mx.array(arr)`,
  `np.array(v)`) live only at `.npy` I/O boundaries. Nothing lands in a per-token
  hot path. No fix-later shims.
- **Test discipline throughout.** Every fix shipped with at least one regression
  test. Parametrized validation for bounds cases. Behavioral contract tested
  from both the create_app API and the CLI layer.
- **Latent-bug pickup.** The initial `/v1/steering/status` fix (`dcdc689`)
  addressed a pre-existing bug that would have continued to mask the live
  `probe_cached` state in ad-hoc diagnostics. Caught while wiring the new load.

## Summary

**Finding counts by final severity:**
- Blocking: 0
- Significant: 3 (all resolved)
- Minor: 2 (all resolved)
- Nit: 0

**Blocking issues resolved:** N/A (none raised).

**Branch state at review end:**
- 9 commits on `feature/probe-vector-persistence` (4 feature + 5 fix/refactor).
- 55 tests passing in `tests/test_serve.py` (21 in `TestProbePersistence`).
- Pushed to `origin`, working tree clean.

**Recommendation:** Ready to merge. Open the PR.
