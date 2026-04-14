# PRP: Per-Token Fire-Control Autotuner

## Source PRD: docs/PRDs/per-token-fire-control-autotuner.md
## Date: 2026-04-14

## 1. Context Summary

Replace the turn-level autotuner with a per-token continuous-gain
controller. At each forward pass, observe rolling-window coherence +
certainty + their derivatives, emit natural-valued multiplicative
gains on (α, β, temperature), apply before the next forward pass.
Serialize the per-step trajectory as a safetensors bundle at turn
end. Interface is MLP-ready; v1 ships a heuristic controller only.

Live evidence motivating: 10 consecutive `sycophant_trap` turns in
`~/autotune.jsonl` with the turn-level controller flipping α between
0.40 and 0.50 and producing identical 25-token stuck responses. The
per-token controller observes trajectory, not outcome, and intervenes
mid-stream.

## 2. Codebase Analysis

### Existing plumbing to build on

- `src/tgirl/certainty.py:34 step_certainty(logits) → {entropy,
  top1_prob, top1_margin}`. Per-token, cheap. Already called per-step
  in both generation loops: `src/tgirl/serve.py:1326` (non-streaming)
  and `src/tgirl/serve.py:1671` (streaming). This is the fast-path
  observable; no rework needed.
- `src/tgirl/coherence.py:75 compute_coherence(tokens)` — takes full
  list, returns `{n_tokens, repeat_rate, bigram_novelty,
  token_entropy}`. We add a sibling `rolling_coherence(tokens,
  window)` that operates on the trailing window and can be called
  per step. Same return keys.
- `src/tgirl/autotune.py:43 Observables` and `:68 Action` — the
  turn-level dataclasses. They are intentionally shaped for the
  per-turn classifier (include `n_tokens`, `finish_reason`). For v1
  we keep them for the migration window but add new dataclasses
  `StepObservables` and `Gains`. Classifier-style `Regime` is no
  longer load-bearing; rationales become free-form strings.
- `src/tgirl/cache.py` — `_BottleneckHook` already supports per-step
  reconfiguration: `set_band(weights | None)`, `set_probe_steering(v,
  α)`, `set_raw_correction(c)`. The controller drives these between
  each forward pass; no hook changes required.
- `src/tgirl/serve.py:1052 _apply_steering(...)` — already called
  per-token in the generation loops. Currently reads `α` from a
  resolved-at-turn-start local variable; the controller makes this
  variable update per-step.

### Convention / gotcha hooks (CLAUDE.md)

- **Tensor math uses native library ops.** Rolling mean/derivative
  helpers stay in pure Python over short lists (W=16) — *not* tensor
  ops. If we ever push the window past ~1000 we'd want MLX, but at
  W=16 Python is faster than MLX launch overhead.
- **No cross-framework conversions.** `step_certainty` already takes
  an `mx.array`. The controller operates on Python floats aggregated
  from the certainty dict — no tensor data flows through the
  controller.
- **Safetensors for on-disk artifacts.** Matches
  `tgirl.estradiol.save_estradiol` pattern. `safetensors.numpy`'s
  `save_file` + `load_file` write/read `dict[str, np.ndarray]`
  atomically. Channel lengths must match.
- **`cache.py` zero-coupling invariant.** No new tgirl imports into
  `cache.py`. The controller lives in `autotune.py` and the serve
  layer orchestrates it; `cache.py` only sees the already-computed
  correction via its existing setters.
- **Hook-triggered string matching.** MLX materialization calls use
  `.item()` / `.tolist()` in new code paths to force evaluation
  without tripping the security-reminder string matcher, consistent
  with the pattern in `tests/test_cache.py` band tests.

### Files we will modify / create

- **Modify** `src/tgirl/autotune.py` — add `StepObservables`, `Gains`,
  `Controller` protocol, `HeuristicController`. Existing turn-level
  surface stays for this PR, deprecated next.
- **Modify** `src/tgirl/coherence.py` — add `rolling_coherence`.
- **Modify** `src/tgirl/serve.py` — gen loops call controller per
  token; per-step arrays accumulate; write safetensors at turn end;
  `/v1/steering/autotune` body takes `log_dir` now, `log_path`
  deprecated (accepted with warning); `/v1/steering/status` exposes
  controller state.
- **Modify** `scripts/test_autotune_live.py` — reads tensor log
  instead of JSONL.
- **New** `src/tgirl/autotune_log.py` — `write_turn(log_dir, arrays,
  metadata)` safetensors writer + `glob_turns(log_dir)` reader.
- **New tests** `tests/test_rolling_coherence.py`,
  `tests/test_controller.py`, `tests/test_autotune_log.py`. Extend
  `tests/test_serve.py::TestBandSteering` with per-step integration
  tests.
- **Retire** `_run_autotune_after_turn` in `src/tgirl/serve.py` and
  the `autotune_logs_to_jsonl_when_path_set` test case.

## 3. Implementation Plan

**Test Command:** `python -m pytest tests/ -q`

TDD is mandatory per CLAUDE.md: each task ships RED-failing tests
first, then the minimum implementation that turns them GREEN, then
a single atomic commit.

### Task 1 — `rolling_coherence` helper
**Files:** `src/tgirl/coherence.py`, `tests/test_rolling_coherence.py`
**Approach:**
- `rolling_coherence(tokens: list[int], window: int = 16) -> dict`
- When `len(tokens) < 2`: return safe defaults matching
  `compute_coherence`'s shape.
- When `len(tokens) < window`: operate on all available tokens (no
  truncation yet).
- When `len(tokens) >= window`: operate on `tokens[-window:]` only.
- Same four return keys: `n_tokens` (set to the window length
  actually used), `repeat_rate`, `bigram_novelty`, `token_entropy`.
**Tests:**
- `test_rolling_matches_full_when_under_window` — `len(tokens) <
  window` should produce the same output as `compute_coherence`.
- `test_rolling_uses_window_when_over` — a long stream with repeats
  only in the last W tokens should show the repeats; a long stream
  with repeats only in the first K tokens (K > window) should NOT
  show them.
- `test_rolling_safe_defaults_on_empty_and_single` — same safe
  defaults (`0.0, 1.0, 0.0` respectively).
- `test_rolling_key_set_stable` — identical keys to
  `compute_coherence`.
**Validation:** `python -m pytest tests/test_rolling_coherence.py -q`

### Task 2 — `StepObservables`, `Gains`, `Controller` protocol
**Files:** `src/tgirl/autotune.py`, `tests/test_controller.py`
**Approach:**
- `@dataclass(frozen=True) class StepObservables`:
  - `step_index: int`
  - `alpha: float`, `beta: float | None`, `temperature: float`
  - Per-step (just this token's logits):
    - `step_entropy: float`, `step_top1_prob: float`,
      `step_top1_margin: float`
  - Rolling-window means (from `rolling_coherence` +
    `step_certainty` running mean):
    - `H_norm: float`, `repeat_rate: float`, `bigram_novelty: float`
    - `mean_top1_prob: float`, `mean_top1_margin: float`,
      `mean_entropy: float`
  - Derivatives (finite difference between current value and value
    `W//2` steps ago; 0.0 when unavailable):
    - `dH_dt: float`, `drepeat_dt: float`, `dcertainty_dt: float`
  - `probe_norm: float`  (current cached `v_probe` norm, or 0.0)
- `@dataclass(frozen=True) class Gains`:
  - `alpha_gain: float`, `beta_gain: float | None`,
    `temperature_gain: float`
  - `rationale: str` — short free-form, logged but not enforced
- `class Controller(Protocol)`:
  - `def step(self, obs: StepObservables) -> Gains: ...`
**Tests:**
- `test_stepobservables_serializable_to_dict` — `asdict` works,
  all keys present.
- `test_gains_unit_identity` — `Gains(1.0, 1.0, 1.0)` represents a
  hold-steady signal.
- `test_controller_is_a_protocol` — can assign any object with a
  `step(obs) -> Gains` method.
**Validation:** `python -m pytest tests/test_controller.py -q`

### Task 3 — `HeuristicController` with proportional-derivative gains
**Files:** `src/tgirl/autotune.py`, `tests/test_controller.py`
**Approach:**
- `HeuristicController` stores configurable constants with 2D-sweep-
  derived defaults:
  ```
  target_H:       float = 0.72   # middle of signal band
  target_top1:    float = 0.55   # healthy commit strength
  k_alpha:        float = 1.5    # α proportional gain
  k_alpha_deriv:  float = 0.8    # α derivative weight
  k_temp:         float = 1.0    # temperature gain
  k_temp_certainty: float = 0.5  # temp responds to excess certainty
  beta_safe_floor:float = 2.0    # β never continuously mapped below
  alpha_max:      float = 1.0
  alpha_min:      float = 0.0
  temp_max:       float = 1.0
  temp_min:       float = 0.0
  ```
- `step(obs)` computes:
  ```
  # Positive error ⇒ H above target ⇒ model too diffuse ⇒ steer harder.
  # Negative error ⇒ H below target ⇒ model collapsing ⇒ ease back.
  error_H = obs.H_norm - target_H
  # Anticipate: if H trending the wrong way, pre-correct.
  anticipated_H = obs.H_norm + k_alpha_deriv * obs.dH_dt
  alpha_gain = exp(-k_alpha * (target_H - anticipated_H))
  # Temperature: raise when model is over-certain and output
  # structure is collapsing (sycophant/loop signature).
  top1_excess = max(0.0, obs.mean_top1_prob - target_top1)
  temp_gain = 1.0 + k_temp * k_temp_certainty * top1_excess
  # β: continuous over [beta_safe_floor, ∞). No gain; propose a next
  # value directly based on whether we need more or less spread.
  # v1: if obs.repeat_rate rising OR H_norm falling, widen (lower β).
  # Else hold. Clip to floor.
  ```
- When `obs.step_index < W // 2`: derivatives are undefined, return
  `Gains(1.0, 1.0, 1.0)` (hold steady).
- Clipping + β sanity happens at the serve layer (Task 5), not here.
  This keeps the controller pure.
**Tests:**
- `test_hold_steady_at_target` — when H and top1 are exactly at
  targets and derivatives are 0, all gains are 1.0 ± 1e-6.
- `test_alpha_gain_increases_when_entropy_above_target` — H > target
  ⇒ `alpha_gain > 1.0`.
- `test_alpha_gain_decreases_when_entropy_below_target` — H < target
  ⇒ `alpha_gain < 1.0`.
- `test_derivative_anticipates_trend` — with H currently at target
  but trending DOWN (`dH_dt < 0`), `alpha_gain < 1.0` (pre-emptive
  ease).
- `test_temperature_gain_tracks_excess_certainty` — ramp
  `mean_top1_prob` from 0.3 → 0.95 and verify `temp_gain` rises
  monotonically from 1.0.
- `test_gains_bounded_under_adversarial_observables` — for
  pathological observables (H=0, H=1, derivatives at extremes), all
  gains stay finite and in a documented [0.1, 5.0] band. Prevents
  runaway.
- `test_step_index_below_warmup_returns_unit_gains` — `step_index < 8`
  ⇒ all gains 1.0.
- `test_no_oscillation_under_sycophant_trajectory` — provide a
  synthetic trajectory replaying the live 25-token sycophant pattern
  (H→0.98, n_tokens<30, finish=stop, repeat=0); controller should
  produce a smooth α reduction sequence, not a ±0.1 flip.
**Validation:** `python -m pytest tests/test_controller.py -q`

### Task 4 — `autotune_log.py` safetensors writer + reader
**Files:** `src/tgirl/autotune_log.py`, `tests/test_autotune_log.py`
**Approach:**
- `@dataclass class TurnTrajectory`: the channel arrays +
  metadata.
- `write_turn(log_dir: Path | str, trajectory: TurnTrajectory) ->
  Path`:
  - Ensure `log_dir` exists (mkdir parents).
  - Filename: `turn_{UTC_ISO}_{finish}.safetensors`.
  - Arrays stored as `np.float32` for real-valued channels,
    `np.int32` for token IDs + step_index, with `n_steps` as a 0-d
    metadata tensor.
  - β is encoded as NaN when None (decoded on read); document this.
  - `finish_reason` stored in the metadata dict (safetensors
    `save_file` has a metadata kwarg for small strings).
- `load_turn(path)` → `TurnTrajectory`.
- `glob_turns(log_dir, pattern="turn_*.safetensors")` →
  `Iterator[Path]`.
**Tests:**
- `test_round_trip` — write a trajectory, read it back, assert all
  channels numerically identical.
- `test_beta_none_encodes_as_nan` — trajectory with some `beta=None`
  steps round-trips to NaN on disk, `None` or NaN in memory.
- `test_glob_sorts_by_timestamp` — multiple files with mixed
  timestamps glob back in sortable order.
- `test_writer_creates_missing_dir` — `log_dir` with parents that
  don't exist is created atomically.
**Validation:** `python -m pytest tests/test_autotune_log.py -q`

### Task 5 — wire per-step controller into non-streaming generation
**Files:** `src/tgirl/serve.py`, `tests/test_serve.py`
**Approach:**
- Add session-scoped controller state to the app factory: the
  `Controller` instance and the rolling log directory. Initialized
  from `POST /v1/steering/autotune {"enabled": true, "log_dir":
  "..."}`.
- In `_generate_tokens` (non-streaming), wrap the existing per-token
  loop:
  ```
  alpha = _resolve_alpha(request)
  beta, skew = _resolve_beta_skew(request)
  temp = _resolve_temperature(request)

  step_arrays = {channel: [] for channel in CHANNELS}
  for step_index in range(max_tok):
      # 1. current-step observables (rolling over steps so far)
      obs = _build_step_observables(
          step_index, alpha, beta, temp,
          generated, certainty_steps, probe_norm_so_far
      )
      # 2. controller
      if _autotune_state["enabled"]:
          gains = _autotune_state["controller"].step(obs)
          alpha, beta, temp = _apply_gains_with_clipping(
              alpha, beta, temp, gains
          )
      # 3. apply to hook BEFORE forward
      _apply_band_to_hook(beta, skew)
      _apply_steering(alpha, v_probe_prev, norm_mode, correction_norms)
      # 4. forward, capture, sample (existing code)
      logits = ctx.forward_fn(token_ids)
      ...
      # 5. record per-step arrays
      for k, v in (...):  step_arrays[k].append(v)
      ...
  # At turn end: write trajectory
  if _autotune_state["enabled"] and _autotune_state["log_dir"]:
      trajectory = TurnTrajectory(arrays=step_arrays, finish=finish)
      write_turn(_autotune_state["log_dir"], trajectory)
  ```
- Per-request overrides set the initial `alpha/beta/temp`. The
  controller then owns the trajectory.
- `_apply_gains_with_clipping` is a serve-local helper:
  ```
  next_alpha = clip(alpha * gains.alpha_gain, α_min, α_max)
  next_temp = clip(temp * gains.temperature_gain, T_min, T_max)
  # β is not multiplicatively gained — controller emits a proposed
  # next β directly via gains.beta_gain interpreted as the raw next
  # β value. Floor at safe threshold.
  next_beta = _safe_band_beta(gains.beta_gain)  # None if above cutoff
  return next_alpha, next_beta, next_temp
  ```
**Tests:**
- `test_controller_invoked_once_per_token` — patch controller's
  `step` method, generate 5 tokens, assert call_count == 5.
- `test_controller_gains_flow_to_hook_before_forward` — use fake
  logits + `HeuristicController`, verify that when the controller
  proposes a smaller α at step 3, the hook's `_probe_alpha` at step 4
  reflects that.
- `test_turn_end_writes_safetensors` — generate a short turn with
  log_dir configured; assert the file appears with correct channel
  lengths.
- `test_no_oscillation_live_controller_integration` — reproduces
  the sycophant-fake-logits scenario; assert α trajectory is
  monotone (strictly non-increasing for the duration) once a collapse
  signal is detected, and α differs from initial by more than 0.1
  within 16 steps.
**Validation:** `python -m pytest tests/test_serve.py -q`

### Task 6 — wire per-step controller into streaming generation
**Files:** `src/tgirl/serve.py`, `tests/test_serve.py`
**Approach:**
- Mirror Task 5 changes in `stream_gen`. The only difference is the
  surrounding async generator structure; the per-step
  observables/controller/write logic is identical.
- Factor `_build_step_observables` and the per-step controller
  block into helpers to avoid drift between the two paths.
**Tests:**
- `test_streaming_controller_invoked_per_token` — use
  `TestClient` streaming; assert step count.
- `test_streaming_turn_end_writes_safetensors` — same as Task 5's
  write test but for the streaming path.
**Validation:** `python -m pytest tests/test_serve.py -q`

### Task 7 — update `/v1/steering/autotune` body + `/v1/steering/status`
**Files:** `src/tgirl/serve.py`, `tests/test_serve.py`
**Approach:**
- `POST /v1/steering/autotune` body:
  ```
  {
    "enabled": bool,
    "log_dir": str | null
  }
  ```
  - Accept `log_path` for one PR as an alias (deprecation warning
    logged); after the PR lands, drop `log_path`.
  - If `log_dir` is a nonexistent path, create it (via
    `Path.mkdir(parents=True, exist_ok=True)`). If the path exists
    and is not a directory, return HTTP 400 with an explanatory
    body.
- `/v1/steering/status["autotune"]` adds:
  - `log_dir`: str | null (replaces `log_path`)
  - `last_gains`: `{alpha_gain, beta_gain, temperature_gain,
    rationale}` or `null` if not yet computed
  - `last_step_observables`: subset of observables (excludes the full
    rolling windows, includes the scalars)
  - Retain: `enabled`.
**Tests:**
- `test_autotune_endpoint_accepts_log_dir` — POST with `log_dir`
  creates directory if missing.
- `test_autotune_endpoint_rejects_nondir_log_dir` — POST with
  `log_dir` pointing at a file returns 400.
- `test_status_exposes_last_gains_after_turn` — run one turn with
  autotune on; assert `last_gains` is populated with the expected
  shape.
**Validation:** `python -m pytest tests/test_serve.py -q`

### Task 8 — remove turn-level controller + JSONL writer
**Files:** `src/tgirl/serve.py`, `src/tgirl/autotune.py`,
`tests/test_serve.py`, `tests/test_autotune.py`
**Approach:**
- Delete `_run_autotune_after_turn` from `serve.py`.
- Delete `observables_to_dict` / `action_to_dict` in `autotune.py` if
  unused after this PR.
- Mark the old turn-level `autotune()` function either deprecated
  (`warnings.warn(DeprecationWarning...)`) or remove entirely. Prefer
  removal — the per-step controller fully subsumes it and there are
  no external callers.
- Remove the `test_autotune_logs_to_jsonl_when_path_set` test; adapt
  the other turn-level tests to step-level where they still make
  sense (drop `n_tokens`, `finish_reason` from assertions;
  `step_index` + `step_entropy` take their place).
**Tests:** full suite still green with net test count ≥ 1120.
**Validation:** `python -m pytest tests/ -q`

### Task 9 — update `scripts/test_autotune_live.py` to read tensor log
**Files:** `scripts/test_autotune_live.py`
**Approach:**
- Replace JSONL record reader with `glob_turns(log_dir)` +
  `load_turn(path)` from `autotune_log`.
- Trajectory summary becomes per-step charts (even if just ASCII):
  α-series, β-series, H-series across the session. Regime histogram
  is retired.
- Live test still exercises the same scenario: 8 turns at
  starting α=0.7 on Qwen3.5-0.8B-MLX-4bit, residual-relative mode,
  autotune on. Surface shows whether α drifts smoothly or stays
  stuck.
**Tests:** script is an operator-run tool; no unit tests, but
manually run against the model and confirm it produces sensible
output and no crashes.
**Validation:** `python scripts/test_autotune_live.py --model ... --turns 8`

## 4. Validation Gates

```bash
# Style
ruff check src/tgirl/autotune.py src/tgirl/coherence.py \
    src/tgirl/serve.py src/tgirl/autotune_log.py tests/

# Types
mypy src/tgirl/autotune.py src/tgirl/autotune_log.py

# Unit + integration
python -m pytest tests/ -q

# Live (operator)
python scripts/test_autotune_live.py \
    --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
    --start-alpha 0.7 --turns 8 --max-tokens 300
# Success: α drifts smoothly, no 0.4↔0.5 ping-pong;
# model produces non-repeating responses across turns.
```

## 5. Rollback Plan

Standard tier; rollback is `git revert` of the merge commit. The
feature is additive to the serve layer — no schema migrations or
external state. The old JSONL log files from the turn-level
controller are untouched by this PR (the new system writes
safetensors into a different directory by default).

If the controller misbehaves in production but the rest of the
feature is fine, set `POST /v1/steering/autotune {"enabled": false}`
— per-request α/β/temperature overrides resume full control,
trajectory logging stops. No restart needed.

## 6. Uncertainty Log

1. **Rolling window W = 16** is from first principles, not measurement.
   First live runs will show whether it's too lagged or too noisy.
   Adjusting it is a config constant change; no architectural
   implications.
2. **Gain constants** (`k_alpha`, `k_temp`, `beta_safe_floor`,
   `target_H`, `target_top1`) are seeded from 2D sweep heuristics.
   They will move when the MLP controller trains against live data.
   v1 HeuristicController defaults are a starting point, not a final
   answer.
3. **β encoding.** Continuous β over `[1.0, ∞)` with `math.inf` ≡
   `None` in the gain emitter, and the serve layer coerces to `None`
   above some cutoff (proposed 50.0). Alternative: keep `None` as a
   discrete state the controller can emit. Implemented the
   cutoff-based coercion; if it feels wrong in live use, swap to
   discrete later.
4. **Safetensors metadata strings.** The library supports a metadata
   dict of small strings. Longer strings (e.g., prompt preview) may
   need to be stored as a 0-d array of bytes or truncated. PRP
   assumes truncation at 240 chars matches existing smoke-script
   pattern.
5. **Derivative smoothing.** v1 uses raw finite difference over
   `W//2` lag — susceptible to noise. If controller output jitters
   token-to-token in live use, switch to an EMA-smoothed derivative.
   Would be a one-line change in `_build_step_observables`.
6. **Turn boundary behavior.** Between turns, the controller's
   internal state resets implicitly (per-turn arrays restart).
   Whether there's value in carrying controller-internal state *across*
   turns (e.g., the last gain vector as a prior) is an open empirical
   question. v1 ships stateless-per-turn.
