# PRD: Per-Token Fire-Control Autotuner

## Status: DRAFT
## Author: claude-code (opus-4.6)
## Date: 2026-04-14
## Branch: feature/per-token-fire-control-autotuner

## 1. Problem Statement

The current autotuner (shipped in `feature/steering-autotune`, merged in
PR #21) runs at *turn boundaries*. It observes the turn that just
finished, computes a next-turn config, and applies it to the next
request. This has two fundamental failures empirically confirmed
against `Qwen3.5-0.8B-MLX-4bit`:

1. **It is categorically the wrong scale.** ESTRADIOL-v2's whole thesis
   is that the bottleneck-residual probe is *alive at every token*.
   Capturing a probe at token N but only using it to tune the NEXT
   turn means the model generates the rest of turn N under stale
   steering. The per-forward-pass feedback loop the architecture was
   built for is never actually engaged.

2. **The per-turn threshold controller oscillates.** Live JSONL evidence
   (`~/autotune.jsonl`, 24 rows): 10 consecutive `sycophant_trap`
   turns with the controller deterministically flipping α between 0.40
   and 0.50. Every turn produces the same 25-26 token stuck response.
   The classifier is correct; the response mechanism is a ±0.1 step
   that can't escape the basin because it reverses direction the
   moment it crosses the basin midpoint. The user's words: "the whole
   thesis is that modulating hyperparameters and performing activation
   steering on a per-token basis supercharges a small model and is
   ultimately better than some SaaS slop that accepts hyperparams from
   devs as a query param."

Both failures trace to the same root: the autotuner is a discrete step
function evaluated once per turn. Neither the **scale** (per-turn) nor
the **shape** (discrete steps, basin flip-flop) fits the problem. The
feedback loop should close at every forward pass, and the response
should be continuous-valued and trajectory-aware — closer to a B-29
Central Fire Control System's lead computing than to a thermostat.

## 2. Proposed Solution

Replace the turn-level autotuner with a **per-token, continuous-gain
controller** that runs inside the generation loop. At each token:

1. The controller reads step-level observables built from the current
   token position and a rolling window of recent tokens — including
   *derivative channels* that capture whether coherence is trending
   up, down, or stable.
2. The controller outputs natural-valued multiplicative **gains** on
   (α, β, temperature) — not `±0.1` steps. A gain of 1.0 is "hold
   steady"; gains smoothly scale the corrections up or down based on
   how far the observables are from target and how fast they are
   moving.
3. The new (α, β, temperature) take effect for the very next forward
   pass, mid-turn. The controller can intervene *between tokens 8 and
   9* when it sees a trajectory headed toward the helpful-template
   short-EOS bailout, rather than waiting for the whole bailout to
   play out and then "planning" a different next turn.

The controller is a `Controller` protocol with two implementations:

- `HeuristicController` — rule-based proportional-derivative gains
  using `exp(-k·error)` and `softplus` response functions. Ships in v1.
- `MLPController` — same interface, learned weights. Trained offline
  from the tensor trajectory log this feature produces. Not in v1; the
  feature ships the training substrate so it can be added later
  without reworking the serve layer.

At turn end, the accumulated per-step arrays (observables,
actions, sampled tokens) are serialized as one safetensors file to the
configured log directory. Training loops glob the directory and
concatenate channels.

The turn-level `autotune()` function remains as a vestigial wrapper or
is removed entirely — the per-step controller subsumes it.

## 3. Architecture Impact

### Files/modules affected

- **`src/tgirl/autotune.py`** — adds `Controller` protocol,
  `HeuristicController`, `StepObservables` dataclass with derivative
  channels, `Gains` output dataclass. Existing `autotune()`,
  `Observables`, `Action` retained for backwards-compat or deprecated
  (decision in the PRP).
- **`src/tgirl/coherence.py`** — adds `rolling_coherence(tokens,
  window)` for O(W) per-step repeat_rate / bigram_novelty /
  token_entropy on the trailing window. Pure function.
- **`src/tgirl/certainty.py`** — no changes; `step_certainty(logits)`
  is already per-token.
- **`src/tgirl/serve.py`** — the two generation loops (non-streaming
  at `_generate_tokens`, streaming at `stream_gen`) gain a per-step
  controller call that applies gains to the hook (α, β) and to the
  sampling temperature. Per-step arrays accumulate in plain Python
  lists; at turn end, one safetensors file is written. The turn-level
  `_run_autotune_after_turn` is removed.
- **New module `src/tgirl/autotune_log.py`** — thin writer that takes
  per-step array dicts + per-turn metadata and emits a safetensors
  bundle to `<log_dir>/turn_<timestamp>_<finish>.safetensors`. Mirror
  of existing `tgirl.estradiol.save_estradiol` pattern.
- **`tests/test_autotune.py`** — existing turn-level tests either
  migrate to step-level or are removed with the turn-level code.
- **New tests** — `test_autotune_step.py`, `test_controller.py`,
  `test_rolling_coherence.py`, `test_autotune_log.py`.
- **`scripts/test_autotune_live.py`** — updated to read the new
  tensor log format and to surface the per-token trajectory instead
  of per-turn summaries.

### Data model changes

- `StepObservables` carries ≈15 channels (6 current-step values + 6
  rolling-window means + 3 derivative channels + scalars like
  step_index, probe_norm).
- `Gains` carries three multiplicative scalars + a rationale string
  for logging.
- The JSONL log is **replaced** by a per-turn safetensors file with
  the fixed channel schema documented in §6.

### API changes

- `POST /v1/steering/autotune` body: `{ "enabled": bool, "log_dir":
  str | null }` (formerly `log_path`, singular file).
- `/v1/steering/status` exposes the current per-step controller's
  last gains + smoothed observable values.
- Per-request `estradiol_alpha/beta/temperature` overrides continue
  to gate turn-start values but are immediately overwritten by the
  controller on step 2+. Interpretation: per-request overrides set
  the initial state; the controller owns the trajectory.

### Dependency additions

- `safetensors` — already a transitive dependency via `mlx-lm`;
  confirm in PRP.
- No new third-party additions.

## 4. Acceptance Criteria

1. The controller is invoked **once per token** during generation in
   both the non-streaming and streaming paths. Unit test confirms the
   number of controller invocations equals the number of tokens
   emitted.
2. Gains are natural-valued (not discrete ±steps). Unit tests verify:
   for observables exactly at target, all gains are 1.0 ± 1e-6; for
   observables off-target, gains depart smoothly from 1.0 as a
   function of the error magnitude.
3. Derivative channels are populated from a rolling window of
   configurable size (default W=16) and behave correctly at the turn
   boundary where the window is not yet full (no NaNs, no unsafe
   division).
4. The sycophant-basin oscillation observed in the turn-level
   controller does **not** reproduce. Integration test with
   deterministic fake-logits that would drive the turn-level
   controller into the 0.4↔0.5 loop produces a monotonically-drifting
   α trajectory instead.
5. At turn end, a safetensors file is written to the configured
   `log_dir` with the documented channel schema and a naming pattern
   that is filesystem-sortable by timestamp. File is readable by
   `safetensors.safe_open` without errors and every channel has the
   expected length.
6. `POST /v1/steering/autotune` accepts `log_dir` and validates that
   the directory is writable; a nonexistent path is created or
   returns a clear 400.
7. `/v1/steering/status["autotune"]` exposes current smoothed
   observables and the last emitted gains, both as inspectable
   numerical values.
8. Full test suite (`pytest tests/`) passes on the branch, with net
   test count ≥ 1120 (current baseline 1118 + new step-controller
   tests, minus any deprecated turn-level tests).
9. Live validation on Qwen3.5-0.8B-MLX-4bit: an 8-turn session
   starting at α=0.4, β=None, temperature=0.0 produces a non-repeating
   response in turn 2 (the scenario that reproducibly failed under
   the turn-level controller).

## 5. Risk Assessment

**Controller stability.** Continuous gains over discrete-time feedback
can oscillate or diverge if the gains are too aggressive or the
derivative smoothing is too short. Mitigation: conservative gain
constants in v1, clipping at each step, and unit tests that drive the
controller with adversarial synthetic observables (step, ramp,
oscillation) to verify the output remains bounded and reasonable.

**Per-step overhead.** The controller runs Python code every token.
Budget: rolling window O(W) for coherence, O(1) for derivatives, O(1)
for gain computation, total maybe 100–200 µs per step. Negligible next
to a Qwen-0.8B forward pass (~20 ms), but measurable and documented.
Mitigation: profile on a real run; if it exceeds 1 ms per token,
reconsider.

**Log file count.** One safetensors file per turn at ~8–16 KB per turn
means a day of heavy chat = thousands of files. Mitigation: document
an expected directory layout (date-sharded subdirs if needed) and
provide a concatenation utility in the training script. Not blocking.

**Backwards compatibility.** The `/v1/steering/autotune` body shape
changes (`log_path` → `log_dir`) and the turn-level `Observables`
dataclass may be removed. No external users of the library are known;
we're breaking our own in-progress integrations. Mitigation: clear
deprecation in the commit message; live scripts (`test_autotune_live`,
`sweep_alpha_cliff`) updated in the same PR.

**Scope creep into learned controller.** The MLP controller is
enticing. Hold the line: v1 ships only the heuristic controller and
the tensor log that would train the MLP. Training and MLP deployment
are follow-up features.

## 6. Open Questions

1. **Rolling window size.** W=16 is a starting guess. Smaller windows
   react faster but are noisier; larger windows smooth better but lag.
   Should be empirically tuned from sweep data. PRP should leave this
   as a configurable constant with a documented default.

2. **Target band for each observable.** From the 2D sweep:
   `H_norm ∈ [0.55, 0.85]` is the signal regime. The center
   `target_H = 0.72` is a reasonable default but the controller should
   allow different target setpoints per deployment (research vs.
   chat). Expose as controller config.

3. **Gain constants.** `k_α`, `k_T`, `k_β`, and the derivative
   weights are free parameters. Hand-pick sensible defaults from the
   2D sweep data in v1; document them as the initial operating point
   that the MLPController will later learn to replace.

4. **What happens on step 1 of a turn?** No rolling window exists
   yet, so derivatives are undefined. Options: (a) hold gains at 1.0
   until window is populated; (b) use fallback values derived from
   the pre-turn state. Recommend (a) for simplicity.

5. **β is hard to multiplicatively update** since it's either `None`
   or a positive float. Continuous β over `[1.0, ∞)` with `None`
   represented as a sentinel (e.g., `math.inf` → single-layer). The
   hook already handles `None` correctly via `set_band(None)`; the
   controller emits float β values and the serve layer coerces
   anything above a threshold (say 50.0) back to `None` for
   single-layer. Bike-sheddable; PRP picks a specific encoding.

6. **Log dir structure.** Flat files vs. date-sharded? For phase-1
   correctness, flat. For production at scale, date-sharded. Deferred
   to PRP.

## 7. Out of Scope

- **Learned controller (MLPController).** Interface shipped; training
  script + deployment path is a separate feature. Specifically *not*
  in scope: gradient-descent training of the gain network, model
  serialization format, hot-swap of learned weights without restart.
- **Bounded KV cache / chunked prefill.** Separate conversation
  (OOM on Whitman), separate feature. This PRD does not address
  arbitrary-length prompts; those will OOM at the same threshold as
  before.
- **Reinforcement learning.** The trajectory log *could* be used for
  RL later. This feature ships supervised-learning substrate only.
  No reward shaping, policy gradients, etc.
- **Per-token β band recomputation overhead.** The
  `band_weights(n_layers, bottleneck_idx, beta, skew)` call is cheap
  (O(layers) on bottleneck neighborhood) but if per-step β changes
  prove hot, cache the `{beta: weights_dict}` mapping. Simple
  optimization; v1 just recomputes each step.
- **Schema migration for existing `autotune.jsonl` files.** The new
  format is orthogonal; old JSONL logs are not backfilled.
- **Additional control axes beyond (α, β, temperature).** Top-p,
  top-k, repetition penalty — all fair game for later. v1 sticks to
  the three axes with existing plumbing.
