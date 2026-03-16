# PRD: ADSR Modulation Matrix

## Status: DRAFT
## Author: Claude (Proposer stance)
## Date: 2026-03-15
## Branch: feature/adsr-modulation-matrix

## 1. Problem Statement

The constrained generation loop uses static inference hooks — each hook produces fixed parameter values regardless of where in the s-expression lifecycle the generation is. `GrammarTemperatureHook` applies the same temperature scaling formula at every position. `NestingDepthHook` only activates near the token budget limit. `RepetitionPenaltyHook` uses fixed window size and bias.

But the lifecycle has distinct phases with different optimal parameter regimes:
- **Attack** (opening structure): high temperature, wide top_p, sharp OT — explore the tool space
- **Decay** (tool selected, args starting): temperature dropping, narrowing — commit to structure
- **Sustain** (argument values): low temperature, tight nucleus — precise values
- **Release** (closing parens): moderate flexibility, strong opener penalty

These phases are detectable from signals already computed per token (grammar freedom, nesting depth, entropy, confidence). The infrastructure exists; the parameter scheduling doesn't.

## 2. Proposed Solution

Replace the static hook pipeline with a configurable **modulation matrix** — an (11, 7) tensor that routes 11 source signals to 7 destination parameters via a single matrix multiplication per token.

**Source signals (11):** grammar_freedom, normalized_entropy, normalized_confidence, grammar_mask_overlap, normalized_depth, position_normalized, phase_attack, phase_decay, phase_sustain, phase_release, cycle_detected.

**Destination parameters (7):** temperature_mod, top_p_mod, repetition_bias_mod, transport_epsilon_mod, opener_bias_mod, backtrack_threshold_mod, presence_penalty_mod.

**Signal chain:** Raw signals → source conditioning (normalize, rectify, slew-limit) → matrix multiply → output conditioning (add to base values, clamp to valid ranges) → `ModelIntervention`.

The matrix *is* the envelope. ADSR phase indicators are source signals; the envelope curves emerge from the matrix weights. The confidence ride is another source signal that modulates through the same matrix. One unified mechanism replaces three separate hooks.

See `docs/design/adsr-envelope.md` for the full signal processing design.

## 3. Architecture Impact

### New files
- `src/tgirl/modulation.py` — `SourceConditioner`, `EnvelopeState`, phase detection, `ModMatrixHook` (torch), `ModMatrixHookMlx` (MLX)
- `tests/test_modulation.py` — unit tests

### Modified files
- `src/tgirl/types.py` — add `transport_epsilon: float | None = None` to `ModelIntervention`
- `src/tgirl/sample.py` — hook auto-conversion for `ModMatrixHook` → `ModMatrixHookMlx`; read `transport_epsilon` from merged intervention in sampling loop
- `src/tgirl/sample_mlx.py` — read `transport_epsilon` from merged intervention in sampling loop
- `benchmarks/run_bfcl.py` — replace hook list with `ModMatrixHook`
- `examples/showcase_unified_api.py` — replace hook list with `ModMatrixHook`

### Unchanged files
- `GrammarTemperatureHook`, `NestingDepthHook` remain in sample.py/sample_mlx.py for backwards compatibility but are superseded by `ModMatrixHook`
- `RepetitionPenaltyHook` stays as-is — cycle detection is incorporated into the mod matrix as a gate signal, but the window-based penalty logic remains in the separate hook for now

### Data model changes
- `ModelIntervention` gains `transport_epsilon: float | None = None`
- New `EnvelopeConfig` dataclass for matrix + base values + conditioner config
- New `EnvelopeState` dataclass for per-generation phase tracking

### No new dependencies
- All compute uses existing `mlx.core` / `torch` ops
- Matrix is 77 parameters — no optimization library needed for v1 (hand-tuned default)

## 4. Acceptance Criteria

1. `ModMatrixHookMlx` implements `InferenceHookMlx` protocol (pre_forward, advance, reset)
2. Source conditioning normalizes all 11 signals to [0, 1] range using native framework ops (no Python iteration)
3. Phase detection correctly identifies attack/decay/sustain/release from nesting depth and grammar freedom signals
4. Matrix multiply produces 7 modulation values via single `mx.matmul` (or `torch.matmul`)
5. Output conditioning adds modulations to base values and clamps to valid ranges
6. `transport_epsilon` field on `ModelIntervention` is read by the sampling loop to construct per-token `TransportConfig`
7. Default matrix reproduces qualitatively similar behavior to current `GrammarTemperatureHook` + `NestingDepthHook` (BFCL accuracy within 5% of baseline)
8. Slew rate limiting on confidence signal prevents parameter discontinuities (no >50% change in temperature between adjacent tokens)
9. Hook auto-conversion in `SamplingSession` maps `ModMatrixHook` → `ModMatrixHookMlx`
10. All existing tests continue to pass
11. Per-token overhead of `ModMatrixHook` is <0.1ms (dominated by source conditioning, not matrix multiply)
12. Telemetry records per-token phase, source vector, and modulation vector

## 5. Risk Assessment

- **Accuracy regression:** The default matrix may not match the finely-tuned behavior of the current hooks. Mitigate: A/B benchmark on BFCL before removing old hooks from defaults.
- **Phase detection edge cases:** Transitions between phases may flicker if grammar freedom oscillates. Mitigate: hysteresis in phase detection (require N consecutive tokens before transition).
- **Slew rate tuning:** Too much smoothing = slow response to real confidence changes. Too little = jerky parameters. Mitigate: expose slew_rate as configurable per-source.
- **Matrix interpretability:** 77 parameters are human-readable but non-trivial to reason about jointly. Mitigate: provide visualization/logging of matrix effect per token.

## 6. Open Questions

1. Should `RepetitionPenaltyHook` be fully absorbed into the mod matrix, or kept as a separate hook? The window-based counting and cycle detection involve Python dict operations on token history — these don't map cleanly to a matrix operation.
2. Should the matrix be per-model (calibrated from BFCL telemetry) or universal (hand-tuned once)?
3. Should phase ramps be linear, exponential, or configurable?
4. Is 11×7 the right matrix size, or should we start smaller (e.g., 6×4 — just CV signals and core destinations)?

## 7. Out of Scope

- Matrix optimization via gradient descent or evolutionary strategies (v2 — requires optimization infrastructure)
- Cross-model matrix transfer experiments (research phase, not v1 engineering)
- Feedback paths (previous-token output as source) — adds complexity, defer to v2
- ESTRADIOL integration (activation_steering remains reserved, not wired)
- Changes to the freeform generation loop (ADSR applies only to constrained generation)
