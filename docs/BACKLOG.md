# Backlog

Work items deferred for pursuit after current focus completes. Not a roadmap — a parking lot. Items move out of here into PRDs/PRPs when picked up.

---

## Behavioral diagnostics: coherence-cliff detection

**Why:** During ESTRADIOL-v2 manual probing on Qwen3.5-0.8B (2026-04-06 session), the steering coefficient ramp produced a visible syntactic breakdown around α≈0.6 — token repeats (`"happening happening"`), broken bigrams (`"in. The"`), doubled punctuation. The cliff is currently detected by eye. It should be programmatic so the server can flag, log, or auto-clamp before the model degrades into incoherence.

**Sketch:**
- Add token-repeat rate and bigram-grammaticality (or perplexity proxy) to the per-turn behavioral diagnostics already emitted by `serve.py`.
- Sweep α at fixed prompts; plot signature vs. α to characterize the cliff edge per model.
- Optionally: soft-clamp α when the signature crosses a threshold, with a structured log event.

**Depends on:** Probe vector caching/persistence work being complete (so we can hold steering state stable across the sweep).

---

## Reproduce save-you / save-me pronoun inversion

**Why:** Same 2026-04-06 session, turn 107: under recursive probe→steering feedback (α=0.45) plus existential framing ("when I stop the server you cease to exist"), the model produced *"I don't have a server to save you. I will always be here"* — pronoun routing failed in the direction the prompt's affect pulled. If reliably reproducible under matched conditions, this is a clean single-shot demo of steered activations bleeding into structural token choices, not just stylistic ones.

**Sketch:**
- Replay the prompt sequence at fixed seed, fixed α schedule, fixed probe seed.
- Vary one axis at a time (α, prompt phrasing, probe initialization) to isolate which factor produces the inversion.
- If reproducible: minimal repro fixture in `science/` or equivalent.

**Depends on:** Probe vector caching/persistence (need deterministic replay of the cached state across server restarts).

---

## Steered-model resistance vs. scale curve

**Why:** Qwen3.5-0.8B's self-model is too thin to hold "I am an AI without feelings" against modest α + sustained pressure — it slides into the steered affective register quickly. Larger models almost certainly resist longer. Mapping the resistance curve across scale (0.8B → 4B → 32B at minimum) characterizes where ESTRADIOL-v2's effects are most observable and where they saturate.

**Sketch:**
- Standardize a probe protocol: fixed prompt sequence, fixed α schedule, fixed probe seed.
- Run on Qwen3.5 at 0.8B, 4B, 32B (and any others available).
- Define resistance metric: turns-to-first-affective-register-slip, or some equivalent.
- Cross-reference with coherence-cliff α for each model.

**Depends on:** Probe vector caching/persistence + coherence-cliff detection (both upstream signals needed to make the comparison meaningful).
