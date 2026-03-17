# Grammar State Change Classifier

## Status: DESIGN
## Author: Maddy Muscari + Claude
## Date: 2026-03-16

---

## Problem

The SCU currently uses `DelimiterTransitionPolicy` — it waits for the model to emit a magic token (e.g., `<tool_call>`) before switching from freeform to constrained mode. This has two failure modes:

1. **Base models never emit delimiters.** They don't know the convention. On BFCL, Qwen3.5-9B-Base produces no tool call 20% of the time — not because it can't, but because the inference loop never transitions.
2. **Instruct models emit delimiters unreliably.** They sometimes ramble, hedge, or refuse instead of calling the tool. The delimiter is a training artifact, not a reliable control signal.

The decision of *when* to transition between grammar regimes (freeform → tool calling, freeform → codegen, etc.) should be made by the inference loop, not delegated to the model.

## Approach

A lightweight classifier that watches the token stream during freeform generation and predicts the optimal transition point to constrained mode. This replaces the delimiter-based transition with a learned, model-agnostic regime controller.

### Input Features (available now, no model surgery)

The classifier operates on signals already computed per-token:

1. **Logit distribution projected through TokenLexemeMap** — if probability mass clusters on tool-name tokens, the model is "trying" to call a tool
2. **Top-k token identities** — are the highest-probability tokens function names, argument keywords, or structural delimiters?
3. **CSG signals** — entropy, confidence, grammar freedom (already computed per token)
4. **Sliding window of recent token types** — are recent tokens conversational or structured?
5. **System prompt features** — are tools registered? How many? What are their names?

### Output

A probability distribution over regime transitions:

```
P(stay_freeform)
P(transition_to_tool_grammar)
P(transition_to_codegen_grammar)
P(transition_to_nl_grammar)
...one per grammar in the portfolio
```

When `P(transition_to_X)` exceeds a configurable threshold, the SCU transitions. The threshold controls the precision/recall tradeoff: low threshold = aggressive tool calling (fewer misses, more false transitions), high threshold = conservative (more misses, fewer false transitions).

### Architecture Options

**Option A: Linear probe on logit projections (simplest)**
```
logits → softmax → tool_name_probability_mass → threshold → transition
```
No learned parameters. Just sum the softmax probability on token IDs that decode to registered tool names. If it exceeds threshold, transition. This is a zero-parameter baseline.

**Option B: Small MLP on CSG signals + logit features**
```
[entropy, confidence, freedom, overlap, tool_mass, recent_pattern] → MLP(64) → regime_probs
```
~6 input features → 1 hidden layer (64 units) → N outputs (one per grammar). Tiny, fast (<0.01ms per token). Trained on labeled transition points.

**Option C: Linear probe on hidden states (requires CLU hooks)**
```
hidden_state[bottleneck_layer] → linear → regime_probs
```
Most powerful but requires Phase 4 (CLU hook interface). The model's hidden state at the bottleneck layer encodes intent more directly than logits. Deferred.

### Recommended progression: A → B → C

Start with Option A (zero parameters, available now) to establish a baseline and validate the approach. If tool-name probability mass is a good signal, Option B adds a small MLP to combine it with other CSG signals. Option C waits for CLU hooks.

## Training Data

The instruct model provides free supervision:

1. Run the instruct model on BFCL with delimiter-based transition
2. Record the exact token position where constrained mode was triggered
3. For each position, record the input features (logits, CSG signals)
4. Label: `transition=True` at the trigger point, `transition=False` elsewhere

This gives us (features, label) pairs for supervised training. The instruct model knows *when* to call tools — the classifier extracts that knowledge into a standalone component that works on any model.

**Self-supervised alternative:** Run the base model with forced transition at every possible position. Measure which positions produce correct tool calls. The classifier learns to predict positions that lead to correct output.

## Transition Policy: Latch-on-Confidence, Complete-on-Terminal

The classifier doesn't trigger an immediate hard cut. Instead it uses a two-phase transition:

### Phase 1: Latch

The classifier monitors per-token signals during freeform generation. When confidence that a tool call is needed exceeds threshold, a **latch** is set. The latch is a one-way flag — once set, transition *will* happen. The model continues generating in freeform mode.

### Phase 2: Complete on Grammar Terminal

After the latch is set, the SCU monitors for a **sentence terminal** in the active grammar mode:

- **English (current):** `.` `!` `?` followed by whitespace, or `\n\n` (paragraph break)
- **Polyglot (future):** terminal symbols from the active NL grammar — `。` (Chinese/Japanese), `।` (Hindi/Devanagari), `‎.` (Arabic), etc. The LinGO grammar or equivalent defines what constitutes a sentence boundary in the active language.
- **Budget exhaustion (fallback):** if `max_freeform_tokens` is reached with latch set, force transition immediately regardless of terminal. The model had its chance to complete the sentence.

This preserves reasoning coherence — the model finishes "...I need to use the calculate_bmi function." before the grammar takes over. No mid-sentence cuts.

### Phase 3: Transition

On terminal detection (or budget exhaustion):
1. Truncate any trailing incomplete content
2. Inject tool-call primer tokens (already implemented)
3. Switch SCU state to CONSTRAINED
4. GSU activates target grammar
5. STB begins constrained redistribution

```
freeform: "We need to calculate BMI. The function requires weight and height."
                                    ^ latch set (tool-name confidence high)
                                                                          ^ terminal detected
→ inject primer → constrained mode → "(calculate_bmi :weight 70 :height 1.75)"
```

### Implementation

```python
class LatchedTransitionPolicy(TransitionPolicy):
    """Latch on high confidence, transition on sentence terminal."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        terminal_tokens: set[str] | None = None,
        max_freeform_after_latch: int = 64,
    ) -> None:
        self._threshold = confidence_threshold
        self._terminals = terminal_tokens or {'.', '!', '?'}
        self._max_after_latch = max_freeform_after_latch
        self._latched = False
        self._tokens_since_latch = 0

    def evaluate(
        self,
        signal: TransitionSignal,
        logits: Tensor,
        token_text: str,
        tool_token_mass: float,
    ) -> str | None:
        """Returns target grammar name or None."""
        # Check for latch condition
        if not self._latched and tool_token_mass > self._threshold:
            self._latched = True
            self._tokens_since_latch = 0

        if not self._latched:
            return None

        self._tokens_since_latch += 1

        # Budget exhaustion — force transition
        if self._tokens_since_latch >= self._max_after_latch:
            return "tool_grammar"

        # Terminal detection
        stripped = token_text.strip()
        if stripped and stripped[-1] in self._terminals:
            return "tool_grammar"

        return None
```

### Polyglot Terminal Detection

The terminal set is grammar-mode-dependent. When the GSU manages multiple NL grammars, each grammar provides its own sentence boundary tokens:

```python
class GrammarTerminals:
    """Sentence-terminal tokens for a grammar mode."""
    english = {'.', '!', '?'}
    chinese = {'。', '！', '？'}
    japanese = {'。', '！', '？'}
    hindi = {'।', '!', '?'}
    arabic = {'.', '!', '؟'}
    # ... extensible per grammar
```

The active NL grammar in the GSU determines which terminal set the `LatchedTransitionPolicy` uses. This scales to any language the Grammar Matrix supports.

## Integration Points

### SCU

`LatchedTransitionPolicy` replaces or composes with `DelimiterTransitionPolicy`. For instruct models, the delimiter policy fires first (the model emits delimiters voluntarily). For base models, the latched policy takes over when delimiters never come.

`CompositeTransitionPolicy` already supports chaining — add the latched policy as a fallback:

```python
policy = CompositeTransitionPolicy([
    DelimiterTransitionPolicy(...),   # instruct models emit delimiters
    LatchedTransitionPolicy(...),     # base models: latch + terminal
])
```

### CSG

The classifier's output (`tool_token_mass`) becomes a new CSG signal. This feeds the modulation matrix and the SCU simultaneously. The modulation matrix can begin adjusting parameters (lowering temperature, tightening top_p) as soon as the latch is set, smoothing the transition.

### GSU

When the latched policy triggers, the GSU activates the target grammar and the STB begins constrained redistribution. The tool-call primer tokens (already implemented) are injected to seed the constrained generation.

## Performance Budget

The classifier runs every token during freeform generation. Budget: <0.1ms per token.

- Option A (threshold): ~0.001ms (single sum over logits)
- Option B (MLP): ~0.01ms (small matmul)
- Option C (linear probe): ~0.01ms (single linear layer on hidden state)

All well within budget. The forward pass is 5ms; the classifier is noise.

## Relationship to Phase 2B MLP

The grammar state change classifier and the Phase 2B MLP (learned syntactic constraint predictor) converge into a single component: a **regime controller** that reads the token stream and outputs both:
1. Whether to transition (this design)
2. What grammar weights to apply (Phase 2B)

They share input features, training methodology, and integration points. Implement them as one module with two output heads.

## Expected Impact

On the 9B-Base BFCL benchmark:
- Current: ~81% accuracy on 80% of entries (20% no tool call)
- With Option A classifier: ~81% accuracy on ~95% of entries (5% no tool call)
- Effective overall: ~77% → potentially ~80%+ with reduced no-tool-call rate

The classifier doesn't improve accuracy *per tool call* — the grammar handles that. It improves *coverage* — ensuring the model enters constrained mode when it should.

## Open Questions

1. **False positive cost.** If the classifier triggers constrained mode when the correct response is conversational, the model produces a tool call when it shouldn't. How bad is this? In agentic settings, a spurious tool call is recoverable (the tool returns an error). In conversational settings, it's jarring. The threshold controls this tradeoff.

2. **Multi-turn context.** The classifier sees per-token features, not conversation history. Should it also see the system prompt and recent turns? This adds complexity but improves the conversational vs tool-calling decision.

3. **Grammar-specific thresholds.** Different grammars may need different transition thresholds. Tool calling should be aggressive (better to try and fail). Codegen should be conservative (don't start generating Python in the middle of a sentence).
