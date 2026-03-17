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

## Integration Points

### SCU

New policy: `ClassifierTransitionPolicy`

```python
class ClassifierTransitionPolicy(TransitionPolicy):
    """Transitions based on learned classifier, not delimiters."""

    def __init__(
        self,
        classifier: GrammarStateClassifier,
        threshold: float = 0.7,
    ) -> None: ...

    def should_transition(
        self,
        signal: TransitionSignal,
        logits: Tensor,  # new: needs logit access
    ) -> str | None:
        """Returns target grammar name or None."""
        ...
```

### CSG

The classifier's output becomes a new CSG signal: `transition_probability`. This feeds the modulation matrix and the SCU simultaneously.

### GSU

When the classifier triggers a transition, the GSU activates the target grammar and the STB begins constrained redistribution. The tool-call primer tokens (already implemented) are injected to seed the constrained generation.

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
