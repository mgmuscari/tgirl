# Native LinGO GSU — TDL-to-Token Constraint Compiler

**Status:** Design
**Date:** 2026-03-16
**Architecture block:** GSU (Grammar Selection Unit)
**Vision:** A universal grammar constraint module for any language model

---

## 1. Vision

A standalone Python module that reads HPSG grammars in TDL format (such as the English Resource Grammar) and produces per-token valid masks for any tokenizer. Not tied to tgirl, not tied to any model or framework. A universal GSU that bridges formal linguistic competence and neural language model performance.

```python
from tgirl.lingo import load_grammar, GrammarConstraint

erg = load_grammar("erg-2025/")
constraint = erg.constrain(tokenizer.get_vocab())

# Per token: which tokens continue well-formed English?
valid_mask = constraint.get_valid_mask(prefix_tokens)
constraint.advance(selected_token)
```

Same `GrammarState` protocol as our CFG grammars. Same OT transport. Same sampling loop. The model doesn't know whether it's being constrained by a Hy tool grammar or the English Resource Grammar — it just sees valid masks.

## 2. Why Not ACE

ACE is a 20-year-old C implementation with:
- Nested functions requiring GCC (not Clang)
- Fixed-address `mmap` for grammar serialization (fails on ARM64 macOS)
- JIT compilation of quick-check filters via runtime `dlopen`
- Architecture-specific frozen grammar images

We tried: compiled `libace.dylib` natively on ARM64, loaded ERG from TDL source. Grammar loads (44k lexemes, 392 rules, 5 seconds). Crashes during the QC JIT phase. The mmap freezer is fundamentally incompatible with Apple Silicon's address space layout.

ACE is a reference implementation. The ERG is the knowledge. We take the knowledge.

## 3. Architecture

### What TDL gives us

The ERG in TDL format provides:
- **Type hierarchy** — ~12,000 types with multiple inheritance and feature constraints
- **Feature structures** — SYNSEM, LOCAL, CAT, HEAD, VAL, etc. with coreferences
- **Lexicon** — 44,543 lexemes mapping surface forms to typed feature structures
- **Construction rules** — 392 rules defining how phrases combine (head-complement, head-subject, head-adjunct, etc.)
- **Lexical rules** — 49 inflectional rules (tense, agreement, etc.)

### What we build

**Layer 1: TDL Parser** (`tgirl/lingo/tdl_parser.py`)
- Parse TDL syntax into an AST: type definitions, feature structures, coreferences, conjunctions
- Handle includes, comments, strings, regex patterns
- Output: structured representation of the type hierarchy and all definitions

**Layer 2: Type System** (`tgirl/lingo/types.py`)
- Type hierarchy with multiple inheritance
- Feature structure representation (frozen Pydantic models or dataclasses)
- Unification engine — the core HPSG operation
- Subsumption checking (is type A more general than type B?)

**Layer 3: Lexicon** (`tgirl/lingo/lexicon.py`)
- Map surface forms (words) to lexeme types
- Unknown word handling via generic lexeme types
- Token-to-lexeme mapping — for each token in a model's vocabulary, which lexeme types can it start/continue?

**Layer 4: Chart Parser** (`tgirl/lingo/chart.py`)
- Bottom-up chart parsing using construction rules
- Incremental: add one token at a time, extend the chart
- Output: which next tokens can produce a well-formed continuation

**Layer 5: GrammarState Adapter** (`tgirl/lingo/grammar_state.py`)
- Implements `GrammarState` protocol (`get_valid_mask`, `is_accepting`, `advance`)
- Wraps the chart parser
- Produces token-level valid masks from the chart's state

### Acceleration strategy

The hot path is unification — checking whether two feature structures are compatible. This is the inner loop of chart parsing (called for every rule application at every chart position).

**MLX acceleration for unification:**
- Feature structures as integer type ID arrays
- Subsumption as matrix lookup (precomputed subsumption table)
- Batch unification for multiple candidate rule applications
- The type hierarchy is static — precompute everything possible

**The chart itself is sparse** — most cells are empty. Sparse tensor operations or dictionary-based representation.

## 4. Simplifications for v1

Full HPSG parsing with the ERG is complex. For v1, we can simplify:

**Coarse grammar** — Instead of full feature structure unification, use the type hierarchy alone for compatibility checking. Two types are compatible if their greatest lower bound (GLB) exists. This is a table lookup, not full unification. The ERG already precomputes GLBs (ACE reports 4,737 GLB types).

**Lexical coverage as coherence signal** — Even without chart parsing, knowing which tokens map to known ERG lexemes vs unknown words gives us the `linguistic_coherence` signal. The ratio of known-lexeme tokens to total tokens is a useful CSG signal.

**Category-based filtering** — If we know the current syntactic category (noun phrase, verb phrase, etc.), we can restrict the valid mask to tokens that could start a phrase of a compatible category. This is coarser than full chart parsing but much faster.

## 5. Integration with VCogPU

The native LinGO GSU plugs into the existing architecture:

```
GSU: tgirl.lingo.grammar_state.LingoGrammarState
  implements: GrammarState protocol
  input: token history
  output: valid_mask (mx.array or torch.Tensor)

CSG: TransitionSignal.linguistic_coherence
  source: LingoGrammarState.coherence_score()
  value: 0.0 (all unknown words) to 1.0 (all known, well-formed)

Modulation Matrix: row 11 → linguistic_coherence
  routes to: temperature, top_p, transport_epsilon, etc.
```

The ADSR envelope gains a new source signal. The modulation matrix grows from (11, 7) to (12, 7). The confidence ride can now factor in linguistic well-formedness.

## 6. Standalone Distribution

The module is designed to be extractable from tgirl:

```
tgirl/lingo/
├── tdl_parser.py      # TDL syntax parser
├── types.py           # Type hierarchy + feature structures
├── lexicon.py         # Word → type mapping
├── chart.py           # Incremental chart parser
├── grammar_state.py   # GrammarState protocol adapter
├── grammars/          # Bundled grammar data (ERG subset or full)
└── __init__.py        # Public API: load_grammar, GrammarConstraint
```

No dependency on tgirl core modules. Uses MLX or torch for acceleration if available, falls back to pure Python. The `GrammarState` protocol is defined as a Python Protocol class — any consumer that speaks the protocol can use it.

## 7. Implementation Sequence

1. **TDL Parser** — parse TDL syntax into AST. Test against ERG files.
2. **Type Hierarchy** — build type lattice with subsumption. Test against ERG's 12k types.
3. **Lexicon Loader** — map words to types. Test coverage against ERG's 44k lexemes.
4. **Token-to-Lexeme Mapping** — for a given tokenizer, which tokens map to which lexemes. This is the vocabulary-grammar bridge.
5. **Coherence Signal** — ratio of known-lexeme tokens. No chart parsing needed.
6. **GrammarState Adapter** — implement protocol, wire into sampling loop.
7. **Modulation Matrix Integration** — add `linguistic_coherence` as source signal row 11.
8. **Chart Parser (v2)** — full incremental chart parsing for per-token valid masks.
9. **MLX Acceleration (v2)** — batch unification, precomputed subsumption tables.
