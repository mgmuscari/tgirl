# PRD: Native LinGO GSU

## Status: DRAFT
## Author: Claude (Proposer stance)
## Date: 2026-03-16
## Branch: feature/lingo-native

## 1. Problem Statement

tgirl's grammar constraint system operates at the tool-call level: a CFG over Hy s-expressions restricts which tokens can form valid tool invocations. But during freeform generation, the model has no structural constraint at all. The token stream between tool calls is unconstrained natural language -- the model can generate anything, including malformed English, hallucinated words, and syntactically incoherent text.

Formal grammars of natural language exist. The English Resource Grammar (ERG) is a broad-coverage HPSG grammar with 51,517 types, 43,729 lexemes, and 293 construction rules, maintained since 1994. It defines precisely which English sentences are well-formed. If we could compile this grammar into per-token valid masks -- the same interface our CFG grammars use -- we could constrain freeform generation toward grammatical English.

The existing implementation of the ERG (ACE) is a 20-year-old C codebase that uses nested functions (GCC-only), fixed-address mmap (fails on ARM64 macOS), and JIT compilation via runtime dlopen. It is fundamentally incompatible with Apple Silicon. ACE is a reference implementation. The ERG is the knowledge. We take the knowledge directly from the TDL source files.

## 2. Proposed Solution

Build a native Python module (`tgirl.lingo`) that reads HPSG grammars in TDL format and produces per-token valid masks for any tokenizer. The module implements the same `GrammarState` protocol as our existing CFG grammars, making it drop-in compatible with the sampling loop and OT transport.

**V1 scope (this PRD):** Type-level compatibility checking, not full feature structure unification. The ERG's 51k+ type hierarchy and 44k lexicon provide enough structure for two capabilities:

1. **Linguistic coherence signal** -- ratio of tokens mapping to known ERG lexemes vs unknown words. Routes through the modulation matrix as source signal row 11.
2. **GrammarState adapter** -- type-level compatibility filtering on next-token candidates. Coarser than full chart parsing but useful and fast.

**V2 scope (deferred):** Full incremental chart parsing with feature structure unification, MLX-accelerated batch unification.

The module is designed to be standalone -- no dependency on tgirl core modules. Uses the `GrammarState` protocol (a Python Protocol class) as its only integration surface.

```python
from tgirl.lingo import load_grammar, LingoGrammarState

erg = load_grammar("erg-2025/")
constraint = erg.constrain(tokenizer.get_vocab())

# Per token: which tokens continue well-formed English?
valid_mask = constraint.get_valid_mask(prefix_tokens)
constraint.advance(selected_token)
```

## 3. Architecture Impact

### New files
- `src/tgirl/lingo/__init__.py` -- public API: `load_grammar`, `LingoGrammarState`
- `src/tgirl/lingo/tdl_parser.py` -- TDL syntax parser producing AST
- `src/tgirl/lingo/types.py` -- type hierarchy with multiple inheritance, subsumption
- `src/tgirl/lingo/lexicon.py` -- word-to-lexeme-type mapping, token-to-lexeme mapping
- `src/tgirl/lingo/grammar_state.py` -- `GrammarState` protocol implementation
- `tests/test_tdl_parser.py` -- TDL parser unit tests
- `tests/test_lingo_types.py` -- type hierarchy unit tests
- `tests/test_lingo_lexicon.py` -- lexicon and token mapping tests
- `tests/test_lingo_grammar_state.py` -- grammar state protocol tests

### Modified files
- `src/tgirl/modulation.py` -- expand modulation matrix from (11, 7) to (12, 7); add `linguistic_coherence` as source signal row 11; update `DEFAULT_MATRIX`, `DEFAULT_CONDITIONERS`, `EnvelopeConfig`, `EnvelopeState`, both hook classes
- `src/tgirl/sample_mlx.py` -- no changes (grammar state protocol unchanged)
- `src/tgirl/sample.py` -- no changes (grammar state protocol unchanged)

### Data model changes
- `EnvelopeConfig.matrix_shape` grows from `(11, 7)` to `(12, 7)`
- `DEFAULT_MATRIX` gains row 11: `linguistic_coherence`
- `DEFAULT_CONDITIONERS` gains entry 11: `SourceConditionerConfig(range_min=0.0, range_max=1.0)`
- `EnvelopeState.prev_smoothed` grows from 11 to 12 elements

### New dependencies
- None for core functionality (pure Python + pydantic)
- ERG TDL source files at `~/ontologi/tools/erg-2025/` (already present)

### API additions
- `tgirl.lingo.load_grammar(path) -> LingoGrammar` -- load TDL grammar from directory
- `LingoGrammar.constrain(vocab) -> LingoGrammarState` -- create per-tokenizer state
- `LingoGrammarState` implements `GrammarStateMlx` protocol

## 4. Acceptance Criteria

1. TDL parser handles all ERG syntax constructs: type definitions with `:=` inheritance, `&` conjunction, `[ ]` feature structures, `#var` coreferences, `< >` lists, `;` comments, `%suffix` directives, docstrings, block comments (`#|...|#`), and `:include` directives
2. Type hierarchy loads 51,517+ types from ERG source in under 5 seconds on Apple Silicon
3. Subsumption checking (`is_subtype(a, b)`) works correctly for the loaded hierarchy, including multiple inheritance
4. Lexicon loads 43,729+ entries with word-to-lexeme-type mapping from `lexicon.tdl`
5. Multi-word lexemes (e.g., `ORTH < "2001", "A", "Space", "Odyssey" >`) are handled correctly
6. Token-to-lexeme mapping covers >80% of a typical model's vocabulary by token frequency
7. `linguistic_coherence` signal produces values in [0.0, 1.0] representing ratio of known-lexeme tokens to total tokens in a window
8. `LingoGrammarState` implements the `GrammarStateMlx` protocol: `get_valid_mask_mx(vocab_size)`, `is_accepting()`, `advance(token_id)`
9. All tensor math uses native framework ops -- no Python iteration on vocab-sized arrays
10. Modulation matrix grows from (11, 7) to (12, 7) with `linguistic_coherence` routed through the matrix
11. ERG source files are referenced by path (not bundled) -- the module discovers them at `load_grammar()` time
12. All existing tests continue to pass (modulation matrix shape change is backward-compatible via config)
13. The `tgirl.lingo` package has zero imports from other tgirl modules (standalone design)

## 5. Risk Assessment

- **TDL parsing complexity:** TDL is a 30-year-old format with numerous syntactic variants, editor directives, and undocumented conventions. The parser may encounter constructs not covered by the initial implementation. Mitigate: test against real ERG files early and often; log unparsed constructs rather than crashing.
- **Type hierarchy scale:** 51k types with multiple inheritance may have deep lattice structures. GLB computation for all type pairs is O(n^2) in the worst case. Mitigate: v1 uses lazy GLB computation (compute on demand, cache results) rather than precomputing the full table.
- **Token mapping coverage:** Subword tokens may not map cleanly to ERG lexemes. A token like "un" could be a prefix of hundreds of words. Mitigate: map tokens to the set of lexeme types they could begin, not exact word matches. Use the tokenizer's full vocabulary scan pattern (same as NestingDepthHookMlx).
- **Modulation matrix regression:** Adding a 12th source row changes the matrix shape. Existing configs with 11-row matrices must continue to work. Mitigate: the new row has zero weights by default, making it a no-op unless explicitly configured.
- **ERG version coupling:** The module is tested against ERG 2025. Future ERG versions may change TDL syntax or type names. Mitigate: parser is generic TDL, not ERG-specific; type hierarchy is loaded dynamically.

## 6. Open Questions

1. Should the GrammarState use type-level compatibility (is GLB non-bottom?) or category-level filtering (is next word's HEAD type compatible with current phrase category)? V1 starts with category-level as it's simpler and faster.
2. Should unknown words (tokens not in the ERG lexicon) be allowed or penalized? Allowing them with a coherence penalty (via the modulation matrix) seems right -- hard-blocking would be too restrictive for creative text.
3. Should the `linguistic_coherence` signal operate on a sliding window or the full generation so far? A sliding window (last N tokens) is more responsive; full history is more stable.
4. How should multi-word lexemes interact with the tokenizer? If a lexeme spans 4 words but the tokenizer splits them into 6 tokens, the token-to-lexeme mapping needs to handle partial matches.

## 7. Out of Scope

- Full feature structure unification (v2 -- requires unification engine)
- Incremental chart parsing (v2 -- requires construction rule application)
- MLX-accelerated batch unification (v2 -- requires matrix representation of feature structures)
- Grammar bundling/packaging (ERG files referenced by path, not packaged)
- Support for grammars other than ERG (TDL parser is generic, but only ERG is tested)
- Unknown word handling via generic lexeme types (v2 -- requires morphological analysis)
- Integration with freeform generation mode (v1 adds the signal; freeform masking is v2)
