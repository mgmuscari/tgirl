# PRP: Native LinGO GSU

## Source PRD: docs/PRDs/lingo-native.md
## Date: 2026-03-16

## 1. Context Summary

Build `tgirl.lingo`, a standalone Python module that reads the English Resource Grammar (ERG) from TDL source files and produces per-token valid masks through the existing `GrammarState` protocol. V1 uses type-level compatibility (not full feature structure unification) and provides a `linguistic_coherence` signal for the ADSR modulation matrix. The module has zero imports from other tgirl modules.

Full design: `docs/design/lingo-native-gsu.md`

## 2. Codebase Analysis

### Protocols to implement
- `GrammarStateMlx` protocol (sample_mlx.py:116-121): `get_valid_mask_mx(vocab_size) -> mx.array`, `is_accepting() -> bool`, `advance(token_id) -> None`
- The sampling loop (sample_mlx.py:469-480) checks `hasattr(grammar_state, "get_valid_mask_mx")` for MLX path, falls back to torch `.get_valid_mask()` + numpy conversion

### Existing patterns to follow
- **Vocab scan at init time:** `NestingDepthHookMlx.__init__` (sample_mlx.py:178-205) iterates `range(vocab_size)`, calls `tokenizer_decode([tid])`, builds per-token metadata dict. Same pattern needed for token-to-lexeme mapping.
- **Grammar state wrapping:** `LLGuidanceGrammarStateMlx` (outlines_adapter.py:132-178) wraps an `LLMatcher` and produces `mx.array` masks. Our `LingoGrammarState` wraps the type hierarchy + lexicon instead.
- **Source signal in ModMatrixHookMlx:** (modulation.py:285-434) The `pre_forward` method computes 11 raw source signals, conditions them, and multiplies by the mod matrix. Row 11 (new) will be `linguistic_coherence`.

### Integration points
- `ModMatrixHookMlx.pre_forward` (modulation.py:285) -- needs access to linguistic_coherence score
- `EnvelopeState.prev_smoothed` (modulation.py:46) -- list grows from 11 to 12 entries
- `DEFAULT_MATRIX` (modulation.py:154-174) -- gains row 11
- `DEFAULT_CONDITIONERS` (modulation.py:181-195) -- gains entry 11
- `EnvelopeConfig.matrix_shape` (modulation.py:229-230) -- returns (12, 7)

### ERG source structure (~/ontologi/tools/erg-2025/)
- `english.tdl` -- top-level load file with `:include` directives
- `fundamentals.tdl` (5,182 lines) -- core type definitions with deep feature structures
- `letypes.tdl` (1,974 lines) -- lexical entry types (956 leaf types for lexeme classification)
- `lexicon.tdl` (226,198 lines) -- 43,729 lexeme entries with ORTH, type, and features
- `constructions.tdl` -- 293 construction rule instances
- `inflr.tdl` -- inflectional rules with `%suffix` directives and `%(letter-set ...)` macros

### TDL syntax constructs to parse
```
; Line comment
#| Block comment |#
type := supertype1 & supertype2.             ; type with supertypes
type := supertype & [ FEAT value ].          ; type with feature structure
type := supertype & [ FEAT.PATH value ].     ; dot-separated feature path
type := supertype & [ FEAT #coref ].         ; coreference variable
type := supertype & [ FEAT < elem1, elem2 > ]. ; list
type := supertype & [ FEAT < ... > ].        ; open list
type := supertype & [ FEAT < elem, ... > ].  ; list with rest
:include "filename".                         ; file inclusion
:begin :type.  :end :type.                   ; section directives
:begin :instance :status rule.               ; instance sections
%suffix (!s !ss) (es eses)                   ; inflectional suffix rules
%(letter-set (!c bdfg...))                   ; letter-set macros
"""docstring"""                              ; triple-quoted docstring
lexeme := type & [ ORTH < "word" > ].        ; lexeme with orthography
```

## 3. Implementation Plan

**Test Command:** `pytest tests/test_tdl_parser.py tests/test_lingo_types.py tests/test_lingo_lexicon.py tests/test_lingo_grammar_state.py -v`

### Task 1: TDL Parser

**Files:** `src/tgirl/lingo/__init__.py` (create), `src/tgirl/lingo/tdl_parser.py` (create), `tests/test_tdl_parser.py` (create)

**Approach:**

Build a recursive descent parser for TDL syntax. The parser produces an AST of `TdlDefinition` objects, each containing:
- `name: str` -- identifier of the type/instance
- `supertypes: list[str]` -- parent types (from `:=` and `&`)
- `features: dict[str, TdlValue]` -- feature structures (nested dicts for paths)
- `coreferences: dict[str, list[str]]` -- `#var` bindings
- `docstring: str | None` -- triple-quoted docstring
- `section: str | None` -- section context (`:begin :type`, `:begin :instance`, etc.)

Key parsing stages:
1. **Tokenizer** -- split TDL text into tokens: identifiers, strings, operators (`:=`, `&`, `.`, `[`, `]`, `<`, `>`, `#`, `,`), comments (`;` to EOL, `#|...|#`), docstrings (`"""..."""`)
2. **Section tracking** -- parse `:begin :type.` / `:end :type.` and `:begin :instance :status <status>.` / `:end :instance.` directives
3. **Definition parser** -- parse `name := body .` where body is supertypes `&` conjunction with optional feature structures
4. **Feature structure parser** -- parse `[ FEAT val, FEAT2 val2 ]` with nesting, dot paths, coreferences, and lists
5. **Include handling** -- parse `:include "filename".` and return as a special AST node (caller resolves paths)
6. **Suffix/letter-set** -- parse `%suffix (...)` and `%(letter-set (...))` as opaque AST nodes (v1 stores them but doesn't interpret)

The parser must handle:
- UTF-8 encoded files with extended characters in letter-sets
- Definitions spanning multiple lines (terminated by `.`)
- Nested feature structures 6-10 levels deep
- Coreferences (`#var`) appearing in feature values
- Lists with elements, open lists (`< ... >`), and cons lists (`< elem, ... >`)
- Block comments `#| ... |#` (including nested)

```python
@dataclass
class TdlToken:
    kind: str  # "ident", "string", "op", "directive", "suffix", "letterset", "docstring"
    value: str
    line: int
    col: int

@dataclass
class TdlFeature:
    """A feature value in a feature structure."""
    pass

@dataclass
class TdlType(TdlFeature):
    name: str

@dataclass
class TdlString(TdlFeature):
    value: str

@dataclass
class TdlCoref(TdlFeature):
    name: str

@dataclass
class TdlList(TdlFeature):
    elements: list[TdlFeature]
    open: bool  # True if ends with ...

@dataclass
class TdlFeatStruct(TdlFeature):
    features: dict[str, TdlFeature]

@dataclass
class TdlConj(TdlFeature):
    parts: list[TdlFeature]

@dataclass
class TdlDefinition:
    name: str
    supertypes: list[str]
    body: TdlFeature | None
    docstring: str | None
    section: str | None
    is_instance: bool

@dataclass
class TdlInclude:
    filename: str

@dataclass
class TdlDirective:
    kind: str  # "begin", "end", "suffix", "letter-set"
    content: str

def tokenize_tdl(text: str) -> list[TdlToken]: ...
def parse_tdl(tokens: list[TdlToken]) -> list[TdlDefinition | TdlInclude | TdlDirective]: ...
def parse_tdl_file(path: Path) -> list[TdlDefinition | TdlInclude | TdlDirective]: ...
```

**Tests:**
- Tokenizer correctly splits `type := super & [ FEAT val ].` into tokens
- Parser handles simple type definition: `a := b.`
- Parser handles conjunction: `a := b & c.`
- Parser handles feature structure: `a := b & [ FEAT val ].`
- Parser handles nested features: `a := b & [ F1 [ F2 val ] ].`
- Parser handles dot paths: `a := b & [ F1.F2.F3 val ].`
- Parser handles coreferences: `a := b & [ F1 #x, F2 #x ].`
- Parser handles lists: `a := b & [ ORTH < "word" > ].`
- Parser handles open lists: `a := b & [ LIST < ... > ].`
- Parser handles lists with rest: `a := b & [ LIST < first, ... > ].`
- Parser handles docstrings: definition with `"""..."""` before `.`
- Parser handles block comments: `#| ... |#` stripped from input
- Parser handles line comments: `; ...` stripped from input
- Parser handles `:include "filename".` directive
- Parser handles `:begin :type.` / `:end :type.` section directives
- Parser handles `%suffix` directives (stored as opaque AST node)
- Parser handles `%(letter-set ...)` macros (stored as opaque AST node)
- Parser handles multi-line definitions (continued until `.`)
- Parser handles real ERG file: `parse_tdl_file("fundamentals.tdl")` produces >100 definitions without errors
- Parser handles real ERG file: `parse_tdl_file("letypes.tdl")` produces lexical entry type definitions
- Parser handles real ERG file: `parse_tdl_file("lexicon.tdl")` parses all 43,729 entries (slow test, mark with `@pytest.mark.slow`)

**Validation:** `pytest tests/test_tdl_parser.py -v`

### Task 2: Type Hierarchy

**Files:** `src/tgirl/lingo/types.py` (create), `tests/test_lingo_types.py` (create)

**Approach:**

Build a type lattice from parsed TDL definitions. Each type has:
- `name: str` -- unique identifier
- `supertypes: frozenset[str]` -- direct parents
- `subtypes: frozenset[str]` -- direct children (computed)
- All ancestor and descendant sets are precomputed at build time for O(1) subsumption checking

```python
@dataclass(frozen=True)
class TypeNode:
    name: str
    supertypes: frozenset[str]  # direct parents

class TypeHierarchy:
    """Type lattice with multiple inheritance and O(1) subsumption."""

    def __init__(self, definitions: list[TdlDefinition]) -> None:
        """Build hierarchy from parsed TDL definitions.

        Computes transitive closure of supertype relations for
        O(1) is_subtype lookups.
        """
        ...

    def is_subtype(self, child: str, parent: str) -> bool:
        """True if child is the same as or a subtype of parent."""
        ...

    def common_supertypes(self, a: str, b: str) -> frozenset[str]:
        """Return all types that are supertypes of both a and b."""
        ...

    def greatest_lower_bound(self, a: str, b: str) -> str | None:
        """Return the GLB of two types, or None if incompatible.

        The GLB is the most specific type that is a supertype of some
        type that is a subtype of both a and b. For v1, approximated
        by checking if the intersection of descendant sets is non-empty.
        """
        ...

    def subtypes_of(self, type_name: str) -> frozenset[str]:
        """Return all (transitive) subtypes of a type."""
        ...

    @property
    def all_types(self) -> frozenset[str]:
        """All type names in the hierarchy."""
        ...

    @property
    def leaf_types(self) -> frozenset[str]:
        """Types with no subtypes."""
        ...
```

Performance strategy: Build the hierarchy in two passes:
1. First pass: collect all type names and their direct supertypes
2. Second pass: compute transitive ancestor sets using topological sort (parents before children). For each type, ancestors = direct supertypes union their ancestors. Store as `dict[str, frozenset[str]]`.
3. Descendant sets: invert the ancestor map. For each type in the ancestor set of X, X is a descendant.

This avoids O(n^2) GLB precomputation. `is_subtype(a, b)` is O(1): check `b in ancestors[a]`. GLB existence is checked lazily.

**Tests:**
- Simple hierarchy: `a := b.` -- `is_subtype("a", "b")` is True
- Transitivity: `a := b. b := c.` -- `is_subtype("a", "c")` is True
- Multiple inheritance: `a := b & c.` -- `is_subtype("a", "b")` and `is_subtype("a", "c")` both True
- `is_subtype("a", "a")` is True (reflexive)
- `is_subtype("b", "a")` is False (not reverse)
- `common_supertypes("a", "b")` includes common ancestors
- `greatest_lower_bound("b", "c")` returns "a" when `a := b & c.`
- `greatest_lower_bound("b", "d")` returns None when no common subtype exists
- `leaf_types` returns types with no children
- Load real ERG: build hierarchy from `fundamentals.tdl` + `letypes.tdl`, verify >1000 types loaded
- Load full ERG: build hierarchy from all TDL files, verify ~51k types loaded in <5 seconds (mark `@pytest.mark.slow`)
- `all_types` count matches expected ERG size
- `sign` type is an ancestor of `word_or_lexrule` (ERG hierarchy check)

**Validation:** `pytest tests/test_lingo_types.py -v`

### Task 3: Lexicon Loader

**Files:** `src/tgirl/lingo/lexicon.py` (create), `tests/test_lingo_lexicon.py` (create)

**Approach:**

Parse `lexicon.tdl` and extract word-to-lexeme-type mappings. Each lexicon entry has:
- `name: str` -- entry identifier (e.g., `100s_n1`)
- `lexeme_type: str` -- the supertype (e.g., `n_-_c-pl-num_le`)
- `orth: list[str]` -- surface forms from `ORTH < "word1", "word2" >`

```python
@dataclass(frozen=True)
class LexEntry:
    name: str
    lexeme_type: str
    orth: tuple[str, ...]  # surface form tokens

class Lexicon:
    """Word-to-lexeme-type mapping from ERG lexicon.

    Maps surface forms (lowercased) to the set of lexeme types they
    can instantiate. Multi-word entries map each word individually
    plus the full phrase.
    """

    def __init__(self, entries: list[LexEntry]) -> None:
        ...

    def types_for_word(self, word: str) -> frozenset[str]:
        """Return lexeme types for a surface form (case-insensitive)."""
        ...

    def is_known_word(self, word: str) -> bool:
        """True if the word appears in any lexicon entry."""
        ...

    @property
    def all_words(self) -> frozenset[str]:
        """All known surface forms."""
        ...

    @property
    def all_lexeme_types(self) -> frozenset[str]:
        """All lexeme types used in the lexicon."""
        ...

def load_lexicon(definitions: list[TdlDefinition]) -> Lexicon:
    """Build lexicon from parsed TDL definitions.

    Extracts ORTH values and supertype from each definition.
    Definitions without ORTH features are skipped (they're type
    definitions, not lexeme entries).
    """
    ...
```

Extracting ORTH from a parsed definition:
- The entry's body contains a feature structure with `ORTH < "word1", ... >`
- Navigate the AST: find the `ORTH` key in the top-level feature structure
- Extract string values from the list

**Tests:**
- Parse single-word entry: `cat_n1 := n_-_c_le & [ ORTH < "cat" > ].` -- `types_for_word("cat")` includes `n_-_c_le`
- Parse multi-word entry: `ORTH < "2001", "A", "Space", "Odyssey" >` -- all individual words map, full phrase maps
- Case insensitivity: `types_for_word("Cat")` same as `types_for_word("cat")`
- Unknown word: `is_known_word("xyzzy")` is False
- Load real ERG lexicon: `load_lexicon` on parsed `lexicon.tdl` produces >43,000 entries (mark `@pytest.mark.slow`)
- `all_lexeme_types` count is in expected range (800-1000 unique types)
- Entry without ORTH is skipped (type definitions mixed into lexicon file)
- Duplicate words map to multiple types: "run" maps to both verb and noun types

**Validation:** `pytest tests/test_lingo_lexicon.py -v`

### Task 4: Token-to-Lexeme Mapping

**Files:** `src/tgirl/lingo/lexicon.py` (modify), `tests/test_lingo_lexicon.py` (modify)

**Approach:**

For each token in a model's vocabulary, determine which lexeme types it could belong to. This is the vocabulary-grammar bridge, following the same pattern as `NestingDepthHookMlx.__init__` (sample_mlx.py:196-205).

```python
class TokenLexemeMap:
    """Maps token IDs to sets of compatible lexeme types.

    Built by scanning the full tokenizer vocabulary once at init time.
    For each token, decode it to text, strip whitespace, lowercase,
    and check which words in the lexicon it could be or be a prefix of.
    """

    def __init__(
        self,
        lexicon: Lexicon,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
    ) -> None:
        """Build token-to-lexeme mapping via vocabulary scan.

        For each token ID in [0, vocab_size):
        1. Decode token to text
        2. Strip leading/trailing whitespace, lowercase
        3. If exact match in lexicon: map to those lexeme types
        4. If prefix match: map to union of matching lexeme types
        5. If no match: token is "unknown" (empty set)
        """
        ...

    def types_for_token(self, token_id: int) -> frozenset[str]:
        """Lexeme types compatible with this token."""
        ...

    def is_known_token(self, token_id: int) -> bool:
        """True if token maps to at least one lexeme type."""
        ...

    @property
    def coverage(self) -> float:
        """Fraction of vocabulary with at least one lexeme mapping."""
        ...

    @property
    def known_token_ids(self) -> frozenset[int]:
        """Token IDs that map to at least one lexeme type."""
        ...
```

Performance: The vocab scan is O(vocab_size * avg_decode_time). For a 128k vocab tokenizer, this is ~1-3 seconds (same order as NestingDepthHookMlx init). The lexicon word lookup uses a dict, so each token's lookup is O(1) amortized.

For prefix matching: build a set of all known words. For each decoded token text, check if `text in known_words` (exact) or if any known word starts with `text` (prefix). To avoid O(n) prefix scan, build a prefix trie or sorted list with binary search. For v1, a simple approach: build `word_prefixes: dict[str, set[str]]` mapping each prefix of length 1-20 to the set of words with that prefix.

**Tests:**
- Exact match: token decoding to "cat" maps to `n_-_c_le` types
- Prefix match: token decoding to "ca" maps to types for "cat", "car", "can", etc.
- Whitespace handling: token decoding to " cat" (with leading space, as in BPE) maps same as "cat"
- Unknown token: special tokens (BOS, EOS) map to empty set
- Coverage property: returns float in [0, 1]
- known_token_ids: returns frozenset of ints
- Mock tokenizer test: create a small vocab with known decodings, verify mapping
- Integration with real tokenizer if available (mark `@pytest.mark.slow`)

**Validation:** `pytest tests/test_lingo_lexicon.py -v`

### Task 5: Linguistic Coherence Signal

**Files:** `src/tgirl/lingo/grammar_state.py` (create), `tests/test_lingo_grammar_state.py` (create)

**Approach:**

The `linguistic_coherence` signal is the ratio of known-lexeme tokens to total tokens in a sliding window. This is computed independently of the full GrammarState -- it's a simple metric that the modulation matrix can use immediately.

```python
class CoherenceTracker:
    """Tracks linguistic coherence over a sliding window.

    Coherence = (number of known-lexeme tokens in window) / window_size.
    A token is "known" if it maps to at least one ERG lexeme type.
    """

    def __init__(
        self,
        token_lexeme_map: TokenLexemeMap,
        window_size: int = 32,
    ) -> None:
        self._map = token_lexeme_map
        self._window_size = window_size
        self._known_count = 0
        self._window: collections.deque[bool] = collections.deque(maxlen=window_size)

    def advance(self, token_id: int) -> None:
        """Record a token and update the coherence window."""
        is_known = self._map.is_known_token(token_id)
        if len(self._window) == self._window_size:
            # Evict oldest
            if self._window[0]:
                self._known_count -= 1
        self._window.append(is_known)
        if is_known:
            self._known_count += 1

    @property
    def coherence(self) -> float:
        """Current coherence score in [0.0, 1.0]."""
        if not self._window:
            return 0.0
        return self._known_count / len(self._window)

    def reset(self) -> None:
        """Clear the window for a new generation pass."""
        self._window.clear()
        self._known_count = 0
```

**Tests:**
- Empty tracker: coherence is 0.0
- All known tokens: coherence is 1.0
- All unknown tokens: coherence is 0.0
- Mixed tokens: coherence is correct ratio
- Window eviction: after window_size+1 advances, oldest token is evicted
- Reset clears state
- Window size of 1: coherence is 0.0 or 1.0

**Validation:** `pytest tests/test_lingo_grammar_state.py -v`

### Task 6: GrammarState Adapter

**Files:** `src/tgirl/lingo/grammar_state.py` (modify), `src/tgirl/lingo/__init__.py` (modify), `tests/test_lingo_grammar_state.py` (modify)

**Approach:**

Implement the `GrammarStateMlx` protocol using type-level compatibility. For v1, the state tracks the current lexeme type context and uses it to filter next tokens.

```python
class LingoGrammarState:
    """GrammarState protocol implementation using ERG type compatibility.

    V1 uses a simple model:
    - Each token maps to a set of lexeme types (from TokenLexemeMap)
    - Tokens are valid if they map to at least one known lexeme type
    - Unknown tokens are allowed but penalized via coherence signal
    - is_accepting() always returns False (freeform mode never "completes")

    The primary value is the coherence signal, not hard masking.
    Hard masking (blocking unknown tokens entirely) is too aggressive
    for freeform generation -- the model needs to produce punctuation,
    numbers, and novel words.
    """

    def __init__(
        self,
        token_lexeme_map: TokenLexemeMap,
        coherence_tracker: CoherenceTracker,
        vocab_size: int,
    ) -> None:
        self._map = token_lexeme_map
        self._coherence = coherence_tracker
        self._vocab_size = vocab_size
        # Pre-compute known-token mask as bool array for O(1) retrieval
        self._known_mask: mx.array  # shape (vocab_size,), dtype bool

    def get_valid_mask_mx(self, tokenizer_vocab_size: int) -> mx.array:
        """Return boolean mask of valid next tokens.

        V1: all tokens are valid (no hard masking). The value comes
        from the coherence signal, not from blocking tokens.
        Returns all-True mask.
        """
        return mx.ones(tokenizer_vocab_size, dtype=mx.bool_)

    def is_accepting(self) -> bool:
        """Freeform grammar never completes -- always False."""
        return False

    def advance(self, token_id: int) -> None:
        """Record token and update coherence tracker."""
        self._coherence.advance(token_id)

    def coherence_score(self) -> float:
        """Current linguistic coherence in [0.0, 1.0]."""
        return self._coherence.coherence

class LingoGrammar:
    """Loaded TDL grammar with type hierarchy and lexicon."""

    def __init__(
        self,
        hierarchy: TypeHierarchy,
        lexicon: Lexicon,
    ) -> None:
        ...

    def constrain(
        self,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
    ) -> LingoGrammarState:
        """Create a per-tokenizer grammar state."""
        ...

def load_grammar(path: str | Path) -> LingoGrammar:
    """Load a TDL grammar from a directory.

    Parses all TDL files referenced from the top-level load file,
    builds the type hierarchy and lexicon.
    """
    ...
```

**Tests:**
- `LingoGrammarState` satisfies `GrammarStateMlx` protocol (isinstance check)
- `get_valid_mask_mx` returns all-True mask of correct size
- `is_accepting()` returns False
- `advance()` updates coherence tracker
- `coherence_score()` returns float in [0.0, 1.0]
- `LingoGrammar.constrain()` produces working `LingoGrammarState`
- `load_grammar()` with real ERG path produces a `LingoGrammar` with >51k types and >43k lexemes (mark `@pytest.mark.slow`)
- Zero imports from `tgirl.sample`, `tgirl.grammar`, `tgirl.transport`, or other tgirl core modules (import isolation test)

**Validation:** `pytest tests/test_lingo_grammar_state.py -v`

### Task 7: Modulation Matrix Integration

**Files:** `src/tgirl/modulation.py` (modify), `tests/test_modulation.py` (modify)

**Approach:**

Expand the modulation matrix from (11, 7) to (12, 7). Add `linguistic_coherence` as source signal row 11.

Changes to `modulation.py`:

1. **`DEFAULT_MATRIX`** -- add row 11:
```python
[ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 11: linguistic_coherence (zero default = no-op)
```

2. **`DEFAULT_CONDITIONERS`** -- add entry 11:
```python
SourceConditionerConfig(range_min=0.0, range_max=1.0),  # 11: linguistic_coherence (already 0-1)
```

3. **`EnvelopeState.prev_smoothed`** -- default grows from `[0.0] * 11` to `[0.0] * 12`

4. **`EnvelopeConfig.matrix_shape`** -- returns `(12, 7)` instead of `(11, 7)`

5. **`ModMatrixHookMlx.__init__`** -- accept optional `coherence_fn: Callable[[], float] | None` parameter. If provided, it's called in `pre_forward` to get the coherence score.

6. **`ModMatrixHookMlx.pre_forward`** -- add `linguistic_coherence` to `raw_sources` list (position 11). If no `coherence_fn`, defaults to 0.0 (no-op with zero matrix weights).

7. **`ModMatrixHook`** (torch variant) -- same changes as MLX variant.

**Backward compatibility:** Existing `EnvelopeConfig` instances with 11-row matrices continue to work because:
- The new row has zero weights by default (no-op)
- `EnvelopeConfig` validates `matrix_flat` length matches `matrix_shape` product
- If a user provides a custom 11x7 matrix (77 values), we need to handle this: pad with zeros to 12x7 (84 values), or raise a clear error. Decision: pad with zeros and log a warning.

**Tests:**
- Default matrix shape is (12, 7) with 84 values
- Default conditioners has 12 entries
- EnvelopeState.prev_smoothed has 12 entries
- ModMatrixHookMlx with coherence_fn=None produces same output as before (linguistic_coherence=0.0 * zero row = no change)
- ModMatrixHookMlx with coherence_fn returning 0.8 and non-zero row 11 weights produces different temperature
- ModMatrixHook (torch) produces same results as MLX variant
- Backward compatibility: old 11-row matrix config pads to 12 rows
- All existing modulation tests still pass

**Validation:** `pytest tests/test_modulation.py -v`

## 4. Validation Gates

```bash
# Lint
ruff check src/tgirl/lingo/ tests/test_tdl_parser.py tests/test_lingo_types.py tests/test_lingo_lexicon.py tests/test_lingo_grammar_state.py

# Type checking
mypy src/tgirl/lingo/

# Unit tests (fast)
pytest tests/test_tdl_parser.py tests/test_lingo_types.py tests/test_lingo_lexicon.py tests/test_lingo_grammar_state.py -v -m "not slow"

# Unit tests (full, including ERG loading)
pytest tests/test_tdl_parser.py tests/test_lingo_types.py tests/test_lingo_lexicon.py tests/test_lingo_grammar_state.py -v

# Modulation regression
pytest tests/test_modulation.py -v

# Full suite (no regressions)
pytest tests/ -v --ignore=tests/test_cache.py \
    --ignore=tests/test_transport.py --ignore=tests/test_transport_mlx.py
```

## 5. Rollback Plan

All new code is in `src/tgirl/lingo/` (new package) and new test files. Modulation changes are additive (row 11 has zero weights by default). Rollback:
1. Delete `src/tgirl/lingo/`
2. Delete `tests/test_tdl_parser.py`, `tests/test_lingo_types.py`, `tests/test_lingo_lexicon.py`, `tests/test_lingo_grammar_state.py`
3. Revert modulation.py changes (shrink matrix from 12 to 11 rows)

## 6. Uncertainty Log

- **TDL parsing completeness:** The ERG uses TDL features accumulated over 30 years. The parser may encounter undocumented syntax in less-commonly-used files. Strategy: parse `fundamentals.tdl`, `letypes.tdl`, and `lexicon.tdl` first (these cover the core hierarchy and lexicon); defer parsing of `constructions.tdl` and `inflr.tdl` until needed. Log unparsed lines rather than crashing.
- **Prefix matching strategy:** The current plan maps tokens to lexeme types via prefix matching against the lexicon. This may over-match: a token "a" is a prefix of thousands of words. Consider filtering to only count exact whole-word matches, using token boundary detection (leading space in BPE tokens indicates word start). This needs experimentation.
- **GLB computation complexity:** The type hierarchy has ~51k types. Full GLB precomputation is O(n^2) = ~2.6 billion pairs. Lazy computation with caching is the right approach, but the cache may grow large for frequently-queried type pairs. Monitor memory usage during testing.
- **Coherence window size:** 32 tokens is a guess. Too small and the signal is noisy; too large and it's unresponsive. The modulation matrix's slew rate conditioning provides some smoothing, but the window size should be validated empirically.
- **ModMatrixHook coherence wiring:** The hook needs a `coherence_fn` callback, which requires the `LingoGrammarState` to be accessible from the hook. In the current architecture, hooks don't have access to other grammar state objects. Options: (a) pass coherence_fn at hook construction time, (b) compute coherence inside the hook using a separate token-lexeme map, (c) add coherence_fn as a field on the hook that gets wired during SamplingSession construction. Option (a) is cleanest and matches the existing pattern where hooks receive configuration at init time.
