"""Linguistic coherence signal and GrammarState adapter for HPSG grammars.

CoherenceTracker: sliding-window ratio of known-lexeme tokens.
LingoGrammarState: implements GrammarStateMlx protocol using type-level
compatibility. V1 provides coherence signal only (no hard masking).
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from tgirl.lingo.lexicon import Lexicon, TokenLexemeMap
from tgirl.lingo.types import TypeHierarchy

logger = logging.getLogger(__name__)


class CoherenceTracker:
    """Tracks linguistic coherence over a sliding window.

    Coherence = (number of known-lexeme tokens in window) / window_size.
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
        if len(self._window) == self._window_size and self._window[0]:
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


class LingoGrammarState:
    """GrammarState protocol implementation using ERG type compatibility.

    V1: all tokens valid (no hard masking). The coherence signal provides
    soft modulation through the ADSR modulation matrix.

    is_accepting() returns True — freeform grammar is always accepting.
    CRITICAL: The sampling loop (sample_mlx.py:489) masks out all stop/EOS
    tokens when is_accepting() returns False. Returning False would prevent
    the model from ever terminating.
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

    def get_valid_mask_mx(self, tokenizer_vocab_size: int) -> mx.array:
        """Return boolean mask of valid next tokens. V1: all True."""
        return mx.ones(tokenizer_vocab_size, dtype=mx.bool_)

    def is_accepting(self) -> bool:
        """Freeform grammar is always accepting — returns True."""
        return True

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
        self.hierarchy = hierarchy
        self.lexicon = lexicon

    def constrain(
        self,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
        window_size: int = 32,
    ) -> LingoGrammarState:
        """Create a per-tokenizer grammar state."""
        token_map = TokenLexemeMap(self.lexicon, tokenizer_decode, vocab_size)
        tracker = CoherenceTracker(token_map, window_size=window_size)
        return LingoGrammarState(token_map, tracker, vocab_size)


def load_grammar(path: str | Path) -> LingoGrammar:
    """Load a TDL grammar from a directory.

    Starting from english.tdl, recursively follows all :include directives
    to parse every referenced TDL file. Builds the type hierarchy and
    lexicon from all collected definitions.
    """
    from tgirl.lingo.lexicon import load_lexicon
    from tgirl.lingo.tdl_parser import TdlDefinition, parse_tdl_directory

    grammar_path = Path(path)
    top_file = grammar_path / "english.tdl"
    if not top_file.exists():
        raise FileNotFoundError(f"Top-level TDL file not found: {top_file}")

    logger.info("Loading grammar from %s", top_file)
    all_nodes = parse_tdl_directory(top_file)
    type_defs = [d for d in all_nodes if isinstance(d, TdlDefinition)]

    logger.info("Building type hierarchy from %d definitions", len(type_defs))
    hierarchy = TypeHierarchy(type_defs)

    logger.info("Building lexicon")
    lexicon = load_lexicon(type_defs)

    logger.info(
        "Grammar loaded: %d types, %d lexeme types, %d words",
        len(hierarchy.all_types),
        len(lexicon.all_lexeme_types),
        len(lexicon.all_words),
    )
    return LingoGrammar(hierarchy, lexicon)
