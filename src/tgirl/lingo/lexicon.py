"""Lexicon loader — word-to-lexeme-type mapping from ERG lexicon.

Parses TDL lexicon entries and builds a mapping from surface forms
(lowercased) to the set of lexeme types they can instantiate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from tgirl.lingo.tdl_parser import (
    TdlDefinition,
    TdlFeatStruct,
    TdlList,
    TdlString,
    TdlConj,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LexEntry:
    """A single lexicon entry."""
    name: str
    lexeme_type: str
    orth: tuple[str, ...]


class Lexicon:
    """Word-to-lexeme-type mapping from ERG lexicon.

    Maps surface forms (lowercased) to the set of lexeme types they
    can instantiate. Multi-word entries map each word individually.
    """

    def __init__(self, entries: list[LexEntry]) -> None:
        self._word_to_types: dict[str, set[str]] = {}
        self._all_lexeme_types: set[str] = set()

        for entry in entries:
            self._all_lexeme_types.add(entry.lexeme_type)
            for word in entry.orth:
                key = word.lower()
                if key not in self._word_to_types:
                    self._word_to_types[key] = set()
                self._word_to_types[key].add(entry.lexeme_type)

    def types_for_word(self, word: str) -> frozenset[str]:
        """Return lexeme types for a surface form (case-insensitive)."""
        return frozenset(self._word_to_types.get(word.lower(), set()))

    def is_known_word(self, word: str) -> bool:
        """True if the word appears in any lexicon entry."""
        return word.lower() in self._word_to_types

    @property
    def all_words(self) -> frozenset[str]:
        """All known surface forms (lowercased)."""
        return frozenset(self._word_to_types.keys())

    @property
    def all_lexeme_types(self) -> frozenset[str]:
        """All lexeme types used in the lexicon."""
        return frozenset(self._all_lexeme_types)


def _extract_orth(body) -> list[str] | None:
    """Extract ORTH values from a TDL definition body.

    Navigates the AST to find ORTH < "word1", ... > in the top-level
    feature structure or conjunction.
    """
    if body is None:
        return None

    # Direct feature structure
    if isinstance(body, TdlFeatStruct):
        orth = body.features.get("ORTH")
        if isinstance(orth, TdlList):
            return [e.value for e in orth.elements if isinstance(e, TdlString)]
        return None

    # Conjunction — look through parts for feature structures
    if isinstance(body, TdlConj):
        for part in body.parts:
            result = _extract_orth(part)
            if result is not None:
                return result

    return None


def load_lexicon(definitions: list[TdlDefinition]) -> Lexicon:
    """Build lexicon from parsed TDL definitions.

    Extracts ORTH values and supertype from each definition.
    Definitions without ORTH features are skipped.
    """
    entries: list[LexEntry] = []

    for defn in definitions:
        if defn.is_addendum:
            continue
        orth = _extract_orth(defn.body)
        if orth is None or len(orth) == 0:
            continue
        if not defn.supertypes:
            continue
        entries.append(LexEntry(
            name=defn.name,
            lexeme_type=defn.supertypes[0],
            orth=tuple(orth),
        ))

    logger.info("Loaded %d lexicon entries", len(entries))
    return Lexicon(entries)


class TokenLexemeMap:
    """Maps token IDs to sets of compatible lexeme types.

    Built by scanning the full tokenizer vocabulary once at init time.
    For each token, decode it to text, normalize, and check for exact
    whole-word matches in the lexicon. No prefix matching.
    """

    def __init__(
        self,
        lexicon: Lexicon,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
    ) -> None:
        self._token_types: dict[int, frozenset[str]] = {}
        self._known_ids: set[int] = set()

        for tid in range(vocab_size):
            text = tokenizer_decode([tid])
            # BPE word boundary: strip leading space
            word = text.lstrip()
            # Also strip trailing whitespace
            word = word.strip()
            # Lowercase for lookup
            word_lower = word.lower()

            types = lexicon.types_for_word(word_lower)
            if types:
                self._token_types[tid] = types
                self._known_ids.add(tid)

        self._vocab_size = vocab_size
        logger.info(
            "Token-lexeme map: %d/%d tokens known (%.1f%%)",
            len(self._known_ids), vocab_size,
            100.0 * len(self._known_ids) / vocab_size if vocab_size > 0 else 0,
        )

    def types_for_token(self, token_id: int) -> frozenset[str]:
        """Lexeme types compatible with this token."""
        return self._token_types.get(token_id, frozenset())

    def is_known_token(self, token_id: int) -> bool:
        """True if token maps to at least one lexeme type."""
        return token_id in self._known_ids

    @property
    def coverage(self) -> float:
        """Fraction of vocabulary with at least one lexeme mapping."""
        if self._vocab_size == 0:
            return 0.0
        return len(self._known_ids) / self._vocab_size

    @property
    def known_token_ids(self) -> frozenset[int]:
        """Token IDs that map to at least one lexeme type."""
        return frozenset(self._known_ids)
