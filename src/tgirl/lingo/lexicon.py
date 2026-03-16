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
