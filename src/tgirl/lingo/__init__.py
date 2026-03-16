"""Native LinGO GSU -- TDL-to-Token Constraint Compiler.

Standalone module that reads HPSG grammars in TDL format (such as the
English Resource Grammar) and produces per-token valid masks through
the GrammarState protocol. Zero imports from other tgirl modules.
"""

from __future__ import annotations

from tgirl.lingo.grammar_state import (
    CoherenceTracker,
    LingoGrammar,
    LingoGrammarState,
    load_grammar,
)
from tgirl.lingo.lexicon import LexEntry, Lexicon, TokenLexemeMap, load_lexicon
from tgirl.lingo.tdl_parser import parse_tdl, parse_tdl_directory, parse_tdl_file, tokenize_tdl
from tgirl.lingo.types import TypeHierarchy

__all__ = [
    "CoherenceTracker",
    "LexEntry",
    "Lexicon",
    "LingoGrammar",
    "LingoGrammarState",
    "TokenLexemeMap",
    "TypeHierarchy",
    "load_grammar",
    "load_lexicon",
    "parse_tdl",
    "parse_tdl_directory",
    "parse_tdl_file",
    "tokenize_tdl",
]
