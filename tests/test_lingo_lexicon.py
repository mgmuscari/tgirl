"""Tests for tgirl.lingo.lexicon — Lexicon loader and token-to-lexeme mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from tgirl.lingo.tdl_parser import (
    TdlDefinition,
    TdlFeatStruct,
    TdlList,
    TdlString,
    TdlType,
    TdlConj,
    tokenize_tdl,
    parse_tdl,
)
from tgirl.lingo.lexicon import LexEntry, Lexicon, TokenLexemeMap, load_lexicon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lex_def(name: str, lexeme_type: str, orth: list[str]) -> TdlDefinition:
    """Create a TdlDefinition that looks like a lexicon entry."""
    orth_list = TdlList([TdlString(w) for w in orth], open=False)
    body = TdlFeatStruct({"ORTH": orth_list})
    return TdlDefinition(
        name=name,
        supertypes=[lexeme_type],
        body=body if len(orth) > 0 else None,
        docstring=None,
        section="instance:lex-entry",
        is_instance=True,
        is_addendum=False,
        suffix=None,
    )


def _type_def(name: str, supertypes: list[str]) -> TdlDefinition:
    """Create a plain type definition (no ORTH)."""
    return TdlDefinition(
        name=name,
        supertypes=supertypes,
        body=None,
        docstring=None,
        section="type",
        is_instance=False,
        is_addendum=False,
        suffix=None,
    )


# ---------------------------------------------------------------------------
# LexEntry and Lexicon tests
# ---------------------------------------------------------------------------

class TestLexicon:

    def test_single_word_entry(self) -> None:
        entries = [LexEntry(name="cat_n1", lexeme_type="n_-_c_le", orth=("cat",))]
        lex = Lexicon(entries)
        assert "n_-_c_le" in lex.types_for_word("cat")

    def test_multi_word_entry(self) -> None:
        entries = [LexEntry(
            name="space_odyssey_n1",
            lexeme_type="n_-_c_le",
            orth=("2001", "A", "Space", "Odyssey"),
        )]
        lex = Lexicon(entries)
        # Individual words
        assert lex.is_known_word("2001")
        assert lex.is_known_word("space")  # case insensitive
        assert lex.is_known_word("odyssey")

    def test_case_insensitive(self) -> None:
        entries = [LexEntry(name="cat_n1", lexeme_type="n_-_c_le", orth=("Cat",))]
        lex = Lexicon(entries)
        assert lex.types_for_word("cat") == lex.types_for_word("CAT")
        assert lex.types_for_word("Cat") == lex.types_for_word("cat")

    def test_unknown_word(self) -> None:
        entries = [LexEntry(name="cat_n1", lexeme_type="n_-_c_le", orth=("cat",))]
        lex = Lexicon(entries)
        assert not lex.is_known_word("xyzzy")
        assert len(lex.types_for_word("xyzzy")) == 0

    def test_duplicate_words_multiple_types(self) -> None:
        entries = [
            LexEntry(name="run_v1", lexeme_type="v_-_le", orth=("run",)),
            LexEntry(name="run_n1", lexeme_type="n_-_c_le", orth=("run",)),
        ]
        lex = Lexicon(entries)
        types = lex.types_for_word("run")
        assert "v_-_le" in types
        assert "n_-_c_le" in types

    def test_all_words(self) -> None:
        entries = [
            LexEntry(name="cat_n1", lexeme_type="n_-_c_le", orth=("cat",)),
            LexEntry(name="dog_n1", lexeme_type="n_-_c_le", orth=("dog",)),
        ]
        lex = Lexicon(entries)
        assert "cat" in lex.all_words
        assert "dog" in lex.all_words

    def test_all_lexeme_types(self) -> None:
        entries = [
            LexEntry(name="cat_n1", lexeme_type="n_-_c_le", orth=("cat",)),
            LexEntry(name="run_v1", lexeme_type="v_-_le", orth=("run",)),
        ]
        lex = Lexicon(entries)
        assert "n_-_c_le" in lex.all_lexeme_types
        assert "v_-_le" in lex.all_lexeme_types


class TestLoadLexicon:

    def test_load_from_definitions(self) -> None:
        defs = [
            _lex_def("cat_n1", "n_-_c_le", ["cat"]),
            _lex_def("dog_n1", "n_-_c_le", ["dog"]),
        ]
        lex = load_lexicon(defs)
        assert lex.is_known_word("cat")
        assert lex.is_known_word("dog")

    def test_skip_type_definitions(self) -> None:
        """Type definitions without ORTH should be skipped."""
        defs = [
            _type_def("n_-_c_le", ["noun_le"]),
            _lex_def("cat_n1", "n_-_c_le", ["cat"]),
        ]
        lex = load_lexicon(defs)
        assert lex.is_known_word("cat")
        # The type definition should not produce a lexicon entry
        assert len(list(lex.all_words)) == 1

    def test_load_from_parsed_tdl(self) -> None:
        """Load from actual parsed TDL text."""
        tdl = 'cat_n1 := n_-_c_le & [ ORTH < "cat" > ].'
        definitions = parse_tdl(tokenize_tdl(tdl))
        lex = load_lexicon(definitions)
        assert lex.is_known_word("cat")
        assert "n_-_c_le" in lex.types_for_word("cat")

    def test_entry_without_orth_skipped(self) -> None:
        """Definition with body but no ORTH feature is skipped."""
        defs = [TdlDefinition(
            name="some_type",
            supertypes=["parent"],
            body=TdlFeatStruct({"SYNSEM": TdlType("val")}),
            docstring=None,
            section=None,
            is_instance=False,
            is_addendum=False,
            suffix=None,
        )]
        lex = load_lexicon(defs)
        assert len(list(lex.all_words)) == 0


# ---------------------------------------------------------------------------
# TokenLexemeMap tests
# ---------------------------------------------------------------------------

def _mock_tokenizer(vocab: dict[int, str]):
    """Create a mock tokenizer_decode from a vocab dict."""
    def decode(ids: list[int]) -> str:
        return "".join(vocab.get(i, "") for i in ids)
    return decode


class TestTokenLexemeMap:

    def _make_map(self, vocab: dict[int, str], words: list[str]) -> TokenLexemeMap:
        entries = [LexEntry(name=f"{w}_n1", lexeme_type=f"{w}_le", orth=(w,)) for w in words]
        lex = Lexicon(entries)
        return TokenLexemeMap(lex, _mock_tokenizer(vocab), len(vocab))

    def test_exact_match(self) -> None:
        vocab = {0: "cat", 1: "dog", 2: "xyz"}
        m = self._make_map(vocab, ["cat", "dog"])
        assert m.is_known_token(0)
        assert m.is_known_token(1)
        assert not m.is_known_token(2)

    def test_bpe_leading_space(self) -> None:
        """Token ' cat' (leading space) should match 'cat'."""
        vocab = {0: " cat", 1: " dog"}
        m = self._make_map(vocab, ["cat", "dog"])
        assert m.is_known_token(0)
        assert "cat_le" in m.types_for_token(0)

    def test_subword_fragment_unknown(self) -> None:
        """Token 'ca' (no leading space, not a word) should be unknown."""
        vocab = {0: "ca", 1: "cat"}
        m = self._make_map(vocab, ["cat"])
        assert not m.is_known_token(0)
        assert m.is_known_token(1)

    def test_single_char_exact_only(self) -> None:
        """'a' maps only if 'a' is an exact lexicon entry."""
        vocab = {0: "a", 1: "b"}
        m = self._make_map(vocab, ["a"])
        assert m.is_known_token(0)
        assert not m.is_known_token(1)

    def test_special_tokens_unknown(self) -> None:
        vocab = {0: "<|endoftext|>", 1: "<s>"}
        m = self._make_map(vocab, ["cat"])
        assert not m.is_known_token(0)
        assert not m.is_known_token(1)

    def test_coverage(self) -> None:
        vocab = {0: "cat", 1: "dog", 2: "xyz", 3: "abc"}
        m = self._make_map(vocab, ["cat", "dog"])
        assert m.coverage == 0.5

    def test_known_token_ids(self) -> None:
        vocab = {0: "cat", 1: "xyz", 2: " dog"}
        m = self._make_map(vocab, ["cat", "dog"])
        known = m.known_token_ids
        assert 0 in known  # "cat"
        assert 2 in known  # " dog" -> "dog"
        assert 1 not in known  # "xyz"

    def test_discriminative_power(self) -> None:
        """Only exact whole-word matches are known."""
        vocab = {0: "cat", 1: "ca", 2: "c", 3: " dog", 4: "xyz"}
        m = self._make_map(vocab, ["cat", "dog"])
        assert m.is_known_token(0)  # "cat" exact
        assert not m.is_known_token(1)  # "ca" fragment
        assert not m.is_known_token(2)  # "c" fragment
        assert m.is_known_token(3)  # " dog" -> "dog"
        assert not m.is_known_token(4)  # "xyz" unknown


# ---------------------------------------------------------------------------
# Real ERG tests (slow)
# ---------------------------------------------------------------------------

ERG_DIR = Path.home() / "ontologi" / "tools" / "erg-2025"


@pytest.mark.slow
class TestRealERGLexicon:

    def test_load_erg_lexicon(self) -> None:
        """Load the full ERG lexicon."""
        from tgirl.lingo.tdl_parser import parse_tdl_file

        path = ERG_DIR / "lexicon.tdl"
        if not path.exists():
            pytest.skip(f"ERG lexicon not found: {path}")

        defs = parse_tdl_file(path)
        type_defs = [d for d in defs if isinstance(d, TdlDefinition)]
        lex = load_lexicon(type_defs)
        assert len(list(lex.all_words)) > 25000  # ERG 2025: ~27k unique words from ~44k entries

    def test_lexeme_type_count(self) -> None:
        """Check range of unique lexeme types."""
        from tgirl.lingo.tdl_parser import parse_tdl_file

        path = ERG_DIR / "lexicon.tdl"
        if not path.exists():
            pytest.skip(f"ERG lexicon not found: {path}")

        defs = parse_tdl_file(path)
        type_defs = [d for d in defs if isinstance(d, TdlDefinition)]
        lex = load_lexicon(type_defs)
        n = len(lex.all_lexeme_types)
        assert 800 <= n <= 1500, f"Expected 800-1500 unique lexeme types, got {n}"
