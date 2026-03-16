"""Tests for the TDL parser.

Covers tokenization, parsing of type definitions, feature structures,
addenda, includes, section directives, suffix/letter-set handling,
and real ERG file parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tgirl.lingo.tdl_parser import (
    TdlCoref,
    TdlDefinition,
    TdlDirective,
    TdlFeatStruct,
    TdlInclude,
    TdlList,
    TdlString,
    TdlType,
    parse_tdl,
    parse_tdl_directory,
    parse_tdl_file,
    resolve_include,
    tokenize_tdl,
)

ERG_DIR = Path.home() / "ontologi" / "tools" / "erg-2025"


# === Tokenizer tests ===


class TestTokenizer:
    def test_simple_type_definition(self) -> None:
        tokens = tokenize_tdl('type := super & [ FEAT val ].')
        kinds = [t.kind for t in tokens]
        assert "ident" in kinds
        assert "op" in kinds

    def test_strips_line_comments(self) -> None:
        tokens = tokenize_tdl('; this is a comment\ntype := super.')
        idents = [t for t in tokens if t.kind == "ident"]
        assert len(idents) == 2  # type, super

    def test_strips_block_comments(self) -> None:
        tokens = tokenize_tdl('type #| block comment |# := super.')
        idents = [t for t in tokens if t.kind == "ident"]
        assert len(idents) == 2

    def test_nested_block_comments(self) -> None:
        tokens = tokenize_tdl('type #| outer #| inner |# still |# := super.')
        idents = [t for t in tokens if t.kind == "ident"]
        assert len(idents) == 2

    def test_string_tokens(self) -> None:
        tokens = tokenize_tdl('[ ORTH < "hello" > ]')
        strings = [t for t in tokens if t.kind == "string"]
        assert len(strings) == 1
        assert strings[0].value == "hello"

    def test_docstring_token(self) -> None:
        tokens = tokenize_tdl('"""\nsome doc\n"""')
        docs = [t for t in tokens if t.kind == "docstring"]
        assert len(docs) == 1

    def test_operators(self) -> None:
        tokens = tokenize_tdl(':= :+ & . [ ] < > , #')
        ops = [t for t in tokens if t.kind == "op"]
        assert len(ops) >= 9

    def test_coref_token(self) -> None:
        tokens = tokenize_tdl('#coref')
        corefs = [t for t in tokens if t.kind == "op" and t.value == "#"]
        idents = [t for t in tokens if t.kind == "ident"]
        assert len(corefs) == 1
        assert idents[0].value == "coref"

    def test_line_numbers_tracked(self) -> None:
        tokens = tokenize_tdl('a\nb\nc')
        lines = [t.line for t in tokens if t.kind == "ident"]
        assert lines == [1, 2, 3]


# === Parser tests: basic definitions ===


class TestParserBasic:
    def test_simple_type(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b.'))
        assert len(result) == 1
        defn = result[0]
        assert isinstance(defn, TdlDefinition)
        assert defn.name == "a"
        assert defn.supertypes == ["b"]
        assert defn.is_addendum is False

    def test_conjunction(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b & c.'))
        defn = result[0]
        assert isinstance(defn, TdlDefinition)
        assert defn.supertypes == ["b", "c"]

    def test_addendum(self) -> None:
        result = parse_tdl(tokenize_tdl('a :+ [ FEAT val ].'))
        defn = result[0]
        assert isinstance(defn, TdlDefinition)
        assert defn.is_addendum is True
        assert defn.supertypes == []
        assert defn.body is not None

    def test_feature_structure(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b & [ FEAT val ].'))
        defn = result[0]
        assert isinstance(defn, TdlDefinition)
        assert defn.body is not None

    def test_nested_features(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b & [ F1 [ F2 val ] ].'))
        defn = result[0]
        assert isinstance(defn, TdlDefinition)
        assert defn.body is not None
        # Navigate: body should be a feat struct with F1
        assert isinstance(defn.body, TdlFeatStruct)
        f1_val = defn.body.features.get("F1")
        assert isinstance(f1_val, TdlFeatStruct)
        f2_val = f1_val.features.get("F2")
        assert isinstance(f2_val, TdlType)
        assert f2_val.name == "val"

    def test_dot_paths(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b & [ F1.F2.F3 val ].'))
        defn = result[0]
        assert isinstance(defn.body, TdlFeatStruct)
        # F1.F2.F3 should expand to nested TdlFeatStruct
        f1 = defn.body.features.get("F1")
        assert isinstance(f1, TdlFeatStruct)
        f2 = f1.features.get("F2")
        assert isinstance(f2, TdlFeatStruct)
        f3 = f2.features.get("F3")
        assert isinstance(f3, TdlType)
        assert f3.name == "val"

    def test_coreferences(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b & [ F1 #x, F2 #x ].'))
        defn = result[0]
        assert isinstance(defn.body, TdlFeatStruct)
        f1 = defn.body.features.get("F1")
        assert isinstance(f1, TdlCoref)
        assert f1.name == "x"

    def test_list(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b & [ ORTH < "word" > ].'))
        defn = result[0]
        assert isinstance(defn.body, TdlFeatStruct)
        orth = defn.body.features.get("ORTH")
        assert isinstance(orth, TdlList)
        assert len(orth.elements) == 1
        assert isinstance(orth.elements[0], TdlString)
        assert orth.elements[0].value == "word"
        assert orth.open is False

    def test_open_list(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b & [ LIST < ... > ].'))
        defn = result[0]
        assert isinstance(defn.body, TdlFeatStruct)
        lst = defn.body.features.get("LIST")
        assert isinstance(lst, TdlList)
        assert lst.open is True
        assert len(lst.elements) == 0

    def test_list_with_rest(self) -> None:
        result = parse_tdl(tokenize_tdl('a := b & [ LIST < first, ... > ].'))
        defn = result[0]
        assert isinstance(defn.body, TdlFeatStruct)
        lst = defn.body.features.get("LIST")
        assert isinstance(lst, TdlList)
        assert lst.open is True
        assert len(lst.elements) == 1

    def test_docstring(self) -> None:
        result = parse_tdl(tokenize_tdl(
            'a :=\n"""\nSome doc\n"""\nb & [ FEAT val ].'
        ))
        defn = result[0]
        assert isinstance(defn, TdlDefinition)
        assert defn.docstring is not None
        assert "Some doc" in defn.docstring

    def test_multiline_definition(self) -> None:
        text = """a := b &
  [ FEAT1 val1,
    FEAT2 val2 ]."""
        result = parse_tdl(tokenize_tdl(text))
        assert len(result) == 1
        defn = result[0]
        assert isinstance(defn, TdlDefinition)
        assert defn.name == "a"


# === Include and section tests ===


class TestIncludeAndSections:
    def test_include_directive(self) -> None:
        result = parse_tdl(tokenize_tdl(':include "filename".'))
        assert len(result) == 1
        assert isinstance(result[0], TdlInclude)
        assert result[0].filename == "filename"

    def test_commented_out_include(self) -> None:
        result = parse_tdl(tokenize_tdl(';:include "bridges".'))
        assert len(result) == 0

    def test_begin_end_type_section(self) -> None:
        result = parse_tdl(tokenize_tdl(
            ':begin :type.\na := b.\n:end :type.'
        ))
        defs = [r for r in result if isinstance(r, TdlDefinition)]
        assert len(defs) == 1

    def test_resolve_include_bare_name(self, tmp_path: Path) -> None:
        (tmp_path / "fundamentals.tdl").touch()
        resolved = resolve_include(tmp_path, "fundamentals")
        assert resolved == tmp_path / "fundamentals.tdl"

    def test_resolve_include_explicit_extension(self, tmp_path: Path) -> None:
        (tmp_path / "lfr.tdl").touch()
        resolved = resolve_include(tmp_path, "lfr.tdl")
        assert resolved == tmp_path / "lfr.tdl"

    def test_resolve_include_subdirectory(self, tmp_path: Path) -> None:
        (tmp_path / "tmr").mkdir()
        (tmp_path / "tmr" / "gml.tdl").touch()
        resolved = resolve_include(tmp_path, "tmr/gml")
        assert resolved == tmp_path / "tmr" / "gml.tdl"

    def test_resolve_include_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_include(tmp_path, "nonexistent")

    def test_section_context_propagates_into_includes(
        self, tmp_path: Path
    ) -> None:
        # Create included file
        (tmp_path / "lex.tdl").write_text(
            'cat_n1 := n_-_c_le & [ ORTH < "cat" > ].\n'
        )
        # Create top file with section context around include
        (tmp_path / "top.tdl").write_text(
            ':begin :instance :status lex-entry.\n'
            ':include "lex".\n'
            ':end :instance.\n'
        )
        result = parse_tdl_directory(tmp_path / "top.tdl")
        defs = [r for r in result if isinstance(r, TdlDefinition)]
        assert len(defs) == 1
        assert defs[0].is_instance is True
        assert defs[0].section == "instance:lex-entry"


# === Suffix and letter-set tests ===


class TestSuffixAndLetterSet:
    def test_letter_set_directive(self) -> None:
        result = parse_tdl(tokenize_tdl(
            '%(letter-set (!c bdfgklmnprstz))'
        ))
        directives = [r for r in result if isinstance(r, TdlDirective)]
        assert len(directives) == 1
        assert directives[0].kind == "letter-set"

    def test_suffix_embedded_in_definition(self) -> None:
        text = """n_pl_olr :=
%suffix (!s !ss) (ch ches)
\"""
Plural noun
\"""
n_pl_inflrule &
[ ND-AFF + ]."""
        result = parse_tdl(tokenize_tdl(text))
        defs = [r for r in result if isinstance(r, TdlDefinition)]
        assert len(defs) == 1
        defn = defs[0]
        assert defn.name == "n_pl_olr"
        assert defn.suffix is not None
        assert "!s !ss" in defn.suffix
        assert "ch ches" in defn.suffix
        assert "n_pl_inflrule" in defn.supertypes


# === Real ERG file tests ===


@pytest.mark.slow
class TestRealERG:
    def test_parse_inflr(self) -> None:
        path = ERG_DIR / "inflr.tdl"
        if not path.exists():
            pytest.skip("ERG not available")
        result = parse_tdl_file(path)
        defs = [r for r in result if isinstance(r, TdlDefinition)]
        assert len(defs) >= 20  # inflr.tdl has ~26 definitions
        suffix_defs = [d for d in defs if d.suffix is not None]
        assert len(suffix_defs) >= 15  # Most have %suffix

    def test_parse_fundamentals(self) -> None:
        path = ERG_DIR / "fundamentals.tdl"
        if not path.exists():
            pytest.skip("ERG not available")
        result = parse_tdl_file(path)
        defs = [r for r in result if isinstance(r, TdlDefinition)]
        assert len(defs) > 100

    def test_parse_lextypes(self) -> None:
        path = ERG_DIR / "lextypes.tdl"
        if not path.exists():
            pytest.skip("ERG not available")
        result = parse_tdl_file(path)
        defs = [r for r in result if isinstance(r, TdlDefinition)]
        assert len(defs) > 2900

    def test_parse_letypes(self) -> None:
        path = ERG_DIR / "letypes.tdl"
        if not path.exists():
            pytest.skip("ERG not available")
        result = parse_tdl_file(path)
        defs = [r for r in result if isinstance(r, TdlDefinition)]
        assert len(defs) >= 200  # ~256 leaf types

    def test_parse_lexicon(self) -> None:
        path = ERG_DIR / "lexicon.tdl"
        if not path.exists():
            pytest.skip("ERG not available")
        result = parse_tdl_file(path)
        defs = [r for r in result if isinstance(r, TdlDefinition)]
        assert len(defs) > 43000
