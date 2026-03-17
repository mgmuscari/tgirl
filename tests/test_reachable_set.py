"""Tests for transport reachable set optimization."""

from __future__ import annotations

import re
import time

import pytest


# ---------------------------------------------------------------------------
# Task 1: compute_reachable_set
# ---------------------------------------------------------------------------


class TestComputeReachableSet:
    """compute_reachable_set extracts grammar terminals and scans vocab."""

    @staticmethod
    def _mock_decode(vocab: dict[int, str]):
        def decode(ids: list[int]) -> str:
            return "".join(vocab.get(i, "\ufffd") for i in ids)
        return decode

    def test_returns_frozenset(self) -> None:
        from tgirl.grammar import compute_reachable_set

        grammar = '?start: "(" ")"'
        vocab = {0: "(", 1: ")", 2: "x"}
        result = compute_reachable_set(grammar, self._mock_decode(vocab), 3)
        assert isinstance(result, frozenset)

    def test_string_literal_tokens_included(self) -> None:
        from tgirl.grammar import compute_reachable_set

        grammar = '?start: "(" tool_name ")" \ntool_name: "greet"'
        vocab = {0: "(", 1: ")", 2: "greet", 3: "x", 4: "y"}
        result = compute_reachable_set(grammar, self._mock_decode(vocab), 5)
        assert 0 in result  # "("
        assert 1 in result  # ")"
        assert 2 in result  # "greet"
        assert 3 not in result  # "x" not in grammar
        assert 4 not in result  # "y" not in grammar

    def test_digit_tokens_included(self) -> None:
        from tgirl.grammar import compute_reachable_set

        grammar = '?start: SIGNED_INT\nSIGNED_INT: /[+-]?[0-9]{1,18}/'
        vocab = {0: "0", 1: "5", 2: "-", 3: "+", 4: "abc"}
        result = compute_reachable_set(grammar, self._mock_decode(vocab), 5)
        assert 0 in result  # "0"
        assert 1 in result  # "5"
        assert 2 in result  # "-"
        assert 3 in result  # "+"
        assert 4 not in result  # "abc" doesn't match

    def test_unreachable_tokens_excluded(self) -> None:
        from tgirl.grammar import compute_reachable_set

        grammar = '?start: "a" | "b"'
        vocab = {0: "a", 1: "b", 2: "\u4e2d", 3: "\U0001f600"}
        result = compute_reachable_set(grammar, self._mock_decode(vocab), 4)
        assert 0 in result
        assert 1 in result
        assert 2 not in result
        assert 3 not in result

    def test_empty_grammar_returns_empty_set(self) -> None:
        from tgirl.grammar import compute_reachable_set

        result = compute_reachable_set("", self._mock_decode({}), 0)
        assert result == frozenset()

    def test_deterministic(self) -> None:
        from tgirl.grammar import compute_reachable_set

        grammar = '?start: "x" | "y"'
        vocab = {0: "x", 1: "y", 2: "z"}
        decode = self._mock_decode(vocab)
        r1 = compute_reachable_set(grammar, decode, 3)
        r2 = compute_reachable_set(grammar, decode, 3)
        assert r1 == r2

    def test_escaped_string_tokens_included(self) -> None:
        from tgirl.grammar import compute_reachable_set

        grammar = '?start: ESCAPED_STRING\n%import common.ESCAPED_STRING'
        vocab = {0: '"', 1: "a", 2: "\\", 3: " ", 4: "\u4e2d"}
        result = compute_reachable_set(grammar, self._mock_decode(vocab), 5)
        assert 0 in result  # quote char
        assert 1 in result  # printable ASCII
        assert 2 in result  # backslash (escape)
        assert 3 in result  # space (printable)

    def test_subword_tokens_with_multiple_chars(self) -> None:
        """Multi-char tokens matching a terminal should be included."""
        from tgirl.grammar import compute_reachable_set

        grammar = '?start: "hello"'
        vocab = {0: "hello", 1: "hel", 2: "lo", 3: "xyz"}
        result = compute_reachable_set(grammar, self._mock_decode(vocab), 4)
        assert 0 in result  # exact match
        # Subword parts may or may not match depending on implementation


# ---------------------------------------------------------------------------
# Task 2: GrammarOutput.reachable_tokens
# ---------------------------------------------------------------------------


class TestGrammarOutputReachableTokens:

    def test_grammar_output_accepts_reachable_tokens(self) -> None:
        from tgirl.grammar import GrammarOutput

        go = GrammarOutput(
            text="test",
            productions=(),
            snapshot_hash="abc",
            tool_quotas={},
            cost_remaining=None,
            reachable_tokens=frozenset({1, 2, 3}),
        )
        assert go.reachable_tokens == frozenset({1, 2, 3})

    def test_grammar_output_default_none(self) -> None:
        from tgirl.grammar import GrammarOutput

        go = GrammarOutput(
            text="test",
            productions=(),
            snapshot_hash="abc",
            tool_quotas={},
            cost_remaining=None,
        )
        assert go.reachable_tokens is None

    def test_grammar_output_frozen_reachable_tokens(self) -> None:
        from tgirl.grammar import GrammarOutput

        go = GrammarOutput(
            text="test",
            productions=(),
            snapshot_hash="abc",
            tool_quotas={},
            cost_remaining=None,
            reachable_tokens=frozenset({1}),
        )
        assert isinstance(go.reachable_tokens, frozenset)
