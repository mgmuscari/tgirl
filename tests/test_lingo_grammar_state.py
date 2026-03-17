"""Tests for tgirl.lingo.grammar_state — CoherenceTracker and LingoGrammarState."""

from __future__ import annotations

from pathlib import Path

import pytest

from tgirl.lingo.lexicon import LexEntry, Lexicon, TokenLexemeMap
from tgirl.lingo.grammar_state import CoherenceTracker, LingoGrammarState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_tokenizer(vocab: dict[int, str]):
    def decode(ids: list[int]) -> str:
        return "".join(vocab.get(i, "") for i in ids)
    return decode


def _make_map(vocab: dict[int, str], words: list[str]) -> TokenLexemeMap:
    entries = [LexEntry(name=f"{w}_n1", lexeme_type=f"{w}_le", orth=(w,)) for w in words]
    lex = Lexicon(entries)
    return TokenLexemeMap(lex, _mock_tokenizer(vocab), len(vocab))


# ---------------------------------------------------------------------------
# CoherenceTracker tests
# ---------------------------------------------------------------------------

class TestCoherenceTracker:

    def test_empty_tracker(self) -> None:
        vocab = {0: "cat", 1: "xyz"}
        m = _make_map(vocab, ["cat"])
        tracker = CoherenceTracker(m, window_size=4)
        assert tracker.coherence == 0.0

    def test_all_known(self) -> None:
        vocab = {0: "cat", 1: "dog"}
        m = _make_map(vocab, ["cat", "dog"])
        tracker = CoherenceTracker(m, window_size=4)
        for _ in range(4):
            tracker.advance(0)
        assert tracker.coherence == 1.0

    def test_all_unknown(self) -> None:
        vocab = {0: "cat", 1: "xyz"}
        m = _make_map(vocab, ["cat"])
        tracker = CoherenceTracker(m, window_size=4)
        for _ in range(4):
            tracker.advance(1)
        assert tracker.coherence == 0.0

    def test_mixed(self) -> None:
        vocab = {0: "cat", 1: "xyz"}
        m = _make_map(vocab, ["cat"])
        tracker = CoherenceTracker(m, window_size=4)
        tracker.advance(0)  # known
        tracker.advance(1)  # unknown
        tracker.advance(0)  # known
        tracker.advance(1)  # unknown
        assert tracker.coherence == 0.5

    def test_window_eviction(self) -> None:
        vocab = {0: "cat", 1: "xyz"}
        m = _make_map(vocab, ["cat"])
        tracker = CoherenceTracker(m, window_size=2)
        tracker.advance(0)  # known
        tracker.advance(0)  # known -> coherence = 1.0
        assert tracker.coherence == 1.0
        tracker.advance(1)  # unknown, evicts first known
        assert tracker.coherence == 0.5
        tracker.advance(1)  # unknown, evicts second known
        assert tracker.coherence == 0.0

    def test_reset(self) -> None:
        vocab = {0: "cat"}
        m = _make_map(vocab, ["cat"])
        tracker = CoherenceTracker(m, window_size=4)
        tracker.advance(0)
        assert tracker.coherence > 0.0
        tracker.reset()
        assert tracker.coherence == 0.0

    def test_window_size_one(self) -> None:
        vocab = {0: "cat", 1: "xyz"}
        m = _make_map(vocab, ["cat"])
        tracker = CoherenceTracker(m, window_size=1)
        tracker.advance(0)
        assert tracker.coherence == 1.0
        tracker.advance(1)
        assert tracker.coherence == 0.0


# ---------------------------------------------------------------------------
# LingoGrammarState tests
# ---------------------------------------------------------------------------

class TestLingoGrammarState:

    def _make_state(self) -> LingoGrammarState:
        vocab = {0: "cat", 1: "dog", 2: "xyz", 3: "abc"}
        m = _make_map(vocab, ["cat", "dog"])
        tracker = CoherenceTracker(m, window_size=4)
        return LingoGrammarState(m, tracker, len(vocab))

    def test_valid_mask_all_true(self) -> None:
        """V1: no hard masking, all tokens valid."""
        import mlx.core as mx
        state = self._make_state()
        mask = state.get_valid_mask_mx(4)
        assert mask.shape == (4,)
        assert mask.dtype == mx.bool_
        assert mask.tolist() == [True, True, True, True]

    def test_is_accepting_true(self) -> None:
        """Freeform grammar always accepting — prevents EOS masking."""
        state = self._make_state()
        assert state.is_accepting() is True

    def test_advance_updates_coherence(self) -> None:
        state = self._make_state()
        state.advance(0)  # "cat" -> known
        assert state.coherence_score() > 0.0

    def test_coherence_score_range(self) -> None:
        state = self._make_state()
        assert 0.0 <= state.coherence_score() <= 1.0
        state.advance(0)
        assert 0.0 <= state.coherence_score() <= 1.0

    def test_zero_imports_from_tgirl_core(self) -> None:
        """grammar_state.py must not import from tgirl core modules."""
        import ast
        import inspect
        import tgirl.lingo.grammar_state as gs
        source = inspect.getsource(gs)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("tgirl.sample"), \
                        f"Forbidden import: {alias.name}"
                    assert not alias.name.startswith("tgirl.grammar"), \
                        f"Forbidden import: {alias.name}"
                    assert not alias.name.startswith("tgirl.transport"), \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("tgirl.sample"), \
                        f"Forbidden import from: {node.module}"
                    assert not node.module.startswith("tgirl.grammar"), \
                        f"Forbidden import from: {node.module}"
                    assert not node.module.startswith("tgirl.transport"), \
                        f"Forbidden import from: {node.module}"
