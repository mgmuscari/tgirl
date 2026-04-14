"""Tests for tgirl.coherence — generation-breakdown signals."""

from __future__ import annotations

import pytest


class TestComputeCoherence:
    """Cheap per-turn coherence signals computed from generated token IDs."""

    def test_empty_sequence_returns_safe_defaults(self) -> None:
        """Zero tokens: nothing to measure, don't trigger alarms."""
        from tgirl.coherence import compute_coherence

        m = compute_coherence([])
        assert m["n_tokens"] == 0
        assert m["repeat_rate"] == 0.0
        assert m["bigram_novelty"] == 1.0

    def test_single_token_returns_safe_defaults(self) -> None:
        """One token: no pairs, no bigrams; safe defaults."""
        from tgirl.coherence import compute_coherence

        m = compute_coherence([42])
        assert m["n_tokens"] == 1
        assert m["repeat_rate"] == 0.0
        assert m["bigram_novelty"] == 1.0

    def test_all_distinct_tokens_are_maximally_coherent(self) -> None:
        """No adjacent repeats + every bigram unique."""
        from tgirl.coherence import compute_coherence

        m = compute_coherence([1, 2, 3, 4])
        assert m["n_tokens"] == 4
        assert m["repeat_rate"] == 0.0
        assert m["bigram_novelty"] == 1.0

    def test_all_same_token_is_maximally_incoherent(self) -> None:
        """Every adjacent pair is a repeat; every bigram collapses to one."""
        from tgirl.coherence import compute_coherence

        m = compute_coherence([5, 5, 5, 5])
        assert m["n_tokens"] == 4
        assert m["repeat_rate"] == 1.0
        # 3 bigrams, all (5,5) → unique set size 1, novelty = 1/3
        assert m["bigram_novelty"] == pytest.approx(1 / 3)

    def test_bigram_loop_has_zero_repeat_rate_but_low_novelty(self) -> None:
        """Classic [1,2,1,2,...] degenerate pattern: no adjacent repeats
        (so repeat_rate misses it), but bigram_novelty catches it.

        This is why both signals matter — repeat_rate alone would say
        "fine" on a tight 2-cycle.
        """
        from tgirl.coherence import compute_coherence

        m = compute_coherence([1, 2, 1, 2, 1, 2])
        assert m["n_tokens"] == 6
        assert m["repeat_rate"] == 0.0
        # 5 bigrams: (1,2),(2,1),(1,2),(2,1),(1,2) → unique = {(1,2),(2,1)} = 2
        assert m["bigram_novelty"] == pytest.approx(2 / 5)

    def test_mixed_sequence_intermediate_signals(self) -> None:
        """Spot-check a realistic-looking pattern with some repetition."""
        from tgirl.coherence import compute_coherence

        # 7 tokens: [10, 10, 20, 30, 20, 30, 40]
        # adjacent pairs: (10,10), (10,20), (20,30), (30,20), (20,30), (30,40)
        # repeats: 1/6
        # bigrams: (10,10), (10,20), (20,30), (30,20), (20,30), (30,40)
        # unique: {(10,10), (10,20), (20,30), (30,20), (30,40)} = 5
        # novelty: 5/6
        m = compute_coherence([10, 10, 20, 30, 20, 30, 40])
        assert m["n_tokens"] == 7
        assert m["repeat_rate"] == pytest.approx(1 / 6)
        assert m["bigram_novelty"] == pytest.approx(5 / 6)

    def test_return_shape_is_stable(self) -> None:
        """Consumers (/v1/steering/status, chat completion metadata) rely
        on a fixed key set. Lock it in.
        """
        from tgirl.coherence import compute_coherence

        m = compute_coherence([1, 2, 3])
        assert set(m.keys()) == {
            "n_tokens",
            "repeat_rate",
            "bigram_novelty",
            "token_entropy",
        }

    def test_token_entropy_zero_for_loop(self) -> None:
        """Mode collapse: all tokens identical → H_norm = 0.
        Distinguishes repetition from fracture when novelty alone
        can't (fracture has high novelty too).
        """
        from tgirl.coherence import compute_coherence

        m = compute_coherence([5, 5, 5, 5, 5, 5])
        assert m["token_entropy"] == 0.0

    def test_token_entropy_one_for_all_unique(self) -> None:
        """Fracture / uniform sampling: every token novel → H_norm = 1.
        Combined with repeat_rate ≈ 0, this is the word-salad signature.
        """
        from tgirl.coherence import compute_coherence

        m = compute_coherence([1, 2, 3, 4, 5, 6, 7, 8])
        assert m["token_entropy"] == pytest.approx(1.0)

    def test_token_entropy_mid_range_for_structured(self) -> None:
        """Healthy language sits in the middle band — Zipfian structure
        gives ~0.6–0.8 H_norm for typical token streams.
        """
        from tgirl.coherence import compute_coherence

        # 10 tokens, some repeats, typical of short coherent text.
        m = compute_coherence([1, 2, 3, 1, 4, 5, 1, 6, 7, 8])
        assert 0.4 < m["token_entropy"] < 1.0

    def test_token_entropy_safe_default_for_short_sequences(self) -> None:
        """n ≤ 1: no variance to measure → H_norm = 0 (not NaN)."""
        from tgirl.coherence import compute_coherence

        assert compute_coherence([]).get("token_entropy") == 0.0
        assert compute_coherence([42]).get("token_entropy") == 0.0

    def test_token_entropy_bigram_loop_is_NOT_fracture(self) -> None:
        """The '[1,2,1,2,...]' degenerate loop has only 2 unique tokens
        out of 6 — low entropy, exposing it as mode collapse even though
        its repeat_rate (0.0) and bigram_novelty (0.4) could be mistaken
        for light disorder. Entropy is the separator.
        """
        from tgirl.coherence import compute_coherence

        m = compute_coherence([1, 2, 1, 2, 1, 2])
        # 2 unique tokens in 6 positions: H = 1 bit, H_max = log2(6) ≈ 2.585
        # H_norm = 1 / 2.585 ≈ 0.387
        assert m["token_entropy"] < 0.5
        # Confirms the signatures disagree on mode-collapse-vs-fracture:
        # bigram_novelty > token_entropy here means "bigrams repeat
        # but the underlying vocabulary is tiny" — exactly what a loop
        # looks like in the feature space.
        assert m["bigram_novelty"] > m["token_entropy"]
