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
        assert set(m.keys()) == {"n_tokens", "repeat_rate", "bigram_novelty"}
