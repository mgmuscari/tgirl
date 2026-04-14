"""Per-turn coherence signals for detecting generation breakdown.

Context: when the ESTRADIOL-v2 steering coefficient α pushes the probe
feedback loop past a critical point, the token distribution collapses
into tight repeats ("happening happening"), broken bigrams ("in. The"),
or 2-cycle loops that the naïve `t[i] == t[i-1]` repeat check misses.
This module exposes cheap O(n) signals computed from the generated
token IDs alone — no reference corpus, no framework coupling, no
allocation beyond a local set.

Intended callers:
    tgirl.serve._generate_tokens — at end of each turn
    tgirl.serve._generate_tokens_streaming — at end of each turn
    any benchmark / α-sweep driver that wants to characterize the cliff

Two signals, chosen for complementary failure-mode coverage:

- **repeat_rate**: fraction of adjacent-token pairs where
  ``tokens[i] == tokens[i-1]``. Catches immediate stutter
  (``"happening happening"``) but misses 2-cycle loops.

- **bigram_novelty**: unique bigrams divided by total bigrams. Catches
  tight cycles and partial loops that slip past repeat_rate.
  ``1.0`` = every bigram new; low values = stuck in a loop.

Both return safe defaults (``0.0`` / ``1.0``) for sequences too short
to measure (<2 tokens), so short turns do not trip downstream alarms.
"""

from __future__ import annotations


def compute_coherence(tokens: list[int]) -> dict[str, float | int]:
    """Compute per-turn coherence signals from generated token IDs.

    Args:
        tokens: Generated token IDs for a single turn. May be empty.

    Returns:
        Dict with keys:
            n_tokens: len(tokens)
            repeat_rate: fraction of adjacent pairs with ``t[i] == t[i-1]``,
                in ``[0.0, 1.0]``. ``0.0`` for sequences with <2 tokens.
            bigram_novelty: ``|set(bigrams)| / len(bigrams)`` in
                ``(0.0, 1.0]``. ``1.0`` for sequences with <2 tokens
                (trivially no repetition to observe).
    """
    n = len(tokens)
    if n < 2:
        return {
            "n_tokens": n,
            "repeat_rate": 0.0,
            "bigram_novelty": 1.0,
        }

    n_pairs = n - 1
    repeats = sum(1 for i in range(1, n) if tokens[i] == tokens[i - 1])
    bigrams = [(tokens[i - 1], tokens[i]) for i in range(1, n)]
    return {
        "n_tokens": n,
        "repeat_rate": repeats / n_pairs,
        "bigram_novelty": len(set(bigrams)) / n_pairs,
    }
