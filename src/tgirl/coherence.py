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

Three signals, chosen for complementary failure-mode coverage:

- **repeat_rate**: fraction of adjacent-token pairs where
  ``tokens[i] == tokens[i-1]``. Catches immediate stutter
  (``"happening happening"``) but misses 2-cycle loops.

- **bigram_novelty**: unique bigrams divided by total bigrams. Catches
  tight cycles and partial loops that slip past repeat_rate. ``1.0`` =
  every bigram new; low values = stuck in a loop.

- **token_entropy**: normalized Shannon entropy over the turn's
  unigram token distribution, ``[0, 1]``. Separates the two breakdown
  attractors that ``bigram_novelty`` alone collapses onto the same
  point:

  - **Mode collapse** (probability mass crushed into a narrow
    attractor): ``H_norm → 0``. Small vocabulary, loop-y.
  - **Semantic fracture** (probability mass pushed off-manifold into
    uniform noise): ``H_norm → 1``. Every token novel, bigrams
    appear diverse, but the stream is structureless.
  - **Coherent language** (healthy Zipfian structure): roughly
    ``0.6–0.8`` in practice.

  When paired with ``repeat_rate`` and ``bigram_novelty``, the triple
  disambiguates loop vs fracture vs signal — a prerequisite for α
  auto-modulation that brakes toward either bound.

All signals return safe defaults for sequences too short to measure
(<2 tokens) so short turns do not trip downstream alarms.
"""

from __future__ import annotations

import math
from collections import Counter


def _normalized_token_entropy(tokens: list[int]) -> float:
    """Shannon entropy of the unigram token distribution, normalized
    by ``log2(n)`` so the result lands in ``[0, 1]`` regardless of
    turn length.

    Returns ``0.0`` for ``n ≤ 1`` (no variance to measure; also avoids
    the ``log2(1) == 0`` denominator).
    """
    n = len(tokens)
    if n < 2:
        return 0.0
    counts = Counter(tokens)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    # log2(n) is the maximum possible entropy when every token is
    # unique (uniform distribution over n distinct values). Dividing
    # by it makes the signal comparable across turn lengths.
    return h / math.log2(n)


def compute_coherence(tokens: list[int]) -> dict[str, float | int]:
    """Compute per-turn coherence signals from generated token IDs.

    Args:
        tokens: Generated token IDs for a single turn. May be empty.

    Returns:
        Dict with keys:
            n_tokens: ``len(tokens)``.
            repeat_rate: fraction of adjacent pairs with
                ``t[i] == t[i-1]`` in ``[0.0, 1.0]``. ``0.0`` for
                sequences with <2 tokens.
            bigram_novelty: ``|set(bigrams)| / len(bigrams)`` in
                ``(0.0, 1.0]``. ``1.0`` for sequences with <2 tokens
                (trivially no repetition to observe).
            token_entropy: normalized Shannon entropy of the unigram
                distribution in ``[0, 1]``. ``0.0`` at mode collapse
                (all-same-token), ``1.0`` at maximum spread (every
                token novel). ``0.0`` for sequences with <2 tokens.
    """
    n = len(tokens)
    if n < 2:
        return {
            "n_tokens": n,
            "repeat_rate": 0.0,
            "bigram_novelty": 1.0,
            "token_entropy": 0.0,
        }

    n_pairs = n - 1
    repeats = sum(1 for i in range(1, n) if tokens[i] == tokens[i - 1])
    bigrams = [(tokens[i - 1], tokens[i]) for i in range(1, n)]
    return {
        "n_tokens": n,
        "repeat_rate": repeats / n_pairs,
        "bigram_novelty": len(set(bigrams)) / n_pairs,
        "token_entropy": _normalized_token_entropy(tokens),
    }
