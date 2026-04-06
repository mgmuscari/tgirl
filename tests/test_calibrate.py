"""Tests for tgirl.calibrate — offline calibration pipeline."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest


class TestEffectiveRank:
    """effective_rank via Shannon entropy of singular values."""

    def test_rank_one_matrix(self) -> None:
        from tgirl.calibrate import effective_rank

        # Rank-1 matrix: outer product of two vectors
        a = mx.array([[1.0, 2.0, 3.0]])
        b = mx.array([[4.0, 5.0, 6.0]])
        M = a.T @ b  # (3, 3) rank 1
        rank = effective_rank(M)
        assert rank == pytest.approx(1.0, abs=0.05)

    def test_identity_full_rank(self) -> None:
        from tgirl.calibrate import effective_rank

        # Identity = all singular values equal → max entropy → rank = d
        d = 10
        M = mx.eye(d)
        rank = effective_rank(M)
        assert rank == pytest.approx(d, abs=0.1)

    def test_known_rank_matrix(self) -> None:
        from tgirl.calibrate import effective_rank

        # Matrix with 3 non-zero singular values in 10D
        # Build from SVD: U @ diag(s) @ Vt
        mx.random.seed(42)
        U = mx.random.normal((20, 10))
        _, _, Vt = mx.linalg.svd(U, stream=mx.cpu)
        s = mx.array([10.0, 8.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        M = Vt.T @ mx.diag(s) @ Vt
        rank = effective_rank(M)
        # Should be close to 3 but not exactly (entropy-based)
        assert 2.0 < rank < 4.0


class TestParticipationRatio:
    """participation_ratio: (Σs)² / Σs² — used for codebook compression."""

    def test_rank_one(self) -> None:
        from tgirl.calibrate import participation_ratio

        s = mx.array([10.0, 0.0, 0.0])
        assert participation_ratio(s) == pytest.approx(1.0, abs=0.01)

    def test_equal_singular_values(self) -> None:
        from tgirl.calibrate import participation_ratio

        s = mx.array([1.0, 1.0, 1.0, 1.0])
        assert participation_ratio(s) == pytest.approx(4.0, abs=0.01)

    def test_typical_behavioral(self) -> None:
        from tgirl.calibrate import participation_ratio

        # Decaying spectrum — effective rank should be between 1 and N
        s = mx.array([10.0, 7.0, 5.0, 3.0, 1.0])
        pr = participation_ratio(s)
        assert 1.0 < pr < 5.0


@pytest.fixture(scope="module")
def qwen_model():
    """Load Qwen3.5-0.8B once for calibration tests."""
    import mlx_lm

    model, tok = mlx_lm.load("Qwen/Qwen3.5-0.8B")
    return model, tok


class TestDiscoverBottleneck:
    """discover_bottleneck finds the minimum effective rank layer."""

    def test_finds_bottleneck_near_14(self, qwen_model) -> None:
        from tgirl.calibrate import discover_bottleneck

        model, tok = qwen_model
        # Need 20+ diverse texts for meaningful rank variation across layers
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning transforms how we process information.",
            "Quantum mechanics describes the behavior of subatomic particles.",
            "The Renaissance period saw a revival of classical learning.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "Democracy requires active participation from all citizens.",
            "The Pythagorean theorem relates the sides of a right triangle.",
            "Ocean currents distribute heat around the globe.",
            "Neural networks loosely mimic the structure of the brain.",
            "The Industrial Revolution transformed manufacturing processes.",
            "DNA carries the genetic instructions for all living organisms.",
            "Financial markets reflect collective expectations about the future.",
            "Climate change affects weather patterns across the planet.",
            "Music theory describes how melodies and harmonies are constructed.",
            "Antibiotics treat bacterial infections but not viral ones.",
            "The theory of relativity changed our understanding of space and time.",
            "Supply and demand determine prices in a market economy.",
            "Volcanic eruptions can affect global temperatures for years.",
            "Computer algorithms solve problems through step-by-step procedures.",
            "The human brain contains approximately eighty-six billion neurons.",
        ]
        bn_layer, ranks = discover_bottleneck(
            model, tok, texts,
            layer_path="language_model.model.layers",
        )
        # Qwen3.5-0.8B shows compression dips around layers 6 and 14
        # with forward-pass extraction (generation-based is more precise)
        assert 4 <= bn_layer <= 18, f"Bottleneck at {bn_layer}, expected 6-14 range"
        # ranks should be a list with one entry per layer
        assert len(ranks) == 24  # Qwen3.5-0.8B has 24 layers
        # The bottleneck should have the minimum rank
        assert ranks[bn_layer] == min(ranks)
