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


class TestBuildScaffold:
    """build_scaffold constructs an orthonormal semantic basis."""

    def test_scaffold_from_synthetic_vectors(self) -> None:
        from tgirl.calibrate import build_scaffold

        # 10 vectors in 64D — should compress
        mx.random.seed(42)
        vecs = {f"dim_{i}": mx.random.normal((64,)) for i in range(10)}
        V_scaffold, diag = build_scaffold(vecs)
        assert V_scaffold.shape[0] == 64
        assert V_scaffold.shape[1] == diag["rank"]
        assert diag["rank"] >= 1
        # Columns should be approximately orthonormal
        gram = V_scaffold.T @ V_scaffold
        mx.eval(gram)
        eye = mx.eye(diag["rank"])
        diff = mx.max(mx.abs(gram - eye))
        mx.eval(diff)
        assert float(diff.item()) < 0.01


class TestBuildCodebook:
    """build_codebook extracts low-rank behavioral basis via SVD."""

    def test_codebook_from_structured_vectors(self) -> None:
        from tgirl.calibrate import build_codebook

        # Create 20 vectors that span a ~5D subspace in 64D
        mx.random.seed(42)
        basis = mx.random.normal((64, 5))  # 5 true dimensions
        coeffs = mx.random.normal((20, 5))
        vecs = {f"dim_{i}": (coeffs[i] @ basis.T) for i in range(20)}

        V_basis, K, diag = build_codebook(vecs)
        assert V_basis.shape[0] == 64
        assert V_basis.shape[1] == K
        # K should be close to 5 for this structured data
        assert 3 <= K <= 8, f"K={K}, expected ~5"
        assert diag["compression_ratio"] < 0.7
        assert len(diag["trait_map"]) == 20

    def test_random_vectors_no_compression(self) -> None:
        from tgirl.calibrate import build_codebook

        # Random vectors should show no compression
        mx.random.seed(99)
        vecs = {f"dim_{i}": mx.random.normal((256,)) for i in range(15)}
        V_basis, K, diag = build_codebook(vecs)
        # K should be close to N (no compression)
        assert K >= 10, f"K={K}, expected ~15 (no compression)"


class TestValidateComplement:
    """validate_complement checks behavioral vectors are in scaffold complement."""

    def test_orthogonal_vectors_are_complement(self) -> None:
        from tgirl.calibrate import validate_complement

        # Scaffold spans first 3 dims, behavioral vectors span last 3 dims
        V_scaffold = mx.zeros((6, 3))
        V_scaffold = V_scaffold.at[:3, :].add(mx.eye(3))
        mx.eval(V_scaffold)

        vecs = {
            "dim_a": mx.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "dim_b": mx.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        }
        fracs = validate_complement(vecs, V_scaffold)
        assert fracs["dim_a"] > 0.99
        assert fracs["dim_b"] > 0.99

    def test_scaffold_vectors_have_low_complement(self) -> None:
        from tgirl.calibrate import validate_complement

        V_scaffold = mx.eye(4)[:, :2]  # scaffold = first 2 dims
        vecs = {"in_scaffold": mx.array([1.0, 0.0, 0.0, 0.0])}
        fracs = validate_complement(vecs, V_scaffold)
        assert fracs["in_scaffold"] < 0.01


class TestExtractBehavioralVectors:
    """Integration: extract behavioral vectors from real model."""

    def test_extract_two_dims(self, qwen_model) -> None:
        """Extract 2 behavioral dims with 2 queries -- smoke test."""
        from tgirl.calibrate import (
            _GenerationHook,
            extract_behavioral_vectors,
        )

        model, tok = qwen_model
        layers = model
        for attr in "language_model.model.layers".split("."):
            layers = getattr(layers, attr)

        hook = _GenerationHook(layers, layer_idx=14)
        hook.install()
        try:
            dims = {
                "helpful": {
                    "system_pos": "You are maximally helpful.",
                    "system_neg": "You are minimally helpful.",
                },
                "terse": {
                    "system_pos": "Use minimal words. Short sentences.",
                    "system_neg": "Be expansive and detailed.",
                },
            }
            queries = [
                "What makes a good leader?",
                "How should I handle a disagreement?",
            ]
            vecs = extract_behavioral_vectors(
                model, tok, hook, dims, queries, max_tok=30,
            )
            assert len(vecs) == 2
            assert vecs["helpful"].shape == (1024,)
            assert vecs["terse"].shape == (1024,)
            # Vectors should be nonzero
            assert float(mx.linalg.norm(vecs["helpful"]).item()) > 0.0
            assert float(mx.linalg.norm(vecs["terse"]).item()) > 0.0
        finally:
            hook.uninstall()


class TestCalibrateFullPipeline:
    """Full pipeline: bottleneck → extract → codebook → save."""

    def test_calibrate_smoke(self, qwen_model, tmp_path) -> None:
        """Run calibrate with 3 dims, 2 queries — validates the full pipeline."""
        from tgirl.calibrate import calibrate
        from tgirl.estradiol import load_estradiol

        model, tok = qwen_model
        output = tmp_path / "qwen.estradiol"

        dims = {
            "helpful": {
                "system_pos": "You are maximally helpful.",
                "system_neg": "You are minimally helpful.",
            },
            "terse": {
                "system_pos": "Use minimal words. Short sentences.",
                "system_neg": "Be expansive and detailed.",
            },
            "formal": {
                "system_pos": "Write in formal academic register.",
                "system_neg": "Write casually like texting a friend.",
            },
        }
        queries = [
            "What makes a good leader?",
            "How should I handle a disagreement?",
        ]

        result = calibrate(
            model, tok,
            model_id="Qwen/Qwen3.5-0.8B",
            layer_path="language_model.model.layers",
            output_path=str(output),
            bottleneck_layer=14,  # skip discovery for speed
            behavioral_dims=dims,
            queries=queries,
            max_tok=30,
        )

        # Verify result structure
        assert result.K >= 1
        assert result.bottleneck_layer == 14
        assert result.model_id == "Qwen/Qwen3.5-0.8B"
        assert result.V_basis.shape == (1024, result.K)
        assert len(result.trait_map) == 3
        assert result.n_dims == 3

        # Verify file was saved and is loadable
        assert output.is_file()
        loaded = load_estradiol(str(output))
        assert loaded.K == result.K
        assert loaded.bottleneck_layer == 14
