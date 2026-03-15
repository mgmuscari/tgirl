"""Tests for tgirl.serve — local inference server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestInferenceContext:
    """Tests for InferenceContext dataclass."""

    def test_inference_context_fields(self) -> None:
        """InferenceContext has all required fields for SamplingSession."""
        from tgirl.serve import InferenceContext

        ctx = InferenceContext(
            registry=MagicMock(),
            forward_fn=lambda ids: None,
            tokenizer_decode=lambda ids: "",
            tokenizer_encode=lambda s: [],
            embeddings=MagicMock(),
            grammar_guide_factory=lambda s: None,
            mlx_grammar_guide_factory=None,
            formatter=MagicMock(),
            backend="torch",
            model_id="test-model",
            stop_token_ids=[0],
        )
        assert ctx.backend == "torch"
        assert ctx.model_id == "test-model"
        assert ctx.stop_token_ids == [0]
        assert ctx.mlx_grammar_guide_factory is None


class TestLoadInferenceContext:
    """Tests for load_inference_context function."""

    def test_load_mlx_backend(self) -> None:
        """MLX backend returns correct InferenceContext."""
        from tgirl.serve import load_inference_context

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="test")
        mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        mock_embed = MagicMock()
        mock_model.model.embed_tokens.weight = mock_embed

        mock_forward_fn = MagicMock()
        mock_grammar_factory = MagicMock()

        with (
            patch("tgirl.serve._try_import_mlx", return_value=True),
            patch(
                "tgirl.serve._load_mlx_model",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch(
                "tgirl.serve._make_mlx_forward",
                return_value=mock_forward_fn,
            ),
            patch(
                "tgirl.serve._make_mlx_grammar_factory",
                return_value=mock_grammar_factory,
            ),
        ):
            ctx = load_inference_context("test-model", backend="mlx")
            assert ctx.backend == "mlx"
            assert ctx.model_id == "test-model"
            assert ctx.forward_fn is mock_forward_fn
            assert ctx.stop_token_ids == [0]

    def test_load_torch_backend(self) -> None:
        """Torch backend returns correct InferenceContext."""
        from tgirl.serve import load_inference_context

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode = MagicMock(return_value="test")
        mock_tokenizer.encode = MagicMock(return_value=[1, 2])

        mock_embeddings = MagicMock()
        mock_model.get_input_embeddings.return_value.weight = mock_embeddings

        mock_forward_fn = MagicMock()
        mock_grammar_factory = MagicMock()

        with (
            patch("tgirl.serve._try_import_mlx", return_value=False),
            patch("tgirl.serve._try_import_torch", return_value=True),
            patch(
                "tgirl.serve._load_torch_model",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch(
                "tgirl.serve._make_torch_forward",
                return_value=mock_forward_fn,
            ),
            patch(
                "tgirl.serve._make_torch_grammar_factory",
                return_value=mock_grammar_factory,
            ),
        ):
            ctx = load_inference_context("test-model", backend="torch")
            assert ctx.backend == "torch"
            assert ctx.model_id == "test-model"
            assert ctx.forward_fn is mock_forward_fn
            assert ctx.stop_token_ids == [2]

    def test_auto_backend_prefers_mlx(self) -> None:
        """Auto backend prefers MLX when available."""
        from tgirl.serve import load_inference_context

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="")
        mock_tokenizer.encode = MagicMock(return_value=[])
        mock_model.model.embed_tokens.weight = MagicMock()

        with (
            patch("tgirl.serve._try_import_mlx", return_value=True),
            patch(
                "tgirl.serve._load_mlx_model",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch(
                "tgirl.serve._make_mlx_forward",
                return_value=MagicMock(),
            ),
            patch(
                "tgirl.serve._make_mlx_grammar_factory",
                return_value=MagicMock(),
            ),
        ):
            ctx = load_inference_context("test-model", backend="auto")
            assert ctx.backend == "mlx"

    def test_mlx_backend_fails_without_mlx(self) -> None:
        """MLX backend fails with clear error when mlx unavailable."""
        from tgirl.serve import load_inference_context

        with (
            patch("tgirl.serve._try_import_mlx", return_value=False),
            pytest.raises(ImportError, match="mlx"),
        ):
            load_inference_context("test-model", backend="mlx")

    def test_torch_backend_fails_without_torch(self) -> None:
        """Torch backend fails with clear error when transformers unavailable."""
        from tgirl.serve import load_inference_context

        with (
            patch("tgirl.serve._try_import_mlx", return_value=False),
            patch("tgirl.serve._try_import_torch", return_value=False),
            pytest.raises(ImportError, match="torch|transformers"),
        ):
            load_inference_context("test-model", backend="torch")
