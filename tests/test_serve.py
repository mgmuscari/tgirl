"""Tests for tgirl.serve — local inference server."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tgirl.registry import ToolRegistry


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


def _make_mock_ctx(
    tools: ToolRegistry | None = None,
) -> Any:
    """Create a mock InferenceContext for server testing."""
    from tgirl.serve import InferenceContext

    registry = tools or ToolRegistry()
    return InferenceContext(
        registry=registry,
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


class TestCreateApp:
    """Tests for the FastAPI app factory."""

    def test_health_endpoint(self) -> None:
        """/health returns model info and tool count."""
        from tgirl.serve import create_app

        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        ctx = _make_mock_ctx(tools=registry)
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "test-model"
        assert data["tools"] == 1
        assert data["backend"] == "torch"

    def test_generate_returns_correct_structure(self) -> None:
        """/generate returns correct response structure."""
        from tgirl.serve import create_app

        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        ctx = _make_mock_ctx(tools=registry)
        app = create_app(ctx)

        # Mock run_chat to return a SamplingResult-like object
        mock_result = MagicMock()
        mock_result.output_text = "Hello world"
        mock_result.tool_calls = []
        mock_result.total_tokens = 42
        mock_result.total_cycles = 1
        mock_result.wall_time_ms = 100.5
        mock_result.quotas_consumed = {}

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch(
            "tgirl.serve._run_session_chat",
            return_value=mock_result,
        ):
            resp = client.post(
                "/generate",
                json={"intent": "Add 2 and 3"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["output"] == "Hello world"
            assert data["total_tokens"] == 42
            assert data["total_cycles"] == 1
            assert data["wall_time_ms"] == 100.5
            assert data["tool_calls"] == []
            assert data["error"] is None

    def test_generate_missing_intent(self) -> None:
        """/generate without intent returns 422."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/generate", json={})
        assert resp.status_code == 422

    def test_generate_error_handling(self) -> None:
        """/generate handles model errors gracefully."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch(
            "tgirl.serve._run_session_chat",
            side_effect=RuntimeError("Model failed"),
        ):
            resp = client.post(
                "/generate",
                json={"intent": "test"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["error"] is not None
            assert "Model failed" in data["error"]


class TestInfoEndpoints:
    """Tests for /tools, /grammar, /grammar/preview, /telemetry."""

    def test_tools_endpoint(self) -> None:
        """/tools lists all registered tools with correct info."""
        from tgirl.serve import create_app

        registry = ToolRegistry()

        @registry.tool(description="Add numbers")
        def add(a: int, b: int) -> int:
            return a + b

        @registry.tool(description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello {name}"

        ctx = _make_mock_ctx(tools=registry)
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        names = {t["name"] for t in data}
        assert names == {"add", "greet"}
        add_tool = next(t for t in data if t["name"] == "add")
        assert add_tool["description"] == "Add numbers"

    def test_grammar_endpoint(self) -> None:
        """/grammar returns valid grammar text and hash."""
        from tgirl.serve import create_app

        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        ctx = _make_mock_ctx(tools=registry)
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/grammar")
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "hash" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0

    def test_grammar_preview_endpoint(self) -> None:
        """/grammar/preview with tool restriction produces filtered grammar."""
        from tgirl.serve import create_app

        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        @registry.tool()
        def sub(a: int, b: int) -> int:
            return a - b

        ctx = _make_mock_ctx(tools=registry)
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post(
            "/grammar/preview",
            json={"restrict_tools": ["add"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "add" in data["text"]

    def test_telemetry_endpoint(self) -> None:
        """/telemetry returns empty list when no sessions have run."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/telemetry")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_telemetry_respects_limit(self) -> None:
        """/telemetry respects limit parameter."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/telemetry?limit=5")
        assert resp.status_code == 200


class TestWebSocket:
    """Tests for /stream WebSocket endpoint."""

    def test_websocket_connection_and_message(self) -> None:
        """WebSocket accepts connection and returns result."""
        from tgirl.serve import create_app

        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        ctx = _make_mock_ctx(tools=registry)
        app = create_app(ctx)

        mock_result = MagicMock()
        mock_result.output_text = "Result: 5"
        mock_result.tool_calls = []
        mock_result.total_tokens = 10
        mock_result.total_cycles = 0
        mock_result.wall_time_ms = 50.0
        mock_result.quotas_consumed = {}

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with (
            patch(
                "tgirl.serve._run_session_chat",
                return_value=mock_result,
            ),
            client.websocket_connect("/stream") as ws,
        ):
            ws.send_json({"intent": "Add 2 and 3"})
            data = ws.receive_json()
            assert data["type"] == "result"
            assert data["output"] == "Result: 5"
            assert data["total_tokens"] == 10

    def test_websocket_error_handling(self) -> None:
        """WebSocket handles errors gracefully."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        app = create_app(ctx)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with (
            patch(
                "tgirl.serve._run_session_chat",
                side_effect=RuntimeError("Inference failed"),
            ),
            client.websocket_connect("/stream") as ws,
        ):
            ws.send_json({"intent": "test"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Inference failed" in data["error"]
