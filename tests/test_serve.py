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
        # mlx_lm returns a wrapper tokenizer; _build_mlx_context
        # extracts ._tokenizer for the fast HF tokenizer
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.eos_token_id = 0
        mock_hf_tokenizer.decode = MagicMock(return_value="test")
        mock_hf_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer

        mock_embed = MagicMock()
        mock_model.language_model.model.embed_tokens.weight.astype.return_value = mock_embed
        mock_embed.shape = (100,)

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


class TestCli:
    """Tests for CLI entrypoint."""

    def test_serve_help(self) -> None:
        """CLI serve command has --help."""
        from click.testing import CliRunner

        from tgirl.cli import serve

        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--port" in result.output
        assert "--backend" in result.output

    def test_serve_default_port(self) -> None:
        """CLI serve command defaults to port 8420."""
        from click.testing import CliRunner

        from tgirl.cli import serve

        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert "8420" in result.output

    def test_serve_has_tools_option(self) -> None:
        """CLI serve command has --tools option."""
        from click.testing import CliRunner

        from tgirl.cli import serve

        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "--tools" in result.output

    def test_serve_loads_tools_from_module(self, tmp_path: Any) -> None:
        """CLI --tools loads a Python module that defines a register() function."""
        from click.testing import CliRunner

        from tgirl.cli import load_tools_from_path, serve

        # Create a temp tool module with a register() function
        tool_file = tmp_path / "my_tools.py"
        tool_file.write_text(
            "def register(registry):\n"
            "    @registry.tool()\n"
            "    def my_add(a: int, b: int) -> int:\n"
            "        return a + b\n"
        )

        registry = ToolRegistry()
        load_tools_from_path(str(tool_file), registry)
        assert len(registry) == 1
        snap = registry.snapshot()
        assert any(t.name == "my_add" for t in snap.tools)

    def test_serve_loads_tools_registry_var(self, tmp_path: Any) -> None:
        """CLI --tools loads a Python module that defines a module-level registry."""
        from tgirl.cli import load_tools_from_path

        tool_file = tmp_path / "tools_with_registry.py"
        tool_file.write_text(
            "from tgirl.registry import ToolRegistry\n"
            "registry = ToolRegistry()\n"
            "@registry.tool()\n"
            "def my_sub(a: int, b: int) -> int:\n"
            "    return a - b\n"
        )

        target_registry = ToolRegistry()
        load_tools_from_path(str(tool_file), target_registry)
        assert len(target_registry) == 1
        snap = target_registry.snapshot()
        assert any(t.name == "my_sub" for t in snap.tools)

    def test_serve_loads_tools_from_directory(self, tmp_path: Any) -> None:
        """CLI --tools loads all .py files from a directory."""
        from tgirl.cli import load_tools_from_path

        # Create two tool modules in a directory
        (tmp_path / "tool_a.py").write_text(
            "def register(registry):\n"
            "    @registry.tool()\n"
            "    def tool_a(x: int) -> int:\n"
            "        return x\n"
        )
        (tmp_path / "tool_b.py").write_text(
            "def register(registry):\n"
            "    @registry.tool()\n"
            "    def tool_b(x: int) -> int:\n"
            "        return x * 2\n"
        )

        registry = ToolRegistry()
        load_tools_from_path(str(tmp_path), registry)
        assert len(registry) == 2


class TestGenerateRequestParams:
    """Tests for wiring GenerateRequest params to SamplingSession."""

    def test_restrict_tools_passed_to_session(self) -> None:
        """restrict_tools filters the registry snapshot passed to session."""
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

        mock_result = MagicMock()
        mock_result.output_text = "ok"
        mock_result.tool_calls = []
        mock_result.total_tokens = 1
        mock_result.total_cycles = 0
        mock_result.wall_time_ms = 1.0
        mock_result.quotas_consumed = {}

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch(
            "tgirl.serve._run_session_chat",
            return_value=mock_result,
        ) as mock_run:
            resp = client.post(
                "/generate",
                json={"intent": "test", "restrict_tools": ["add"]},
            )
            assert resp.status_code == 200
            # Check that _run_session_chat was called with a registry
            # that only contains the restricted tools
            call_args = mock_run.call_args
            called_ctx = call_args[0][0] if call_args[0] else call_args[1].get("ctx")
            # The ctx should have a filtered registry
            snap = called_ctx.registry.snapshot()
            tool_names = {t.name for t in snap.tools}
            assert tool_names == {"add"}

    def test_ot_epsilon_passed_to_transport_config(self) -> None:
        """ot_epsilon creates a TransportConfig with custom epsilon."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        app = create_app(ctx)

        mock_result = MagicMock()
        mock_result.output_text = "ok"
        mock_result.tool_calls = []
        mock_result.total_tokens = 1
        mock_result.total_cycles = 0
        mock_result.wall_time_ms = 1.0
        mock_result.quotas_consumed = {}

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch(
            "tgirl.serve._run_session_chat",
            return_value=mock_result,
        ) as mock_run:
            resp = client.post(
                "/generate",
                json={"intent": "test", "ot_epsilon": 0.05},
            )
            assert resp.status_code == 200
            call_args = mock_run.call_args
            # transport_config is the 4th positional arg (index 3)
            transport_config = call_args[0][3]
            assert transport_config.epsilon == 0.05

    def test_base_temperature_passed_to_hooks(self) -> None:
        """base_temperature creates a GrammarTemperatureHook."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        app = create_app(ctx)

        mock_result = MagicMock()
        mock_result.output_text = "ok"
        mock_result.tool_calls = []
        mock_result.total_tokens = 1
        mock_result.total_cycles = 0
        mock_result.wall_time_ms = 1.0
        mock_result.quotas_consumed = {}

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch(
            "tgirl.serve._run_session_chat",
            return_value=mock_result,
        ) as mock_run:
            resp = client.post(
                "/generate",
                json={"intent": "test", "base_temperature": 0.5},
            )
            assert resp.status_code == 200
            call_args = mock_run.call_args
            # hooks is the 5th positional arg (index 4)
            hooks_arg = call_args[0][4]
            assert hooks_arg is not None
            assert len(hooks_arg) >= 1
            # Find the GrammarTemperatureHook
            from tgirl.sample import GrammarTemperatureHook
            temp_hooks = [
                h for h in hooks_arg
                if isinstance(h, GrammarTemperatureHook)
            ]
            assert len(temp_hooks) == 1
            assert temp_hooks[0].base_temperature == 0.5

    def test_scopes_passed_to_snapshot(self) -> None:
        """scopes filters registry snapshot via scopes parameter."""
        from tgirl.serve import create_app

        registry = ToolRegistry()

        @registry.tool(scope="admin")
        def admin_tool(x: int) -> int:
            return x

        @registry.tool()
        def public_tool(x: int) -> int:
            return x

        ctx = _make_mock_ctx(tools=registry)
        app = create_app(ctx)

        mock_result = MagicMock()
        mock_result.output_text = "ok"
        mock_result.tool_calls = []
        mock_result.total_tokens = 1
        mock_result.total_cycles = 0
        mock_result.wall_time_ms = 1.0
        mock_result.quotas_consumed = {}

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch(
            "tgirl.serve._run_session_chat",
            return_value=mock_result,
        ) as mock_run:
            # Request with no admin scope — admin_tool should be excluded
            resp = client.post(
                "/generate",
                json={"intent": "test", "scopes": ["user"]},
            )
            assert resp.status_code == 200
            call_args = mock_run.call_args
            called_ctx = call_args[0][0]
            snap = called_ctx.registry.snapshot()
            tool_names = {t.name for t in snap.tools}
            # admin_tool requires "admin" scope, only "user" was given
            assert "admin_tool" not in tool_names
            # public_tool has no scope restriction
            assert "public_tool" in tool_names

    def test_max_cost_passed_to_session_config(self) -> None:
        """max_cost sets session_cost_budget in SessionConfig."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        app = create_app(ctx)

        mock_result = MagicMock()
        mock_result.output_text = "ok"
        mock_result.tool_calls = []
        mock_result.total_tokens = 1
        mock_result.total_cycles = 0
        mock_result.wall_time_ms = 1.0
        mock_result.quotas_consumed = {}

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch(
            "tgirl.serve._run_session_chat",
            return_value=mock_result,
        ) as mock_run:
            resp = client.post(
                "/generate",
                json={"intent": "test", "max_cost": 10.0},
            )
            assert resp.status_code == 200
            call_args = mock_run.call_args
            # session_config is the 3rd positional arg (index 2)
            session_config = call_args[0][2]
            assert session_config.session_cost_budget == 10.0

    def test_none_params_use_defaults(self) -> None:
        """When request params are None, defaults from create_app are used."""
        from tgirl.serve import create_app
        from tgirl.transport import TransportConfig
        from tgirl.types import SessionConfig

        default_session_config = SessionConfig(session_cost_budget=99.0)
        default_transport = TransportConfig(epsilon=0.2)

        ctx = _make_mock_ctx()
        app = create_app(
            ctx,
            session_config=default_session_config,
            transport_config=default_transport,
        )

        mock_result = MagicMock()
        mock_result.output_text = "ok"
        mock_result.tool_calls = []
        mock_result.total_tokens = 1
        mock_result.total_cycles = 0
        mock_result.wall_time_ms = 1.0
        mock_result.quotas_consumed = {}

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch(
            "tgirl.serve._run_session_chat",
            return_value=mock_result,
        ) as mock_run:
            resp = client.post(
                "/generate",
                json={"intent": "test"},
            )
            assert resp.status_code == 200
            call_args = mock_run.call_args
            # Should use original ctx (not filtered)
            called_ctx = call_args[0][0]
            assert called_ctx is ctx
            # Should use default session_config
            session_cfg = call_args[0][2]
            assert session_cfg is default_session_config
            # Should use default transport_config
            transport_cfg = call_args[0][3]
            assert transport_cfg is default_transport


# --- OpenAI-compatible endpoint tests ---


class TestOpenAIModels:
    """GET /v1/models returns model list."""

    def test_models_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from tgirl.serve import InferenceContext, create_app

        ctx = InferenceContext(
            registry=MagicMock(),
            forward_fn=lambda ids: None,
            tokenizer_decode=lambda ids: "",
            tokenizer_encode=lambda s: [],
            embeddings=MagicMock(),
            grammar_guide_factory=lambda s: None,
            mlx_grammar_guide_factory=None,
            formatter=MagicMock(),
            backend="mlx",
            model_id="Qwen/Qwen3.5-0.8B",
            stop_token_ids=[0],
        )
        app = create_app(ctx)
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "Qwen/Qwen3.5-0.8B"
        assert data["data"][0]["object"] == "model"


class TestOpenAIChatCompletions:
    """POST /v1/chat/completions — non-streaming."""

    @pytest.fixture()
    def mock_app(self):
        """App with a mock forward_fn that returns predictable logits."""
        import mlx.core as mx
        from fastapi.testclient import TestClient
        from tgirl.serve import InferenceContext, create_app

        vocab_size = 100
        call_count = [0]

        def fake_forward(token_ids):
            call_count[0] += 1
            logits = mx.zeros((vocab_size,))
            # Always pick token 42, then EOS (token 0) on second call
            if call_count[0] <= 3:
                logits = logits.at[42].add(mx.array(100.0))
            else:
                logits = logits.at[0].add(mx.array(100.0))  # EOS
            return logits

        ctx = InferenceContext(
            registry=MagicMock(),
            forward_fn=fake_forward,
            tokenizer_decode=lambda ids: "hello world",
            tokenizer_encode=lambda s: [1, 2, 3],
            embeddings=MagicMock(),
            grammar_guide_factory=lambda s: None,
            mlx_grammar_guide_factory=None,
            formatter=MagicMock(),
            backend="mlx",
            model_id="test-model",
            stop_token_ids=[0],
        )
        app = create_app(ctx)
        return TestClient(app), call_count

    def test_basic_completion(self, mock_app) -> None:
        client, _ = mock_app
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "hello world"
        assert data["choices"][0]["finish_reason"] in ("stop", "length")
        assert data["usage"]["prompt_tokens"] == 3
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0

    def test_missing_model_returns_422(self, mock_app) -> None:
        client, _ = mock_app
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 422

    def test_missing_messages_returns_422(self, mock_app) -> None:
        client, _ = mock_app
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
        })
        assert resp.status_code == 422

    def test_reasoning_content_split(self) -> None:
        """When think_end_token_id is set, output splits into reasoning + content."""
        import mlx.core as mx
        from fastapi.testclient import TestClient
        from tgirl.serve import InferenceContext, create_app

        vocab_size = 100
        THINK_END = 99  # </think> token
        call_count = [0]

        def fake_forward(token_ids):
            call_count[0] += 1
            logits = mx.full((vocab_size,), -100.0)
            if call_count[0] <= 2:
                # First 2 tokens: reasoning (tokens 10, 11)
                logits = logits.at[10 + call_count[0] - 1].add(mx.array(200.0))
            elif call_count[0] == 3:
                # Token 3: </think>
                logits = logits.at[THINK_END].add(mx.array(200.0))
            elif call_count[0] <= 5:
                # Tokens 4-5: response (tokens 20, 21)
                logits = logits.at[20 + call_count[0] - 4].add(mx.array(200.0))
            else:
                # EOS
                logits = logits.at[0].add(mx.array(200.0))
            return logits

        decoded = {
            10: "think", 11: "ing",
            20: "ans", 21: "wer",
        }

        ctx = InferenceContext(
            registry=MagicMock(),
            forward_fn=fake_forward,
            tokenizer_decode=lambda ids: "".join(decoded.get(i, "") for i in ids),
            tokenizer_encode=lambda s: [1, 2, 3],
            embeddings=MagicMock(),
            grammar_guide_factory=lambda s: None,
            mlx_grammar_guide_factory=None,
            formatter=MagicMock(),
            backend="mlx",
            model_id="test-model",
            stop_token_ids=[0],
            think_end_token_id=THINK_END,
        )
        # Make formatter.format_messages work
        ctx.formatter.format_messages = MagicMock(return_value="prompt")

        app = create_app(ctx)
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 20,
            "temperature": 0.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        msg = data["choices"][0]["message"]
        assert msg["content"] == "answer"
        assert msg["reasoning_content"] == "thinking"


class TestProbePersistence:
    """Tests for CLI-driven probe vector persistence across server restarts."""

    def test_create_app_loads_probe_at_startup(self, tmp_path: Any) -> None:
        """When probe_load_path is set, lifespan startup populates the cache."""
        import numpy as np
        from fastapi.testclient import TestClient

        from tgirl.serve import create_app

        probe_path = tmp_path / "session_probe.npy"
        np.save(probe_path, np.arange(8, dtype=np.float32))

        ctx = _make_mock_ctx()
        app = create_app(ctx, probe_load_path=str(probe_path))

        with TestClient(app) as client:
            resp = client.get("/v1/steering/status")
            assert resp.status_code == 200
            assert resp.json()["probe_cached"] is True

    def test_create_app_saves_probe_at_shutdown(self, tmp_path: Any) -> None:
        """When probe_save_path is set, lifespan shutdown writes the cache."""
        import numpy as np
        from fastapi.testclient import TestClient

        from tgirl.serve import create_app

        # Prime the cache by loading, so shutdown has something to save.
        load_path = tmp_path / "in.npy"
        save_path = tmp_path / "out.npy"
        expected = np.arange(8, dtype=np.float32)
        np.save(load_path, expected)

        ctx = _make_mock_ctx()
        app = create_app(
            ctx,
            probe_load_path=str(load_path),
            probe_save_path=str(save_path),
        )

        with TestClient(app):
            pass  # exit triggers lifespan shutdown

        assert save_path.exists(), "shutdown should have written the probe"
        assert np.allclose(np.load(save_path), expected)

    def test_create_app_shutdown_save_skips_when_cache_empty(
        self, tmp_path: Any
    ) -> None:
        """Shutdown save is a no-op when no probe has been cached."""
        from fastapi.testclient import TestClient

        from tgirl.serve import create_app

        save_path = tmp_path / "out.npy"
        ctx = _make_mock_ctx()
        app = create_app(ctx, probe_save_path=str(save_path))

        with TestClient(app):
            pass

        assert not save_path.exists(), (
            "shutdown should not write when cache is empty"
        )

    def test_cli_exposes_probe_flags(self) -> None:
        """CLI serve --help lists --probe-load and --probe-save-on-shutdown."""
        from click.testing import CliRunner

        from tgirl.cli import serve

        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "--probe-load" in result.output
        assert "--probe-save-on-shutdown" in result.output

    def test_create_app_autosaves_probe_on_interval(
        self, tmp_path: Any
    ) -> None:
        """A non-zero autosave interval writes the probe periodically."""
        import time

        import numpy as np
        from fastapi.testclient import TestClient

        from tgirl.serve import create_app

        load_path = tmp_path / "in.npy"
        save_path = tmp_path / "out.npy"
        expected = np.arange(8, dtype=np.float32)
        np.save(load_path, expected)

        ctx = _make_mock_ctx()
        app = create_app(
            ctx,
            probe_load_path=str(load_path),
            probe_save_path=str(save_path),
            probe_autosave_interval_s=0.05,
        )

        with TestClient(app):
            # Wait long enough for at least one autosave tick to fire
            # before the lifespan-shutdown final save would also run.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not save_path.exists():
                time.sleep(0.02)
            first_write_observed = save_path.exists()

        assert first_write_observed, (
            "autosave should have written within the lifespan, "
            "not just at shutdown"
        )
        assert np.allclose(np.load(save_path), expected)

    def test_create_app_autosave_requires_save_path(self) -> None:
        """Autosave interval without a save path is a configuration error."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        with pytest.raises(ValueError, match="probe_save_path"):
            create_app(ctx, probe_autosave_interval_s=0.1)

    @pytest.mark.parametrize("bad_interval", [0, 0.0, -1.0, -0.001])
    def test_create_app_rejects_non_positive_autosave_interval(
        self, tmp_path: Any, bad_interval: float
    ) -> None:
        """Non-positive intervals turn the autosave loop into a tight
        write storm via asyncio.sleep(<=0). Reject at config time."""
        from tgirl.serve import create_app

        ctx = _make_mock_ctx()
        with pytest.raises(ValueError, match="must be positive"):
            create_app(
                ctx,
                probe_save_path=str(tmp_path / "out.npy"),
                probe_autosave_interval_s=bad_interval,
            )

    @pytest.mark.parametrize("bad_interval", ["0", "0.0", "-1", "-0.5"])
    def test_cli_rejects_non_positive_autosave_interval(
        self, tmp_path: Any, bad_interval: str
    ) -> None:
        """CLI surfaces non-positive autosave intervals as UsageError
        before any model load work happens."""
        from click.testing import CliRunner

        from tgirl.cli import serve

        runner = CliRunner()
        result = runner.invoke(
            serve,
            [
                "--model", "test-model",
                "--tools", "dummy",
                "--probe-save-on-shutdown", str(tmp_path / "out.npy"),
                "--probe-autosave-interval", bad_interval,
            ],
        )
        assert result.exit_code != 0
        assert "must be positive" in result.output

    def test_cli_forwards_probe_paths_to_create_app(
        self, tmp_path: Any
    ) -> None:
        """CLI --probe-load / --probe-save-on-shutdown reach create_app."""
        from click.testing import CliRunner

        from tgirl.cli import serve

        tool_file = tmp_path / "t.py"
        tool_file.write_text(
            "def register(registry):\n"
            "    @registry.tool()\n"
            "    def noop() -> int:\n"
            "        return 0\n"
        )

        load_p = str(tmp_path / "in.npy")
        save_p = str(tmp_path / "out.npy")

        captured: dict[str, Any] = {}

        def fake_create_app(ctx: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock()

        runner = CliRunner()
        with (
            patch(
                "tgirl.serve.load_inference_context",
                return_value=_make_mock_ctx(),
            ),
            patch("tgirl.serve.create_app", side_effect=fake_create_app),
            patch("uvicorn.run"),
        ):
            result = runner.invoke(
                serve,
                [
                    "--model", "test-model",
                    "--tools", str(tool_file),
                    "--probe-load", load_p,
                    "--probe-save-on-shutdown", save_p,
                    "--probe-autosave-interval", "2.5",
                ],
            )

        assert result.exit_code == 0, result.output
        assert captured.get("probe_load_path") == load_p
        assert captured.get("probe_save_path") == save_p
        assert captured.get("probe_autosave_interval_s") == 2.5

    def test_cli_exposes_autosave_interval_flag(self) -> None:
        """CLI serve --help lists --probe-autosave-interval."""
        from click.testing import CliRunner

        from tgirl.cli import serve

        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "--probe-autosave-interval" in result.output

    def test_write_is_atomic_destination_not_corrupted_on_rename_failure(
        self, tmp_path: Any
    ) -> None:
        """A crash/failure between write and rename must leave the
        destination path either untouched (previous good contents)
        or absent — never a partially-written corrupt file."""
        import time

        import numpy as np
        from fastapi.testclient import TestClient

        from tgirl.serve import create_app

        # Pre-seed the destination with a known-good file so we can
        # prove failed writes leave it untouched, not corrupted.
        save_path = tmp_path / "out.npy"
        good = np.arange(4, dtype=np.float32)
        np.save(save_path, good)

        # Load a different vector so the cache has fresh content
        # that would be written on shutdown (and autosave ticks).
        load_path = tmp_path / "in.npy"
        np.save(load_path, np.arange(8, dtype=np.float32))

        ctx = _make_mock_ctx()
        app = create_app(
            ctx,
            probe_load_path=str(load_path),
            probe_save_path=str(save_path),
            probe_autosave_interval_s=0.03,
        )

        # Force every rename to fail. This simulates a crash after
        # the tmp file is written but before the atomic rename lands.
        with patch("os.replace", side_effect=OSError("simulated crash")):
            with TestClient(app):
                time.sleep(0.15)  # let at least one autosave tick run

        # Destination must still hold the original good contents —
        # no partial bytes from the failed writes leaked through.
        reloaded = np.load(save_path)
        assert np.allclose(reloaded, good), (
            "rename failure corrupted the destination file"
        )

    def test_shutdown_save_roundtrips_without_npy_suffix(
        self, tmp_path: Any
    ) -> None:
        """User-supplied path is used verbatim for the save; np.save's
        suffix auto-append must not land the file at path+.npy."""
        import os

        import numpy as np
        from fastapi.testclient import TestClient

        from tgirl.serve import create_app

        load_path = tmp_path / "in.npy"
        expected = np.arange(8, dtype=np.float32)
        np.save(load_path, expected)

        # Explicitly no .npy suffix.
        save_path = str(tmp_path / "probe")

        ctx = _make_mock_ctx()
        app = create_app(
            ctx,
            probe_load_path=str(load_path),
            probe_save_path=save_path,
        )

        with TestClient(app):
            pass

        assert os.path.exists(save_path), (
            "file should land at user-supplied path, not path+.npy"
        )
        assert not os.path.exists(save_path + ".npy"), (
            "np.save suffix auto-append leaked through"
        )
        assert np.allclose(np.load(save_path), expected)

    def test_autosave_write_failure_does_not_block_shutdown_save(
        self, tmp_path: Any
    ) -> None:
        """Autosave I/O failures must not kill the task or skip the
        final shutdown save — that path is exactly why persistence
        exists."""
        import time

        import numpy as np
        from fastapi.testclient import TestClient

        from tgirl.serve import create_app

        load_path = tmp_path / "in.npy"
        save_path = tmp_path / "out.npy"
        np.save(load_path, np.arange(8, dtype=np.float32))

        original_save = np.save
        call_log: list[str] = []

        def flaky_save(p: Any, arr: Any) -> None:
            call_log.append(str(p))
            # Fail every autosave (interval tick), but allow the
            # final shutdown save (identified by call count — after
            # we've seen at least one failure, subsequent ones still
            # fail to prove the task survives, then shutdown passes).
            if len(call_log) < 3:
                raise OSError("simulated disk full")
            original_save(p, arr)

        ctx = _make_mock_ctx()
        app = create_app(
            ctx,
            probe_load_path=str(load_path),
            probe_save_path=str(save_path),
            probe_autosave_interval_s=0.03,
        )

        with patch("numpy.save", side_effect=flaky_save):
            # Must not raise on exit despite repeated autosave failures.
            with TestClient(app):
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and len(call_log) < 3:
                    time.sleep(0.02)

        # At least one autosave attempt + the shutdown save attempt
        # should have been logged. The shutdown save is the critical
        # assertion — without the fix, the dead-task exception would
        # propagate out of the finally block and skip it entirely.
        assert len(call_log) >= 2, (
            f"expected autosave retries + shutdown save; got {call_log}"
        )
