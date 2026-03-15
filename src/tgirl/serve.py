"""FastAPI local inference server for tgirl.

Provides a REST API for grammar-constrained inference with registered
tools. Supports both MLX and torch backends with automatic detection.

Optional dependency: requires ``fastapi`` and either ``mlx-lm`` or
``transformers`` depending on the chosen backend.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from tgirl.format import ChatTemplateFormatter
from tgirl.registry import ToolRegistry
from tgirl.types import PromptFormatter, SessionConfig

logger = structlog.get_logger()


@dataclass(frozen=True)
class InferenceContext:
    """Everything needed to create SamplingSession instances.

    Constructed once at startup via load_inference_context, then
    shared across all requests. Each request creates a new
    SamplingSession from this context.
    """

    registry: ToolRegistry
    forward_fn: Callable[[list[int]], Any]
    tokenizer_decode: Callable[[list[int]], str]
    tokenizer_encode: Callable[[str], list[int]]
    embeddings: Any
    grammar_guide_factory: Callable[[str], Any]
    mlx_grammar_guide_factory: Callable | None
    formatter: PromptFormatter
    backend: Literal["torch", "mlx"]
    model_id: str
    stop_token_ids: list[int]


# --- Backend detection helpers ---


def _try_import_mlx() -> bool:
    """Check if mlx-lm is available."""
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def _try_import_torch() -> bool:
    """Check if transformers is available."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


# --- Backend-specific loaders (extracted for mockability) ---


def _load_mlx_model(model_id: str) -> tuple[Any, Any]:
    """Load model and tokenizer via mlx-lm."""
    import mlx_lm

    model, tokenizer = mlx_lm.load(model_id)
    return model, tokenizer


def _make_mlx_forward(model: Any) -> Callable[[list[int]], Any]:
    """Create cached forward function for MLX model."""
    from tgirl.cache import make_mlx_forward_fn

    return make_mlx_forward_fn(model)


def _make_mlx_grammar_factory(tokenizer: Any) -> Callable:
    """Create MLX grammar guide factory."""
    from tgirl.outlines_adapter import make_outlines_grammar_factory_mlx

    return make_outlines_grammar_factory_mlx(tokenizer)


def _load_torch_model(model_id: str) -> tuple[Any, Any]:
    """Load model and tokenizer via HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tokenizer


def _make_torch_forward(model: Any) -> Callable[[list[int]], Any]:
    """Create cached forward function for HF model."""
    from tgirl.cache import make_hf_forward_fn

    return make_hf_forward_fn(model)


def _make_torch_grammar_factory(tokenizer: Any) -> Callable:
    """Create torch grammar guide factory."""
    from tgirl.outlines_adapter import make_outlines_grammar_factory

    return make_outlines_grammar_factory(tokenizer)


# --- Main loader ---


def load_inference_context(
    model_id: str,
    backend: str = "auto",
) -> InferenceContext:
    """Load model and construct all SamplingSession dependencies.

    Backend detection:
    - "auto": try MLX first (Apple Silicon), fall back to torch
    - "mlx": require mlx-lm, fail if unavailable
    - "torch": require transformers, fail if unavailable

    Args:
        model_id: HuggingFace model ID or local path.
        backend: Backend to use ("auto", "mlx", "torch").

    Returns:
        InferenceContext ready for creating SamplingSession instances.
    """
    resolved_backend = _resolve_backend(backend)

    if resolved_backend == "mlx":
        return _build_mlx_context(model_id)
    return _build_torch_context(model_id)


def _resolve_backend(backend: str) -> Literal["mlx", "torch"]:
    """Resolve backend string to concrete backend."""
    if backend == "mlx":
        if not _try_import_mlx():
            msg = (
                "mlx-lm is required for MLX backend. "
                "Install with: pip install mlx-lm"
            )
            raise ImportError(msg)
        return "mlx"

    if backend == "torch":
        if not _try_import_torch():
            msg = (
                "transformers is required for torch backend. "
                "Install with: pip install transformers torch"
            )
            raise ImportError(msg)
        return "torch"

    # auto: prefer MLX
    if _try_import_mlx():
        return "mlx"
    if _try_import_torch():
        return "torch"

    msg = (
        "No inference backend available. "
        "Install mlx-lm (Apple Silicon) or transformers + torch."
    )
    raise ImportError(msg)


def _build_mlx_context(model_id: str) -> InferenceContext:
    """Build InferenceContext for MLX backend."""
    model, tokenizer = _load_mlx_model(model_id)
    forward_fn = _make_mlx_forward(model)
    grammar_factory = _make_mlx_grammar_factory(tokenizer)
    embeddings = model.model.embed_tokens.weight

    eos = tokenizer.eos_token_id
    stop_ids = [eos] if isinstance(eos, int) else list(eos)

    logger.info(
        "mlx_model_loaded",
        model_id=model_id,
        vocab_size=len(tokenizer),
    )

    return InferenceContext(
        registry=ToolRegistry(),
        forward_fn=forward_fn,
        tokenizer_decode=tokenizer.decode,
        tokenizer_encode=tokenizer.encode,
        embeddings=embeddings,
        grammar_guide_factory=grammar_factory,
        mlx_grammar_guide_factory=grammar_factory,
        formatter=ChatTemplateFormatter(tokenizer),
        backend="mlx",
        model_id=model_id,
        stop_token_ids=stop_ids,
    )


def _build_torch_context(model_id: str) -> InferenceContext:
    """Build InferenceContext for torch/HF backend."""
    model, tokenizer = _load_torch_model(model_id)
    forward_fn = _make_torch_forward(model)
    grammar_factory = _make_torch_grammar_factory(tokenizer)
    embeddings = model.get_input_embeddings().weight

    eos = tokenizer.eos_token_id
    stop_ids = [eos] if isinstance(eos, int) else list(eos)

    logger.info(
        "torch_model_loaded",
        model_id=model_id,
        vocab_size=len(tokenizer),
    )

    return InferenceContext(
        registry=ToolRegistry(),
        forward_fn=forward_fn,
        tokenizer_decode=tokenizer.decode,
        tokenizer_encode=tokenizer.encode,
        embeddings=embeddings,
        grammar_guide_factory=grammar_factory,
        mlx_grammar_guide_factory=None,
        formatter=ChatTemplateFormatter(tokenizer),
        backend="torch",
        model_id=model_id,
        stop_token_ids=stop_ids,
    )


# --- FastAPI request/response models ---

try:
    from pydantic import BaseModel as _PydanticBase

    class GenerateRequest(_PydanticBase):
        """Request body for /generate endpoint."""

        intent: str
        scopes: list[str] | None = None
        max_cost: float | None = None
        restrict_tools: list[str] | None = None
        ot_epsilon: float | None = None
        base_temperature: float | None = None

    class ToolCallResponse(_PydanticBase):
        """Single tool call in a generate response."""

        pipeline: str
        result: Any | None = None
        error: str | None = None
        cycle_number: int = 0
        tool_invocations: dict[str, int] = {}

    class GenerateResponse(_PydanticBase):
        """Response body for /generate endpoint."""

        output: str
        tool_calls: list[ToolCallResponse]
        total_tokens: int
        total_cycles: int
        wall_time_ms: float
        quotas_consumed: dict[str, int]
        error: str | None = None

except ImportError:
    pass  # pydantic not available (shouldn't happen — it's a core dep)


# --- FastAPI server ---


def _run_session_chat(
    ctx: InferenceContext,
    messages: list[dict[str, str]],
    session_config: SessionConfig | None = None,
    transport_config: Any = None,
    hooks: list[Any] | None = None,
) -> Any:
    """Create a SamplingSession and run chat. Extracted for mockability."""
    from tgirl.sample import SamplingSession

    session = SamplingSession(
        registry=ctx.registry,
        forward_fn=ctx.forward_fn,
        tokenizer_decode=ctx.tokenizer_decode,
        tokenizer_encode=ctx.tokenizer_encode,
        embeddings=ctx.embeddings,
        grammar_guide_factory=ctx.grammar_guide_factory,
        mlx_grammar_guide_factory=ctx.mlx_grammar_guide_factory,
        formatter=ctx.formatter,
        backend=ctx.backend,
        config=session_config,
        transport_config=transport_config,
        hooks=hooks,
        stop_token_ids=ctx.stop_token_ids,
    )
    return session.run_chat(messages)


def create_app(
    ctx: InferenceContext,
    session_config: SessionConfig | None = None,
    transport_config: Any = None,
    hooks: list[Any] | None = None,
) -> Any:
    """Create FastAPI app from a pre-loaded InferenceContext.

    Each /generate request creates a new SamplingSession from
    the shared InferenceContext.

    Args:
        ctx: Pre-loaded InferenceContext with model and dependencies.
        session_config: Optional session configuration override.
        transport_config: Optional transport configuration override.
        hooks: Optional inference hooks.

    Returns:
        A FastAPI application instance.
    """
    from fastapi import FastAPI

    app = FastAPI(title="tgirl")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "model": ctx.model_id,
            "tools": len(ctx.registry),
            "backend": ctx.backend,
            "status": "ok",
        }

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> GenerateResponse:
        messages = [{"role": "user", "content": request.intent}]
        try:
            result = await asyncio.to_thread(
                _run_session_chat,
                ctx,
                messages,
                session_config,
                transport_config,
                hooks,
            )
            tool_call_responses = [
                ToolCallResponse(
                    pipeline=tc.pipeline,
                    result=tc.result,
                    error=(
                        tc.error.message if tc.error else None
                    ),
                    cycle_number=tc.cycle_number,
                    tool_invocations=tc.tool_invocations,
                )
                for tc in result.tool_calls
            ]
            return GenerateResponse(
                output=result.output_text,
                tool_calls=tool_call_responses,
                total_tokens=result.total_tokens,
                total_cycles=result.total_cycles,
                wall_time_ms=result.wall_time_ms,
                quotas_consumed=result.quotas_consumed,
            )
        except Exception as e:
            logger.error("generate_error", error=str(e))
            return GenerateResponse(
                output="",
                tool_calls=[],
                total_tokens=0,
                total_cycles=0,
                wall_time_ms=0.0,
                quotas_consumed={},
                error=str(e),
            )

    return app
