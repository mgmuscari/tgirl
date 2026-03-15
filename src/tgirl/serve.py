"""FastAPI local inference server for tgirl.

Provides a REST API for grammar-constrained inference with registered
tools. Supports both MLX and torch backends with automatic detection.

Optional dependency: requires ``fastapi`` and either ``mlx-lm`` or
``transformers`` depending on the chosen backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from tgirl.format import ChatTemplateFormatter
from tgirl.registry import ToolRegistry
from tgirl.types import PromptFormatter

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
