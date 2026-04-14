"""FastAPI local inference server for tgirl.

Provides a REST API for grammar-constrained inference with registered
tools. Supports both MLX and torch backends with automatic detection.

Optional dependency: requires ``fastapi`` and either ``mlx-lm`` or
``transformers`` depending on the chosen backend.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

import structlog

from tgirl.format import ChatTemplateFormatter
from tgirl.registry import ToolRegistry
from tgirl.types import PromptFormatter, SessionConfig

try:
    from fastapi import WebSocket as _FastAPIWebSocket  # noqa: F401
except ImportError:
    _FastAPIWebSocket = None  # type: ignore[assignment,misc]

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
    think_end_token_id: int | None = None  # </think> token for reasoning models
    bottleneck_hook: Any | None = None  # _BottleneckHook for probe feedback
    estradiol_file: Any | None = None  # CalibrationResult for behavioral diagnostics


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
    bottleneck_layer: int | None = None,
    layer_path: str | None = None,
    estradiol_path: str | None = None,
) -> InferenceContext:
    """Load model and construct all SamplingSession dependencies.

    Backend detection:
    - "auto": try MLX first (Apple Silicon), fall back to torch
    - "mlx": require mlx-lm, fail if unavailable
    - "torch": require transformers, fail if unavailable

    Args:
        model_id: HuggingFace model ID or local path.
        backend: Backend to use ("auto", "mlx", "torch").
        bottleneck_layer: Layer index for ESTRADIOL probe feedback.
            If provided, installs a bottleneck hook for self-steering.
        layer_path: Dot-separated path to model layers list
            (e.g. "language_model.model.layers"). Auto-detected if None.

    Returns:
        InferenceContext ready for creating SamplingSession instances.
    """
    resolved_backend = _resolve_backend(backend)

    if resolved_backend == "mlx":
        return _build_mlx_context(model_id, bottleneck_layer, layer_path, estradiol_path)
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


def _build_mlx_context(
    model_id: str,
    bottleneck_layer: int | None = None,
    layer_path: str | None = None,
    estradiol_path: str | None = None,
) -> InferenceContext:
    """Build InferenceContext for MLX backend."""
    import mlx.core as mx

    model, mlx_tokenizer = _load_mlx_model(model_id)
    hf_tokenizer = mlx_tokenizer._tokenizer

    forward_fn = _make_mlx_forward(model)
    grammar_factory = _make_mlx_grammar_factory(hf_tokenizer)
    embeddings = model.language_model.model.embed_tokens.weight.astype(
        mx.float32
    )
    mx.eval(embeddings)

    stop_ids: list[int] = []
    if hf_tokenizer.eos_token_id is not None:
        stop_ids.append(hf_tokenizer.eos_token_id)
    for token_str in ["<|im_end|>", "<|endoftext|>"]:
        ids = hf_tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in stop_ids:
            stop_ids.append(ids[0])

    # Detect </think> token for reasoning models
    think_end_id = None
    think_ids = hf_tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_ids) == 1:
        think_end_id = think_ids[0]
        logger.info("reasoning_model_detected", think_end_token_id=think_end_id)

    # Install bottleneck hook for probe feedback steering
    hook = None
    if bottleneck_layer is not None:
        from tgirl.cache import _BottleneckHook

        if layer_path is None:
            for candidate in ["language_model.model.layers", "model.layers"]:
                obj = model
                try:
                    for attr in candidate.split("."):
                        obj = getattr(obj, attr)
                    layer_path = candidate
                    break
                except AttributeError:
                    continue

        if layer_path is not None:
            layers = model
            for attr in layer_path.split("."):
                layers = getattr(layers, attr)
            hook = _BottleneckHook(layers, layer_idx=bottleneck_layer)
            hook.install()
            logger.info(
                "bottleneck_hook_installed",
                layer=bottleneck_layer,
                layer_path=layer_path,
            )

    # Load .estradiol calibration for behavioral diagnostics
    cal = None
    if estradiol_path is not None:
        from tgirl.estradiol import load_estradiol

        cal = load_estradiol(estradiol_path)
        logger.info("estradiol_loaded", path=estradiol_path, K=cal.K)
        # Use calibration's bottleneck layer if hook not already set
        if hook is None and cal.bottleneck_layer is not None:
            from tgirl.cache import _BottleneckHook

            if layer_path is None:
                for candidate in ["language_model.model.layers", "model.layers"]:
                    obj = model
                    try:
                        for attr in candidate.split("."):
                            obj = getattr(obj, attr)
                        layer_path = candidate
                        break
                    except AttributeError:
                        continue
            if layer_path is not None:
                layers = model
                for attr in layer_path.split("."):
                    layers = getattr(layers, attr)
                hook = _BottleneckHook(layers, layer_idx=cal.bottleneck_layer)
                hook.install()
                logger.info(
                    "bottleneck_hook_from_estradiol",
                    layer=cal.bottleneck_layer,
                )
    else:
        # Auto-detect: look for <model_id_slug>.estradiol in cwd
        import pathlib

        slug = model_id.replace("/", "_")
        for candidate in [f"{slug}.estradiol", f"{model_id.split('/')[-1]}.estradiol"]:
            p = pathlib.Path(candidate)
            if p.exists():
                from tgirl.estradiol import load_estradiol

                cal = load_estradiol(str(p))
                logger.info("estradiol_autodetected", path=str(p), K=cal.K)
                if hook is None:
                    from tgirl.cache import _BottleneckHook

                    if layer_path is None:
                        for lp_candidate in ["language_model.model.layers", "model.layers"]:
                            obj = model
                            try:
                                for attr in lp_candidate.split("."):
                                    obj = getattr(obj, attr)
                                layer_path = lp_candidate
                                break
                            except AttributeError:
                                continue
                    if layer_path is not None:
                        layers = model
                        for attr in layer_path.split("."):
                            layers = getattr(layers, attr)
                        hook = _BottleneckHook(layers, layer_idx=cal.bottleneck_layer)
                        hook.install()
                        logger.info(
                            "bottleneck_hook_from_estradiol",
                            layer=cal.bottleneck_layer,
                        )
                break

    logger.info(
        "mlx_model_loaded",
        model_id=model_id,
        vocab_size=embeddings.shape[0],
        self_steering=hook is not None,
        behavioral_diagnostics=cal is not None,
    )

    return InferenceContext(
        registry=ToolRegistry(),
        forward_fn=forward_fn,
        tokenizer_decode=hf_tokenizer.decode,
        tokenizer_encode=hf_tokenizer.encode,
        embeddings=embeddings,
        grammar_guide_factory=grammar_factory,
        mlx_grammar_guide_factory=grammar_factory,
        formatter=ChatTemplateFormatter(hf_tokenizer),
        backend="mlx",
        model_id=model_id,
        stop_token_ids=stop_ids,
        think_end_token_id=think_end_id,
        bottleneck_hook=hook,
        estradiol_file=cal,
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

    class GrammarPreviewRequest(_PydanticBase):
        """Request body for /grammar/preview endpoint."""

        scopes: list[str] | None = None
        restrict_tools: list[str] | None = None

    # --- OpenAI-compatible models ---

    class ChatMessage(_PydanticBase):
        role: str
        content: str | None = None
        reasoning_content: str | None = None  # thinking block for reasoning models

    class ChatCompletionRequest(_PydanticBase):
        model: str
        messages: list[ChatMessage]
        temperature: float = 1.0
        top_p: float = 1.0
        max_tokens: int | None = None
        max_completion_tokens: int | None = None
        stream: bool = False
        frequency_penalty: float = 0.0
        presence_penalty: float = 0.0
        stop: str | list[str] | None = None
        n: int = 1
        seed: int | None = None
        logprobs: bool = False
        top_logprobs: int | None = None
        logit_bias: dict[str, float] | None = None
        user: str | None = None
        # tgirl extensions
        estradiol_alpha: float | None = None  # per-request override; uses server default
        enable_thinking: bool = False  # set True for reasoning mode

    class ChatCompletionChoice(_PydanticBase):
        index: int
        message: ChatMessage
        finish_reason: str | None

    class CompletionUsage(_PydanticBase):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    class ChatCompletionResponse(_PydanticBase):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: list[ChatCompletionChoice]
        usage: CompletionUsage

    class DeltaMessage(_PydanticBase):
        role: str | None = None
        content: str | None = None
        reasoning_content: str | None = None

    class StreamChoice(_PydanticBase):
        index: int
        delta: DeltaMessage
        finish_reason: str | None = None

    class ChatCompletionChunk(_PydanticBase):
        id: str
        object: str = "chat.completion.chunk"
        created: int
        model: str
        choices: list[StreamChoice]

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


def _filter_registry(
    registry: ToolRegistry,
    *,
    restrict_tools: list[str] | None = None,
    scopes: list[str] | None = None,
) -> ToolRegistry:
    """Create a new ToolRegistry containing only the requested tools.

    Uses the snapshot filtering mechanism to determine which tools pass,
    then copies matching tool definitions and callables into a new registry.
    """
    snapshot = registry.snapshot(
        restrict_to=restrict_tools,
        scopes=set(scopes) if scopes else None,
    )
    filtered = ToolRegistry()
    for tool_def in snapshot.tools:
        # Copy the tool definition and callable into the new registry
        callable_fn = registry.get_callable(tool_def.name)
        filtered._tools[tool_def.name] = tool_def
        filtered._callables[tool_def.name] = callable_fn
    return filtered


def create_app(
    ctx: InferenceContext,
    session_config: SessionConfig | None = None,
    transport_config: Any = None,
    hooks: list[Any] | None = None,
    probe_load_path: str | None = None,
    probe_save_path: str | None = None,
    probe_autosave_interval_s: float | None = None,
) -> Any:
    """Create FastAPI app from a pre-loaded InferenceContext.

    Each /generate request creates a new SamplingSession from
    the shared InferenceContext.

    Args:
        ctx: Pre-loaded InferenceContext with model and dependencies.
        session_config: Optional session configuration override.
        transport_config: Optional transport configuration override.
        hooks: Optional inference hooks.
        probe_load_path: If set, load a probe vector from this path on
            server startup (populates the self-steering cache).
        probe_save_path: If set, save the probe cache to this path on
            server shutdown (persists behavioral continuity across
            restarts). No-op if the cache is empty at shutdown.
        probe_autosave_interval_s: If set, save the probe cache to
            probe_save_path every N seconds during the server lifetime
            in addition to the final shutdown save. Requires
            probe_save_path to be set.

    Returns:
        A FastAPI application instance.
    """
    import asyncio
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    if probe_autosave_interval_s is not None and probe_save_path is None:
        msg = (
            "probe_autosave_interval_s requires probe_save_path: the "
            "autosave loop has nowhere to write."
        )
        raise ValueError(msg)

    # Probe cache: persists the last bottleneck activation across turns.
    # Hoisted above `app` construction so the lifespan handler can close
    # over it. Referenced by /generate, /v1/steering/*, and _generate_tokens.
    _probe_cache: dict[str, Any] = {"v_probe": None}

    def _write_probe(path: str, event: str) -> None:
        import numpy as np

        v = _probe_cache.get("v_probe")
        if v is None:
            logger.info(event + "_skipped", reason="cache_empty", path=path)
            return
        np.save(path, np.array(v))
        logger.info(event, path=path, shape=list(v.shape))

    async def _autosave_loop(path: str, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            _write_probe(path, "probe_autosaved")

    @asynccontextmanager
    async def _lifespan(_app: Any):
        if probe_load_path is not None:
            import mlx.core as _mx
            import numpy as np

            arr = np.load(probe_load_path)
            _probe_cache["v_probe"] = _mx.array(arr)
            logger.info(
                "probe_loaded_at_startup",
                path=probe_load_path,
                shape=list(arr.shape),
            )

        autosave_task: asyncio.Task[None] | None = None
        if (
            probe_autosave_interval_s is not None
            and probe_save_path is not None
        ):
            autosave_task = asyncio.create_task(
                _autosave_loop(probe_save_path, probe_autosave_interval_s)
            )

        try:
            yield
        finally:
            if autosave_task is not None:
                autosave_task.cancel()
                try:
                    await autosave_task
                except asyncio.CancelledError:
                    pass
            if probe_save_path is not None:
                _write_probe(probe_save_path, "probe_saved_at_shutdown")

    app = FastAPI(title="tgirl", lifespan=_lifespan)

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
            # Build per-request overrides from GenerateRequest params
            req_ctx = ctx
            req_session_config = session_config
            req_transport_config = transport_config
            req_hooks = hooks

            # Filter registry by restrict_tools and/or scopes
            if request.restrict_tools is not None or request.scopes is not None:
                filtered_registry = _filter_registry(
                    ctx.registry,
                    restrict_tools=request.restrict_tools,
                    scopes=request.scopes,
                )
                req_ctx = replace(ctx, registry=filtered_registry)

            # Override transport epsilon
            if request.ot_epsilon is not None:
                from tgirl.transport import TransportConfig

                if transport_config is not None:
                    req_transport_config = transport_config.model_copy(
                        update={"epsilon": request.ot_epsilon}
                    )
                else:
                    req_transport_config = TransportConfig(epsilon=request.ot_epsilon)

            # Override base temperature via hook
            if request.base_temperature is not None:
                from tgirl.sample import GrammarTemperatureHook

                temp_hook = GrammarTemperatureHook(
                    base_temperature=request.base_temperature
                )
                # Replace any existing GrammarTemperatureHook, keep others
                base_hooks = [
                    h
                    for h in (hooks or [])
                    if not isinstance(h, GrammarTemperatureHook)
                ]
                req_hooks = base_hooks + [temp_hook]

            # Override session cost budget
            if request.max_cost is not None:
                if session_config is not None:
                    req_session_config = session_config.model_copy(
                        update={"session_cost_budget": request.max_cost}
                    )
                else:
                    req_session_config = SessionConfig(
                        session_cost_budget=request.max_cost
                    )

            result = await asyncio.to_thread(
                _run_session_chat,
                req_ctx,
                messages,
                req_session_config,
                req_transport_config,
                req_hooks,
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

    # In-memory telemetry buffer
    telemetry_buffer: list[dict[str, Any]] = []

    @app.get("/tools")
    async def list_tools() -> list[dict[str, Any]]:
        snapshot = ctx.registry.snapshot()
        return [
            {
                "name": td.name,
                "description": td.description,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type_repr.model_dump(),
                        "required": not p.has_default,
                    }
                    for p in td.parameters
                ],
                "quota": td.quota,
                "scope": td.scope,
            }
            for td in snapshot.tools
        ]

    @app.get("/grammar")
    async def get_grammar() -> dict[str, Any]:
        from tgirl.grammar import generate as generate_grammar

        snapshot = ctx.registry.snapshot()
        output = generate_grammar(snapshot)
        return {"text": output.text, "hash": output.snapshot_hash}

    @app.post("/grammar/preview")
    async def preview_grammar(
        request: GrammarPreviewRequest,
    ) -> dict[str, Any]:
        from tgirl.grammar import generate as generate_grammar

        snapshot = ctx.registry.snapshot(
            scopes=set(request.scopes) if request.scopes else None,
            restrict_to=request.restrict_tools,
        )
        output = generate_grammar(snapshot)
        return {"text": output.text, "hash": output.snapshot_hash}

    @app.get("/telemetry")
    async def get_telemetry(limit: int = 100) -> list[dict[str, Any]]:
        return telemetry_buffer[-limit:]

    @app.websocket("/stream")
    async def stream(websocket: _FastAPIWebSocket) -> None:
        await websocket.accept()
        try:
            data = await websocket.receive_json()
            intent = data.get("intent", "")
            messages = [{"role": "user", "content": intent}]

            try:
                result = await asyncio.to_thread(
                    _run_session_chat,
                    ctx,
                    messages,
                    session_config,
                    transport_config,
                    hooks,
                )
                await websocket.send_json({
                    "type": "result",
                    "output": result.output_text,
                    "tool_calls": [
                        {
                            "pipeline": tc.pipeline,
                            "result": tc.result,
                            "error": (
                                tc.error.message
                                if tc.error
                                else None
                            ),
                            "cycle_number": tc.cycle_number,
                            "tool_invocations": tc.tool_invocations,
                        }
                        for tc in result.tool_calls
                    ],
                    "total_tokens": result.total_tokens,
                    "total_cycles": result.total_cycles,
                    "wall_time_ms": result.wall_time_ms,
                    "quotas_consumed": result.quotas_consumed,
                })
            except Exception as e:
                logger.error("stream_error", error=str(e))
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                })
        except Exception:
            pass
        finally:
            await websocket.close()

    # --- OpenAI-compatible endpoints ---

    # _probe_cache is declared at the top of create_app so the lifespan
    # handler can close over it for probe load/save on startup/shutdown.
    _steering_config: dict[str, float] = {"alpha": 0.05}
    _steering_stats: dict[str, Any] = {
        "requests": 0,
        "last_probe_norm": 0.0,
        "last_correction_norm": 0.0,
        "last_alpha": 0.0,
        "probe_cached": False,
    }

    def _generate_tokens(
        request: ChatCompletionRequest,
        prompt_tokens: list[int],
    ) -> tuple[list[int], str]:
        """Autoregressive generation with probe feedback self-steering.

        When ctx.bottleneck_hook is installed and estradiol_alpha > 0:
          v_steer(n+1) = alpha * v_probe(n)

        The probe from the last token is cached in _probe_cache so the
        next turn picks up the same behavioral state.

        Returns (generated_token_ids, finish_reason).
        """
        import mlx.core as mx

        max_tok = request.max_completion_tokens or request.max_tokens or 1024
        temp = request.temperature
        hook = ctx.bottleneck_hook
        alpha = request.estradiol_alpha if request.estradiol_alpha is not None else _steering_config["alpha"]

        if request.seed is not None:
            mx.random.seed(request.seed)

        # Build stop token set
        stop_ids = set(ctx.stop_token_ids)
        if request.stop:
            stops = [request.stop] if isinstance(request.stop, str) else request.stop
            for s in stops:
                ids = ctx.tokenizer_encode(s)
                if len(ids) == 1:
                    stop_ids.add(ids[0])

        # Initialize probe from cache (behavioral continuity across turns)
        v_probe_prev = _probe_cache.get("v_probe")
        _steering_stats["requests"] += 1
        _steering_stats["last_alpha"] = alpha
        _steering_stats["probe_cached"] = v_probe_prev is not None
        correction_norms: list[float] = []

        generated: list[int] = []
        token_ids = list(prompt_tokens)

        for _ in range(max_tok):
            # Steer: inject scaled probe from previous token (or previous turn)
            if hook is not None and alpha > 0 and v_probe_prev is not None:
                correction = alpha * v_probe_prev
                hook.set_raw_correction(correction)
                correction_norms.append(float(mx.linalg.norm(correction).item()))
            elif hook is not None:
                hook.clear_steering()

            logits = ctx.forward_fn(token_ids)
            if hasattr(logits, "logits"):
                logits = logits.logits

            # Capture probe for next token
            if hook is not None:
                v_probe_prev = hook.get_captured()
                if v_probe_prev is not None:
                    mx.eval(v_probe_prev)
                    _steering_stats["last_probe_norm"] = float(
                        mx.linalg.norm(v_probe_prev).item()
                    )

            if temp > 0:
                logits = logits / temp
                next_token = int(mx.random.categorical(logits).item())
            else:
                next_token = int(mx.argmax(logits).item())

            if next_token in stop_ids:
                _probe_cache["v_probe"] = v_probe_prev
                if correction_norms:
                    _steering_stats["last_correction_norm"] = correction_norms[-1]
                    _steering_stats["mean_correction_norm"] = (
                        sum(correction_norms) / len(correction_norms)
                    )
                return generated, "stop"

            generated.append(next_token)
            token_ids = token_ids + [next_token]

        _probe_cache["v_probe"] = v_probe_prev
        if correction_norms:
            _steering_stats["last_correction_norm"] = correction_norms[-1]
            _steering_stats["mean_correction_norm"] = (
                sum(correction_norms) / len(correction_norms)
            )
        return generated, "length"

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": ctx.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/steering/alpha")
    async def set_alpha(alpha: float) -> dict[str, Any]:
        _steering_config["alpha"] = alpha
        logger.info("alpha_updated", alpha=alpha)
        return {"alpha": alpha}

    @app.post("/v1/steering/probe/save")
    async def save_probe(path: str = "session_probe.npy") -> dict[str, Any]:
        """Save the cached probe vector to disk."""
        import mlx.core as _mx
        import numpy as np

        v = _probe_cache.get("v_probe")
        if v is None:
            return {"error": "no probe cached"}
        np.save(path, np.array(v))
        norm = float(_mx.linalg.norm(v).item())
        logger.info("probe_saved", path=path, norm=norm)
        return {"saved": path, "norm": norm, "shape": list(v.shape)}

    @app.post("/v1/steering/probe/load")
    async def load_probe(path: str = "session_probe.npy") -> dict[str, Any]:
        """Load a probe vector from disk into the cache."""
        import mlx.core as _mx
        import numpy as np

        arr = np.load(path)
        _probe_cache["v_probe"] = _mx.array(arr)
        norm = float(_mx.linalg.norm(_probe_cache["v_probe"]).item())
        logger.info("probe_loaded", path=path, norm=norm)
        return {"loaded": path, "norm": norm, "shape": list(arr.shape)}

    @app.get("/v1/steering/status")
    async def steering_status() -> dict[str, Any]:
        # Unpack stats first, then override with live-computed fields so stale
        # stats entries (e.g. the stats dict's own "probe_cached" that is only
        # touched during generation) do not mask the source-of-truth cache.
        result: dict[str, Any] = {
            **_steering_stats,
            "hook_installed": ctx.bottleneck_hook is not None,
            "probe_cached": _probe_cache.get("v_probe") is not None,
        }
        # Project cached probe onto behavioral codebook for diagnostics
        v = _probe_cache.get("v_probe")
        if v is not None and ctx.estradiol_file is not None:
            import mlx.core as mx

            cal = ctx.estradiol_file
            coords = {}
            for name, trait_vec in cal.trait_map.items():
                # Cosine similarity with trait direction in activation space
                full_vec = cal.V_basis @ trait_vec  # (d_model,)
                cos = float((v @ full_vec).item()) / (
                    float(mx.linalg.norm(v).item())
                    * float(mx.linalg.norm(full_vec).item())
                    + 1e-10
                )
                coords[name] = round(cos, 4)
            result["behavioral_state"] = coords
        return result

    def _format_prompt(
        request: ChatCompletionRequest,
    ) -> list[int]:
        """Format messages and tokenize using the model's chat template."""
        messages = [
            {"role": m.role, "content": m.content or ""}
            for m in request.messages
        ]
        try:
            prompt_text = ctx.formatter.format_messages(
                messages,
                enable_thinking=request.enable_thinking,
            )
        except Exception:
            prompt_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
        return ctx.tokenizer_encode(prompt_text)

    def _split_reasoning(
        token_ids: list[int],
    ) -> tuple[str | None, str]:
        """Split generated tokens into reasoning_content and content.

        If the model produced a </think> token, everything before it
        is reasoning_content; everything after is content. If no
        </think> token, everything is content.
        """
        think_end = ctx.think_end_token_id
        if think_end is not None and think_end in token_ids:
            idx = token_ids.index(think_end)
            reasoning_tokens = token_ids[:idx]
            content_tokens = token_ids[idx + 1:]
            reasoning = ctx.tokenizer_decode(reasoning_tokens).strip()
            content = ctx.tokenizer_decode(content_tokens).strip()
            return (reasoning or None), content
        return None, ctx.tokenizer_decode(token_ids)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        import time

        import mlx.core as mx

        created = int(time.time())
        completion_id = f"chatcmpl-{created}"
        prompt_tokens = _format_prompt(request)

        if request.stream:
            from fastapi.responses import StreamingResponse

            async def stream_gen():
                # First chunk: role
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[StreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant"),
                    )],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                max_tok = request.max_completion_tokens or request.max_tokens or 1024
                temp = request.temperature
                stop_ids = set(ctx.stop_token_ids)
                token_ids = list(prompt_tokens)
                think_end = ctx.think_end_token_id
                in_thinking = request.enable_thinking and think_end is not None
                hook = ctx.bottleneck_hook
                alpha = request.estradiol_alpha if request.estradiol_alpha is not None else _steering_config["alpha"]
                v_probe_prev = _probe_cache.get("v_probe")
                generated_tokens: list[int] = []
                _steering_stats["requests"] += 1
                _steering_stats["last_alpha"] = alpha
                _steering_stats["probe_cached"] = v_probe_prev is not None

                if request.seed is not None:
                    mx.random.seed(request.seed)

                for _ in range(max_tok):
                    # Probe feedback steering
                    if hook is not None and alpha > 0 and v_probe_prev is not None:
                        hook.set_raw_correction(alpha * v_probe_prev)
                    elif hook is not None:
                        hook.clear_steering()

                    logits = ctx.forward_fn(token_ids)
                    if hasattr(logits, "logits"):
                        logits = logits.logits

                    if hook is not None:
                        v_probe_prev = hook.get_captured()
                        if v_probe_prev is not None:
                            mx.eval(v_probe_prev)
                            _steering_stats["last_probe_norm"] = float(
                                mx.linalg.norm(v_probe_prev).item()
                            )

                    if temp > 0:
                        logits = logits / temp
                        next_token = int(mx.random.categorical(logits).item())
                    else:
                        next_token = int(mx.argmax(logits).item())

                    if next_token in stop_ids:
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[StreamChoice(
                                index=0,
                                delta=DeltaMessage(),
                                finish_reason="stop",
                            )],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                        _probe_cache["v_probe"] = v_probe_prev
                        response_text = ctx.tokenizer_decode(generated_tokens)
                        logger.info(
                            "stream_complete",
                            tokens=len(generated_tokens),
                            response=response_text[:200],
                        )
                        break

                    # Detect end-of-thinking transition
                    if in_thinking and next_token == think_end:
                        in_thinking = False
                        token_ids = token_ids + [next_token]
                        continue  # don't emit the </think> token

                    token_ids = token_ids + [next_token]
                    generated_tokens.append(next_token)
                    text = ctx.tokenizer_decode([next_token])

                    if in_thinking:
                        # Emit as reasoning_content delta
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[StreamChoice(
                                index=0,
                                delta=DeltaMessage(reasoning_content=text),
                            )],
                        )
                    else:
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[StreamChoice(
                                index=0,
                                delta=DeltaMessage(content=text),
                            )],
                        )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                else:
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[StreamChoice(
                            index=0,
                            delta=DeltaMessage(),
                            finish_reason="length",
                        )],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    _probe_cache["v_probe"] = v_probe_prev
                    response_text = ctx.tokenizer_decode(generated_tokens)
                    logger.info(
                        "stream_complete_length",
                        tokens=len(generated_tokens),
                        response=response_text[:200],
                    )

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_gen(),
                media_type="text/event-stream",
            )

        # Non-streaming
        generated, finish_reason = await asyncio.to_thread(
            _generate_tokens, request, prompt_tokens,
        )
        reasoning_content, content = _split_reasoning(generated)

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning_content,
                ),
                finish_reason=finish_reason,
            )],
            usage=CompletionUsage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(generated),
                total_tokens=len(prompt_tokens) + len(generated),
            ),
        )

    return app
