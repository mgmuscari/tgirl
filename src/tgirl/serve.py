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
    from fastapi import Request as _FastAPIRequest  # noqa: F401
    from fastapi import WebSocket as _FastAPIWebSocket  # noqa: F401
except ImportError:
    _FastAPIRequest = None  # type: ignore[assignment,misc]
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
    mlx_grammar_guide_factory: Callable[[str], Any] | None
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


def _make_mlx_grammar_factory(tokenizer: Any) -> Callable[[str], Any]:
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


def _make_torch_grammar_factory(tokenizer: Any) -> Callable[[str], Any]:
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
    auto_calibrate: bool = True,
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
        estradiol_path: Explicit path to a .estradiol calibration. If
            None, autodetect by model basename in cwd.
        auto_calibrate: When True (default), if no estradiol file is
            found via autodetect, run the full ESTRADIOL calibration
            pipeline on first start and save it to disk for reuse.
            Set False to skip calibration (steering will be disabled).

    Returns:
        InferenceContext ready for creating SamplingSession instances.
    """
    resolved_backend = _resolve_backend(backend)

    if resolved_backend == "mlx":
        return _build_mlx_context(
            model_id,
            bottleneck_layer,
            layer_path,
            estradiol_path,
            auto_calibrate=auto_calibrate,
        )
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
    auto_calibrate: bool = True,
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
        candidates = [
            f"{slug}.estradiol",
            f"{model_id.split('/')[-1]}.estradiol",
        ]
        for candidate in candidates:
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

        # Bootstrap: no estradiol on disk and the user wants one. The
        # full ESTRADIOL calibration pipeline runs once (~30s-2min on
        # 0.8B), saves to the basename autodetect path, and returns
        # the result so the hook can install. Without this, a fresh
        # install silently lacks steering — every chat completion is
        # the deterministic baseline output and the autotuner has no
        # signal to act on.
        if cal is None and auto_calibrate:
            import tgirl.calibrate as _cal_mod

            # Discover layer_path BEFORE calibrating: tgirl.calibrate
            # defaults to "model.layers", but VL-style MLX wrappers
            # (Qwen3.5-*-MLX-*, etc.) expose layers under
            # "language_model.model.layers". Mismatched path → calibrate
            # raises AttributeError walking the model graph. We probe
            # both candidates against the loaded model and forward the
            # one that resolves.
            if layer_path is None:
                for lp_candidate in [
                    "language_model.model.layers",
                    "model.layers",
                ]:
                    obj = model
                    try:
                        for attr in lp_candidate.split("."):
                            obj = getattr(obj, attr)
                        layer_path = lp_candidate
                        break
                    except AttributeError:
                        continue
            if layer_path is None:
                msg = (
                    "Cannot bootstrap estradiol calibration: layers "
                    "not found at any known path on the loaded model. "
                    "Pass --no-auto-calibrate to skip, or provide an "
                    "explicit layer_path."
                )
                raise RuntimeError(msg)

            bootstrap_path = candidates[1]  # "<basename>.estradiol"
            logger.info(
                "estradiol_bootstrap_starting",
                model_id=model_id,
                output_path=bootstrap_path,
                layer_path=layer_path,
            )
            cal = _cal_mod.calibrate(
                model=model,
                tokenizer=hf_tokenizer,
                model_id=model_id,
                layer_path=layer_path,
                output_path=bootstrap_path,
            )
            logger.info(
                "estradiol_bootstrap_complete",
                path=bootstrap_path,
                K=cal.K,
                bottleneck_layer=cal.bottleneck_layer,
            )
            if hook is None and cal.bottleneck_layer is not None:
                from tgirl.cache import _BottleneckHook

                layers = model
                for attr in layer_path.split("."):
                    layers = getattr(layers, attr)
                hook = _BottleneckHook(
                    layers, layer_idx=cal.bottleneck_layer
                )
                hook.install()
                logger.info(
                    "bottleneck_hook_from_bootstrap",
                    layer=cal.bottleneck_layer,
                )

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
        estradiol_beta: float | None = None  # band sharpness (inverse σ, layers⁻¹). None = single-layer.
        estradiol_skew: float | None = None  # band σ_up/σ_down ratio. None = server default (1.0).
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
    if (
        probe_autosave_interval_s is not None
        and probe_autosave_interval_s <= 0
    ):
        # asyncio.sleep(<=0) yields immediately, turning the autosave
        # loop into a tight write storm that pins the event loop and
        # saturates disk. Reject at config time.
        msg = (
            "probe_autosave_interval_s must be positive "
            f"(got {probe_autosave_interval_s})."
        )
        raise ValueError(msg)

    # Probe cache: persists the last bottleneck activation across turns.
    # Hoisted above `app` construction so the lifespan handler can close
    # over it. Referenced by /generate, /v1/steering/*, and _generate_tokens.
    _probe_cache: dict[str, Any] = {"v_probe": None}

    def _write_probe(path: str, event: str) -> None:
        # Atomic write: stream into a sibling .tmp, then os.replace.
        # Non-atomic writes leave a truncated file at `path` on crash,
        # and the next --probe-load on the same path raises from
        # np.load — exactly the recovery path this feature promises.
        # Using an open file object (not a string path) bypasses
        # np.save's .npy suffix auto-append so the final file lands
        # at exactly `path`, keeping --probe-save and --probe-load
        # symmetric regardless of whether the user included .npy.
        import os
        import numpy as np

        v = _probe_cache.get("v_probe")
        if v is None:
            logger.info(event + "_skipped", reason="cache_empty", path=path)
            return
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            np.save(f, np.array(v))
        os.replace(tmp, path)
        logger.info(event, path=path, shape=list(v.shape))

    async def _autosave_loop(path: str, interval_s: float) -> None:
        # Swallow write failures so a transient I/O error (disk full,
        # permissions, flaky fs) does not kill the task. A dead task
        # would make `await autosave_task` in the lifespan finally
        # re-raise its original exception, bypassing the
        # `except CancelledError` guard and preventing the final
        # shutdown save — the very error path persistence protects.
        while True:
            await asyncio.sleep(interval_s)
            try:
                _write_probe(path, "probe_autosaved")
            except Exception:
                logger.exception("probe_autosave_failed", path=path)

    @asynccontextmanager
    async def _lifespan(_app: Any):
        if probe_load_path is not None:
            import mlx.core as _mx
            import numpy as np

            # Log the path before loading so operators can tell which
            # file triggered a load failure (np.load's error messages
            # don't always include the path).
            logger.info("probe_load_attempt", path=probe_load_path)
            arr = np.load(probe_load_path)

            # Validate shape against the model's hidden dim. A probe
            # saved from a different model (or a stale file from an
            # earlier checkpoint) would otherwise silently corrupt
            # steering — broadcast-safe arithmetic hides shape drift
            # until deep inside the bottleneck hook's forward pass,
            # or worse, produces wrong steering with no error at all
            # when the dims happen to match but the basis is foreign.
            #
            # Source the expected dim from the estradiol codebook when
            # available: V_basis has shape (d_model, K) and survives
            # model quantization. Fall back to embeddings.shape[-1]
            # for the non-estradiol hook path, but note that for 4-bit
            # quantized MLX models the embeddings trailing dim is the
            # *packed* dim, not the true d_model — so that fallback
            # cannot validate the quantized path.
            if ctx.estradiol_file is not None:
                expected_dim = int(ctx.estradiol_file.V_basis.shape[0])
            else:
                expected_dim = int(ctx.embeddings.shape[-1])
            if arr.shape != (expected_dim,):
                msg = (
                    f"Loaded probe shape {tuple(arr.shape)} does not "
                    f"match model hidden dim ({expected_dim},). The "
                    f"file at {probe_load_path} was likely saved from "
                    f"a different model. Remove --probe-load or pass a "
                    f"probe saved from this model."
                )
                raise ValueError(msg)

            # Cast to float32 to match the hook's native capture
            # dtype (see _BottleneckHook where captures are cast to
            # float32). Prevents silent dtype promotion inside the
            # alpha * v_probe multiply, which could shift the
            # correction magnitude calibration.
            _probe_cache["v_probe"] = _mx.array(arr.astype(np.float32))
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
                # Swallow failures here for the same reason as the
                # autosave loop: a best-effort save shouldn't raise
                # out of the lifespan `finally` and obscure the real
                # shutdown cause in logs.
                try:
                    _write_probe(
                        probe_save_path, "probe_saved_at_shutdown"
                    )
                except Exception:
                    logger.exception(
                        "probe_shutdown_save_failed",
                        path=probe_save_path,
                    )

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
    _steering_config: dict[str, Any] = {
        "alpha": 0.05,
        # Band steering defaults: None = single-layer (current behavior).
        # skew=1.0 is symmetric. These act as fallbacks when a request
        # does not override via estradiol_beta / estradiol_skew.
        "beta": None,
        "skew": 1.0,
        # Steering normalization mode. "absolute" scales v_probe
        # by α directly (pre-this-feature behavior). "residual_relative"
        # strips v_probe's magnitude and rescales the correction to
        # α * |residual_last| per forward pass — α becomes a structural
        # fraction of the signal power being overwritten. Default is
        # "absolute" so existing clients see no change.
        "normalization": "absolute",
    }

    _VALID_NORM_MODES = {"absolute", "residual_relative"}

    def _apply_steering(
        alpha: float,
        v_probe_prev: Any,
        norm_mode: str,
        correction_norms: list[float] | None = None,
    ) -> None:
        """Push the current correction configuration onto the bottleneck
        hook. Dispatches by normalization mode so the generation loops
        remain a single call site regardless of which mode is active.
        """
        import mlx.core as _mx

        hook_local = ctx.bottleneck_hook
        if hook_local is None:
            return
        if alpha <= 0 or v_probe_prev is None:
            hook_local.clear_steering()
            return
        if norm_mode == "residual_relative":
            hook_local.set_probe_steering(v_probe_prev, alpha)
            if correction_norms is not None:
                # Magnitude is α * |residual_last|, not knowable ahead
                # of the forward pass; record α * |v_probe| as a proxy
                # so existing telemetry stays populated without a
                # double-eval.
                correction_norms.append(
                    float(_mx.linalg.norm(v_probe_prev).item()) * alpha
                )
        else:
            correction = alpha * v_probe_prev
            hook_local.set_raw_correction(correction)
            if correction_norms is not None:
                correction_norms.append(
                    float(_mx.linalg.norm(correction).item())
                )
    _steering_stats: dict[str, Any] = {
        "requests": 0,
        "last_probe_norm": 0.0,
        "last_correction_norm": 0.0,
        "last_alpha": 0.0,
        "last_coherence": None,
        # Mean per-token logit-distribution signals from the most recent
        # turn — feeds the autotuner alongside last_coherence.
        "last_certainty": None,
        # Finish disposition of the last turn: "stop" (EOS) or "length".
        # Half the cliff signal — sycophant collapse looks like coherent
        # short-EOS output; only finish_reason disambiguates.
        "last_finish_reason": None,
    }

    # Autotuner state. When enabled, the rule-based controller in
    # tgirl.autotune drives (α, β, temperature) from the per-turn
    # telemetry rather than honoring per-request overrides.
    _autotune_state: dict[str, Any] = {
        "enabled": False,
        "log_path": None,
        # Next-turn config the controller picked at the end of the
        # most recent turn. Initialized lazily from _steering_config
        # when autotune is first enabled.
        "next_alpha": None,
        "next_beta": None,
        "next_temperature": None,
        # Diagnostics from the most recent autotune call.
        "last_regime": None,
        "last_rationale": None,
    }

    def _resolve_alpha(request: Any) -> float:
        """Per-turn α resolution. When autotune is on it owns the knob;
        otherwise per-request override wins, then server default.
        """
        if _autotune_state["enabled"] and _autotune_state["next_alpha"] is not None:
            return float(_autotune_state["next_alpha"])
        if request.estradiol_alpha is not None:
            return float(request.estradiol_alpha)
        return float(_steering_config["alpha"])

    def _resolve_beta_skew(request: Any) -> tuple[float | None, float]:
        """Same precedence pattern for (β, skew)."""
        if _autotune_state["enabled"] and _autotune_state["next_beta"] is not None:
            return float(_autotune_state["next_beta"]), float(
                _steering_config["skew"]
            )
        beta = (
            request.estradiol_beta
            if request.estradiol_beta is not None
            else _steering_config["beta"]
        )
        skew = (
            request.estradiol_skew
            if request.estradiol_skew is not None
            else _steering_config["skew"]
        )
        return beta, skew

    def _resolve_temperature(request: Any) -> float:
        """Autotuner can override the request's temperature too — the
        third knob in the (α, β, temp) control space.
        """
        if (
            _autotune_state["enabled"]
            and _autotune_state["next_temperature"] is not None
        ):
            return float(_autotune_state["next_temperature"])
        return float(request.temperature)

    def _run_autotune_after_turn(finish_reason: str) -> None:
        """Build Observables from the just-completed turn's stats and
        run the controller. Logs (obs, action) to JSONL if configured.
        """
        if not _autotune_state["enabled"]:
            return
        coh = _steering_stats.get("last_coherence") or {}
        cert = _steering_stats.get("last_certainty") or {}
        if not coh:
            return  # nothing to learn from a turn with no telemetry

        from tgirl.autotune import (
            Observables,
            action_to_dict,
            autotune,
            observables_to_dict,
        )

        obs = Observables(
            repeat_rate=float(coh.get("repeat_rate") or 0.0),
            bigram_novelty=float(coh.get("bigram_novelty") or 1.0),
            token_entropy=float(coh.get("token_entropy") or 0.0),
            n_tokens=int(coh.get("n_tokens") or 0),
            finish_reason=finish_reason,
            mean_entropy=float(cert.get("mean_entropy") or 0.0),
            mean_top1_prob=float(cert.get("mean_top1_prob") or 1.0),
            mean_top1_margin=float(cert.get("mean_top1_margin") or 1.0),
            alpha=float(_autotune_state.get("next_alpha") or _steering_config["alpha"]),
            beta=_autotune_state.get("next_beta") or _steering_config["beta"],
            temperature=float(
                _autotune_state.get("next_temperature") or 0.0
            ),
        )
        action = autotune(obs)
        _autotune_state["next_alpha"] = action.next_alpha
        _autotune_state["next_beta"] = action.next_beta
        _autotune_state["next_temperature"] = action.next_temperature
        _autotune_state["last_regime"] = action.regime
        _autotune_state["last_rationale"] = action.rationale

        log_path = _autotune_state.get("log_path")
        if log_path:
            import json
            from pathlib import Path

            with Path(log_path).open("a") as f:
                f.write(
                    json.dumps({
                        "observables": observables_to_dict(obs),
                        "action": action_to_dict(action),
                    }) + "\n"
                )

    def _apply_band_to_hook(
        request_beta: float | None,
        request_skew: float | None,
    ) -> None:
        """Resolve per-request/server band config and push to the hook.

        Call before generation starts. A request with no explicit
        override inherits the server default; an explicit ``None`` on
        the request falls back to server default too (Pydantic can't
        distinguish "field omitted" from "field explicitly null", but
        practically the semantics are the same — use the server
        default). Server default of ``beta=None`` means single-layer
        injection (bit-compatible with the pre-band hook).

        Always called so that stale band config from a prior request
        cannot leak into the next one.
        """
        hook = ctx.bottleneck_hook
        if hook is None:
            return
        beta = (
            request_beta
            if request_beta is not None
            else _steering_config["beta"]
        )
        skew = (
            request_skew
            if request_skew is not None
            else _steering_config["skew"]
        )
        if beta is None:
            hook.set_band(None)
            return
        from tgirl.band import band_weights

        weights = band_weights(
            n_layers=len(hook._layers),
            bottleneck_idx=hook._layer_idx,
            beta=beta,
            skew=skew,
        )
        hook.set_band(weights)

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
        hook = ctx.bottleneck_hook
        alpha = _resolve_alpha(request)
        temp = _resolve_temperature(request)
        _beta_resolved, _skew_resolved = _resolve_beta_skew(request)
        _apply_band_to_hook(_beta_resolved, _skew_resolved)

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
        correction_norms: list[float] = []

        generated: list[int] = []
        token_ids = list(prompt_tokens)

        norm_mode = _steering_config["normalization"]
        # Per-token logit-distribution signals; surfaced as last_certainty
        # alongside last_coherence at turn end so the autotuner can read
        # both the model's pre-sampling confidence and its post-generation
        # output structure.
        certainty_steps: list[dict[str, float]] = []

        for _ in range(max_tok):
            # Steer: inject scaled probe from previous token (or previous turn).
            # Dispatch to absolute vs residual-relative mode.
            _apply_steering(alpha, v_probe_prev, norm_mode, correction_norms)

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

            # Record certainty BEFORE temperature scaling — raw distribution.
            from tgirl.certainty import step_certainty as _step_cert

            certainty_steps.append(_step_cert(logits))

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
                from tgirl.certainty import mean_certainty as _mean_cert
                from tgirl.coherence import compute_coherence

                _steering_stats["last_coherence"] = compute_coherence(generated)
                _steering_stats["last_certainty"] = _mean_cert(certainty_steps)
                _steering_stats["last_finish_reason"] = "stop"
                _run_autotune_after_turn("stop")
                return generated, "stop"

            generated.append(next_token)
            token_ids = token_ids + [next_token]

        _probe_cache["v_probe"] = v_probe_prev
        if correction_norms:
            _steering_stats["last_correction_norm"] = correction_norms[-1]
            _steering_stats["mean_correction_norm"] = (
                sum(correction_norms) / len(correction_norms)
            )
        from tgirl.certainty import mean_certainty as _mean_cert
        from tgirl.coherence import compute_coherence

        _steering_stats["last_coherence"] = compute_coherence(generated)
        _steering_stats["last_certainty"] = _mean_cert(certainty_steps)
        _steering_stats["last_finish_reason"] = "length"
        _run_autotune_after_turn("length")
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

    @app.post("/v1/steering/beta")
    async def set_beta(request: _FastAPIRequest) -> dict[str, Any]:
        # Accept query params (handy for curl) or a JSON body. A JSON
        # `beta: null` is how callers explicitly clear the band —
        # query params can't express null cleanly. Body wins when
        # both are present.
        body: dict[str, Any] = {}
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                body = await request.json()
            except Exception:
                body = {}

        q = request.query_params
        if "beta" in body:
            new_beta = body["beta"]
        elif "beta" in q:
            new_beta = float(q["beta"])
        else:
            new_beta = _steering_config["beta"]

        if "skew" in body:
            new_skew = body["skew"]
        elif "skew" in q:
            new_skew = float(q["skew"])
        else:
            new_skew = _steering_config["skew"]

        _steering_config["beta"] = new_beta
        if new_skew is not None:
            _steering_config["skew"] = new_skew
        logger.info(
            "beta_updated",
            beta=_steering_config["beta"],
            skew=_steering_config["skew"],
        )
        return {
            "beta": _steering_config["beta"],
            "skew": _steering_config["skew"],
        }

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

    @app.post("/v1/steering/autotune")
    async def set_autotune(request: _FastAPIRequest) -> dict[str, Any]:
        """Enable / disable the rule-based steering controller.

        Body: ``{"enabled": bool, "log_path": str | null}``.

        When enabled, the controller drives (α, β, temperature) from
        the per-turn coherence + certainty telemetry and ignores
        per-request overrides on those fields. When ``log_path`` is
        set, every (observables, action) tuple is appended to that
        path as JSONL — the training set for the perceptron that will
        eventually replace the rule-based logic.
        """
        body: dict[str, Any] = {}
        if "application/json" in request.headers.get("content-type", ""):
            try:
                body = await request.json()
            except Exception:
                body = {}
        if "enabled" in body:
            _autotune_state["enabled"] = bool(body["enabled"])
            # Initialize the controller's working knobs from the
            # current server config so the first autotune cycle has
            # a sensible baseline to adjust from.
            if _autotune_state["enabled"]:
                if _autotune_state["next_alpha"] is None:
                    _autotune_state["next_alpha"] = _steering_config["alpha"]
                if _autotune_state["next_beta"] is None:
                    _autotune_state["next_beta"] = _steering_config["beta"]
                if _autotune_state["next_temperature"] is None:
                    _autotune_state["next_temperature"] = 0.0
        if "log_path" in body:
            _autotune_state["log_path"] = body["log_path"]
        logger.info(
            "autotune_updated",
            enabled=_autotune_state["enabled"],
            log_path=_autotune_state["log_path"],
        )
        return {
            "enabled": _autotune_state["enabled"],
            "log_path": _autotune_state["log_path"],
        }

    @app.post("/v1/steering/normalization")
    async def set_normalization(request: _FastAPIRequest) -> Any:
        from fastapi.responses import JSONResponse

        body: dict[str, Any] = {}
        if "application/json" in request.headers.get("content-type", ""):
            try:
                body = await request.json()
            except Exception:
                body = {}
        mode = body.get("mode") or request.query_params.get("mode")
        if mode not in _VALID_NORM_MODES:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"unknown mode: {mode!r}. "
                    f"Must be one of: {sorted(_VALID_NORM_MODES)}."
                },
            )
        _steering_config["normalization"] = mode
        logger.info("normalization_updated", mode=mode)
        return {"mode": mode}

    @app.post("/v1/steering/probe/clear")
    async def clear_probe() -> dict[str, Any]:
        """Reset the probe cache to empty. Use between α/β configurations
        in a parameter sweep so accumulated probe state from one step
        doesn't bleed into the next."""
        _probe_cache["v_probe"] = None
        logger.info("probe_cleared")
        return {"cleared": True}

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
        # `probe_cached` and `hook_installed` are computed live from the
        # source of truth (_probe_cache / ctx.bottleneck_hook), not stored
        # in _steering_stats. Keep it that way: mirroring them into stats
        # would reintroduce a mask-the-cache bug if any reader spread
        # stats without an explicit override.
        #
        # beta/skew are live config, not turn stats — also overriden here.
        result: dict[str, Any] = {
            **_steering_stats,
            "hook_installed": ctx.bottleneck_hook is not None,
            "probe_cached": _probe_cache.get("v_probe") is not None,
            "beta": _steering_config["beta"],
            "skew": _steering_config["skew"],
            "normalization": _steering_config["normalization"],
            "autotune": dict(_autotune_state),
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
                stop_ids = set(ctx.stop_token_ids)
                token_ids = list(prompt_tokens)
                think_end = ctx.think_end_token_id
                in_thinking = request.enable_thinking and think_end is not None
                hook = ctx.bottleneck_hook
                alpha = _resolve_alpha(request)
                temp = _resolve_temperature(request)
                _beta_resolved_s, _skew_resolved_s = _resolve_beta_skew(request)
                _apply_band_to_hook(_beta_resolved_s, _skew_resolved_s)
                v_probe_prev = _probe_cache.get("v_probe")
                generated_tokens: list[int] = []
                _steering_stats["requests"] += 1
                _steering_stats["last_alpha"] = alpha

                if request.seed is not None:
                    mx.random.seed(request.seed)

                norm_mode = _steering_config["normalization"]
                certainty_steps_s: list[dict[str, float]] = []

                for _ in range(max_tok):
                    # Probe feedback steering (absolute or residual-relative).
                    _apply_steering(alpha, v_probe_prev, norm_mode)

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

                    from tgirl.certainty import step_certainty as _step_cert_s

                    certainty_steps_s.append(_step_cert_s(logits))

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
                        from tgirl.certainty import mean_certainty as _mean_cert_s
                        from tgirl.coherence import compute_coherence

                        _steering_stats["last_coherence"] = compute_coherence(
                            generated_tokens
                        )
                        _steering_stats["last_certainty"] = _mean_cert_s(
                            certainty_steps_s
                        )
                        _steering_stats["last_finish_reason"] = "stop"
                        _run_autotune_after_turn("stop")
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
                    from tgirl.certainty import mean_certainty as _mean_cert_s
                    from tgirl.coherence import compute_coherence

                    _steering_stats["last_coherence"] = compute_coherence(
                        generated_tokens
                    )
                    _steering_stats["last_certainty"] = _mean_cert_s(
                        certainty_steps_s
                    )
                    _steering_stats["last_finish_reason"] = "length"
                    _run_autotune_after_turn("length")
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
