"""Constrained sampling engine for dual-mode inference.

Integrates grammar constraints, optimal transport redistribution,
and Hy pipeline execution into a dual-mode sampling loop.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tgirl.registry import ToolRegistry
    from tgirl.types import PromptFormatter

import structlog
import torch
from pydantic import BaseModel, ConfigDict

from tgirl.transport import TransportConfig, redistribute_logits
from tgirl.types import (
    ModelIntervention,
    PipelineError,
    RegistrySnapshot,
    RerankConfig,
    SessionConfig,
    TelemetryRecord,
)

logger = structlog.get_logger()


@runtime_checkable
class GrammarState(Protocol):
    """Protocol for grammar state trackers."""

    def get_valid_mask(self, tokenizer_vocab_size: int) -> torch.Tensor: ...
    def is_accepting(self) -> bool: ...
    def advance(self, token_id: int) -> None: ...


@runtime_checkable
class InferenceHook(Protocol):
    """Protocol for per-token inference hooks."""

    def pre_forward(
        self,
        position: int,
        grammar_state: GrammarState,
        token_history: list[int],
        logits: torch.Tensor,
    ) -> ModelIntervention: ...


def merge_interventions(interventions: list[ModelIntervention]) -> ModelIntervention:
    """Merge multiple hook interventions. Last non-None value wins per field."""
    merged: dict[str, object] = {}
    for intervention in interventions:
        for field_name in ModelIntervention.model_fields:
            val = getattr(intervention, field_name)
            if val is not None:
                merged[field_name] = val
    return ModelIntervention(**merged)


class GrammarTemperatureHook:
    """Default hook: grammar-implied temperature scheduling.

    Adjusts temperature based on the fraction of valid tokens (freedom).
    When few tokens are valid, temperature is low (more deterministic).
    When many tokens are valid, temperature approaches base_temperature.
    """

    def __init__(
        self, base_temperature: float = 0.3, scaling_exponent: float = 0.5
    ) -> None:
        self.base_temperature = base_temperature
        self.scaling_exponent = scaling_exponent

    def pre_forward(
        self,
        position: int,
        grammar_state: GrammarState,
        token_history: list[int],
        logits: torch.Tensor,
    ) -> ModelIntervention:
        vocab_size = logits.shape[-1]
        valid_mask = grammar_state.get_valid_mask(vocab_size)
        valid_count = valid_mask.sum().item()
        if valid_count <= 1:
            return ModelIntervention(temperature=0.0)
        freedom = valid_count / vocab_size
        temp = self.base_temperature * (freedom**self.scaling_exponent)
        return ModelIntervention(temperature=temp)


def apply_penalties(
    logits: torch.Tensor,
    intervention: ModelIntervention,
    token_history: list[int],
) -> torch.Tensor:
    """Pre-OT: apply repetition, presence, frequency penalties and logit bias.

    These modify the model's original logit distribution before OT redistribution,
    so that OT operates on a penalty-adjusted signal.
    """
    result = logits.clone()

    # Repetition penalty
    if (
        intervention.repetition_penalty is not None
        and intervention.repetition_penalty != 1.0
    ):
        for token_id in set(token_history):
            if result[token_id] > 0:
                result[token_id] /= intervention.repetition_penalty
            else:
                result[token_id] *= intervention.repetition_penalty

    # Presence penalty
    if (
        intervention.presence_penalty is not None
        and intervention.presence_penalty != 0.0
    ):
        for token_id in set(token_history):
            result[token_id] -= intervention.presence_penalty

    # Frequency penalty
    if (
        intervention.frequency_penalty is not None
        and intervention.frequency_penalty != 0.0
    ):
        counts = Counter(token_history)
        for token_id, count in counts.items():
            result[token_id] -= intervention.frequency_penalty * count

    # Logit bias
    if intervention.logit_bias is not None:
        for token_id, bias in intervention.logit_bias.items():
            result[token_id] += bias

    return result


def apply_shaping(
    logits: torch.Tensor,
    intervention: ModelIntervention,
) -> torch.Tensor:
    """Post-OT: apply temperature, top-k, top-p to redistributed logits."""
    result = logits.clone()

    # Temperature
    if intervention.temperature is not None and intervention.temperature > 0:
        result = result / intervention.temperature
    elif intervention.temperature is not None and intervention.temperature == 0:
        # Greedy: set all but max to -inf. torch.argmax returns the first
        # (lowest-index) maximum, which is deterministic and consistent.
        max_idx = result.argmax()
        mask = torch.ones_like(result, dtype=torch.bool)
        mask[max_idx] = False
        result[mask] = float("-inf")

    # Top-k
    if intervention.top_k is not None and intervention.top_k > 0:
        top_k_vals, _ = torch.topk(result, min(intervention.top_k, result.shape[-1]))
        threshold = top_k_vals[-1]
        result[result < threshold] = float("-inf")

    # Top-p (nucleus)
    if intervention.top_p is not None and intervention.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(result, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff_mask = cumulative - probs > intervention.top_p
        sorted_logits[cutoff_mask] = float("-inf")
        # Unsort: scatter modified sorted values back to original positions
        result = torch.empty_like(sorted_logits)
        result.scatter_(0, sorted_indices, sorted_logits)

    return result


class DelimiterDetector:
    """Detects tool call delimiters in generated token stream.

    Maintains a sliding window of decoded text rather than accumulating
    all token IDs. The window is bounded to 2x the delimiter's character
    length, which is sufficient to detect any delimiter that spans a
    token boundary.
    """

    def __init__(
        self,
        delimiter: str,
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self.delimiter = delimiter
        self.decode = tokenizer_decode
        self._decoded_window: str = ""
        self._max_window = len(delimiter) * 2

    def feed(self, token_id: int) -> bool:
        """Feed a token. Returns True if delimiter is detected."""
        new_text = self.decode([token_id])
        self._decoded_window += new_text
        if len(self._decoded_window) > self._max_window:
            self._decoded_window = self._decoded_window[-self._max_window :]
        return self.delimiter in self._decoded_window

    def reset(self) -> None:
        """Clear the detection window."""
        self._decoded_window = ""


class ConstrainedGenerationResult(BaseModel):
    """Result of a constrained generation pass."""

    model_config = ConfigDict(frozen=True)
    tokens: list[int]
    hy_source: str
    grammar_valid_counts: list[int]
    temperatures_applied: list[float]
    wasserstein_distances: list[float]
    top_p_applied: list[float]
    token_log_probs: list[float]
    ot_computation_total_ms: float
    ot_bypassed_count: int
    ot_bypass_reasons: list[str | None]
    ot_iterations: list[int]
    grammar_generation_ms: float


def run_constrained_generation(
    grammar_state: GrammarState,
    forward_fn: Callable[[list[int]], torch.Tensor],
    tokenizer_decode: Callable[[list[int]], str],
    embeddings: torch.Tensor,
    hooks: list[InferenceHook],
    transport_config: TransportConfig,
    max_tokens: int = 512,
    context_tokens: list[int] | None = None,
) -> ConstrainedGenerationResult:
    """Run constrained token generation until grammar accepts or max_tokens.

    Per-token processing order (TGIRL.md 8.5, penalties moved pre-OT):
    1. forward_fn(context_tokens) -> raw logits
    2. grammar_state.get_valid_mask(vocab_size) -> valid_mask
    3. call all InferenceHooks -> merge_interventions() -> merged
    4. apply_penalties(logits, intervention, token_history) -> adjusted
    5. redistribute_logits(adjusted, valid_mask, embeddings, config) -> OT
    6. apply_shaping(ot_logits, intervention) -> shaped logits
    7. sample token from shaped logits
    8. grammar_state.advance(token_id)
    9. record telemetry
    """
    start_time = time.monotonic()
    tokens: list[int] = []
    token_history = list(context_tokens) if context_tokens else []
    grammar_valid_counts: list[int] = []
    temperatures_applied: list[float] = []
    wasserstein_distances: list[float] = []
    top_p_applied: list[float] = []
    token_log_probs: list[float] = []
    ot_computation_total_ms = 0.0
    ot_bypassed_count = 0
    ot_bypass_reasons: list[str | None] = []
    ot_iterations: list[int] = []

    vocab_size = embeddings.shape[0]

    for position in range(max_tokens):
        # 1. Forward pass
        raw_logits = forward_fn(token_history)

        # 2. Grammar mask
        valid_mask = grammar_state.get_valid_mask(vocab_size)
        valid_count = int(valid_mask.sum().item())
        grammar_valid_counts.append(valid_count)

        # 3. Hooks
        interventions = [
            hook.pre_forward(position, grammar_state, token_history, raw_logits)
            for hook in hooks
        ]
        merged = merge_interventions(interventions)

        # 4. Pre-OT penalties
        adjusted = apply_penalties(raw_logits, merged, token_history)

        # 5. OT redistribution
        ot_start = time.monotonic()
        ot_result = redistribute_logits(
            adjusted, valid_mask, embeddings, config=transport_config
        )
        ot_elapsed_ms = (time.monotonic() - ot_start) * 1000
        ot_computation_total_ms += ot_elapsed_ms
        wasserstein_distances.append(ot_result.wasserstein_distance)
        ot_bypass_reasons.append(ot_result.bypass_reason)
        ot_iterations.append(ot_result.iterations)
        if ot_result.bypassed:
            ot_bypassed_count += 1

        # 6. Post-OT shaping
        shaped = apply_shaping(ot_result.logits, merged)

        # Record temperature and top_p applied
        temperatures_applied.append(
            merged.temperature if merged.temperature is not None else -1.0
        )
        top_p_applied.append(merged.top_p if merged.top_p is not None else -1.0)

        # 7. Sample token
        probs = torch.softmax(shaped, dim=-1)
        probs = torch.clamp(probs, min=0.0)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            # Fallback: uniform over valid tokens
            probs = valid_mask.float()
            probs = probs / probs.sum()

        token_id = int(torch.multinomial(probs, 1).item())
        tokens.append(token_id)
        token_history.append(token_id)

        # Record log prob
        log_prob = (
            torch.log(probs[token_id]).item() if probs[token_id] > 0 else float("-inf")
        )
        token_log_probs.append(log_prob)

        # 8. Advance grammar
        grammar_state.advance(token_id)

        # Check if grammar accepts
        if grammar_state.is_accepting():
            break

    elapsed_ms = (time.monotonic() - start_time) * 1000
    hy_source = tokenizer_decode(tokens)

    return ConstrainedGenerationResult(
        tokens=tokens,
        hy_source=hy_source,
        grammar_valid_counts=grammar_valid_counts,
        temperatures_applied=temperatures_applied,
        wasserstein_distances=wasserstein_distances,
        top_p_applied=top_p_applied,
        token_log_probs=token_log_probs,
        ot_computation_total_ms=ot_computation_total_ms,
        ot_bypassed_count=ot_bypassed_count,
        ot_bypass_reasons=ot_bypass_reasons,
        ot_iterations=ot_iterations,
        grammar_generation_ms=elapsed_ms,
    )


class ToolCallRecord(BaseModel):
    """Record of a single tool call cycle."""

    model_config = ConfigDict(frozen=True)
    pipeline: str  # Hy source
    result: Any = None
    error: PipelineError | None = None
    cycle_number: int
    tool_invocations: dict[str, int]


class SamplingResult(BaseModel):
    """Result of a complete dual-mode sampling session."""

    model_config = ConfigDict(frozen=True)
    output_text: str
    tool_calls: list[ToolCallRecord]
    telemetry: list[TelemetryRecord]
    total_tokens: int
    total_cycles: int
    wall_time_ms: float
    quotas_consumed: dict[str, int]


class SamplingSession:
    """Orchestrates the dual-mode sampling loop.

    Generates tokens in freeform mode, detects tool delimiters,
    switches to constrained mode for Hy pipeline generation,
    executes pipelines, and returns to freeform mode.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        forward_fn: Callable[[list[int]], Any],
        tokenizer_decode: Callable[[list[int]], str],
        tokenizer_encode: Callable[[str], list[int]],
        embeddings: torch.Tensor | Any,
        grammar_guide_factory: Callable[[str], GrammarState],
        config: SessionConfig | None = None,
        hooks: list[InferenceHook] | None = None,
        transport_config: TransportConfig | None = None,
        rerank_config: RerankConfig | None = None,
        formatter: PromptFormatter | None = None,
        backend: Literal["torch", "mlx", "auto"] = "auto",
        mlx_grammar_guide_factory: Callable | None = None,
        transition_policy: Any | None = None,
    ) -> None:
        from tgirl.rerank import ToolRouter
        from tgirl.state_machine import DelimiterTransitionPolicy

        self._registry = registry
        self._forward_fn = forward_fn
        self._decode = tokenizer_decode
        self._encode = tokenizer_encode
        self._embeddings = embeddings
        self._grammar_guide_factory = grammar_guide_factory
        self._mlx_grammar_guide_factory = mlx_grammar_guide_factory
        self._config = config or SessionConfig()
        self._hooks = hooks or []
        self._transport_config = transport_config or TransportConfig()
        self._consumed_quotas: dict[str, int] = {}
        self._rerank_config = rerank_config
        self._formatter = formatter
        self._last_user_content: str | None = None
        self._backend = backend
        self._transition_policy = transition_policy or DelimiterTransitionPolicy(
            delimiter=self._config.tool_open_delimiter,
            tokenizer_decode=self._decode,
        )
        # Set on first forward call if backend="auto"
        self._is_mlx: bool | None = (
            True if backend == "mlx"
            else False if backend == "torch"
            else None
        )
        self._embeddings_mlx: Any = None  # Lazy-converted on first MLX use
        self._mlx_hooks: list | None = None  # Lazy-converted on first MLX use
        # Map session backend to router backend ("auto" -> "torch" default)
        _router_backend = "mlx" if backend == "mlx" else "torch"
        self._router = (
            ToolRouter(
                grammar_guide_factory=grammar_guide_factory,
                forward_fn=forward_fn,
                tokenizer_decode=tokenizer_decode,
                embeddings=embeddings,
                config=rerank_config,
                backend=_router_backend,
            )
            if rerank_config is not None
            else None
        )

    def run_chat(self, messages: list[dict[str, str]]) -> SamplingResult:
        """Format messages and run the dual-mode sampling loop.

        Generates a system prompt from the registry snapshot, prepends it,
        formats via the configured PromptFormatter, encodes, and delegates
        to run(). Callers should not include their own system message —
        run_chat() always prepends the registry-generated system prompt.

        Args:
            messages: Chat messages (role/content dicts). Should not
                include a system message; one will be generated automatically.

        Returns:
            SamplingResult from the sampling loop.

        Raises:
            ValueError: If no formatter is configured.
        """
        if self._formatter is None:
            msg = "run_chat() requires a formatter — pass formatter= to SamplingSession"
            raise ValueError(msg)

        from tgirl.instructions import generate_system_prompt

        snapshot = self._registry.snapshot(
            cost_budget=self._config.session_cost_budget
        )
        system_prompt = generate_system_prompt(
            snapshot,
            tool_open=self._config.tool_open_delimiter,
            tool_close=self._config.tool_close_delimiter,
        )

        # Store last user message content for routing context
        for msg_item in reversed(messages):
            if msg_item.get("role") == "user":
                self._last_user_content = msg_item["content"]
                break

        # Prepend system message
        full_messages = [{"role": "system", "content": system_prompt}] + list(messages)

        # Format and encode
        formatted = self._formatter.format_messages(full_messages)
        prompt_tokens = self._encode(formatted)

        return self.run(prompt_tokens)

    def run(self, prompt_tokens: list[int]) -> SamplingResult:
        """Run the full dual-mode sampling loop.

        Note: _last_user_content is consumed and cleared here. It is only
        valid when set by run_chat() immediately before this call.
        """
        # Capture and clear routing state to prevent stale leaks
        last_user_content = self._last_user_content
        self._last_user_content = None

        # Deferred imports to break circular dependency at module level
        from tgirl.compile import run_pipeline
        from tgirl.grammar import generate as generate_grammar

        start_time = time.monotonic()
        token_history = list(prompt_tokens)
        output_parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        telemetry_records: list[TelemetryRecord] = []
        total_tokens = 0
        cycle_count = 0

        from tgirl.state_machine import SessionState, TransitionSignal

        # Reset transition policy for fresh run
        if hasattr(self._transition_policy, "reset"):
            self._transition_policy.reset()

        for _ in range(self._config.max_tool_cycles + 1):
            # --- Freeform mode ---
            freeform_tokens: list[int] = []
            delimiter_found = False
            timed_out = False

            for freeform_pos in range(self._config.freeform_max_tokens):
                # Check timeout
                elapsed = (time.monotonic() - start_time) * 1000
                if elapsed > self._config.session_timeout * 1000:
                    timed_out = True
                    break

                _t0 = time.monotonic()
                raw_logits = self._forward_fn(token_history)
                _t_forward = time.monotonic()

                # Auto-detect backend on first forward call
                if self._is_mlx is None:
                    try:
                        import mlx.core as mx
                        self._is_mlx = isinstance(raw_logits, mx.array)
                    except ImportError:
                        self._is_mlx = False

                if self._is_mlx:
                    import mlx.core as mx
                    logits = raw_logits / max(
                        self._config.freeform_temperature, 1e-10
                    )
                    _t_div = time.monotonic()
                    probs_check = mx.softmax(logits, axis=-1)
                    prob_sum = float(mx.sum(probs_check).item())
                    _t_softmax = time.monotonic()
                    if prob_sum > 0:
                        token_id = int(
                            mx.random.categorical(logits).item()
                        )
                    else:
                        # Fallback: uniform over vocab
                        n = logits.shape[0]
                        token_id = int(
                            mx.random.categorical(
                                mx.zeros((n,))
                            ).item()
                        )
                    _t_sample = time.monotonic()
                    if total_tokens < 5 or total_tokens % 50 == 0:
                        logger.info(
                            "freeform_mlx_timing",
                            token=total_tokens,
                            forward_ms=round((_t_forward - _t0) * 1000, 1),
                            div_ms=round((_t_div - _t_forward) * 1000, 1),
                            softmax_ms=round((_t_softmax - _t_div) * 1000, 1),
                            sample_ms=round((_t_sample - _t_softmax) * 1000, 1),
                            total_ms=round((_t_sample - _t0) * 1000, 1),
                        )
                else:
                    # Torch path (unchanged)
                    logits = raw_logits / max(
                        self._config.freeform_temperature, 1e-10
                    )
                    probs = torch.softmax(logits, dim=-1)
                    probs = torch.clamp(probs, min=0.0)
                    prob_sum = probs.sum()
                    if prob_sum > 0:
                        probs = probs / prob_sum
                    token_id = int(
                        torch.multinomial(probs, 1).item()
                    )

                freeform_tokens.append(token_id)
                token_history.append(token_id)
                total_tokens += 1

                # Evaluate transition policy
                signal = TransitionSignal(
                    token_position=freeform_pos,
                    grammar_mask_overlap=0.0,
                    token_entropy=0.0,
                    token_log_prob=0.0,
                    grammar_freedom=0.0,
                )
                decision = self._transition_policy.evaluate(
                    SessionState.FREEFORM,
                    signal,
                    token_id=token_id,
                )
                if decision.should_transition:
                    delimiter_found = True
                    break

            if freeform_tokens:
                output_parts.append(self._decode(freeform_tokens))

            if timed_out or not delimiter_found:
                break  # No tool call or timeout, session ends

            if cycle_count >= self._config.max_tool_cycles:
                break

            # --- Constrained mode ---
            cycle_count += 1
            freeform_count_before = len(freeform_tokens)
            if hasattr(self._transition_policy, "reset"):
                self._transition_policy.reset()

            # Generate grammar from snapshot with reduced quotas
            snapshot = self._snapshot_with_remaining_quotas()

            # Optional re-ranking pass
            rerank_active = (
                self._router is not None
                and self._rerank_config
                and self._rerank_config.enabled
            )
            if rerank_active:
                # Build routing context tokens
                if self._formatter is not None and last_user_content is not None:
                    from tgirl.instructions import generate_routing_prompt

                    routing_prompt = generate_routing_prompt(snapshot)
                    routing_messages = [
                        {"role": "system", "content": routing_prompt},
                        {"role": "user", "content": last_user_content},
                    ]
                    routing_context = self._encode(
                        self._formatter.format_messages(routing_messages)
                    )
                else:
                    routing_context = token_history

                rerank_result = self._router.route(
                    snapshot=snapshot,
                    context_tokens=routing_context,
                    transport_config=self._transport_config,
                )
                total_tokens += rerank_result.routing_tokens
                # Restrict the EXISTING quota-adjusted snapshot
                selected = frozenset(rerank_result.selected_tools)
                snapshot = RegistrySnapshot(
                    tools=tuple(t for t in snapshot.tools if t.name in selected),
                    quotas={k: v for k, v in snapshot.quotas.items() if k in selected},
                    cost_remaining=snapshot.cost_remaining,
                    scopes=snapshot.scopes,
                    timestamp=snapshot.timestamp,
                    type_grammars=snapshot.type_grammars,
                )

            grammar_output = generate_grammar(snapshot)
            grammar_state = self._grammar_guide_factory(grammar_output.text)

            # Run constrained generation (dispatch by backend)
            if self._is_mlx:
                from tgirl.sample_mlx import (
                    GrammarTemperatureHookMlx,
                    run_constrained_generation_mlx,
                )

                # Lazy-convert embeddings to mx.array once
                if self._embeddings_mlx is None:
                    import mlx.core as mx
                    if isinstance(self._embeddings, torch.Tensor):
                        self._embeddings_mlx = mx.array(
                            self._embeddings.numpy()
                        )
                    else:
                        self._embeddings_mlx = self._embeddings

                # Lazy-convert hooks to MLX versions
                if self._mlx_hooks is None:
                    self._mlx_hooks = []
                    for hook in self._hooks:
                        if isinstance(hook, GrammarTemperatureHook):
                            self._mlx_hooks.append(
                                GrammarTemperatureHookMlx(
                                    base_temperature=hook.base_temperature,
                                    scaling_exponent=hook.scaling_exponent,
                                )
                            )
                        # Skip torch-only hooks — they can't operate on mx.array

                # Use MLX grammar factory if available, else fallback
                if self._mlx_grammar_guide_factory is not None:
                    mlx_grammar_state = self._mlx_grammar_guide_factory(
                        grammar_output.text
                    )
                else:
                    mlx_grammar_state = grammar_state

                gen_result = run_constrained_generation_mlx(
                    grammar_state=mlx_grammar_state,
                    forward_fn=self._forward_fn,
                    tokenizer_decode=self._decode,
                    embeddings=self._embeddings_mlx,
                    hooks=self._mlx_hooks,
                    transport_config=self._transport_config,
                    max_tokens=self._config.constrained_max_tokens,
                    context_tokens=token_history,
                )
            else:
                gen_result = run_constrained_generation(
                    grammar_state=grammar_state,
                    forward_fn=self._forward_fn,
                    tokenizer_decode=self._decode,
                    embeddings=self._embeddings,
                    hooks=self._hooks,
                    transport_config=self._transport_config,
                    max_tokens=self._config.constrained_max_tokens,
                    context_tokens=token_history,
                )

            token_history.extend(gen_result.tokens)
            total_tokens += len(gen_result.tokens)

            # Count tool invocations
            tool_names = set(self._registry.names())
            invocations = self._count_tool_invocations(gen_result.hy_source, tool_names)

            # Update consumed quotas
            for name, count in invocations.items():
                self._consumed_quotas[name] = self._consumed_quotas.get(name, 0) + count

            # Execute the pipeline
            pipeline_result = run_pipeline(gen_result.hy_source, self._registry)

            error = None
            result_val = None
            if isinstance(pipeline_result, PipelineError):
                error = pipeline_result
            else:
                result_val = getattr(pipeline_result, "result", pipeline_result)

            record = ToolCallRecord(
                pipeline=gen_result.hy_source,
                result=result_val,
                error=error,
                cycle_number=cycle_count,
                tool_invocations=invocations,
            )
            tool_calls.append(record)

            # Build TelemetryRecord for this cycle (B2 fix)
            cycle_wall_ms = (time.monotonic() - start_time) * 1000
            import hashlib

            snap_hash = hashlib.sha256(snapshot.model_dump_json().encode()).hexdigest()[
                :16
            ]
            telemetry_records.append(
                TelemetryRecord(
                    pipeline_id=f"cycle-{cycle_count}",
                    tokens=gen_result.tokens,
                    grammar_valid_counts=gen_result.grammar_valid_counts,
                    temperatures_applied=gen_result.temperatures_applied,
                    wasserstein_distances=gen_result.wasserstein_distances,
                    top_p_applied=gen_result.top_p_applied,
                    token_log_probs=gen_result.token_log_probs,
                    grammar_generation_ms=gen_result.grammar_generation_ms,
                    ot_computation_total_ms=gen_result.ot_computation_total_ms,
                    ot_bypassed_count=gen_result.ot_bypassed_count,
                    ot_bypass_reasons=gen_result.ot_bypass_reasons,
                    ot_iterations=gen_result.ot_iterations,
                    hy_source=gen_result.hy_source,
                    execution_result=result_val,
                    execution_error=error,
                    cycle_number=cycle_count,
                    freeform_tokens_before=freeform_count_before,
                    wall_time_ms=cycle_wall_ms,
                    total_tokens=total_tokens,
                    model_id="unknown",
                    registry_snapshot_hash=snap_hash,
                    rerank_selected_tool=(
                        rerank_result.selected_tools[0] if rerank_active else None
                    ),
                    rerank_routing_tokens=(
                        rerank_result.routing_tokens if rerank_active else None
                    ),
                    rerank_latency_ms=(
                        rerank_result.routing_latency_ms if rerank_active else None
                    ),
                )
            )

            # Inject result into context
            result_text = (
                f"{self._config.result_open_delimiter}"
                f"{result_val}"
                f"{self._config.result_close_delimiter}"
            )
            result_tokens = self._encode(result_text)
            token_history.extend(result_tokens)
            total_tokens += len(result_tokens)
            output_parts.append(result_text)

        wall_time_ms = (time.monotonic() - start_time) * 1000

        return SamplingResult(
            output_text="".join(output_parts),
            tool_calls=tool_calls,
            telemetry=telemetry_records,
            total_tokens=total_tokens,
            total_cycles=cycle_count,
            wall_time_ms=wall_time_ms,
            quotas_consumed=dict(self._consumed_quotas),
        )

    def _snapshot_with_remaining_quotas(self) -> RegistrySnapshot:
        """Produce a snapshot with quotas reduced by consumed counts.

        Tools that have exhausted their quota will have quota=0 in the
        snapshot, making them inexpressible in the grammar for subsequent
        cycles (TGIRL.md 3.3 safety by construction).
        """
        base = self._registry.snapshot(cost_budget=self._config.session_cost_budget)
        remaining_quotas = {
            name: max(0, limit - self._consumed_quotas.get(name, 0))
            for name, limit in base.quotas.items()
        }
        return RegistrySnapshot(
            tools=base.tools,
            quotas=remaining_quotas,
            cost_remaining=base.cost_remaining,
            scopes=base.scopes,
            timestamp=base.timestamp,
        )

    def _count_tool_invocations(
        self, hy_source: str, tool_names: set[str]
    ) -> dict[str, int]:
        """Count tool invocations in Hy source.

        Matches tool names appearing as:
        - Direct calls: (tool_name ...)
        - Bare symbols in threading: (-> ... (tool_name) ...)
        - Bare symbols in threading position: (-> ... tool_name)

        Uses word-boundary matching to avoid false positives on
        tool names that are substrings of other identifiers.
        """
        counts: dict[str, int] = {}
        for name in tool_names:
            # Match after ( with optional whitespace, OR as bare symbol
            # preceded by whitespace (threading position)
            pattern = rf"(?:\(\s*{re.escape(name)}\b|(?<=\s){re.escape(name)}\b)"
            matches = re.findall(pattern, hy_source)
            if matches:
                counts[name] = len(matches)
        return counts
