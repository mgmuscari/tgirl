"""Constrained sampling engine for dual-mode inference.

Integrates grammar constraints, optimal transport redistribution,
and Hy pipeline execution into a dual-mode sampling loop.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import structlog
import torch
from pydantic import BaseModel, ConfigDict

from tgirl.transport import TransportConfig, redistribute_logits
from tgirl.types import ModelIntervention, PipelineError, SessionConfig

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
        if ot_result.bypassed:
            ot_bypassed_count += 1

        # 6. Post-OT shaping
        shaped = apply_shaping(ot_result.logits, merged)

        # Record temperature and top_p applied
        temperatures_applied.append(
            merged.temperature if merged.temperature is not None else -1.0
        )
        top_p_applied.append(
            merged.top_p if merged.top_p is not None else -1.0
        )

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
            torch.log(probs[token_id]).item()
            if probs[token_id] > 0
            else float("-inf")
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
    telemetry: list[dict[str, Any]]
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
        registry: object,  # ToolRegistry
        forward_fn: Callable[[list[int]], torch.Tensor],
        tokenizer_decode: Callable[[list[int]], str],
        tokenizer_encode: Callable[[str], list[int]],
        embeddings: torch.Tensor,
        grammar_guide_factory: Callable[[str], GrammarState],
        config: SessionConfig | None = None,
        hooks: list[InferenceHook] | None = None,
        transport_config: TransportConfig | None = None,
    ) -> None:
        self._registry = registry
        self._forward_fn = forward_fn
        self._decode = tokenizer_decode
        self._encode = tokenizer_encode
        self._embeddings = embeddings
        self._grammar_guide_factory = grammar_guide_factory
        self._config = config or SessionConfig()
        self._hooks = hooks or []
        self._transport_config = transport_config or TransportConfig()
        self._consumed_quotas: dict[str, int] = {}

    def run(self, prompt_tokens: list[int]) -> SamplingResult:
        """Run the full dual-mode sampling loop."""
        start_time = time.monotonic()
        token_history = list(prompt_tokens)
        output_parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        telemetry: list[dict[str, Any]] = []
        total_tokens = 0
        cycle_count = 0

        # Set up delimiter detector
        open_detector = DelimiterDetector(
            self._config.tool_open_delimiter, self._decode
        )

        for _ in range(self._config.max_tool_cycles + 1):
            # --- Freeform mode ---
            freeform_tokens: list[int] = []
            delimiter_found = False

            for _ in range(self._config.freeform_max_tokens):
                # Check timeout
                elapsed = (time.monotonic() - start_time) * 1000
                if elapsed > self._config.session_timeout * 1000:
                    break

                raw_logits = self._forward_fn(token_history)
                # Simple freeform sampling with config temperature
                logits = raw_logits / max(self._config.freeform_temperature, 1e-10)
                probs = torch.softmax(logits, dim=-1)
                probs = torch.clamp(probs, min=0.0)
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum

                token_id = int(torch.multinomial(probs, 1).item())
                freeform_tokens.append(token_id)
                token_history.append(token_id)
                total_tokens += 1

                if open_detector.feed(token_id):
                    delimiter_found = True
                    break

            if freeform_tokens:
                output_parts.append(self._decode(freeform_tokens))

            if not delimiter_found:
                break  # No tool call, session ends

            if cycle_count >= self._config.max_tool_cycles:
                break

            # --- Constrained mode ---
            cycle_count += 1
            open_detector.reset()

            # Generate grammar from registry snapshot
            from tgirl.grammar import generate as generate_grammar

            snapshot = self._registry.snapshot(
                cost_budget=self._config.session_cost_budget
            )
            grammar_output = generate_grammar(snapshot)
            grammar_state = self._grammar_guide_factory(grammar_output.text)

            # Run constrained generation
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
            invocations = self._count_tool_invocations(
                gen_result.hy_source, tool_names
            )

            # Update consumed quotas
            for name, count in invocations.items():
                self._consumed_quotas[name] = (
                    self._consumed_quotas.get(name, 0) + count
                )

            # Execute the pipeline
            from tgirl.compile import run_pipeline

            pipeline_result = run_pipeline(
                gen_result.hy_source, self._registry
            )

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
            telemetry=telemetry,
            total_tokens=total_tokens,
            total_cycles=cycle_count,
            wall_time_ms=wall_time_ms,
            quotas_consumed=dict(self._consumed_quotas),
        )

    def _count_tool_invocations(
        self, hy_source: str, tool_names: set[str]
    ) -> dict[str, int]:
        """Count tool invocations in Hy source via simple pattern matching.

        Counts occurrences of registered tool names that appear as function
        calls (after open paren or in threading position).
        """
        counts: dict[str, int] = {}
        # Match tool names that appear after ( or as bare symbols
        for name in tool_names:
            # Count occurrences of the tool name as a symbol
            pattern = rf"\(\s*{re.escape(name)}\b"
            matches = re.findall(pattern, hy_source)
            if matches:
                counts[name] = len(matches)
        return counts
