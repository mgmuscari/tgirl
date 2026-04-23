"""Grammar-constrained tool re-ranking.

Routes user requests to the best tool via a short grammar-constrained
generation pass that only accepts tool names.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import structlog
import torch

from tgirl.grammar import generate_routing_grammar
from tgirl.sample import GrammarState, run_constrained_generation
from tgirl.transport import TransportConfig
from tgirl.types import RegistrySnapshot, RerankConfig, RerankResult

logger = structlog.get_logger()


class ToolRouter:
    """Routes user requests to the best tool via grammar-constrained re-ranking."""

    def __init__(
        self,
        grammar_guide_factory: Callable[[str], GrammarState],
        forward_fn: Callable[[list[int]], Any],
        tokenizer_decode: Callable[[list[int]], str],
        embeddings: torch.Tensor | Any,
        config: RerankConfig | None = None,
        backend: Literal["torch", "mlx"] = "torch",
    ) -> None:
        self._grammar_guide_factory = grammar_guide_factory
        self._forward_fn = forward_fn
        self._tokenizer_decode = tokenizer_decode
        self._embeddings = embeddings
        self._config = config or RerankConfig()
        self._backend = backend
        self._embeddings_mlx: Any = None  # Lazy-converted on first MLX route
        # Cache compiled routing grammar states, keyed on sorted tool names.
        # Stores grammar TEXT (not mutable GrammarState objects) to allow
        # fresh GrammarState creation per route() call.
        self._routing_grammar_cache: dict[tuple[str | int, ...], str] = {}

    def route(
        self,
        snapshot: RegistrySnapshot,
        context_tokens: list[int],
        transport_config: TransportConfig | None = None,
    ) -> RerankResult:
        """Run grammar-constrained re-ranking to select the best tool.

        1. Filter quota-exhausted tools
        2. Short-circuit if 0 or 1 tools remain
        3. Generate routing grammar (caller provides context_tokens)
        4. Run short constrained generation pass
        5. Parse selected tool name from output
        6. Return RerankResult
        """
        # If disabled, return all tools
        if not self._config.enabled:
            return RerankResult(
                selected_tools=tuple(t.name for t in snapshot.tools),
                routing_tokens=0,
                routing_latency_ms=0.0,
                routing_grammar_text="",
            )

        start_time = time.monotonic()

        # Step 1: Filter quota-exhausted tools
        filtered_tools = tuple(
            t
            for t in snapshot.tools
            if t.name not in snapshot.quotas or snapshot.quotas[t.name] > 0
        )
        filtered_snapshot = RegistrySnapshot(
            tools=filtered_tools,
            quotas={
                k: v
                for k, v in snapshot.quotas.items()
                if k in {t.name for t in filtered_tools}
            },
            cost_remaining=snapshot.cost_remaining,
            scopes=snapshot.scopes,
            timestamp=snapshot.timestamp,
            type_grammars=snapshot.type_grammars,
        )

        # Step 2: Short-circuit for 0 or 1 tools
        if len(filtered_tools) == 0:
            # Will raise ValueError from generate_routing_grammar
            generate_routing_grammar(filtered_snapshot)

        if len(filtered_tools) == 1:
            latency_ms = (time.monotonic() - start_time) * 1000
            return RerankResult(
                selected_tools=(filtered_tools[0].name,),
                routing_tokens=0,
                routing_latency_ms=latency_ms,
                routing_grammar_text="",
            )

        # Step 3: Use context_tokens as-is (caller provides routing context)
        routing_context_tokens = list(context_tokens)

        # Step 4: Get routing grammar text (with caching)
        top_k = self._config.top_k
        cache_key: tuple[str, ...] = tuple(sorted(t.name for t in filtered_tools))
        # Include top_k in cache key so different top_k values don't collide
        grammar_cache_key: tuple[str | int, ...] = (*cache_key, top_k)
        if grammar_cache_key in self._routing_grammar_cache:
            routing_grammar_text = self._routing_grammar_cache[grammar_cache_key]
        else:
            routing_grammar_text = generate_routing_grammar(
                filtered_snapshot, top_k=top_k,
            )
            self._routing_grammar_cache[grammar_cache_key] = routing_grammar_text

        # Step 5: Compile grammar into fresh GrammarState
        # (GrammarState is mutated by advance(), so we need a fresh one each call)
        grammar_state = self._grammar_guide_factory(routing_grammar_text)

        # Step 6: Run constrained generation with empty hooks
        tc = transport_config or TransportConfig()
        if self._backend == "mlx":
            from tgirl.sample_mlx import run_constrained_generation_mlx

            # Lazy-convert embeddings to mx.array once
            if self._embeddings_mlx is None:
                import mlx.core as mx
                if isinstance(self._embeddings, torch.Tensor):
                    self._embeddings_mlx = mx.array(
                        self._embeddings.numpy()
                    )
                else:
                    self._embeddings_mlx = self._embeddings

            gen_result = run_constrained_generation_mlx(
                grammar_state=grammar_state,
                forward_fn=self._forward_fn,
                tokenizer_decode=self._tokenizer_decode,
                embeddings=self._embeddings_mlx,
                hooks=[],
                transport_config=tc,
                max_tokens=self._config.max_tokens,
                context_tokens=routing_context_tokens,
            )
        else:
            gen_result = run_constrained_generation(
                grammar_state=grammar_state,
                forward_fn=self._forward_fn,
                tokenizer_decode=self._tokenizer_decode,
                embeddings=self._embeddings,
                hooks=[],
                transport_config=tc,
                max_tokens=self._config.max_tokens,
                context_tokens=routing_context_tokens,
            )

        # Step 7: Parse and validate output (multi-tool aware)
        raw_tools = gen_result.hy_source.strip().split()
        valid_names = {t.name for t in filtered_tools}

        # Deduplicate while preserving order, skip invalid names
        seen: set[str] = set()
        ordered: list[str] = []
        for name in raw_tools:
            if name not in seen and name in valid_names:
                seen.add(name)
                ordered.append(name)

        if not ordered:
            msg = (
                f"Routing produced '{gen_result.hy_source.strip()}' which "
                f"contains no valid tools from {valid_names}"
            )
            raise ValueError(msg)

        # Respect top_k
        selected = ordered[:top_k]

        latency_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "routing_complete",
            selected_tools=selected,
            routing_tokens=len(gen_result.tokens),
            latency_ms=latency_ms,
        )

        return RerankResult(
            selected_tools=tuple(selected),
            routing_tokens=len(gen_result.tokens),
            routing_latency_ms=latency_ms,
            routing_grammar_text=routing_grammar_text,
        )
