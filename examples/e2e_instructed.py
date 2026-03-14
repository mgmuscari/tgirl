#!/usr/bin/env python3
"""Instructed e2e demo using run_chat() — the unified inference API.

Demonstrates the simplified API where SamplingSession handles:
1. System prompt generation from registered tools
2. Chat template formatting via PromptFormatter
3. Routing context construction for re-ranking
4. The full dual-mode sampling loop

Requirements:
    pip install 'tgirl[grammar,compile,transport,sample]' llguidance mlx-lm

Usage:
    python examples/e2e_instructed.py
"""

from __future__ import annotations

import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import numpy as np
import structlog
import torch

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

MODEL_ID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"


def main() -> int:
    from tgirl.format import ChatTemplateFormatter
    from tgirl.registry import ToolRegistry
    from tgirl.sample import GrammarTemperatureHook, SamplingSession
    from tgirl.transport import TransportConfig
    from tgirl.types import RerankConfig, SessionConfig

    # --- 1. Register tools ---
    registry = ToolRegistry()

    @registry.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @registry.tool()
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    @registry.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    log.info("tools_registered", tools=list(registry.names()))

    # --- 2. Load model ---
    from mlx_lm import load as mlx_load

    log.info("loading_model", model=MODEL_ID)
    mlx_model, mlx_tokenizer = mlx_load(MODEL_ID)
    hf_tokenizer = mlx_tokenizer._tokenizer

    mlx_embed = mlx_model.language_model.model.embed_tokens.weight.astype(
        mx.float32
    )
    mx.eval(mlx_embed)
    embeddings = torch.from_numpy(np.array(mlx_embed, copy=False))
    vocab_size = embeddings.shape[0]

    log.info("model_loaded", vocab_size=vocab_size)

    def forward_fn(token_ids: list[int]) -> torch.Tensor:
        input_ids = mx.array([token_ids])
        logits = mlx_model(input_ids)
        last = logits[0, -1, :].astype(mx.float32)
        mx.eval(last)
        return torch.from_numpy(np.array(last, copy=False))

    def tokenizer_decode(ids: list[int]) -> str:
        return hf_tokenizer.decode(ids)

    def tokenizer_encode(text: str) -> list[int]:
        return hf_tokenizer.encode(text)

    # --- 3. Grammar factory ---
    from tgirl.outlines_adapter import make_outlines_grammar_factory

    grammar_factory = make_outlines_grammar_factory(hf_tokenizer)

    # --- 4. Create formatter ---
    formatter = ChatTemplateFormatter(hf_tokenizer)

    # --- 5. Create session with run_chat() API ---
    config = SessionConfig(
        max_tool_cycles=1,
        freeform_max_tokens=100,
        constrained_max_tokens=64,
        session_timeout=30.0,
    )

    rerank_config = RerankConfig(max_tokens=16, temperature=0.3)

    session = SamplingSession(
        registry=registry,
        forward_fn=forward_fn,
        tokenizer_decode=tokenizer_decode,
        tokenizer_encode=tokenizer_encode,
        embeddings=embeddings,
        grammar_guide_factory=grammar_factory,
        config=config,
        hooks=[GrammarTemperatureHook(base_temperature=0.5)],
        transport_config=TransportConfig(bypass_ratio=0.5),
        rerank_config=rerank_config,
        formatter=formatter,
    )

    # --- 6. Run test cases via run_chat() ---
    test_cases = [
        ("Add 3 and 5 together", "add"),
        ("Say hello to Alice", "greet"),
        ("What is 6 times 7?", "multiply"),
        ("Greet the world", "greet"),
        ("Compute 100 plus 200", "add"),
        ("What is 12 multiplied by 8?", "multiply"),
        ("Say hi to Bob", "greet"),
    ]

    results = []
    for i, (request, expected_tool) in enumerate(test_cases):
        log.info(
            "test_case",
            index=i + 1,
            request=request,
            expected=expected_tool,
        )

        messages = [{"role": "user", "content": request}]
        result = session.run_chat(messages)

        # Extract tool info from result
        tool_call = result.tool_calls[0] if result.tool_calls else None
        hy_source = tool_call.pipeline if tool_call else "(none)"
        tool_name = hy_source.strip().split()[0].lstrip("(") if tool_call else "none"
        is_error = tool_call.error is not None if tool_call else True
        result_val = tool_call.result if tool_call and not is_error else "error"

        results.append(
            {
                "request": request,
                "expected": expected_tool,
                "hy_source": hy_source,
                "tool": tool_name,
                "correct": tool_name == expected_tool,
                "tokens": result.total_tokens,
                "elapsed_ms": round(result.wall_time_ms, 1),
                "result": result_val,
                "error": is_error,
            }
        )

    # --- 7. Report ---
    n = len(test_cases)
    correct = sum(1 for r in results if r["correct"])
    valid = sum(1 for r in results if not r["error"])

    print("\n" + "=" * 80)
    print("  run_chat() API -- Qwen3.5-0.8B-MLX-4bit on Apple Silicon")
    print("=" * 80)

    for r in results:
        mark = "+" if r["correct"] else "-"
        print(f"\n  Request: \"{r['request']}\"  (expected: {r['expected']})")
        print(
            f"    [{mark}] {r['hy_source']:<30s} -> {r['result']}"
            f"  ({r['tokens']}tok, {r['elapsed_ms']}ms)"
        )

    print(f"\n  {'Tool selection accuracy:':40s} {correct}/{n}")
    print(f"  {'Valid executable calls:':40s} {valid}/{n}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
