#!/usr/bin/env python3
"""Instructed e2e demo: teach the model the grammar, then constrain output.

Unlike the basic demo, this populates the context with:
1. An explanation of Hy s-expression tool call syntax
2. The available tools with their signatures
3. An explicit request to generate a tool call

Then grammar-constrained generation forces the output to be valid.

Requirements:
    pip install 'tgirl[grammar,compile,transport,sample]' llguidance mlx-lm

Usage:
    python examples/e2e_instructed.py
"""

from __future__ import annotations

import os
import sys
import time

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

# --- Hard-coded tool documentation ---

TOOL_DOCS = """\
### add
  Signature: (add <int> <int>) -> int
  Description: Add two integers
  Examples: (add 3 5), (add -1 100), (add 0 0)

### greet
  Signature: (greet <string>) -> string
  Description: Greet someone by name
  Examples: (greet "Alice"), (greet "world")

### multiply
  Signature: (multiply <int> <int>) -> int
  Description: Multiply two integers
  Examples: (multiply 6 7), (multiply -3 4)"""

SYSTEM_PROMPT = """\
You are a tool-calling assistant. You respond with exactly one tool call \
expression in Hy s-expression syntax. No other text.

## Syntax

Tool calls use Lisp-style s-expressions:
  (tool_name arg1 arg2 ...)

Strings are double-quoted: "hello"
Integers are bare numbers: 42, -7, 0
Negative integers use a leading minus: -3

## Available Tools

{tool_docs}

## Task

Given the user's request, respond with exactly one tool call expression. \
Output ONLY the s-expression, nothing else.
"""


def build_prompt(user_request: str) -> str:
    """Build the full prompt with tool docs and user request."""
    system = SYSTEM_PROMPT.format(tool_docs=TOOL_DOCS)
    return f"{system}\n\nUser: {user_request}\nAssistant: "


def main() -> int:
    from tgirl.registry import ToolRegistry

    # --- 1. Register tools ---
    registry = ToolRegistry()

    @registry.tool()
    def add(a: int, b: int) -> int:
        return a + b

    @registry.tool()
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    @registry.tool()
    def multiply(a: int, b: int) -> int:
        return a * b

    log.info("tools_registered", tools=list(registry.names()))

    # --- 2. Generate grammar (flat, no composition) ---
    from tgirl.grammar import generate as generate_grammar

    snapshot = registry.snapshot()
    grammar_output = generate_grammar(snapshot)

    flat_grammar = grammar_output.text.replace(
        "expr: tool_call | composition", "expr: tool_call"
    )
    log.info("grammar_generated", hash=grammar_output.snapshot_hash)

    # --- 3. Load model ---
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

    # --- 4. Grammar factory ---
    from tgirl.outlines_adapter import make_outlines_grammar_factory

    grammar_factory = make_outlines_grammar_factory(hf_tokenizer)

    # --- 5. Run test cases ---
    from tgirl.compile import run_pipeline
    from tgirl.sample import GrammarTemperatureHook, run_constrained_generation
    from tgirl.transport import TransportConfig
    from tgirl.types import PipelineError

    test_cases = [
        "Add 3 and 5 together",
        "Say hello to Alice",
        "What is 6 times 7?",
        "Greet the world",
        "Compute 100 plus 200",
    ]

    hooks = [GrammarTemperatureHook(base_temperature=0.5)]
    transport_config = TransportConfig(bypass_ratio=0.5)

    results = []
    for i, request in enumerate(test_cases):
        prompt = build_prompt(request)
        prompt_tokens = hf_tokenizer.encode(prompt)

        grammar_state = grammar_factory(flat_grammar)

        log.info(
            "test_case",
            index=i + 1,
            request=request,
            prompt_tokens=len(prompt_tokens),
        )

        t0 = time.monotonic()
        gen_result = run_constrained_generation(
            grammar_state=grammar_state,
            forward_fn=forward_fn,
            tokenizer_decode=tokenizer_decode,
            embeddings=embeddings,
            hooks=hooks,
            transport_config=transport_config,
            max_tokens=64,
            context_tokens=prompt_tokens,
        )
        elapsed = time.monotonic() - t0

        # Execute
        pipeline_result = run_pipeline(gen_result.hy_source, registry)
        is_error = isinstance(pipeline_result, PipelineError)
        result_val = (
            pipeline_result.message
            if is_error
            else getattr(pipeline_result, "result", pipeline_result)
        )

        results.append(
            {
                "request": request,
                "hy_source": gen_result.hy_source,
                "tokens": len(gen_result.tokens),
                "elapsed_ms": round(elapsed * 1000, 1),
                "result": result_val,
                "error": is_error,
            }
        )

    # --- 6. Report ---
    print("\n" + "=" * 72)
    print("  INSTRUCTED E2E RESULTS -- Qwen3.5-0.8B-MLX-4bit on Apple Silicon")
    print("=" * 72)

    for r in results:
        status = "ERR" if r["error"] else "OK "
        print(f"\n  [{status}] \"{r['request']}\"")
        print(f"        Hy: {r['hy_source']}")
        print(f"        Result: {r['result']}")
        print(f"        ({r['tokens']} tokens, {r['elapsed_ms']}ms)")

    valid = sum(1 for r in results if not r["error"])
    print(f"\n  {valid}/{len(results)} produced valid, executable tool calls.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
