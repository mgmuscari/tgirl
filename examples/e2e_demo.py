#!/usr/bin/env python3
"""End-to-end demo: real model + real grammar constraint + real execution.

Validates tgirl's core claim: grammar-constrained generation forces
syntactically valid Hy s-expressions from any model, even one that
has never seen Hy.

Runs Qwen3.5-0.8B (4-bit MLX) on Apple Silicon via Metal, with
llguidance providing grammar state tracking.

Requirements:
    pip install 'tgirl[grammar,compile,transport,sample]' llguidance mlx-lm

Usage:
    python examples/e2e_demo.py
"""

from __future__ import annotations

import os
import sys
import time

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    # --- 1. Register tools ---
    from tgirl.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @registry.tool()
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    log.info("tools_registered", tools=list(registry.names()))

    # --- 2. Generate grammar ---
    from tgirl.grammar import generate as generate_grammar

    snapshot = registry.snapshot()
    grammar_output = generate_grammar(snapshot)
    log.info(
        "grammar_generated",
        hash=grammar_output.snapshot_hash,
        lines=grammar_output.text.count("\n"),
    )
    print("\n--- Generated Grammar ---")
    print(grammar_output.text)
    print("--- End Grammar ---\n")

    # --- 3. Load Qwen3.5-0.8B via MLX ---
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load as mlx_load

    log.info("loading_model", model=MODEL_ID)
    mlx_model, mlx_tokenizer = mlx_load(MODEL_ID)

    # Extract embedding weights as torch tensor for OT cost matrix
    mlx_embed = mlx_model.language_model.model.embed_tokens.weight.astype(mx.float32)
    mx.eval(mlx_embed)
    embeddings = torch.from_numpy(np.array(mlx_embed, copy=False))
    vocab_size = embeddings.shape[0]

    log.info(
        "model_loaded",
        vocab_size=vocab_size,
        tokenizer_vocab_size=mlx_tokenizer.vocab_size,
        embedding_dim=embeddings.shape[1],
    )

    # Bridge functions: MLX model → torch tensors for tgirl sampling loop
    def forward_fn(token_ids: list[int]) -> torch.Tensor:
        input_ids = mx.array([token_ids])
        logits = mlx_model(input_ids)
        # Last position logits → numpy → torch
        last_logits = logits[0, -1, :].astype(mx.float32)
        mx.eval(last_logits)
        return torch.from_numpy(np.array(last_logits, copy=False))

    hf_tokenizer = mlx_tokenizer._tokenizer

    def tokenizer_decode(ids: list[int]) -> str:
        return hf_tokenizer.decode(ids)

    # --- 4. Create grammar factory ---
    from tgirl.outlines_adapter import make_outlines_grammar_factory

    grammar_factory = make_outlines_grammar_factory(hf_tokenizer)
    log.info("grammar_factory_ready")

    # --- 5. Run constrained generation ---
    from tgirl.sample import GrammarTemperatureHook, run_constrained_generation
    from tgirl.transport import TransportConfig

    # Use a simplified grammar without composition rules for the demo.
    # The full grammar allows recursive threading/let/if/try/pmap, which
    # causes small models to get stuck in infinite nesting. Flat tool
    # calls are what we're validating here.
    flat_grammar = grammar_output.text
    # Remove composition from expr alternatives and strip composition rules
    flat_grammar = flat_grammar.replace(
        "expr: tool_call | composition", "expr: tool_call"
    )

    grammar_state = grammar_factory(flat_grammar)
    hooks = [GrammarTemperatureHook(base_temperature=0.3)]
    transport_config = TransportConfig(
        bypass_ratio=0.5,
    )

    prompt = "The tool call is: "
    prompt_tokens = hf_tokenizer.encode(prompt)
    log.info("starting_constrained_generation", prompt=prompt)
    t0 = time.monotonic()

    result = run_constrained_generation(
        grammar_state=grammar_state,
        forward_fn=forward_fn,
        tokenizer_decode=tokenizer_decode,
        embeddings=embeddings,
        hooks=hooks,
        transport_config=transport_config,
        max_tokens=128,
        context_tokens=prompt_tokens,
    )

    elapsed = time.monotonic() - t0
    log.info(
        "generation_complete",
        tokens=len(result.tokens),
        elapsed_ms=round(elapsed * 1000, 1),
        ot_bypassed=result.ot_bypassed_count,
    )

    print(f"\n--- Generated Hy Source ({len(result.tokens)} tokens) ---")
    print(result.hy_source)
    print("--- End Hy Source ---\n")

    # Show per-token details
    print("Token trace:")
    for i, tid in enumerate(result.tokens):
        text = hf_tokenizer.decode([tid])
        valid = result.grammar_valid_counts[i]
        temp = result.temperatures_applied[i]
        print(
            f"  [{i:3d}] token={tid:5d}  {text!r:12s}"
            f"  valid={valid:5d}/{vocab_size}  temp={temp:.3f}"
        )

    # --- 6. Parse and execute ---
    print("\n--- Pipeline Execution ---")
    from tgirl.compile import run_pipeline
    from tgirl.types import PipelineError

    pipeline_result = run_pipeline(result.hy_source, registry)

    if isinstance(pipeline_result, PipelineError):
        log.error(
            "pipeline_failed",
            stage=pipeline_result.stage,
            error=pipeline_result.message,
        )
        print(f"FAILED at {pipeline_result.stage}: {pipeline_result.message}")
        print(
            "\nNote: Qwen3.5-0.8B has never seen Hy s-expressions."
            " Syntactic validity is the claim, not semantic correctness."
        )
        return 0
    else:
        result_val = getattr(pipeline_result, "result", pipeline_result)
        log.info("pipeline_succeeded", result=result_val)
        print(f"Result: {result_val}")
        print(
            "\nThe grammar constraint forced Qwen3.5-0.8B to produce a valid,"
            " executable Hy s-expression on Apple Silicon."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
