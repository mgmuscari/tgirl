#!/usr/bin/env python
"""BFCL benchmark runner for tgirl grammar-constrained inference.

Runs BFCL test entries through tgirl's grammar-constrained sampling
pipeline and produces JSONL result files for evaluation.

Usage:
    PYTHONUNBUFFERED=1 python -u benchmarks/run_bfcl.py \
        --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
        --category simple_python \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BFCL benchmark with tgirl grammar-constrained inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model path or HuggingFace model ID "
            "(e.g., mlx-community/Qwen3.5-0.8B-MLX-4bit)"
        ),
    )
    parser.add_argument(
        "--category",
        type=str,
        default="simple_python",
        help="BFCL category to evaluate (default: simple_python)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process (for testing)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="result",
        help="Directory for result output (default: result/)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tgirl-grammar",
        help="Model name for BFCL evaluation (default: tgirl-grammar)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run BFCL AST evaluation after generation",
    )
    return parser.parse_args()


def register_model_config(model_name: str) -> None:
    """Register a ModelConfig entry in BFCL's MODEL_CONFIG_MAPPING.

    Required for ast_checker.convert_func_name() to work without KeyError.
    """
    try:
        from bfcl_eval.constants.model_config import (
            MODEL_CONFIG_MAPPING,
            ModelConfig,
        )
    except ImportError as e:
        log.warning(
            "bfcl_eval_import_failed",
            error=str(e),
            msg="Install bfcl_eval with all dependencies for evaluation",
        )
        return

    if model_name not in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[model_name] = ModelConfig(
            underscore_to_dot=False,
        )
        log.info("registered_model_config", model_name=model_name)


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the BFCL benchmark pipeline."""
    from mlx_lm import load as mlx_load

    from tgirl.bfcl import load_test_data, register_bfcl_tools, sexpr_to_bfcl
    from tgirl.format import ChatTemplateFormatter
    from tgirl.outlines_adapter import make_outlines_grammar_factory
    from tgirl.registry import ToolRegistry
    from tgirl.sample import GrammarTemperatureHook, SamplingSession
    from tgirl.transport import TransportConfig
    from tgirl.types import RerankConfig, SessionConfig

    # --- 1. Load model ---
    log.info("loading_model", model=args.model)
    mlx_model, mlx_tokenizer = mlx_load(args.model)
    hf_tokenizer = mlx_tokenizer._tokenizer

    mlx_embed = mlx_model.language_model.model.embed_tokens.weight.astype(mx.float32)
    mx.eval(mlx_embed)
    embeddings = torch.from_numpy(np.array(mlx_embed, copy=False))
    log.info("model_loaded", vocab_size=embeddings.shape[0])

    def forward_fn(token_ids: list[int]) -> torch.Tensor:
        input_ids = mx.array([token_ids])
        logits = mlx_model(input_ids)
        last = logits[0, -1, :].astype(mx.float32)
        mx.eval(last)
        return torch.from_numpy(np.array(last, copy=False))

    # --- 2. Grammar factory + formatter ---
    grammar_factory = make_outlines_grammar_factory(hf_tokenizer)
    formatter = ChatTemplateFormatter(hf_tokenizer)

    # --- 3. Session config ---
    session_config = SessionConfig(
        max_tool_cycles=1,
        freeform_max_tokens=4096,
        constrained_max_tokens=4096,
        session_timeout=30.0,
    )
    session_hooks = [GrammarTemperatureHook(base_temperature=0.5)]
    transport_config = TransportConfig(bypass_ratio=0.5)
    rerank_config = RerankConfig(max_tokens=16, temperature=0.3)

    # --- 4. Register model config for BFCL checker ---
    register_model_config(args.model_name)

    # --- 5. Load test data ---
    entries = load_test_data(args.category)
    if args.limit is not None:
        entries = entries[: args.limit]

    log.info("loaded_test_data", category=args.category, count=len(entries))

    # --- 6. Prepare result directory ---
    result_dir = Path(args.result_dir) / args.model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"BFCL_v4_{args.category}_result.json"

    # --- 7. Run inference ---
    results: list[dict[str, str]] = []
    total_tokens = 0
    total_ms = 0.0
    errors = 0

    for i, entry in enumerate(entries):
        entry_id = entry["id"]
        messages = entry["question"][0]  # First turn: list[dict]

        # Fresh registry per entry
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, entry["function"])

        # Fresh session per entry (prevents quota/state leakage)
        session = SamplingSession(
            registry=registry,
            forward_fn=forward_fn,
            tokenizer_decode=hf_tokenizer.decode,
            tokenizer_encode=hf_tokenizer.encode,
            embeddings=embeddings,
            grammar_guide_factory=grammar_factory,
            config=session_config,
            hooks=session_hooks,
            transport_config=transport_config,
            rerank_config=rerank_config,
            formatter=formatter,
        )

        t0 = time.monotonic()
        try:
            result = session.run_chat(messages)
            elapsed = (time.monotonic() - t0) * 1000

            if result.tool_calls:
                hy_source = result.tool_calls[0].pipeline
                bfcl_output = sexpr_to_bfcl(hy_source, registry, name_map)
            else:
                # No tool call — output freeform text (will fail BFCL decode)
                bfcl_output = result.output_text

            total_tokens += result.total_tokens
            total_ms += elapsed

            log.info(
                "entry_complete",
                index=i + 1,
                total=len(entries),
                id=entry_id,
                tokens=result.total_tokens,
                elapsed_ms=round(elapsed, 1),
                has_tool_call=bool(result.tool_calls),
                output=bfcl_output[:80],
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            total_ms += elapsed
            errors += 1
            bfcl_output = ""
            log.error(
                "entry_failed",
                index=i + 1,
                id=entry_id,
                error=str(e),
                elapsed_ms=round(elapsed, 1),
            )

        results.append({"id": entry_id, "result": bfcl_output})

    # --- 8. Write results ---
    with open(result_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    log.info("results_written", path=str(result_path), count=len(results))

    # --- 9. Report ---
    n = len(entries)
    print("\n" + "=" * 70)
    print(f"  tgirl BFCL benchmark — {args.category}")
    print(f"  Model: {args.model}")
    print("=" * 70)
    print(f"  Entries processed:  {n}")
    print(f"  Errors:             {errors}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Total wall time:    {round(total_ms)}ms")
    if n > 0:
        print(f"  Avg per entry:      {round(total_ms / n)}ms")
    print(f"  Results at:         {result_path}")
    print("=" * 70)

    # --- 10. Optional: run BFCL AST checker ---
    if args.evaluate:
        try:
            from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

            log.info("running_ast_checker", category=args.category)
            ast_checker(args.model_name, str(result_path), args.category)
        except ImportError as e:
            log.warning(
                "checker_skipped",
                reason="bfcl_eval not fully installed",
                error=str(e),
            )
        except Exception as e:
            log.error("checker_failed", error=str(e))


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
