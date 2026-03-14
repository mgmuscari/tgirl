#!/usr/bin/env python
"""BFCL benchmark runner for tgirl grammar-constrained inference.

Runs BFCL test entries through tgirl's grammar-constrained sampling
pipeline and produces JSONL result files for evaluation.

Usage:
    python benchmarks/run_bfcl.py \\
        --model mlx-community/Qwen3.5-0.8B-MLX-4bit \\
        --category simple_python \\
        --limit 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

logger = structlog.get_logger()


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
    Requires bfcl_eval with all its transitive dependencies installed.
    """
    try:
        from bfcl_eval.constants.model_config import (
            MODEL_CONFIG_MAPPING,
            ModelConfig,
        )
    except ImportError as e:
        logger.warning(
            "bfcl_eval_import_failed",
            error=str(e),
            msg="Install bfcl_eval with all dependencies for evaluation",
        )
        return

    if model_name not in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[model_name] = ModelConfig(
            underscore_to_dot=False,
        )
        logger.info(
            "registered_model_config",
            model_name=model_name,
        )


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the BFCL benchmark pipeline.

    This function loads the model, processes test entries through
    grammar-constrained inference, and writes results as JSONL.
    """
    from tgirl.bfcl import (
        load_test_data,
        register_bfcl_tools,
    )
    from tgirl.registry import ToolRegistry

    # Register model config for BFCL evaluation
    register_model_config(args.model_name)

    # Load test data
    entries = load_test_data(args.category)
    if args.limit:
        entries = entries[: args.limit]

    logger.info(
        "loaded_test_data",
        category=args.category,
        count=len(entries),
    )

    # Prepare result directory
    result_dir = Path(args.result_dir) / args.model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"BFCL_v4_{args.category}_result.json"

    # NOTE: Model loading and inference require GPU/MLX hardware.
    # This script provides the structure; actual inference is run
    # on machines with appropriate hardware.
    #
    # The inference loop would:
    # 1. Load model via mlx_lm or transformers
    # 2. For each entry:
    #    a. Create fresh ToolRegistry
    #    b. Register BFCL functions via register_bfcl_tools()
    #    c. Create SamplingSession with registry
    #    d. Run session.run_chat(messages)
    #    e. Translate output via sexpr_to_bfcl()
    #    f. Append result to JSONL

    logger.warning(
        "inference_not_implemented",
        msg="Model loading and inference require GPU/MLX hardware. "
        "This script provides structure only.",
    )

    # Placeholder: write empty results
    results: list[dict[str, str]] = []
    for entry in entries:
        # Each entry would go through the inference pipeline
        registry = ToolRegistry()
        register_bfcl_tools(registry, entry["function"])

        # Placeholder result — in real run, this comes from
        # SamplingSession.run_chat() + sexpr_to_bfcl()
        results.append(
            {
                "id": entry["id"],
                "result": "",  # Would be filled by inference
            }
        )

    with open(result_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(
        "results_written",
        path=str(result_path),
        count=len(results),
    )

    # Optional: run evaluation
    if args.evaluate:
        logger.info("evaluation_skipped", reason="requires inference results")


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
