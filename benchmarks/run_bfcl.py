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
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import structlog

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
        "--transition-policy",
        type=str,
        default="delimiter",
        help=(
            "Transition policy: 'delimiter' (default), 'immediate', "
            "or 'budget:N' (e.g., budget:3)"
        ),
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
            model_name=model_name,
            display_name=model_name,
            url="",
            org="tgirl",
            license="Apache-2.0",
            model_handler="default",
            underscore_to_dot=False,
        )
        log.info("registered_model_config", model_name=model_name)


def make_transition_policy(policy_str: str, tokenizer_decode: Any = None) -> Any:
    """Parse a transition policy string into a policy object.

    Args:
        policy_str: One of 'delimiter', 'immediate', or 'budget:N'.
        tokenizer_decode: Required for 'delimiter' policy.

    Returns:
        A TransitionPolicy instance.
    """
    from tgirl.state_machine import (
        BudgetTransitionPolicy,
        DelimiterTransitionPolicy,
        ImmediateTransitionPolicy,
    )

    if policy_str == "delimiter":
        if tokenizer_decode is None:
            msg = "delimiter policy requires tokenizer_decode"
            raise ValueError(msg)
        return DelimiterTransitionPolicy(
            delimiter="<tool>",
            tokenizer_decode=tokenizer_decode,
        )
    elif policy_str == "immediate":
        return ImmediateTransitionPolicy()
    elif policy_str.startswith("budget:"):
        try:
            budget = int(policy_str.split(":", 1)[1])
        except (ValueError, IndexError):
            msg = (
                f"Invalid budget policy: {policy_str!r}. "
                "Use 'budget:N' where N is an integer."
            )
            raise ValueError(msg) from None
        return BudgetTransitionPolicy(budget=budget)
    else:
        msg = (
            f"Unknown transition policy: {policy_str!r}. "
            "Use 'delimiter', 'immediate', or 'budget:N'."
        )
        raise ValueError(msg)


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the BFCL benchmark pipeline."""
    from mlx_lm import load as mlx_load

    from tgirl.bfcl import (
        load_ground_truth,
        load_test_data,
        register_bfcl_tools,
        sexpr_to_bfcl,
        sexpr_to_bfcl_dict,
    )
    from tgirl.cache import CacheStats, make_mlx_forward_fn
    from tgirl.format import ChatTemplateFormatter
    from tgirl.modulation import EnvelopeConfig, ModMatrixHook
    from tgirl.outlines_adapter import (
        make_outlines_grammar_factory,
        make_outlines_grammar_factory_mlx,
    )
    from tgirl.registry import ToolRegistry
    from tgirl.sample import SamplingSession
    from tgirl.transport import TransportConfig
    from tgirl.types import RerankConfig, SessionConfig

    # --- 1. Load model ---
    log.info("loading_model", model=args.model)
    mlx_model, mlx_tokenizer = mlx_load(args.model)
    hf_tokenizer = mlx_tokenizer._tokenizer

    mlx_embed = mlx_model.language_model.model.embed_tokens.weight.astype(mx.float32)
    mx.eval(mlx_embed)  # materialize embedding weights
    embeddings = mlx_embed  # Keep as mx.array for MLX backend
    log.info("model_loaded", vocab_size=embeddings.shape[0])

    cache_stats = CacheStats()
    forward_fn = make_mlx_forward_fn(mlx_model, stats=cache_stats)

    # --- 2. Grammar factories + formatter ---
    grammar_factory = make_outlines_grammar_factory(hf_tokenizer)
    mlx_grammar_factory = make_outlines_grammar_factory_mlx(hf_tokenizer)
    formatter = ChatTemplateFormatter(hf_tokenizer)

    # --- 2b. Collect stop token IDs ---
    stop_token_ids = []
    if hf_tokenizer.eos_token_id is not None:
        stop_token_ids.append(hf_tokenizer.eos_token_id)
    # Some models have additional stop tokens (e.g., <|im_end|>)
    for token_str in ["<|im_end|>", "<|endoftext|>"]:
        ids = hf_tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in stop_token_ids:
            stop_token_ids.append(ids[0])
    log.info("stop_token_ids", ids=stop_token_ids)

    # --- 2c. Extract tool call primer tokens ---
    from tgirl.sample import extract_tool_call_primer
    added_tokens = getattr(hf_tokenizer, 'added_tokens_encoder', None)
    tool_call_primer = extract_tool_call_primer(
        tokenizer_encode=hf_tokenizer.encode,
        added_tokens=added_tokens,
    )
    log.info("tool_call_primer", tokens=tool_call_primer)

    # --- 3. Session config + transition policy ---
    transition_policy = make_transition_policy(
        args.transition_policy,
        tokenizer_decode=hf_tokenizer.decode,
    )
    log.info("transition_policy", policy=args.transition_policy)

    session_config = SessionConfig(
        max_tool_cycles=1,
        freeform_max_tokens=512,
        constrained_max_tokens=512,
        session_timeout=30.0,
        force_tool_call=True,
    )
    session_hooks = [
        ModMatrixHook(
            config=EnvelopeConfig(base_temperature=0.5),
            tokenizer_decode=hf_tokenizer.decode,
            vocab_size=embeddings.shape[0],
            max_tokens=session_config.constrained_max_tokens,
        ),
    ]
    transport_config = TransportConfig(valid_ratio_threshold=0.5)

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
    telemetry_path = result_dir / f"BFCL_v4_{args.category}_telemetry.jsonl"

    # --- 7. Load ground truth for evaluation ---
    ground_truth_entries = load_ground_truth(args.category)
    ground_truth_by_id = {e["id"]: e["ground_truth"] for e in ground_truth_entries}

    try:
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        from bfcl_eval.constants.enums import Language

        has_checker = True
    except ImportError:
        has_checker = False
        log.warning("ast_checker_unavailable", msg="pip install bfcl_eval for accuracy scoring")

    # --- 8. Run inference ---
    results: list[dict[str, str]] = []
    parsed_outputs: list[list[dict] | None] = []
    total_tokens = 0
    total_ms = 0.0
    errors = 0
    ast_correct = 0
    ast_checked = 0

    for i, entry in enumerate(entries):
        entry_id = entry["id"]
        messages = entry["question"][0]  # First turn: list[dict]

        # Fresh registry per entry
        registry = ToolRegistry()
        name_map = register_bfcl_tools(registry, entry["function"])

        # Fresh session per entry (prevents quota/state leakage)
        # Set top_k to number of tools so router can return the full set
        n_tools = len(entry["function"])
        rerank_config = RerankConfig(
            max_tokens=16, temperature=0.3, top_k=n_tools,
        )
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
            backend="mlx",
            mlx_grammar_guide_factory=mlx_grammar_factory,
            transition_policy=transition_policy,
            stop_token_ids=stop_token_ids,
            tool_call_primer_tokens=tool_call_primer,
        )

        t0 = time.monotonic()
        bfcl_output = ""
        bfcl_parsed = None
        result = None
        hy_source = ""
        ast_result_str = ""
        try:
            result = session.run_chat(messages)
            elapsed = (time.monotonic() - t0) * 1000

            if result.tool_calls:
                hy_source = result.tool_calls[0].pipeline
                bfcl_output = sexpr_to_bfcl(hy_source, registry, name_map)
                bfcl_parsed = sexpr_to_bfcl_dict(hy_source, registry, name_map)
            else:
                bfcl_output = result.output_text

            total_tokens += result.total_tokens
            total_ms += elapsed

            # AST accuracy check
            if has_checker and bfcl_parsed is not None and entry_id in ground_truth_by_id:
                gt = ground_truth_by_id[entry_id]
                check = ast_checker(
                    entry["function"], bfcl_parsed, gt,
                    Language.PYTHON, args.category, args.model_name,
                )
                ast_checked += 1
                if check["valid"]:
                    ast_correct += 1
                    ast_result_str = "PASS"
                else:
                    ast_result_str = f"FAIL: {check.get('error', ['?'])[0]}"
            elif bfcl_parsed is None and result.tool_calls:
                ast_result_str = "NO_TOOL_CALL"
            elif not result.tool_calls:
                ast_result_str = "NO_TOOL_CALL"

            log.info(
                "entry_complete",
                index=i + 1,
                total=len(entries),
                id=entry_id,
                tokens=result.total_tokens,
                elapsed_ms=round(elapsed, 1),
                has_tool_call=bool(result.tool_calls),
                ast=ast_result_str or "N/A",
                output=bfcl_output[:80],
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            total_ms += elapsed
            errors += 1
            ast_result_str = f"ERROR: {e!s}"
            log.error(
                "entry_failed",
                index=i + 1,
                id=entry_id,
                error=str(e),
                elapsed_ms=round(elapsed, 1),
            )

        results.append({"id": entry_id, "result": bfcl_output})
        parsed_outputs.append(bfcl_parsed)

        # --- Write telemetry entry (append mode for crash recovery) ---
        telemetry_entry: dict[str, Any] = {
            "id": entry_id,
            "ast_result": ast_result_str or "N/A",
            "hy_source": hy_source,
            "bfcl_output": bfcl_output,
            "ground_truth": ground_truth_by_id.get(entry_id),
            "hyperparams": {
                "base_temperature": session_hooks[0].config.base_temperature,
                "modulation_matrix_hash": hash(session_hooks[0].config.matrix_flat),
                "ot_epsilon": transport_config.epsilon,
                "ot_valid_ratio_threshold": transport_config.valid_ratio_threshold,
                "ot_max_iterations": transport_config.max_iterations,
            },
        }
        if result is not None and result.telemetry:
            tel = result.telemetry[0]
            telemetry_entry.update({
                "tokens": tel.tokens,
                "grammar_valid_counts": tel.grammar_valid_counts,
                "temperatures_applied": tel.temperatures_applied,
                "wasserstein_distances": tel.wasserstein_distances,
                "token_log_probs": tel.token_log_probs,
                "ot_bypass_reasons": tel.ot_bypass_reasons,
                "ot_iterations": tel.ot_iterations,
                "freeform_tokens_before": tel.freeform_tokens_before,
            })
        with open(telemetry_path, "a") as tf:
            tf.write(json.dumps(telemetry_entry) + "\n")

    # --- 8. Write results ---
    with open(result_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    log.info("results_written", path=str(result_path), count=len(results))

    # --- 10. Report ---
    n = len(entries)
    tool_call_count = sum(1 for p in parsed_outputs if p is not None)
    print("\n" + "=" * 70)
    print(f"  tgirl BFCL benchmark — {args.category}")
    print(f"  Model: {args.model}")
    print(f"  Transition policy:  {args.transition_policy}")
    print("=" * 70)
    print(f"  Entries processed:  {n}")
    print(f"  Tool calls made:    {tool_call_count}/{n} ({round(100*tool_call_count/n) if n else 0}%)")
    print(f"  Errors:             {errors}")
    if ast_checked > 0:
        print(f"  AST accuracy:       {ast_correct}/{ast_checked} ({round(100*ast_correct/ast_checked)}%)")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Total wall time:    {round(total_ms)}ms")
    if n > 0:
        print(f"  Avg per entry:      {round(total_ms / n)}ms")
    print(f"  Results at:         {result_path}")
    print(f"  Telemetry at:       {telemetry_path}")
    print(f"  Cache hits:         {cache_stats.hits}")
    print(f"  Cache misses:       {cache_stats.misses}")
    print(f"  Cache resets:       {cache_stats.resets}")
    print(f"  Tokens saved:       {cache_stats.tokens_saved}")
    print("=" * 70)


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
