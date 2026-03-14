#!/usr/bin/env python3
"""Hyperparameter sweep for JSON tools: isolate OT vs model vs temperature.

Runs a focused subset of test cases across parameter configurations:
1. Pure masking (no OT) — valid_ratio_threshold=0.0, so OT is always bypassed
2. Default OT — valid_ratio_threshold=0.5
3. Aggressive OT — valid_ratio_threshold=0.9 (OT on most steps)
4. Temperature sweep — base_temperature in [0.1, 0.3, 0.5, 1.0, 2.0]
5. Greedy (temp=0) — no sampling randomness at all

Usage:
    python examples/sweep_json_tools.py
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import numpy as np
import structlog
import torch

# Minimal logging
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

# Focused test cases — one per tool, clear intent
TEST_CASES = [
    ('Get the "name" field from {"name": "Alice", "age": 30}', "get_field"),
    ('How many keys are in {"x": 1, "y": 2, "z": 3}?', "count_keys"),
    ('Convert "hello world" to uppercase', "to_upper"),
    ('What is the length of "grammar-constrained"?', "string_length"),
    ('Set the "status" field to "active" in {"id": 42}', "set_field"),
    ('Merge {"name": "Bob"} with {"role": "admin"}', "merge_objects"),
    ("Add 10 and 20", "add"),
]


def setup():
    """One-time setup: registry, grammar, model, factory."""
    from tgirl.grammar import generate as generate_grammar
    from tgirl.outlines_adapter import make_outlines_grammar_factory
    from tgirl.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.tool()
    def get_field(obj: str, key: str) -> str:
        """Extract a field value from a JSON object string."""
        return str(json.loads(obj)[key])

    @registry.tool()
    def set_field(obj: str, key: str, value: str) -> str:
        """Set a field in a JSON object string, returns updated JSON."""
        d = json.loads(obj)
        try:
            d[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            d[key] = value
        return json.dumps(d)

    @registry.tool()
    def count_keys(obj: str) -> int:
        """Count the number of keys in a JSON object."""
        return len(json.loads(obj))

    @registry.tool()
    def merge_objects(a: str, b: str) -> str:
        """Merge two JSON objects. Keys in b override keys in a."""
        merged = {**json.loads(a), **json.loads(b)}
        return json.dumps(merged)

    @registry.tool()
    def to_upper(s: str) -> str:
        """Convert a string to uppercase."""
        return s.upper()

    @registry.tool()
    def string_length(s: str) -> int:
        """Return the length of a string."""
        return len(s)

    @registry.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    # Grammar
    snapshot = registry.snapshot()
    grammar_output = generate_grammar(snapshot)
    flat_grammar = grammar_output.text.replace(
        "expr: tool_call | composition", "expr: tool_call"
    )

    # Model
    from mlx_lm import load as mlx_load

    print(f"Loading {MODEL_ID}...")
    mlx_model, mlx_tokenizer = mlx_load(MODEL_ID)
    hf_tokenizer = mlx_tokenizer._tokenizer

    mlx_embed = mlx_model.language_model.model.embed_tokens.weight.astype(
        mx.float32
    )
    mx.eval(mlx_embed)
    embeddings = torch.from_numpy(np.array(mlx_embed, copy=False))

    def forward_fn(token_ids: list[int]) -> torch.Tensor:
        input_ids = mx.array([token_ids])
        logits = mlx_model(input_ids)
        last = logits[0, -1, :].astype(mx.float32)
        mx.eval(last)
        return torch.from_numpy(np.array(last, copy=False))

    def tokenizer_decode(ids: list[int]) -> str:
        return hf_tokenizer.decode(ids)

    grammar_factory = make_outlines_grammar_factory(hf_tokenizer)

    # System prompt
    system_msg = """\
You call tools using s-expressions. Reply with ONLY one s-expression, nothing else.

## Syntax
  (tool_name arg1 arg2 ...)
  Strings are double-quoted. Integers are bare numbers.

## Available Tools

(get_field <json_string> <key_string>) -> string
  Extract a field value from a JSON object.
  Example: (get_field "{\\"name\\": \\"Alice\\", \\"age\\": 30}" "name") -> "Alice"

(set_field <json_string> <key_string> <value_string>) -> string
  Set a field in a JSON object, returns updated JSON.
  Example: (set_field "{\\"x\\": 1}" "y" "2") -> "{\\"x\\": 1, \\"y\\": 2}"

(count_keys <json_string>) -> int
  Count the number of keys in a JSON object.
  Example: (count_keys "{\\"a\\": 1, \\"b\\": 2, \\"c\\": 3}") -> 3

(merge_objects <json_string> <json_string>) -> string
  Merge two JSON objects. Keys in second override first.
  Example: (merge_objects "{\\"a\\": 1}" "{\\"b\\": 2}") -> "{\\"a\\": 1, \\"b\\": 2}"

(to_upper <string>) -> string
  Convert a string to uppercase.
  Example: (to_upper "hello") -> "HELLO"

(string_length <string>) -> int
  Return the length of a string.
  Example: (string_length "hello") -> 5

(add <int> <int>) -> int
  Add two integers.
  Example: (add 3 5) -> 8"""

    return {
        "registry": registry,
        "flat_grammar": flat_grammar,
        "grammar_factory": grammar_factory,
        "forward_fn": forward_fn,
        "tokenizer_decode": tokenizer_decode,
        "embeddings": embeddings,
        "hf_tokenizer": hf_tokenizer,
        "system_msg": system_msg,
    }


def run_config(ctx, config_name, transport_config, base_temp, scaling_exp=0.5):
    """Run all test cases with a given config. Returns list of result dicts."""
    from tgirl.compile import run_pipeline
    from tgirl.sample import GrammarTemperatureHook, run_constrained_generation
    from tgirl.types import PipelineError

    hooks = [GrammarTemperatureHook(
        base_temperature=base_temp,
        scaling_exponent=scaling_exp,
    )]

    results = []
    for request, expected_tool in TEST_CASES:
        messages = [
            {"role": "system", "content": ctx["system_msg"]},
            {"role": "user", "content": request},
        ]
        chat_prompt = ctx["hf_tokenizer"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = ctx["hf_tokenizer"].encode(chat_prompt)
        grammar_state = ctx["grammar_factory"](ctx["flat_grammar"])

        t0 = time.monotonic()
        gen_result = run_constrained_generation(
            grammar_state=grammar_state,
            forward_fn=ctx["forward_fn"],
            tokenizer_decode=ctx["tokenizer_decode"],
            embeddings=ctx["embeddings"],
            hooks=hooks,
            transport_config=transport_config,
            max_tokens=256,
            context_tokens=prompt_tokens,
        )
        elapsed = time.monotonic() - t0

        pipeline_result = run_pipeline(gen_result.hy_source, ctx["registry"])
        is_error = isinstance(pipeline_result, PipelineError)

        # Extract tool name from (tool_name ...)
        hy = gen_result.hy_source.strip()
        selected_tool = ""
        if hy.startswith("("):
            parts = hy[1:].split(None, 1)
            if parts:
                selected_tool = parts[0].rstrip(")")

        correct = selected_tool == expected_tool

        results.append({
            "request": request[:50],
            "expected": expected_tool,
            "selected": selected_tool,
            "correct": correct,
            "error": is_error,
            "hy": hy[:80],
            "tokens": len(gen_result.tokens),
            "elapsed_ms": round(elapsed * 1000, 1),
            "ot_bypassed": gen_result.ot_bypassed_count,
            "total_steps": len(gen_result.tokens),
        })

    return results


def print_results(config_name, results):
    """Print a compact results table."""
    correct = sum(1 for r in results if r["correct"])
    executable = sum(1 for r in results if not r["error"])
    total = len(results)

    print(f"\n{'─' * 80}")
    print(f"  {config_name}")
    print(f"  Tool selection: {correct}/{total} correct | Executable: {executable}/{total}")
    print(f"{'─' * 80}")
    for r in results:
        mark = "V" if r["correct"] else "X"
        err = " ERR" if r["error"] else ""
        ot_info = f"OT-skip:{r['ot_bypassed']}/{r['total_steps']}"
        print(
            f"  {mark} {r['expected']:15s} -> {r['selected']:15s} "
            f"{ot_info:16s} {r['tokens']:3d}tok "
            f"{r['elapsed_ms']:7.0f}ms{err}"
        )
        if not r["correct"] or r["error"]:
            print(f"    hy: {r['hy']}")

    return correct, executable


def main() -> int:
    from tgirl.transport import TransportConfig

    ctx = setup()
    print("\n" + "=" * 80)
    print("  HYPERPARAMETER SWEEP -- JSON Tools x Qwen3.5-0.8B-MLX-4bit")
    print("=" * 80)

    summary = []

    # -- Phase 1: OT bypass test --
    # valid_ratio_threshold=0.0 means OT bypass triggers whenever
    # valid_ratio > 0.0, which is always true (there's always >= 1 valid token).
    # This gives us pure grammar masking with no redistribution.
    print("\n\n>> PHASE 1: OT vs Pure Masking")

    configs = [
        ("NO OT (pure mask)", TransportConfig(valid_ratio_threshold=0.0), 1.0),
        ("OT default (thresh=0.5)", TransportConfig(valid_ratio_threshold=0.5), 1.0),
        ("OT aggressive (thresh=0.9)", TransportConfig(valid_ratio_threshold=0.9), 1.0),
    ]

    for name, tc, temp in configs:
        results = run_config(ctx, name, tc, base_temp=temp)
        c, e = print_results(name, results)
        summary.append((name, c, e, len(results)))

    # -- Phase 2: Temperature sweep (with OT disabled) --
    print("\n\n>> PHASE 2: Temperature Sweep (no OT)")

    no_ot = TransportConfig(valid_ratio_threshold=0.0)
    for temp in [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        name = f"temp={temp} (no OT)"
        results = run_config(ctx, name, no_ot, base_temp=temp)
        c, e = print_results(name, results)
        summary.append((name, c, e, len(results)))

    # -- Phase 3: Temperature sweep (with OT) --
    print("\n\n>> PHASE 3: Temperature Sweep (with OT, thresh=0.5)")

    default_ot = TransportConfig(valid_ratio_threshold=0.5)
    for temp in [0.01, 0.1, 0.3, 0.5, 1.0, 2.0]:
        name = f"temp={temp} (OT thresh=0.5)"
        results = run_config(ctx, name, default_ot, base_temp=temp)
        c, e = print_results(name, results)
        summary.append((name, c, e, len(results)))

    # -- Phase 4: Scaling exponent sweep --
    print("\n\n>> PHASE 4: Scaling Exponent Sweep (no OT, base_temp=1.0)")

    for exp in [0.1, 0.25, 0.5, 1.0, 2.0]:
        name = f"exp={exp} (no OT, temp=1.0)"
        results = run_config(ctx, name, no_ot, base_temp=1.0, scaling_exp=exp)
        c, e = print_results(name, results)
        summary.append((name, c, e, len(results)))

    # -- Summary --
    print("\n\n" + "=" * 80)
    print("  SWEEP SUMMARY")
    print("=" * 80)
    print(f"  {'Config':<40s} {'Correct':>8s} {'Exec':>8s}")
    print(f"  {'_' * 40} {'_' * 8} {'_' * 8}")
    for name, c, e, t in summary:
        print(f"  {name:<40s} {c}/{t:>5d}   {e}/{t:>5d}")
    print("=" * 80)

    best = max(summary, key=lambda x: (x[1], x[2]))
    print(f"\n  Best: {best[0]} -- {best[1]}/{best[3]} correct, {best[2]}/{best[3]} executable")

    return 0


if __name__ == "__main__":
    sys.exit(main())
