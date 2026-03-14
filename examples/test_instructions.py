#!/usr/bin/env python3
"""Test auto-generated instructions vs hand-written, with and without enrichment.

Runs 3 configs:
1. Hand-written system prompt (baseline -- the one from e2e_json_tools.py)
2. Auto-generated from snapshot (parameter names only)
3. Auto-generated with enrichment (semantic type tags)

Usage:
    python examples/test_instructions.py
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

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

MODEL_ID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"

TEST_CASES = [
    ('Get the "name" field from {"name": "Alice", "age": 30}', "get_field"),
    ('How many keys are in {"x": 1, "y": 2, "z": 3}?', "count_keys"),
    ('Convert "hello world" to uppercase', "to_upper"),
    ('What is the length of "grammar-constrained"?', "string_length"),
    ('Set the "status" field to "active" in {"id": 42}', "set_field"),
    ('Merge {"name": "Bob"} with {"role": "admin"}', "merge_objects"),
    ("Add 10 and 20", "add"),
]

HAND_WRITTEN_PROMPT = """\
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


def make_registry():
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

    return registry


def run_tests(system_msg, registry, hf_tokenizer, grammar_factory, flat_grammar,
              forward_fn, tokenizer_decode, embeddings):
    from tgirl.compile import run_pipeline
    from tgirl.sample import GrammarTemperatureHook, run_constrained_generation
    from tgirl.transport import TransportConfig
    from tgirl.types import PipelineError

    # Pure masking, no OT -- isolate instruction quality
    transport_config = TransportConfig(valid_ratio_threshold=0.0)
    hooks = [GrammarTemperatureHook(base_temperature=1.0)]

    results = []
    for request, expected_tool in TEST_CASES:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": request},
        ]
        chat_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = hf_tokenizer.encode(chat_prompt)
        grammar_state = grammar_factory(flat_grammar)

        t0 = time.monotonic()
        gen_result = run_constrained_generation(
            grammar_state=grammar_state,
            forward_fn=forward_fn,
            tokenizer_decode=tokenizer_decode,
            embeddings=embeddings,
            hooks=hooks,
            transport_config=transport_config,
            max_tokens=256,
            context_tokens=prompt_tokens,
        )
        elapsed = time.monotonic() - t0

        pipeline_result = run_pipeline(gen_result.hy_source, registry)
        is_error = isinstance(pipeline_result, PipelineError)

        hy = gen_result.hy_source.strip()
        selected_tool = ""
        if hy.startswith("("):
            parts = hy[1:].split(None, 1)
            if parts:
                selected_tool = parts[0].rstrip(")")

        correct = selected_tool == expected_tool

        results.append({
            "expected": expected_tool,
            "selected": selected_tool,
            "correct": correct,
            "error": is_error,
            "hy": hy[:80],
            "tokens": len(gen_result.tokens),
            "elapsed_ms": round(elapsed * 1000, 1),
        })

    return results


def print_results(config_name, results):
    correct = sum(1 for r in results if r["correct"])
    executable = sum(1 for r in results if not r["error"])
    total = len(results)

    print(f"\n{'_' * 80}")
    print(f"  {config_name}")
    print(f"  Tool selection: {correct}/{total} correct | Executable: {executable}/{total}")
    print(f"{'_' * 80}")
    for r in results:
        mark = "V" if r["correct"] else "X"
        err = " ERR" if r["error"] else ""
        print(
            f"  {mark} {r['expected']:15s} -> {r['selected']:15s} "
            f"{r['tokens']:3d}tok {r['elapsed_ms']:7.0f}ms{err}"
        )
        if not r["correct"] or r["error"]:
            print(f"    hy: {r['hy']}")

    return correct, executable


def main() -> int:
    from tgirl.grammar import generate as generate_grammar
    from tgirl.instructions import generate_system_prompt
    from tgirl.outlines_adapter import make_outlines_grammar_factory

    registry = make_registry()

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

    mlx_embed = mlx_model.language_model.model.embed_tokens.weight.astype(mx.float32)
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

    print("\n" + "=" * 80)
    print("  INSTRUCTION QUALITY TEST -- JSON Tools x Qwen3.5-0.8B-MLX-4bit")
    print("=" * 80)

    summary = []

    # --- Config 1: Hand-written prompt (baseline) ---
    r1 = run_tests(HAND_WRITTEN_PROMPT, registry, hf_tokenizer,
                   grammar_factory, flat_grammar, forward_fn, tokenizer_decode, embeddings)
    c1, e1 = print_results("1. HAND-WRITTEN (baseline)", r1)
    summary.append(("Hand-written", c1, e1))

    # --- Config 2: Auto-generated, no enrichment ---
    auto_prompt = generate_system_prompt(snapshot)
    print(f"\n  [Auto-generated prompt, {len(auto_prompt)} chars]")
    r2 = run_tests(auto_prompt, registry, hf_tokenizer,
                   grammar_factory, flat_grammar, forward_fn, tokenizer_decode, embeddings)
    c2, e2 = print_results("2. AUTO-GENERATED (param names only)", r2)
    summary.append(("Auto (names)", c2, e2))

    # --- Config 3: Auto-generated with enrichment ---
    registry.enrich("get_field",
        param_tags={"obj": "JsonObject", "key": "FieldName"},
        examples=['(get_field "{\\"name\\": \\"Alice\\", \\"age\\": 30}" "name") -> "Alice"'],
    )
    registry.enrich("set_field",
        param_tags={"obj": "JsonObject", "key": "FieldName", "value": "JsonValue"},
        examples=['(set_field "{\\"x\\": 1}" "y" "2") -> "{\\"x\\": 1, \\"y\\": 2}"'],
    )
    registry.enrich("count_keys",
        param_tags={"obj": "JsonObject"},
        examples=['(count_keys "{\\"a\\": 1, \\"b\\": 2, \\"c\\": 3}") -> 3'],
    )
    registry.enrich("merge_objects",
        param_tags={"a": "JsonObject", "b": "JsonObject"},
        examples=['(merge_objects "{\\"a\\": 1}" "{\\"b\\": 2}") -> "{\\"a\\": 1, \\"b\\": 2}"'],
    )
    registry.enrich("to_upper", examples=['(to_upper "hello") -> "HELLO"'])
    registry.enrich("string_length", examples=['(string_length "hello") -> 5'])
    registry.enrich("add", examples=['(add 3 5) -> 8'])

    enriched_snapshot = registry.snapshot()
    enriched_prompt = generate_system_prompt(enriched_snapshot)
    print(f"\n  [Enriched prompt, {len(enriched_prompt)} chars]")
    r3 = run_tests(enriched_prompt, registry, hf_tokenizer,
                   grammar_factory, flat_grammar, forward_fn, tokenizer_decode, embeddings)
    c3, e3 = print_results("3. AUTO-GENERATED + ENRICHMENT (semantic tags)", r3)
    summary.append(("Enriched", c3, e3))

    # --- Summary ---
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for name, c, e in summary:
        print(f"  {name:<30s}  correct: {c}/7  executable: {e}/7")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
