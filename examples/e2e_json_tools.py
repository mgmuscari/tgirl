#!/usr/bin/env python3
"""Advanced e2e demo: JSON manipulation tools with grammar constraint.

Demonstrates grammar-constrained generation on tasks requiring:
- String arguments containing JSON
- Multi-step reasoning about data structures
- Correct tool selection from a larger registry

Uses Qwen3.5-0.8B chat template for instruction following.

Requirements:
    pip install 'tgirl[grammar,compile,transport,sample]' llguidance mlx-lm

Usage:
    python examples/e2e_json_tools.py
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
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

MODEL_ID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"


def main() -> int:
    from tgirl.registry import ToolRegistry

    # --- 1. Register JSON manipulation tools ---
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

    log.info("tools_registered", tools=list(registry.names()))

    # --- 2. Generate grammar ---
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

    # --- 5. Build system prompt ---
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

    # --- 6. Test cases ---
    from tgirl.compile import run_pipeline
    from tgirl.sample import GrammarTemperatureHook, run_constrained_generation
    from tgirl.transport import TransportConfig
    from tgirl.types import PipelineError

    test_cases = [
        'Get the "name" field from {"name": "Alice", "age": 30}',
        'How many keys are in {"x": 1, "y": 2, "z": 3}?',
        'Convert "hello world" to uppercase',
        'What is the length of "grammar-constrained"?',
        'Set the "status" field to "active" in {"id": 42}',
        'Merge {"name": "Bob"} with {"role": "admin"}',
        'How many keys does {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5} have?',
    ]

    hooks = [GrammarTemperatureHook(base_temperature=1.0)]
    transport_config = TransportConfig(bypass_ratio=0.5)
    max_gen_tokens = 256

    results = []
    for i, request in enumerate(test_cases):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": request},
        ]
        chat_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = hf_tokenizer.encode(chat_prompt)

        grammar_state = grammar_factory(flat_grammar)

        log.info("test_case", index=i + 1, request=request)

        t0 = time.monotonic()
        gen_result = run_constrained_generation(
            grammar_state=grammar_state,
            forward_fn=forward_fn,
            tokenizer_decode=tokenizer_decode,
            embeddings=embeddings,
            hooks=hooks,
            transport_config=transport_config,
            max_tokens=max_gen_tokens,
            context_tokens=prompt_tokens,
        )
        elapsed = time.monotonic() - t0

        pipeline_result = run_pipeline(gen_result.hy_source, registry)
        is_error = isinstance(pipeline_result, PipelineError)
        if is_error:
            result_val = f"[{pipeline_result.stage}] {pipeline_result.message}"
        else:
            result_val = getattr(pipeline_result, "result", pipeline_result)

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

    # --- 7. Report ---
    print("\n" + "=" * 76)
    print("  JSON TOOLS E2E -- Qwen3.5-0.8B-MLX-4bit on Apple Silicon")
    print("=" * 76)

    for r in results:
        status = "ERR" if r["error"] else "OK "
        print(f"\n  [{status}] \"{r['request']}\"")
        print(f"        Hy:     {r['hy_source']}")
        print(f"        Result: {r['result']}")
        print(f"        ({r['tokens']} tokens, {r['elapsed_ms']}ms)")

    valid = sum(1 for r in results if not r["error"])
    print(f"\n  {valid}/{len(results)} produced valid, executable tool calls.")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
