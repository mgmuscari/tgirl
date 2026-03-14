#!/usr/bin/env python3
"""Test tool re-ranking strategies before constrained generation.

Compares four re-ranking approaches:
1. Baseline: typed grammar, all tools, no re-ranking
2. Log-prob scoring: single forward pass, rank tools by P(name|context)
3. Grammar-constrained re-ranking: tiny grammar allows only tool names
4. Semi-structured: freeform generation with "reply with tool name" prompt

Each re-ranker's top-1 pick restricts the grammar for constrained generation.

Usage:
    python -u examples/test_rerank.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import numpy as np
import structlog
import torch

import logging
logging.disable(logging.CRITICAL)

def _drop_event(_, __, ___):
    raise structlog.DropEvent

structlog.configure(
    processors=[_drop_event],
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

JSONOBJECT_RULE = r'/\"\{[^\"]*\}\"/'
FIELDNAME_RULE = r'/\"[a-zA-Z_][a-zA-Z0-9_]*\"/'


def make_registry():
    from tgirl.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_type("JsonObject", JSONOBJECT_RULE)
    registry.register_type("FieldName", FIELDNAME_RULE)

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

    registry.enrich("get_field",
        param_tags={"obj": "JsonObject", "key": "FieldName"},
        examples=['(get_field "{\\"name\\": \\"Alice\\"}" "name") -> "Alice"'],
    )
    registry.enrich("set_field",
        param_tags={"obj": "JsonObject", "key": "FieldName"},
        examples=['(set_field "{\\"x\\": 1}" "y" "2") -> "{\\"x\\": 1, \\"y\\": 2}"'],
    )
    registry.enrich("count_keys",
        param_tags={"obj": "JsonObject"},
        examples=['(count_keys "{\\"a\\": 1, \\"b\\": 2}") -> 2'],
    )
    registry.enrich("merge_objects",
        param_tags={"a": "JsonObject", "b": "JsonObject"},
        examples=['(merge_objects "{\\"a\\": 1}" "{\\"b\\": 2}") -> "{\\"a\\": 1, \\"b\\": 2}"'],
    )
    registry.enrich("to_upper", examples=['(to_upper "hello") -> "HELLO"'])
    registry.enrich("string_length", examples=['(string_length "hello") -> 5'])
    registry.enrich("add", examples=['(add 3 5) -> 8'])

    return registry


# --- Re-ranking strategies ---


RERANK_PROMPT = """You are a tool routing assistant. Given a user request, pick the best tool.

Available tools:
- get_field(obj, key): Extract a field value from a JSON object string
- set_field(obj, key, value): Set a field in a JSON object string
- count_keys(obj): Count the number of keys in a JSON object
- merge_objects(a, b): Merge two JSON objects
- to_upper(s): Convert a string to uppercase
- string_length(s): Return the length of a string
- add(a, b): Add two integers

Reply with ONLY the tool name, nothing else. Example:
User: What is 3 + 5?
Tool: add"""


def freeform_generate(mlx_model, hf_tokenizer, prompt_tokens, max_tokens=16):
    """Simple greedy freeform generation."""
    tokens = list(prompt_tokens)
    generated = []
    eos_id = hf_tokenizer.eos_token_id

    for _ in range(max_tokens):
        input_ids = mx.array([tokens])
        logits = mlx_model(input_ids)
        last = logits[0, -1, :]
        mx.eval(last)
        next_id = int(mx.argmax(last).item())
        if next_id == eos_id:
            break
        generated.append(next_id)
        tokens.append(next_id)

    return hf_tokenizer.decode(generated).strip()


def rerank_semi_structured(mlx_model, hf_tokenizer, request, tool_names):
    """Semi-structured: ask model to name the tool in freeform mode."""
    messages = [
        {"role": "system", "content": RERANK_PROMPT},
        {"role": "user", "content": request},
    ]
    chat_prompt = hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = hf_tokenizer.encode(chat_prompt)
    response = freeform_generate(mlx_model, hf_tokenizer, prompt_tokens, max_tokens=32)

    found = []
    response_lower = response.lower()
    for name in tool_names:
        if name in response_lower and name not in found:
            found.append(name)

    return found, response


def rerank_logprob(mlx_model, hf_tokenizer, request, tool_names):
    """Log-prob scoring: single forward pass, rank by P(tool_name | context).

    For each tool name, compute the sum of log-probs for its token sequence
    given the re-ranking prompt context. No generation needed — just scoring.
    """
    messages = [
        {"role": "system", "content": RERANK_PROMPT},
        {"role": "user", "content": request},
        {"role": "assistant", "content": ""},
    ]
    chat_prompt = hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Strip trailing whitespace/newline that the template might add after empty assistant
    chat_prompt = chat_prompt.rstrip()
    context_tokens = hf_tokenizer.encode(chat_prompt)

    # Tokenize each tool name (without special tokens)
    tool_token_seqs = {}
    for name in tool_names:
        toks = hf_tokenizer.encode(name, add_special_tokens=False)
        tool_token_seqs[name] = toks

    # Score each tool name by running forward passes token-by-token
    scores = {}
    for name in tool_names:
        name_tokens = tool_token_seqs[name]
        total_logprob = 0.0
        current_tokens = list(context_tokens)

        for tok in name_tokens:
            input_ids = mx.array([current_tokens])
            logits = mlx_model(input_ids)
            last = logits[0, -1, :].astype(mx.float32)
            mx.eval(last)

            # Convert to log-probs via log-softmax
            last_np = np.array(last, copy=False)
            max_logit = float(np.max(last_np))
            log_sum_exp = max_logit + math.log(float(np.sum(np.exp(last_np - max_logit))))
            token_logprob = float(last_np[tok]) - log_sum_exp

            total_logprob += token_logprob
            current_tokens.append(tok)

        # Normalize by length to avoid bias toward shorter names
        scores[name] = total_logprob / len(name_tokens)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_names = [name for name, _ in ranked]
    score_str = ", ".join(f"{n}={s:.2f}" for n, s in ranked[:3])

    return ranked_names, score_str


def rerank_grammar_constrained(mlx_model, hf_tokenizer, request, tool_names,
                                grammar_factory, forward_fn, tokenizer_decode,
                                embeddings):
    """Grammar-constrained re-ranking: tiny grammar that only allows tool names."""
    from tgirl.sample import GrammarTemperatureHook, run_constrained_generation
    from tgirl.transport import TransportConfig

    # Build a minimal grammar that only accepts one of the tool names
    alternatives = " | ".join(f'"{name}"' for name in tool_names)
    grammar_text = f"""start: tool_choice
tool_choice: {alternatives}
"""

    messages = [
        {"role": "system", "content": RERANK_PROMPT},
        {"role": "user", "content": request},
    ]
    chat_prompt = hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = hf_tokenizer.encode(chat_prompt)

    grammar_state = grammar_factory(grammar_text)
    transport_config = TransportConfig(valid_ratio_threshold=0.0)
    hooks = [GrammarTemperatureHook(base_temperature=0.3)]

    gen_result = run_constrained_generation(
        grammar_state=grammar_state,
        forward_fn=forward_fn,
        tokenizer_decode=tokenizer_decode,
        embeddings=embeddings,
        hooks=hooks,
        transport_config=transport_config,
        max_tokens=16,
        context_tokens=prompt_tokens,
    )

    response = gen_result.hy_source.strip()
    # The grammar should produce exactly one tool name
    found = [response] if response in tool_names else []

    return found, response


# --- Test runner ---


def run_tests_restricted(registry, hf_tokenizer, grammar_factory,
                         forward_fn, tokenizer_decode, embeddings,
                         tool_selections):
    """Run constrained generation with per-test-case tool restrictions."""
    from tgirl.compile import run_pipeline
    from tgirl.grammar import generate as generate_grammar
    from tgirl.instructions import generate_system_prompt
    from tgirl.sample import GrammarTemperatureHook, run_constrained_generation
    from tgirl.transport import TransportConfig
    from tgirl.types import PipelineError

    transport_config = TransportConfig(valid_ratio_threshold=0.0)
    hooks = [GrammarTemperatureHook(base_temperature=1.0)]

    results = []
    for i, (request, expected_tool) in enumerate(TEST_CASES):
        restricted_tools = tool_selections[i]

        snap = registry.snapshot(restrict_to=restricted_tools)
        prompt = generate_system_prompt(snap)
        grammar = generate_grammar(snap)
        flat = grammar.text.replace("expr: tool_call | composition", "expr: tool_call")

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": request},
        ]
        chat_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = hf_tokenizer.encode(chat_prompt)
        grammar_state = grammar_factory(flat)

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
            "restricted_to": restricted_tools,
            "correct": correct,
            "error": is_error,
            "hy": hy[:80],
            "tokens": len(gen_result.tokens),
            "elapsed_ms": round(elapsed * 1000, 1),
            "valid_counts": gen_result.grammar_valid_counts[:5],
        })

    return results


def print_results(config_name, results, rerank_info=None):
    correct = sum(1 for r in results if r["correct"])
    executable = sum(1 for r in results if not r["error"])
    total = len(results)

    print(f"\n{'_' * 80}")
    print(f"  {config_name}")
    print(f"  Tool selection: {correct}/{total} correct | Executable: {executable}/{total}")
    print(f"{'_' * 80}")
    for i, r in enumerate(results):
        mark = "V" if r["correct"] else "X"
        err = " ERR" if r["error"] else ""
        restricted = r.get("restricted_to", [])
        vc = str(r["valid_counts"])
        print(
            f"  {mark} {r['expected']:15s} -> {r['selected']:15s} "
            f"{r['tokens']:3d}tok {r['elapsed_ms']:7.0f}ms{err}"
        )
        if restricted:
            print(f"    restricted_to: {restricted}")
        if rerank_info and i < len(rerank_info):
            print(f"    rerank: {rerank_info[i]!r}")
        print(f"    valid_counts(first 5): {vc}")
        if not r["correct"] or r["error"]:
            print(f"    hy: {r['hy']}")

    return correct, executable


def main() -> int:
    from tgirl.grammar import generate as generate_grammar
    from tgirl.instructions import generate_system_prompt
    from tgirl.outlines_adapter import make_outlines_grammar_factory
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
    registry = make_registry()
    tool_names = registry.names()

    print("\n" + "=" * 80)
    print("  RE-RANK COMPARISON -- Qwen3.5-0.8B-MLX-4bit")
    print("=" * 80)

    summary = []

    # ==========================================================================
    # Phase 1: Run all three re-rankers
    # ==========================================================================

    # --- 1a: Log-prob scoring ---
    print("\n  --- Log-prob re-ranking ---")
    logprob_picks = []
    logprob_info = []
    for request, expected in TEST_CASES:
        t0 = time.monotonic()
        ranked, score_str = rerank_logprob(mlx_model, hf_tokenizer, request, tool_names)
        elapsed = time.monotonic() - t0
        logprob_picks.append(ranked)
        logprob_info.append(score_str)
        pick = ranked[0] if ranked else "???"
        mark = "V" if pick == expected else "X"
        print(f"  {mark} {expected:15s} -> {pick:15s}  {elapsed*1000:5.0f}ms  [{score_str}]")

    logprob_correct = sum(
        1 for i, (_, exp) in enumerate(TEST_CASES)
        if logprob_picks[i] and logprob_picks[i][0] == exp
    )
    print(f"  Log-prob accuracy: {logprob_correct}/7")

    # --- 1b: Grammar-constrained re-ranking ---
    print("\n  --- Grammar-constrained re-ranking ---")
    grammar_picks = []
    grammar_info = []
    for request, expected in TEST_CASES:
        t0 = time.monotonic()
        found, response = rerank_grammar_constrained(
            mlx_model, hf_tokenizer, request, tool_names,
            grammar_factory, forward_fn, tokenizer_decode, embeddings,
        )
        elapsed = time.monotonic() - t0
        grammar_picks.append(found)
        grammar_info.append(response)
        pick = found[0] if found else "???"
        mark = "V" if pick == expected else "X"
        print(f"  {mark} {expected:15s} -> {pick:15s}  {elapsed*1000:5.0f}ms  raw: {response!r}")

    grammar_correct = sum(
        1 for i, (_, exp) in enumerate(TEST_CASES)
        if grammar_picks[i] and grammar_picks[i][0] == exp
    )
    print(f"  Grammar-constrained accuracy: {grammar_correct}/7")

    # --- 1c: Semi-structured re-ranking ---
    print("\n  --- Semi-structured re-ranking ---")
    semi_picks = []
    semi_info = []
    for request, expected in TEST_CASES:
        t0 = time.monotonic()
        found, response = rerank_semi_structured(mlx_model, hf_tokenizer, request, tool_names)
        elapsed = time.monotonic() - t0
        semi_picks.append(found)
        semi_info.append(response)
        pick = found[0] if found else "???"
        mark = "V" if pick == expected else "X"
        print(f"  {mark} {expected:15s} -> {pick:15s}  {elapsed*1000:5.0f}ms  raw: {response!r}")

    semi_correct = sum(
        1 for i, (_, exp) in enumerate(TEST_CASES)
        if semi_picks[i] and semi_picks[i][0] == exp
    )
    print(f"  Semi-structured accuracy: {semi_correct}/7")

    # ==========================================================================
    # Phase 2: Constrained generation with each re-ranker's top-1 pick
    # ==========================================================================

    # --- Config 1: Baseline (no re-ranking) ---
    print("\n  --- Running baseline (no re-ranking) ---")
    snap_all = registry.snapshot()
    prompt_all = generate_system_prompt(snap_all)
    grammar_all = generate_grammar(snap_all)
    flat_all = grammar_all.text.replace("expr: tool_call | composition", "expr: tool_call")

    from tgirl.compile import run_pipeline
    from tgirl.sample import GrammarTemperatureHook, run_constrained_generation
    from tgirl.transport import TransportConfig
    from tgirl.types import PipelineError

    transport_config = TransportConfig(valid_ratio_threshold=0.0)
    hooks = [GrammarTemperatureHook(base_temperature=1.0)]

    baseline_results = []
    for request, expected_tool in TEST_CASES:
        messages = [
            {"role": "system", "content": prompt_all},
            {"role": "user", "content": request},
        ]
        chat_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = hf_tokenizer.encode(chat_prompt)
        grammar_state = grammar_factory(flat_all)

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

        pipeline_result = run_pipeline(gen_result.hy_source, registry)
        is_error = isinstance(pipeline_result, PipelineError)
        hy = gen_result.hy_source.strip()
        selected_tool = ""
        if hy.startswith("("):
            parts = hy[1:].split(None, 1)
            if parts:
                selected_tool = parts[0].rstrip(")")

        baseline_results.append({
            "expected": expected_tool,
            "selected": selected_tool,
            "correct": selected_tool == expected_tool,
            "error": is_error,
            "hy": hy[:80],
            "tokens": len(gen_result.tokens),
            "elapsed_ms": round(elapsed * 1000, 1),
            "valid_counts": gen_result.grammar_valid_counts[:5],
        })

    c1, e1 = print_results("1. BASELINE (no re-ranking)", baseline_results)
    summary.append(("Baseline", c1, e1))

    # --- Config 2: Log-prob top-1 ---
    print("\n  --- Running log-prob top-1 ---")
    lp_selections = [
        [ranked[0]] if ranked else tool_names
        for ranked in logprob_picks
    ]
    r2 = run_tests_restricted(
        registry, hf_tokenizer, grammar_factory,
        forward_fn, tokenizer_decode, embeddings, lp_selections,
    )
    c2, e2 = print_results("2. LOG-PROB TOP-1", r2, logprob_info)
    summary.append(("Log-prob top-1", c2, e2))

    # --- Config 3: Grammar-constrained top-1 ---
    print("\n  --- Running grammar-constrained top-1 ---")
    gc_selections = [
        [found[0]] if found else tool_names
        for found in grammar_picks
    ]
    r3 = run_tests_restricted(
        registry, hf_tokenizer, grammar_factory,
        forward_fn, tokenizer_decode, embeddings, gc_selections,
    )
    c3, e3 = print_results("3. GRAMMAR-CONSTRAINED TOP-1", r3, grammar_info)
    summary.append(("Grammar top-1", c3, e3))

    # --- Config 4: Semi-structured top-1 ---
    print("\n  --- Running semi-structured top-1 ---")
    ss_selections = [
        [found[0]] if found else tool_names
        for found in semi_picks
    ]
    r4 = run_tests_restricted(
        registry, hf_tokenizer, grammar_factory,
        forward_fn, tokenizer_decode, embeddings, ss_selections,
    )
    c4, e4 = print_results("4. SEMI-STRUCTURED TOP-1", r4, semi_info)
    summary.append(("Semi-structured top-1", c4, e4))

    # --- Summary ---
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  {'Strategy':<35s}  {'rerank':>7s}  {'correct':>7s}  {'exec':>5s}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*5}")
    rerank_accs = ["-", f"{logprob_correct}/7", f"{grammar_correct}/7", f"{semi_correct}/7"]
    for (name, c, e), ra in zip(summary, rerank_accs):
        print(f"  {name:<35s}  {ra:>7s}  {c}/7      {e}/7")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
