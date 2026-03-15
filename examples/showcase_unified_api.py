#!/usr/bin/env python3
"""Showcase: unified run_chat() API with a non-trivial tool set.

Registers 8 tools spanning math, text, data, and utility domains.
Tests routing accuracy on 15 requests designed to be ambiguous —
the model must understand intent, not just match keywords.

This demonstrates that grammar-constrained generation on a 0.8B model
can reliably select and invoke the correct tool from a large set.

Requirements:
    pip install 'tgirl[grammar,compile,transport,sample]' llguidance mlx-lm

Usage:
    PYTHONUNBUFFERED=1 python -u examples/showcase_unified_api.py
"""

from __future__ import annotations

import os
import sys

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

MODEL_ID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"


def main() -> int:
    from tgirl.cache import CacheStats, make_mlx_forward_fn
    from tgirl.format import ChatTemplateFormatter
    from tgirl.modulation import EnvelopeConfig, ModMatrixHook
    from tgirl.registry import ToolRegistry
    from tgirl.sample import SamplingSession
    from tgirl.transport import TransportConfig
    from tgirl.types import RerankConfig, SessionConfig

    # --- 1. Register 8 tools across 4 domains ---
    registry = ToolRegistry()

    # Math domain
    @registry.tool(description="Add two numbers together")
    def add(a: int, b: int) -> int:
        return a + b

    @registry.tool(description="Multiply two numbers")
    def multiply(a: int, b: int) -> int:
        return a * b

    @registry.tool(description="Compute the remainder when dividing a by b")
    def modulo(a: int, b: int) -> int:
        return a % b

    # Text domain
    @registry.tool(description="Reverse a string of text")
    def reverse(text: str) -> str:
        return text[::-1]

    @registry.tool(description="Count the number of characters in a string")
    def length(text: str) -> int:
        return len(text)

    @registry.tool(description="Convert text to uppercase letters")
    def uppercase(text: str) -> str:
        return text.upper()

    # Data domain
    @registry.tool(description="Find the maximum of two integers")
    def maximum(a: int, b: int) -> int:
        return max(a, b)

    # Utility domain
    @registry.tool(description="Repeat a string n times")
    def repeat(text: str, n: int) -> str:
        return text * n

    tool_names = list(registry.names())
    log.info("tools_registered", count=len(tool_names), tools=tool_names)

    # --- 2. Load model ---
    from mlx_lm import load as mlx_load

    log.info("loading_model", model=MODEL_ID)
    mlx_model, mlx_tokenizer = mlx_load(MODEL_ID)
    hf_tokenizer = mlx_tokenizer._tokenizer

    mlx_embed = mlx_model.language_model.model.embed_tokens.weight.astype(
        mx.float32
    )
    mx.eval(mlx_embed)
    embeddings = mlx_embed  # Keep as mx.array for MLX backend

    log.info("model_loaded", vocab_size=embeddings.shape[0])

    cache_stats = CacheStats()
    forward_fn = make_mlx_forward_fn(mlx_model, stats=cache_stats)

    # --- 3. Grammar factory + formatter ---
    from tgirl.outlines_adapter import (
        make_outlines_grammar_factory,
        make_outlines_grammar_factory_mlx,
    )

    grammar_factory = make_outlines_grammar_factory(hf_tokenizer)
    mlx_grammar_factory = make_outlines_grammar_factory_mlx(hf_tokenizer)
    formatter = ChatTemplateFormatter(hf_tokenizer)

    # --- 4. Session factory (fresh session per request — no quota/state leakage) ---
    session_config = SessionConfig(
        max_tool_cycles=10,
        freeform_max_tokens=100,
        constrained_max_tokens=64,
        session_timeout=30.0,
    )
    session_hooks = [
        ModMatrixHook(
            config=EnvelopeConfig(base_temperature=0.5),
            tokenizer_decode=hf_tokenizer.decode,
            vocab_size=embeddings.shape[0],
            max_tokens=64,
        ),
    ]
    transport_config = TransportConfig(bypass_ratio=0.5)
    rerank_config = RerankConfig(max_tokens=16, temperature=0.3, top_k=len(tool_names))

    # Collect stop token IDs to mask during constrained generation
    stop_token_ids = []
    if hf_tokenizer.eos_token_id is not None:
        stop_token_ids.append(hf_tokenizer.eos_token_id)
    for token_str in ["<|im_end|>", "<|endoftext|>"]:
        ids = hf_tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in stop_token_ids:
            stop_token_ids.append(ids[0])

    def make_session() -> SamplingSession:
        return SamplingSession(
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
            stop_token_ids=stop_token_ids,
        )

    # --- 5. Test cases: 15 requests, designed to be non-obvious ---
    # Each tests a different kind of understanding:
    #   - Synonyms ("product" = multiply, "remainder" = modulo)
    #   - Indirect phrasing ("how long is" = length)
    #   - Ambiguity between similar tools (add vs multiply vs maximum)
    test_cases = [
        # Math — straightforward
        ("What is 42 plus 17?", "add", 59),
        ("Compute the product of 8 and 9", "multiply", 72),
        ("What's the remainder of 17 divided by 5?", "modulo", 2),
        # Math — tricky phrasing
        ("How much is 250 and 750 combined?", "add", 1000),
        ("If I have 13 groups of 7, how many total?", "multiply", 91),
        ("Which is larger, 42 or 99?", "maximum", 99),
        # Text — straightforward
        ("Reverse the word 'hello'", "reverse", "olleh"),
        ("Make 'whisper' all caps", "uppercase", "WHISPER"),
        ("How many characters in 'abracadabra'?", "length", 11),
        # Text — tricky phrasing
        ("Spell 'racecar' backwards", "reverse", "racecar"),
        ("How long is the string 'hello world'?", "length", 11),
        ("SHOUT the word 'quiet'", "uppercase", "QUIET"),
        # Utility
        ("Say 'ha' three times", "repeat", "hahaha"),
        ("Echo 'boom' 5 times in a row", "repeat", "boomboomboomboomboom"),
        # Cross-domain ambiguity
        ("What's bigger, 1000 or 999?", "maximum", 1000),
    ]

    # --- 6. Run all test cases ---
    results = []
    for i, (request, expected_tool, expected_result) in enumerate(test_cases):
        log.info("test_case", index=i + 1, request=request, expected=expected_tool)

        session = make_session()
        result = session.run_chat([{"role": "user", "content": request}])

        tool_call = result.tool_calls[0] if result.tool_calls else None
        hy_source = tool_call.pipeline if tool_call else "(none)"
        tool_name = hy_source.strip().split()[0].lstrip("(") if tool_call else "none"
        is_error = tool_call.error is not None if tool_call else True
        result_val = tool_call.result if tool_call and not is_error else "error"

        # Check both routing and execution correctness
        routed_correctly = tool_name == expected_tool
        executed_correctly = (
            not is_error and str(result_val) == str(expected_result)
        )

        results.append({
            "request": request,
            "expected_tool": expected_tool,
            "expected_result": expected_result,
            "hy_source": hy_source,
            "tool": tool_name,
            "routed": routed_correctly,
            "executed": executed_correctly,
            "tokens": result.total_tokens,
            "elapsed_ms": round(result.wall_time_ms, 1),
            "result": result_val,
            "error": is_error,
        })

    # --- 7. Report ---
    n = len(test_cases)
    routed = sum(1 for r in results if r["routed"])
    executed = sum(1 for r in results if r["executed"])
    valid = sum(1 for r in results if not r["error"])
    total_tokens = sum(r["tokens"] for r in results)
    total_ms = sum(r["elapsed_ms"] for r in results)

    print("\n" + "=" * 90)
    print("  tgirl unified API showcase — 8 tools, 15 requests")
    print(f"  Model: {MODEL_ID}")
    print("=" * 90)

    for r in results:
        route_mark = "+" if r["routed"] else "-"
        exec_mark = "+" if r["executed"] else "!"
        status = f"[{route_mark}{exec_mark}]"
        print(f"\n  \"{r['request']}\"")
        print(
            f"    {status} {r['hy_source']:<40s} -> {r['result']}"
            f"  (expect: {r['expected_result']}, {r['tokens']}tok, {r['elapsed_ms']}ms)"
        )

    print(f"\n  {'':50s} Score")
    print(f"  {'Tool routing accuracy:':50s} {routed}/{n}")
    print(f"  {'End-to-end correctness (route + args + exec):':50s} {executed}/{n}")
    print(f"  {'Valid executable calls:':50s} {valid}/{n}")
    print(f"  {'Total tokens:':50s} {total_tokens}")
    print(f"  {'Total wall time:':50s} {round(total_ms)}ms")
    print(f"  {'Avg per request:':50s} {round(total_ms / n)}ms")
    print(f"  {'Cache hits:':50s} {cache_stats.hits}")
    print(f"  {'Cache misses:':50s} {cache_stats.misses}")
    print(f"  {'Cache resets:':50s} {cache_stats.resets}")
    print(f"  {'Tokens saved by cache:':50s} {cache_stats.tokens_saved}")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    sys.exit(main())
