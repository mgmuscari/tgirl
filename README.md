# tgirl

**Transformational Generator for Inference-Restricting Languages**

A Python library for local LLM inference with grammar-constrained compositional tool calling. tgirl makes invalid tool calls inexpressible at the token level — not caught at runtime, not retried, not hallucinated. Impossible.

## What it does

tgirl supports two generation modes within a single inference session:

- **Freeform mode** — unconstrained natural language generation (thinking, conversing, explaining)
- **Constrained mode** — grammar-constrained [Hy](https://hylang.org) s-expression tool pipelines with mathematical safety guarantees

The model generates freely until it needs to invoke tools. The grammar then dynamically constrains output to only permit well-formed Hy s-expressions composing registered tools within their type, quota, and scope constraints. After execution, the model returns to freeform generation.

```
;; Single compositional expression — one inference pass, not five
(-> user-query
    (search-docs)
    (summarize)
    (translate "es"))

;; Fan-out with pmap
(pmap [sentiment-analysis entity-extraction] document)

;; Error handling
(try (risky-tool input)
     (catch [e Exception] (fallback input)))
```

## Why not JSON/XML tool calling?

| | JSON/XML (MCP, OpenAI, etc.) | tgirl |
|---|---|---|
| Structural errors | Caught at runtime, retried | Inexpressible at token level |
| Composition | Sequential round-trips | Single s-expression per pipeline |
| Model requirements | Instruct-tuned + tool-call fine-tuned | Any model with a vocabulary distribution |
| Safety guarantees | Validation after generation | Enforced during generation |
| Token cost | N calls = N inference passes | 1 pass for entire pipeline |

## Architecture

Core modules, each independently importable:

```
tgirl/
├── registry.py          # Tool registration, type extraction, snapshots
├── grammar.py           # Dynamic CFG generation from registry state
├── compile.py           # Hy parsing, AST compilation, sandboxed execution
├── transport.py         # Optimal transport logit redistribution (torch)
├── transport_mlx.py     # OT redistribution — MLX-native for Apple Silicon
├── sample.py            # Constrained sampling engine (torch)
├── sample_mlx.py        # Constrained sampling — MLX-native (zero torch in hot loop)
├── outlines_adapter.py  # llguidance grammar constraint bridge
├── cache.py             # KV cache management, forward_fn factories
├── rerank.py            # Grammar-constrained tool reranking
├── bridge.py            # MCP compatibility layer (import/export)
└── serve.py             # Optional: FastAPI local inference server
```

### Defense-in-depth (compile module)

Three layers ensure safety — any one is sufficient, all three are active:

1. **Grammar** — prevents invalid expressions at the token level during generation
2. **Static analysis** — Hy AST + Python AST analysis catches anything the grammar might miss
3. **Sandbox** — restricted namespace with only registered tools and safe builtins

### Optimal transport logit redistribution

When grammar constraints mask out most of the vocabulary, naive masking discards the model's probability mass on invalid tokens. tgirl uses optimal transport (Sinkhorn algorithm) to redistribute that mass to semantically similar valid tokens, preserving the model's intent while enforcing grammatical correctness.

### MLX-native inference

On Apple Silicon, the entire inference hot loop runs in MLX with zero torch or numpy conversions per token. Grammar masks are applied via `llguidance.mlx`, penalties use functional scatter, and shaping uses pure MLX ops. Typical constrained generation speed: **~7-10ms/token** (comparable to freeform generation).

## Installation

```bash
pip install tgirl                     # Core (registry + types)
pip install tgirl[grammar]            # + grammar generation
pip install tgirl[compile]            # + Hy compilation + sandbox
pip install tgirl[sample]             # + sampling engine + OT
pip install tgirl[mlx]                # + MLX backend (Apple Silicon)
pip install tgirl[all]                # Everything
```

Requires Python 3.11+.

## Quick start

### Register tools and execute pipelines

```python
from tgirl import ToolRegistry, generate_grammar, run_pipeline

registry = ToolRegistry()

@registry.tool()
def search(query: str) -> list[str]:
    return db.search(query)

@registry.tool()
def summarize(texts: list[str]) -> str:
    return llm.summarize(texts)

# Generate grammar from registry state
grammar = generate_grammar(registry)

# Execute a pipeline
result = run_pipeline(
    '(-> "transformers" (search) (summarize))',
    registry,
)
```

### Run inference with grammar-constrained sampling (MLX)

```python
from tgirl import (
    ToolRegistry, SamplingSession, GrammarTemperatureHook,
    TransportConfig, SessionConfig,
    make_mlx_forward_fn,
)
from tgirl.outlines_adapter import (
    make_outlines_grammar_factory,
    make_outlines_grammar_factory_mlx,
)
from tgirl.format import ChatTemplateFormatter
from mlx_lm import load as mlx_load

# Load model
model, tokenizer = mlx_load("mlx-community/Qwen3.5-0.8B-MLX-4bit")
hf_tokenizer = tokenizer._tokenizer
embeddings = model.language_model.model.embed_tokens.weight

# Register tools
registry = ToolRegistry()

@registry.tool()
def add(a: int, b: int) -> int:
    return a + b

# Create session
session = SamplingSession(
    registry=registry,
    forward_fn=make_mlx_forward_fn(model),
    tokenizer_decode=hf_tokenizer.decode,
    tokenizer_encode=hf_tokenizer.encode,
    embeddings=embeddings,
    grammar_guide_factory=make_outlines_grammar_factory(hf_tokenizer),
    hooks=[GrammarTemperatureHook()],
    transport_config=TransportConfig(),
    formatter=ChatTemplateFormatter(hf_tokenizer),
    backend="mlx",
    mlx_grammar_guide_factory=make_outlines_grammar_factory_mlx(hf_tokenizer),
)

result = session.run_chat([{"role": "user", "content": "What is 2 + 3?"}])
```

## Status

**v0.1.0** — All core modules implemented and tested: `registry`, `grammar`, `compile`, `transport`, `sample`, `cache`, `rerank`, `outlines_adapter`. MLX-native backend operational with ~7-10ms/token constrained generation on Apple Silicon. BFCL benchmark integration for evaluation.

## License

MIT

## Author

Maddy Muscari / [Ontologi LLC](https://ontologi.dev)
