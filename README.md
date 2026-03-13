# tgirl

**Transformational Grammar for Inference-Restricting Languages**

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

Six core modules, each independently importable:

```
tgirl/
├── registry.py    # Tool registration, type extraction, snapshots
├── grammar.py     # Dynamic CFG generation from registry state
├── compile.py     # Hy parsing, AST compilation, sandboxed execution
├── transport.py   # Optimal transport logit redistribution
├── sample.py      # Constrained sampling engine
└── bridge.py      # MCP compatibility layer (import/export)
```

### Defense-in-depth (compile module)

Three layers ensure safety — any one is sufficient, all three are active:

1. **Grammar** — prevents invalid expressions at the token level during generation
2. **Static analysis** — Hy AST + Python AST analysis catches anything the grammar might miss
3. **Sandbox** — restricted namespace with only registered tools and safe builtins

## Installation

```bash
pip install tgirl                     # Core (registry + types)
pip install tgirl[grammar]            # + grammar generation
pip install tgirl[compile]            # + Hy compilation + sandbox
pip install tgirl[all]                # Everything
```

Requires Python 3.11+.

## Quick start

```python
from tgirl import ToolRegistry, generate_grammar, run_pipeline

# Register tools
registry = ToolRegistry()

@registry.tool()
def search(query: str) -> list[str]:
    return db.search(query)

@registry.tool()
def summarize(texts: list[str]) -> str:
    return llm.summarize(texts)

# Generate grammar from registry state
grammar = generate_grammar(registry)

# Execute a pipeline (from model output or manual)
result = run_pipeline(
    '(-> "transformers" (search) (summarize))',
    registry,
)
# result.result == "Transformers are a neural network architecture..."
```

## Status

**v0.1.0** — Active development. `registry`, `grammar`, and `compile` modules are implemented and tested. `transport`, `sample`, `bridge`, and `serve` are next.

## License

MIT

## Author

Maddy Muscari / [Ontologi LLC](https://ontologi.dev)
