# tgirl

**Linguistic Synthesis Engine**

A grammar-aware inference engine for local LLMs. tgirl takes any language model and boosts its structured output accuracy through grammar constraints, optimal transport logit redistribution, and auto-tuning hyperparameters. Invalid tool calls are inexpressible at the token level — not caught at runtime, not retried, not hallucinated. Impossible.

**Key result:** Qwen3.5-9B-Base (zero instruct tuning, zero RLHF) achieves **79% AST accuracy** on BFCL v4 with 98% tool call rate — matching the instruct model's 80%. A base model that has never been trained to call tools matches a model explicitly post-trained for it, purely through inference-time grammar engineering.

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

## Benchmark Results

BFCL v4 `simple_python` (400 entries, AST accuracy evaluation):

| Model | Instruct? | Tool Rate | AST Accuracy |
|-------|-----------|-----------|-------------|
| Qwen3.5-9B | Yes (SFT+RL) | 98% | 80% (314/391) |
| **Qwen3.5-9B-Base** | **No** | **98%** | **79%** (308/392) |
| Qwen3.5-0.8B | Yes (SFT+RL) | 96% | 48% (182/382) |
| Qwen3.5-0.8B-Base | No | 84% | 46% (153/334) |

Base models match instruct models under grammar constraints. The inference loop compensates for the absence of alignment training. At 9B scale, the base model's broader distribution is equally amenable to grammar-constrained structured output.

## Architecture

tgirl implements a **Virtual Cognitive Processing Unit** (VCogPU) — 9 functional blocks that separate grammar competence from model performance:

- **CLU** (Cognitive Logic Unit) — any transformer model via `forward_fn`
- **GSU** (Grammar Selection Unit) — CFG + HPSG grammar compilation to token masks
- **SCU** (Stochastic Control Unit) — regime transitions, confidence-gated mode switching
- **STB** (Semantic Transport Bus) — optimal transport logit redistribution
- **CSG** (Confidence Signal Generator) — entropy, confidence, coherence signals
- **ADSR Modulation** — (12,7) matrix routing signals to auto-tuned parameters

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full block architecture.

### Core modules

```
tgirl/
├── registry.py          # Tool registration, type extraction, snapshots
├── grammar.py           # Dynamic CFG generation from registry state
├── compile.py           # Hy parsing, AST compilation, sandboxed execution
├── transport.py         # Optimal transport logit redistribution (torch)
├── transport_mlx.py     # OT redistribution — MLX-native for Apple Silicon
├── sample.py            # Constrained sampling engine (torch)
├── sample_mlx.py        # Constrained sampling — MLX-native
├── outlines_adapter.py  # llguidance grammar constraint bridge
├── modulation.py        # ADSR modulation matrix — auto-tuning parameters
├── state_machine.py     # SCU — transition policies, regime control
├── cache.py             # KV cache management, forward_fn factories
├── rerank.py            # Grammar-constrained tool reranking
├── lingo/               # Native LinGO GSU — ERG type hierarchy + coherence
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

### ADSR modulation matrix

A (12,7) matrix routes 12 per-token signals (entropy, confidence, nesting depth, ADSR phase gates, cycle detection, linguistic coherence) to 7 destination parameters (temperature, top_p, repetition bias, OT epsilon, opener penalty, backtrack threshold, presence penalty). The matrix multiply auto-tunes hyperparameters per token — no manual tuning needed.

### Latched transition policy

For base models that don't emit tool-call delimiters, the SCU uses an entropy-based confidence latch: when the model's distribution becomes sharp (low entropy = it knows what it wants to do), the latch is set. The model finishes its sentence (terminal detection), then transitions to constrained mode. Freeform generation is implicit chain-of-thought reasoning — it goes into the KV cache as context for the grammar-constrained tool call.

### LinGO-native grammar support

The `tgirl.lingo` module reads HPSG grammars in TDL format (English Resource Grammar 2025 — 51k types, 43k lexemes) and produces per-token linguistic coherence signals. This extends the grammar portfolio beyond CFGs to formal linguistic grammars for natural language well-formedness.

### MLX-native inference

On Apple Silicon, the entire inference hot loop runs in MLX with zero torch or numpy conversions per token. Typical constrained generation speed: **~7-10ms/token**.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full engineering roadmap. Key next steps:

- **Phase 2B**: MLP learned grammar controller — extract implicit grammatical knowledge from logit distributions
- **Phase 3**: Codegen grammar portfolio — Python, Rust, JS CFGs for code generation benchmarks
- **Phase 4**: ESTRADIOL — activation steering via residual stream read/write

## Status

**v0.1.0** — Core inference engine operational. 849 tests. LinGO-native GSU with ERG 2025 support. ADSR modulation matrix with 12 source signals. Latched transition policy for base model tool calling. BFCL benchmark integration.

## License

MIT

## Author

Maddy Muscari / [Ontologi LLC](https://ontologi.dev)
