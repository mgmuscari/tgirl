# Ollama Feature Gap Analysis

**Date:** 2026-04-23
**Author:** Maddy Muscari (with Claude Opus 4.7 research agent)
**Purpose:** Inform v0.2 launch scoping by honestly comparing tgirl's current capabilities against Ollama's production surface. Used downstream as input for the plugin architecture PRD and revised launch thesis.

## 1. Ollama Feature Surface (April 2026)

### CLI

| Command | Purpose |
|---|---|
| `ollama serve` | Start the daemon (binds `127.0.0.1:11434`) |
| `ollama run <model>` | Pull if needed, load, REPL chat |
| `ollama pull <model>` | Download from registry |
| `ollama push <ns/model:tag>` | Upload to registry |
| `ollama ls` / `list` | List local models |
| `ollama show <model>` | Modelfile, parameters, template, license, `model_info`, `capabilities` |
| `ollama cp`, `ollama rm` | Copy / delete locally |
| `ollama create -f Modelfile <name>` | Build custom model |
| `ollama stop <model>` | Evict from memory immediately |
| `ollama ps` | Show loaded models + TTL + VRAM |
| `ollama signin` / `signout` | Auth with ollama.com |
| `ollama launch [claude\|openclaw\|hermes\|...]` | Zero-config integration bootstrap |

No `ollama embed` subcommand — embeddings are API-only.

### HTTP API

**Native endpoints (NDJSON streaming by default):**
- `POST /api/generate` — `prompt`, `suffix`, `images[]`, `format` (`"json"` or JSON Schema), `options`, `stream`, `keep_alive`, `think`, `raw`
- `POST /api/chat` — `messages[]`, `tools[]` (OpenAI function-schema), `format`, `options`, `stream`, `keep_alive`. Response emits `message.tool_calls[]` and `message.thinking` for reasoning models
- `POST /api/embed` — accepts string or array, `dimensions`, `truncate`; returns `embeddings[][]`
- `POST /api/embeddings` — legacy singular-`prompt` form, retained
- `GET /api/tags` — local models
- `POST /api/show` — metadata + `capabilities[]` (`tools`, `vision`, `embedding`, `thinking`)
- `POST /api/create` — `from`, `files` (blob digests), `adapters`, `template`, `license`, `system`, `parameters`, `quantize`
- `POST /api/copy`, `DELETE /api/delete`
- `POST /api/pull`, `POST /api/push` — streaming status
- `HEAD /api/blobs/:digest`, `POST /api/blobs/:digest` — blob-addressed content upload
- `GET /api/ps`, `GET /api/version`

**OpenAI-compat layer:**
- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/models/{model}`, `/v1/embeddings`
- Experimental: `/v1/images/generations`, `/v1/responses` (non-stateful only)
- Supported chat params: `model, messages, temperature, top_p, max_tokens, stream, seed, stop, frequency_penalty, presence_penalty, response_format, tools, reasoning_effort`
- **Not supported on `/v1`:** `tool_choice`, `response_format: json_schema` (only basic `json_object`), `logprobs`, stateful `/v1/responses`

### Model Management

- Registry: `ollama.com/library` default. Models addressed as `[namespace/]model:tag`. `:latest` is the default tag convention. Quant suffixes as tags (`:q4_K_M`, `:q8_0`, `:fp16`).
- Content-addressed on disk: each model is a manifest of SHA256-digested blobs (weights, template, params, license). Pull resolves manifest → downloads missing blobs → links into store.
- Storage paths:
  - macOS: `~/.ollama/models`
  - Linux: `/usr/share/ollama/.ollama/models`
  - Windows: `C:\Users\%USERNAME%\.ollama\models`
  - Override: `OLLAMA_MODELS`
- Private registry support is not officially documented beyond DNS-redirect / reverse-proxy tricks.
- Cloud ("hosted") models available with auth-keyed access; structured outputs currently unsupported on cloud models.

### Modelfile

Directives: `FROM` (base model or local `./path.gguf` / safetensors dir), `PARAMETER`, `TEMPLATE`, `SYSTEM`, `ADAPTER` ((Q)LoRA), `LICENSE`, `MESSAGE` (seed history), `REQUIRES` (min Ollama version).

Documented `PARAMETER` keys: `num_ctx` (default 2048), `repeat_last_n` (64), `repeat_penalty` (1.1), `temperature` (0.8), `seed`, `stop`, `num_predict` (-1), `top_k` (40), `top_p` (0.9), `min_p` (0.0). Wider set via API `options`.

Multi-modal projectors travel with `FROM` base. No separate `PROJECTOR` directive. No conditional logic. Single base only. No Modelfile-level tool-schema declaration (tools are per-request only).

### Runtime Behavior

- **Keep-alive:** default 5 minutes. Override via `OLLAMA_KEEP_ALIVE` env (duration string; `-1` = indefinite; `0` = unload immediately) or `keep_alive` in request body. `ollama stop` forces eviction.
- **Multi-model concurrency:**
  - `OLLAMA_NUM_PARALLEL` — concurrent requests per model (default 1; memory scales)
  - `OLLAMA_MAX_LOADED_MODELS` — default `3 × GPU_count`, or 3 on CPU
  - `OLLAMA_MAX_QUEUE` — 512
- **Context length:** `OLLAMA_CONTEXT_LENGTH` env, `/set parameter num_ctx` in REPL, or `options.num_ctx` in API (default 4096 on serve; 2048 in Modelfile PARAMETER default)
- **GPU offload:** `options.num_gpu` = layers; autodetected on first load
- **Flash attention:** `OLLAMA_FLASH_ATTENTION=1`
- **Bind:** `OLLAMA_HOST` (default `127.0.0.1:11434`). No built-in auth.
- **Apple Silicon:** native MLX runner in preview since 2026-03-30; v0.21.1+ adds logprobs + fused top-p/top-k to the MLX runner.

### Tool Calling / Structured Output

- `/api/chat` `tools[]` accepts OpenAI-style function schemas. Response emits `message.tool_calls[]` when the model decides to call. Also on `/v1/chat/completions` (but without `tool_choice`).
- **Mechanism: template-pattern-matched, NOT grammar-constrained.** The parser "directly references each model's template to understand the prefix of the tool call" and parses structured content from the model's own output. Relies on the fine-tuned model emitting a conventional token prefix.
- **Structured outputs:** `format` accepts `"json"` (loose) or a JSON Schema. Under the hood Ollama uses GBNF internally for both, converting JSON Schema to a constrained grammar. **Arbitrary user-supplied GBNF is NOT exposed** — PRs adding this (e.g. PR #1606) have been rejected as a product decision. Cloud endpoints don't support structured outputs.
- **Empirical reliability:**
  - Issue #5769: streaming `tool_calls` deltas not emitted correctly on `/v1` layer — breaks many agent frameworks
  - Issue #15258: v0.20.0 `/api/chat` and `/api/generate` hang on Apple Silicon M4 (patched)
  - Tool-call fidelity improves sharply at `num_ctx ≥ 32k`

### Integrations Ecosystem

Official SDKs: `ollama` (Python), `ollama` (JS/TS). Framework adapters: `langchain-ollama`, LlamaIndex `Ollama` LLM class, Haystack, Firebase Genkit, Langfuse. Language bindings: LangChainDart, Ollama4j (Java), Ollama for Laravel (PHP), LLPhant (PHP), Ollama for Swift, Ollama-hpp (C++), R. UIs: Open WebUI, Raycast, Continue, VS Code extensions, macOS app (daemon), OpenClaw, Claude Code compat (Anthropic Messages API layer), OpenAI Codex CLI via `/v1`, Kimi CLI.

### Recent Noteworthy Additions (~last 6 months)

- MLX runner on Apple Silicon (preview Mar 2026; v0.21.1 added logprobs + fused sampling)
- `ollama launch` (Jan 2026) — zero-config integration bootstrap
- Image generation (experimental, Jan 2026, macOS only)
- Anthropic Messages API compat layer
- Streaming tool calling with parallel calls (late 2025; refined through v0.20.6)
- Web search API (Sep 2025; free tier + cloud rate limits)
- New scheduler (Sep 2025; better multi-GPU, fewer OOMs)
- Cloud models preview (Sep 2025)
- Structured outputs with JSON Schema (Q1 2026, v0.18)
- Thinking/reasoning surface: `think` param, `reasoning_effort`, `message.thinking`

### Known Limitations / Common Complaints

- **No arbitrary grammar exposure.** GBNF is internal-only; product team has declined community PRs. Single biggest gap for structured-decoding use cases.
- **Tool calling is template-pattern-matched, not grammar-constrained** — reliability depends on the model and prompt context size; fails on base / non-instruct models.
- **Streaming `tool_calls` on `/v1` buggy** (issue #5769) — many frameworks fall back to non-streaming.
- **Concurrency is single-replica-per-model.** `OLLAMA_NUM_PARALLEL>1` batches within one engine; no multi-instance sharding. vLLM / SGLang dominate high-QPS.
- **No built-in authentication.** `OLLAMA_HOST` exposure has led to documented mass-exploitation of internet-exposed instances (Jan 2026 incident).
- **Per-request model override incurs cold-start.** Swapping between loaded models triggers load/unload cycles.
- **No per-request sampler overrides beyond the documented `options` set** on `/v1` (no Mirostat 2.0 variants, no typical sampling).
- **No `tool_choice` on `/v1` layer.**
- **`response_format: json_schema` not on `/v1` layer** (only on native `format` field).
- **Private/self-hosted registries:** not officially supported/documented beyond DNS tricks.
- **Cloud models:** structured outputs currently unsupported.

## 2. Gap Analysis — What tgirl Has / Lacks

### tgirl capabilities already present (some informal)

| Capability | tgirl state | Notes |
|---|---|---|
| OpenAI-compat chat API | ✓ | `serve.py` `/v1/chat/completions` with SSE streaming |
| Model auto-load | ✓ | `tgirl serve --model <hf_id>` downloads from HF Hub on first run |
| BFCL benchmark runner | ✓ | `benchmarks/run_bfcl.py` standalone |
| Arbitrary grammar constraint | ✓ (**Ollama can't do this**) | `/grammar/preview`, `llguidance` backend via `outlines_adapter.py` |
| Tool calling via grammar | ✓ (**Ollama can't do this correctly**) | Grammar-constrained tool emission; works on base models |
| Compositional tool pipelines | ✓ (**No Ollama equivalent**) | Hy s-expressions, single-pass multi-tool composition |
| Optimal transport logit redistribution | ✓ (**No Ollama equivalent**) | `transport.py` / `transport_mlx.py` |
| Steering endpoints | ✓ (experimental) | `/v1/steering/*` — ESTRADIOL self-steering |
| MLX-native inference | ✓ | Also newly available in Ollama since Mar 2026 |

### Adoption-blocking gaps (users expect these from "a local server")

| Capability | Present in Ollama | tgirl state | Effort |
|---|---|---|---|
| `pip install` packaging | yes (`ollama` binary) | partial (`tgirl[serve,mlx]` extras) | S |
| Model management (pull/list/rm/show) | yes | no; passes `--model` per start | L (requires model store) |
| Daemon + hot-swap / keep-alive | yes | no; process-per-serve | L |
| Interactive chat CLI (`ollama run`) | yes | no | M |
| Embeddings endpoint | yes | no | M |
| Modelfile / composition format | yes | no | M (but disputable need) |
| Private registry / auth | limited | no | L |

### Comparative gaps (Ollama has it but badly — potential leapfrog opportunities)

- **Built-in auth.** Ollama lacks. A minimal auth layer (API key + rate limit) is L effort and would be a real positioning point.
- **Concurrency.** Ollama single-replica-per-model. Multi-replica sharding of the same model is beyond v0.2 but worth noting.
- **`tool_choice` + `response_format: json_schema` on `/v1`.** Ollama doesn't support these on OpenAI-compat. tgirl can, because grammar is first-class.
- **Arbitrary grammar exposure.** Ollama *refuses to ship this*. tgirl already has it. Making it first-class in the API is S effort.

### tgirl-unique capabilities (no Ollama equivalent — positioning moat)

- **User-supplied GBNF / Lark grammars in the API.** Direct extension of `/grammar/preview`.
- **Base-model tool calling.** Ollama's template-pattern approach can't do this; tgirl's grammar-constrained approach can.
- **Compositional Hy pipelines.** Single turn, multiple tools in one s-expression, grammar-constrained.
- **Inline grammar-switched computation** (proposed in plugin architecture PRD — not yet implemented). Letter-counting, arithmetic, string ops executed by a verified runtime subroutine in the middle of natural-language generation. No Ollama equivalent.

## 3. Implications for v0.2 Positioning

**The current "Apple Silicon Server MVP" PRD frames v0.2 as a competitor to Ollama on UX maturity.** This comparison is unwinnable in the short term: Ollama has years of daemon / model-management polish, an ecosystem of SDKs and UI adapters, and a recognizable brand. Competing head-on requires parity with features that have nothing to do with tgirl's unique value.

**The actual wedge is sharper and narrower:** tgirl is *the local server that does what Ollama won't*. Specifically:
1. **Arbitrary grammar exposure in the API** — Ollama has refused this.
2. **Grammar-constrained tool calling that works on base models** — Ollama's mechanism can't do this.
3. **Compositional tool pipelines** — single turn, multiple tools, structurally validated.
4. **Inline grammar-switched computation** — if we ship it, a positioning moat with no peer in the local-server space.

**Recommendation:** treat Ollama parity features as *explicitly deferred*, not as aspirational. The launch narrative leans into what the grammar-first architecture makes possible — not into what makes tgirl look like a second-place Ollama.

### Features to EXPLICITLY DEFER from v0.2

- Daemon + hot-swap + `keep_alive` (Ollama turf; 10× our runway)
- `tgirl pull / list / rm` model management (expect users to pass HF Hub IDs)
- Modelfile / Ollama-style composition (chat templates from HF repos cover this)
- Interactive chat REPL (not our differentiator)
- Private registry support

### Features to KEEP in v0.2 scope

- `pip install tgirl[serve,mlx]` packaging polish
- OpenAI-format `tools` in request body (already planned; grammar-constrained adapter)
- `tgirl bench bfcl` CLI subcommand wrapping the existing runner
- Evidence-gate BFCL runs (9B base vs instruct)
- Launch writeup citing committed numbers

### Features to ADD to v0.2 scope (new, from the inline-compute discussion)

- **Plugin architecture** — formalize `@tool()` into a capability-tiered public plugin API
- **Stdlib v1** — bundled I/O-free tool pack (math, strings, lists, comparison, type coercion)
- **Inline Hy executor** — grammar-switched mid-generation computation. Uses `inference-irq-controller` substrate
- **Letter-counting / arithmetic demo** in the launch writeup — the narrative hook

### Launch narrative revision

Old: *"Apple Silicon local server for grammar-constrained tool calling."*
New: *"The local chatbot that can't miscount R's in strawberry. Because a verified Hy subroutine is doing it, not the model. Grammar-switched inline computation, on Apple Silicon, locally."*

The BFCL 9B base vs instruct evidence gate survives but becomes proof the grammar substrate is sound — not the lead claim. The lead claim is the demo.

## 4. Sources

- [ollama/ollama README](https://github.com/ollama/ollama/blob/main/README.md)
- [docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [docs/cli.mdx](https://github.com/ollama/ollama/blob/main/docs/cli.mdx)
- [docs/modelfile.mdx](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx)
- [docs/openapi.yaml](https://github.com/ollama/ollama/blob/main/docs/openapi.yaml)
- [docs/faq.mdx](https://github.com/ollama/ollama/blob/main/docs/faq.mdx)
- [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Tool calling capability](https://docs.ollama.com/capabilities/tool-calling)
- [Structured outputs capability](https://docs.ollama.com/capabilities/structured-outputs)
- [Streaming tool blog](https://ollama.com/blog/streaming-tool)
- [MLX blog](https://ollama.com/blog/mlx)
- [Releases](https://github.com/ollama/ollama/releases)
- [Issue #5769 — streaming tool_calls bug](https://github.com/ollama/ollama)
- [Issue #6237 — Ollama stance on grammar feature](https://github.com/ollama/ollama/issues/6237)
- [Issue #15258 — v0.20.0 Apple Silicon hang](https://github.com/ollama/ollama/issues/15258)
- [Issue #9054 — no multi-instance concurrency](https://github.com/ollama/ollama/issues/9054)
- [PR #1606 — rejected GBNF support](https://github.com/ollama/ollama/pull/1606)
- [Running Ollama in production (aicompetence)](https://aicompetence.org/ollama-production-limitations/)
- [MLX vs llama.cpp vs Ollama (contracollective)](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026)
- [Comparative study arXiv 2511.05502](https://arxiv.org/pdf/2511.05502)
