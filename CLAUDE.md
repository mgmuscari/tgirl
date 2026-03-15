# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Operating Principles

**Default stance: Team Lead.** When receiving user instructions or feedback, orchestrate dialectical teams — do not solo-implement. Use `/new-feature` to initiate work. Use team commands (`/review-plan-team`, `/execute-team`) for standard/full tier. The proposer implements; the training partner tests balance; the team lead orchestrates.

**Pressure demands the *right* structure.** Scope pressure → tighten methodology (more review, more planning). Performance/bug pressure → tighten feedback loops (faster iteration, empirical validation). Never abandon structure — match it to the problem type.

**The dialectic is the product.** The structured tension between proposer and training partner is not overhead — it is the mechanism that produces quality. Bypassing it is like skipping the partner and practicing alone.

**TDD is mandatory.** All implementation follows RED → GREEN → REFACTOR. Tests are written before implementation code. Never mock to make tests pass — fix the real issue. Never weaken tests to get green.

## Project Overview

tgirl (**Transformational Generator for Inference-Restricting Languages**) is a Python library for local LLM inference with grammar-constrained compositional tool calling. It is not an application or framework — it is a library that other systems import and use.

The library supports two generation modes within a single inference session:

- **Freeform mode**: unconstrained natural language generation (thinking, conversing, explaining)
- **Constrained mode**: grammar-constrained Hy s-expression tool pipelines with mathematical safety guarantees

The model generates freely until it needs to invoke tools, then the grammar dynamically constrains output to only permit well-formed Hy s-expressions composing registered tools within type, quota, and scope constraints. After execution, the model returns to freeform generation.

**Core claim**: grammar-constrained compositional generation produces valid tool call pipelines at lower token cost, fewer inference round-trips, and zero structural error rate compared to sequential JSON/XML tool calling — including on base (non-instruct) models.

The canonical specification is `TGIRL.md` (the technical requirements document).

## Module Architecture

Core modules, each independently importable and testable:

```
tgirl/
├── registry.py          # Tool registration, type extraction, snapshots
├── grammar.py           # Dynamic CFG generation from registry state (Jinja2 templates)
├── compile.py           # Hy parsing, AST compilation, sandboxed execution
├── transport.py         # Optimal transport logit redistribution (torch, zero coupling)
├── transport_mlx.py     # OT redistribution — MLX-native (zero coupling)
├── sample.py            # Constrained sampling engine, hooks, telemetry (torch)
├── sample_mlx.py        # Constrained sampling — MLX-native (zero torch/numpy in hot loop)
├── outlines_adapter.py  # llguidance bridge: GrammarState (torch) + GrammarStateMlx
├── cache.py             # KV cache management, forward_fn factories (HF + MLX)
├── rerank.py            # Grammar-constrained tool reranking
├── instructions.py      # System prompt generation from registry snapshots
├── format.py            # Chat template formatting (ChatML, plain)
├── bfcl.py              # BFCL benchmark integration
├── bridge.py            # MCP compatibility layer (import/export)
├── serve.py             # Optional: FastAPI local inference server
├── cli.py               # Optional: CLI entrypoint
├── templates/           # Jinja grammar templates (base, tools, types, composition)
├── telemetry.py         # Telemetry data structures and logging
└── types.py             # Shared type definitions
```

### Dependency Graph

```
registry  (pydantic, annotated-types)
    ↓
grammar   (jinja2, depends on registry)
    ↓
compile   (hy, RestrictedPython, depends on registry)
    ↓
transport      (torch, POT — NO other tgirl deps)
transport_mlx  (mlx — NO other tgirl deps except TransportConfig)
    ↓
sample      (torch — depends on grammar, transport)
sample_mlx  (mlx — depends on transport_mlx, zero torch/numpy)
    ↓
outlines_adapter  (llguidance — torch via llguidance.torch, mlx via llguidance.mlx)
    ↓
cache    (transformers | mlx-lm — forward_fn factories)
rerank   (depends on sample, grammar)
    ↓
bridge   (mcp — depends on registry)
    ↓
serve    (fastapi, transformers — depends on all above)
```

All modules use `structlog` for structured logging. `transport`/`transport_mlx` must have zero coupling to grammar or registry — they operate on raw logit tensors and valid token sets. `sample_mlx` has zero torch and zero numpy imports; all per-token math stays in `mx.array`.

## Setup

```bash
./scripts/setup.sh    # Install git hooks, verify directory structure and templates
```

### Python Environment

- Python 3.11, 3.12, 3.13 (test against all three)
- Pin `hy>=1.0,<2.0`
- Core deps: `pydantic`, `annotated-types`, `structlog`, `jinja2`, `hy`, `RestrictedPython`, `torch`, `POT`, `outlines`, `llguidance`
- MLX deps (Apple Silicon): `mlx`, `mlx-lm`, `llguidance` (includes `llguidance.mlx`)
- Optional deps: `fastapi`, `transformers`, `mcp`
- Dev tools: `ruff`, `mypy`, `pytest`, `hypothesis`

## Key Design Constraints

- **Safety by construction, not validation.** Invalid tool calls are inexpressible at the token level, not caught at runtime. Quotas enforced in grammar, not post-hoc.
- **Model-agnostic.** Any model with a probability distribution over a vocabulary can be grammar-constrained. No instruct tuning required.
- **Composable over sequential.** Single compositional Hy expression per pipeline, not multiple inference passes.
- **transport is independent.** `tgirl.transport` must be usable without any other tgirl module. Test this isolation.
- **Sandbox is defense-in-depth.** Grammar prevents invalid expressions; static analysis (Hy AST + Python AST) catches template bugs; execution sandbox restricts namespace. All three layers required.
- **Deterministic grammars.** Same registry snapshot must produce same grammar. Grammar diffing is a first-class feature.

## Development Lifecycle Pipeline

Four workflow tiers (light/iterative/standard/full) match process weight to change size. Default is standard. Tier metadata is stored in `.push-hands-tier` on feature branches — this file must never reach `main`.

```
Light:     Feature Branch → Implement (TDD) → /review-code (optional) → PR → Merge
Iterative: Feature Branch → Implement (TDD) → Benchmark/Test → Fix → Loop → /review-code → PR → Merge
Standard:  /new-feature → PRD + PRP → /review-plan → /execute-prp (TDD) → /review-code → PR → Merge
Full:      Standard pipeline + /security-audit (expected) → PR → Merge
```

Each arrow is a gate. Work does not proceed until the gate passes.

**Iterative tier** — For benchmark-driven optimization, performance investigation, and bug hunt loops. Direct implementation allowed, TDD mandatory, code review before PR. The developer is the training partner in real-time. Use `/investigate` to enter this workflow.

**Security audits are strongly recommended** for work touching `compile.py` (sandbox), `transport.py` (numerical correctness), and `sample.py` (sampling loop integrity).

### Team Mode (default for standard/full tiers)

```
Team Review:  /review-plan-team → concurrent PRP review + revision
Team Execute: /execute-team → message-gated implementation (TDD) + incremental review
Team Audit:   /security-audit-team → dual-agent security audit with peer challenge
```

Requires: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in Claude Code settings.

**Known team bugs and hardening:**
- Agent definitions don't load for team members (bug #24316). Team commands inline stance definitions.
- Model inheritance is unreliable (bug #32368). All team spawns require explicit `model: "opus"`.
- Team commands include health checks; teams abort rather than silently degrade.

## Agent Stances (AGENTS.md)

Five defined stances with distinct optimization targets:

- **Proposer** — Completion-oriented, used for `/new-feature`, `/generate-prp`, `/execute-prp`
- **Senior Training Partner** — Senses structural weakness in plans, used for `/review-plan`. Cannot write code.
- **Code Review Partner** — Reviews diff against PRP spec, used for `/review-code`
- **Security Auditor** (hard stance) — Exploit-minded, requires PoC for HIGH+ findings
- **Skeptical Client** (hard stance) — Challenges severity inflation, demands proof

## Conventions

- Branches: `feature/<slug>` from `main`
- Commits: [Conventional Commits](https://www.conventionalcommits.org/), max 72 chars first line
- PRDs: `docs/PRDs/<feature-slug>.md`
- PRPs: `docs/PRPs/<feature-slug>.md` (matches PRD slug)
- Reviews: `docs/reviews/{plans,code}/<feature-slug>-review.md`
- Audits: `docs/audits/<feature-slug>-audit.md`
- Tier metadata: `.push-hands-tier` (feature branches only, never on `main`)
- Slugs: lowercase, hyphen-separated, alphanumeric only (`^[a-z0-9]+(-[a-z0-9]+)*$`)

## Execution Rules

- Each PRP task = one atomic commit (test + implementation together)
- TDD is mandatory: RED (write failing test) → GREEN (implement to pass) → REFACTOR → COMMIT
- The PRP must include a `Test Command` field — if missing, ask the user before proceeding
- If no test framework exists yet, the first task must set one up
- If validation fails, fix before proceeding — never mock to make tests pass
- Never weaken or delete tests to get green — fix the implementation instead
- If stuck (3+ failed attempts on same task), stop and flag for human review
- After any developer correction, update CLAUDE.md Known Gotchas via `/update-claude-md`
- **No cross-framework conversions.** If a function needs both MLX and torch, implement two variants with matching interfaces. Never `mx.array(tensor.numpy())` or `torch.from_numpy(np.array(mx_array))` in any code path. Each variant uses only its own framework's native operations.
- **No Python-fu on tensor data.** No list comprehensions, no `for v in tensor`, no `sum(list)` when the data came from a tensor. Use the framework's native ops (`mx.sum`, `torch.softmax`, etc.). If you must convert to Python, use `.tolist()`, never element-wise iteration.
- **No "fix later" shims.** If a task requires functionality that doesn't exist yet (e.g., a missing registry method, a missing type mapping), implement it as part of the task. Never stub it, never label it "out of scope," never implement a workaround. The shim becomes load-bearing debt.
- **Performance-aware implementation.** Any function that runs per-token or touches tensors with vocab-sized dimensions must be profiled mentally: if it iterates over 248k elements in Python, it's wrong. Use native framework ops.

## Methodology Enforcement Hooks

Three hooks in `.claude/hooks/` enforce the methodology:

1. **`push-hands-guard.sh`** — Fires on `UserPromptSubmit`. Detects implementation-intent keywords and injects methodology reminders.
2. **`block-solo-implementation.sh`** — Fires on `PreToolUse` for Edit/Write. Blocks direct edits to `src/` and `tests/` on standard/full tier feature branches.
3. **`enforce-opus-teams.sh`** — Fires on `PreToolUse` for Agent. Denies team agent spawns without `model: "opus"`.

## Known Gotchas

2026-02-10: GitHub Actions `${{ }}` expressions in `run:` blocks are script injection vectors when using attacker-controlled values like `github.head_ref` → Always pass untrusted context via `env:` variables, never inline interpolation

2026-02-10: `grep -qE "$PATTERN" "$FILE"` checks ALL lines, not just the first → Use `head -1 "$FILE" | grep -qE "$PATTERN"` when validating only the first line (e.g., commit message hooks)

2026-03-01: Claude Code's built-in TaskCreate system reminders can override CLAUDE.md operating principles → The `push-hands-guard.sh` hook counteracts this by injecting methodology reminders on UserPromptSubmit before system nudges take effect

2026-03-09: Claude Code agent definitions (`.claude/agents/*.md`) don't load for team members — `subagent_type` is ignored when `team_name` is set (bug #24316, OPEN) → Team commands inline stance definitions in spawn prompts as a workaround

2026-03-09: Claude Code `model: inherit` doesn't resolve properly for agent spawns (bug #32368) → Use `model: opus` explicitly in all agent definitions and team spawn calls

2026-03-09: `~/.claude/teams/` directory detection is unreliable for checking active teams → `block-solo-implementation.sh` uses tier + branch + file path checks instead

2026-03-14: **All tensor/matrix math must use the accelerated library for the context (MLX or PyTorch). Never use Python list comprehensions on tensor data.** `[float(v) for v in mx_array]` on 248k elements takes 8.3 seconds; `mx_array.tolist()` takes 6ms — a 1,400x difference. Conversion between linalg libraries (e.g., mx→torch, torch→numpy) must be avoided unless there is absolutely no alternative; exhaust the library's own API and documentation before resorting to cross-library conversion.
