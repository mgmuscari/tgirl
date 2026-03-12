# PRD: Multi-Agent Platform Parity

## Status: IMPLEMENTED
## Author: agent (Proposer)
## Date: 2026-02-11
## Branch: feature/multi-agent-parity

## 1. Problem Statement

Ontologi Push Hands is architecturally coupled to Claude Code. Every execution surface — slash commands (`.claude/commands/`), agent definitions (`.claude/agents/`), agent teams (`TeamCreate`/`SendMessage`), tool-level stance enforcement, and permission sandboxing — is Claude Code-specific. The spec itself says "designed for solo developers and small teams building with Claude Code (or comparable agentic coding tools)" but never delivers on the parenthetical.

This matters because:

1. **The methodology is valuable independent of the runtime.** The PRD → PRP → review-gate → implementation pipeline, the stance-based dialectical review, and the tiered workflow system are agent-agnostic ideas locked inside a Claude Code-specific implementation.

2. **The AGENTS.md convention is gaining traction.** OpenCode, Cursor, and other tools are converging on `AGENTS.md` as a cross-platform agent configuration standard. Push Hands should meet developers where they are.

3. **Teams expand what's possible.** Claude Code agent teams enable concurrent dialectical workflows (review-plan-team, execute-team, security-audit-team). Other platforms lack this capability. Rather than abandoning the dialectical model for those platforms, we can design a portable execution layer where the sequential commands work everywhere and team commands work where the platform supports them.

**Who needs this:** Any developer using Push Hands methodology with an AI coding agent other than Claude Code — or developers who want to evaluate Push Hands without committing to Claude Code as their primary tool.

## 2. Proposed Solution

Introduce a **platform adapter layer** that decouples the Push Hands methodology from Claude Code's runtime primitives. The core idea: every Push Hands workflow is expressed as a **portable prompt template** that any agent can consume, plus **platform-specific adapters** that wire those templates into each agent's native command/invocation system.

### Architecture: Three Layers

```
┌─────────────────────────────────────────────────┐
│  Platform Adapters                               │
│  .claude/commands/  .agents/opencode/  .cursor/  │
│  (wiring layer — platform-specific)              │
├─────────────────────────────────────────────────┤
│  Portable Prompt Templates                       │
│  prompts/{new-feature,generate-prp,...}.md        │
│  (the methodology — platform-agnostic)           │
├─────────────────────────────────────────────────┤
│  Shared Infrastructure                           │
│  docs/, scripts/, hooks, AGENTS.md, CLAUDE.md    │
│  (already portable)                              │
└─────────────────────────────────────────────────┘
```

**Layer 1: Shared Infrastructure** (already exists, ~95% portable)
- `docs/PRDs/`, `docs/PRPs/`, `docs/reviews/`, `docs/audits/` — pure markdown templates
- `scripts/` — shell scripts for branch management, hooks, setup
- `AGENTS.md` — stance definitions (portable concept, currently mixed with Claude Code tool specifics)
- Git hooks — language-agnostic shell scripts

**Layer 2: Portable Prompt Templates** (new)
- One markdown file per workflow stage in `prompts/`
- Contains the full instructions for each stage (what currently lives in `.claude/commands/`)
- Platform-neutral: references file paths, git operations, and document templates — never platform-specific tools
- Includes stance primes as inline sections (no dependency on `.claude/agents/`)
- These are the **canonical source of truth** for what each workflow stage does

**Layer 3: Platform Adapters** (new + refactored)
- Thin wiring layers that import from `prompts/` and adapt to each platform's native system
- Claude Code: `.claude/commands/*.md` import from `prompts/` and add Claude Code-specific syntax (agent spawning, team mode, tool restrictions)
- OpenCode: `.agents/opencode/` or equivalent, using AGENTS.md convention
- Cursor: `.cursor/` rules/commands pointing at the same prompt templates
- Generic/CLI: `scripts/run-stage.sh` that loads a prompt template and pipes it to any CLI-based agent (aider, opencode, etc.)

### Stance Portability

Current stance enforcement has two levels:
1. **Prompt-level** (portable): "You cannot write code — only test the plan's balance"
2. **Tool-level** (Claude Code-specific): Training partner agent definition omits Write/Edit tools

The portable prompt templates use prompt-level enforcement. Claude Code's adapter layer adds tool-level enforcement on top. Other platforms get behavioral enforcement only — degraded but functional.

### Team Mode Strategy

Team commands (`/review-plan-team`, `/execute-team`, `/security-audit-team`) remain Claude Code-exclusive capabilities. The portable layer ensures that every team workflow has a sequential equivalent that produces the same artifact types:

| Team Command | Sequential Equivalent | Artifact Path |
|---|---|---|
| `/review-plan-team` | `/review-plan` | `docs/reviews/plans/{slug}-review.md` |
| `/execute-team` | `/execute-prp` + `/review-code` | Implementation + `docs/reviews/code/{slug}-review.md` |
| `/security-audit-team` | `/security-audit` | `docs/audits/{slug}-audit.md` |

This is already true today. The change is making the sequential equivalents genuinely portable.

## 3. Architecture Impact

### New Files/Directories

```
prompts/                           # Portable prompt templates (new)
  stances/                         # Stance definitions (extracted from .claude/agents/)
    proposer.md
    training-partner.md
    code-reviewer.md
    security-auditor.md
    skeptical-client.md
  workflows/                       # Workflow stage prompts (extracted from .claude/commands/)
    new-feature.md
    generate-prp.md
    review-plan.md
    execute-prp.md
    review-code.md
    security-audit.md
    update-project-docs.md         # Renamed from update-claude-md (portable)
  README.md                        # How the prompt template system works

adapters/                          # Platform adapter configs + docs (new)
  claude-code/
    README.md                      # How Claude Code integration works
  opencode/
    README.md                      # How OpenCode integration works
    agents.md                      # OpenCode-formatted agent definitions
  cursor/
    README.md                      # How Cursor integration works
  generic/
    README.md                      # How to use with any CLI agent
    run-stage.sh                   # Shell wrapper for CLI agents
```

### Modified Files

| File | Change |
|---|---|
| `.claude/commands/*.md` | Refactor to import from `prompts/workflows/`, add Claude Code-specific wiring (team spawning, tool restrictions) |
| `.claude/agents/*.md` | Keep as-is (Claude Code-specific tool enforcement), but extract portable stance content to `prompts/stances/` |
| `AGENTS.md` | Refactor: remove Claude Code-specific tool references from the main stance definitions, add a "Platform-Specific Enforcement" section |
| `CLAUDE.md` | Update to reference new structure, add multi-platform section |
| `push-hands.md` | Update architecture section to describe three-layer model |
| `docs/guides/agent-teams.md` | No change (already scoped to Claude Code) |
| `scripts/setup.sh` | Add adapter detection/setup for non-Claude-Code platforms |
| `README.md` | Add supported platforms section |

### No New Dependencies

This feature is pure documentation and prompt engineering — no runtime dependencies, no packages, no build tools.

## 4. Acceptance Criteria

1. A `prompts/` directory exists containing portable prompt templates for all six sequential workflow stages (new-feature, generate-prp, review-plan, execute-prp, review-code, security-audit) plus stances.
2. Each portable prompt template is self-contained: a developer can copy-paste it into any AI coding agent's context and execute the workflow stage without referencing Claude Code-specific features.
3. Claude Code `.claude/commands/*.md` files are refactored to delegate to `prompts/workflows/` for methodology content, adding only Claude Code-specific wiring (slash command syntax, agent team spawning, tool lists).
4. Claude Code `.claude/agents/*.md` files remain unchanged (they are the tool-enforcement layer), but their behavioral content is also available in `prompts/stances/`.
5. An `adapters/` directory exists with at least: `claude-code/` (documenting current integration), `opencode/` (with AGENTS.md-formatted definitions), `cursor/` (with rules-based integration), and `generic/` (with a shell wrapper `run-stage.sh`).
6. `adapters/generic/run-stage.sh` can load any prompt template and pipe it to stdout (for use with `aider --message`, `opencode`, or any CLI agent that accepts prompt input).
7. `AGENTS.md` is refactored to separate portable stance definitions from platform-specific enforcement details.
8. `push-hands.md` spec is updated to describe the three-layer architecture (shared infra, portable prompts, platform adapters).
9. `README.md` includes a "Supported Platforms" section listing Claude Code (full support), OpenCode, Cursor, and generic CLI (sequential support).
10. `scripts/setup.sh` detects which platforms are available and reports which adapter(s) are active.
11. All existing Claude Code functionality continues to work identically — this is additive, not a rewrite.
12. The `/update-claude-md` command is generalized to `/update-project-docs` in the portable layer (updates CLAUDE.md, AGENTS.md, or equivalent per platform).

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| **Prompt template drift** — Claude Code commands diverge from portable templates over time | HIGH | Lint/CI check that `.claude/commands/` reference (not duplicate) `prompts/workflows/` content. Single source of truth enforcement. |
| **Stance degradation without tool enforcement** — Non-Claude-Code platforms rely on prompt-level "you cannot write code" which models may violate | MEDIUM | Document the limitation clearly. Prompt-level enforcement works well in practice for capable models. The degradation is behavioral, not catastrophic. |
| **Platform churn** — OpenCode, Cursor, etc. change their agent config formats | MEDIUM | Adapters are thin and documented. Keep the portable layer stable; adapters are cheap to update. |
| **Scope creep into runtime orchestration** — Temptation to build a cross-platform agent orchestrator | HIGH | Explicitly out of scope. The portable layer is prompts and docs, not a runtime. |
| **AGENTS.md convention not stabilized** — The cross-platform standard may shift | LOW | Track the convention. Current implementation is simple enough to adapt. |

## 6. Open Questions

1. **OpenCode AGENTS.md format:** What is the exact schema OpenCode expects? Do we need YAML frontmatter, specific field names, or just markdown? Need to verify against current OpenCode docs before writing the adapter.

2. **Cursor rules format:** Cursor has evolved its agent/rules system several times. What is the current (Feb 2026) recommended format for custom commands and agent roles? `.cursor/rules/` vs `.cursorrules` vs something else?

3. **Import mechanism for Claude Code commands:** Can `.claude/commands/*.md` reference external files (e.g., `{% include prompts/workflows/review-plan.md %}`), or must they be self-contained? If self-contained, the "import" may need to be a build step or a convention where commands explicitly read the prompt file.

4. **`update-claude-md` generalization:** Should the portable version update a generic `PROJECT.md` or should each platform adapter know its own config file name (CLAUDE.md, .cursorrules, etc.)?

## 7. Out of Scope

- **Cross-platform agent orchestrator/runtime.** This feature produces prompts and documentation, not a tool that runs agents.
- **Team mode for non-Claude-Code platforms.** Agent teams remain Claude Code-exclusive. Sequential equivalents serve other platforms.
- **Testing the portable prompts against every target platform.** Initial delivery validates Claude Code parity and provides adapters. Community testing against other platforms is expected.
- **IDE extension development.** No VS Code extensions, Cursor plugins, or similar.
- **Automated format conversion tooling.** Adapters are hand-authored, not generated.
