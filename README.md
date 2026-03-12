# Ontologi Push Hands

A structured, dialectical, AI-assisted software development lifecycle designed for solo developers and small teams building with Claude Code, OpenCode, Cursor, Windsurf, Cline, Aider, or other agentic coding tools.

The name comes from **tui shou** (推手), the t'ai chi practice where two partners test each other's structural integrity through continuous contact. In push hands, you don't fight your partner — you *listen* through pressure. The same dynamic governs this methodology: agent pairs probe each other's work, finding where the structure is weak by applying sustained, calibrated pressure at every stage of development.

## Core Thesis

**Structured tension between training partners produces better outcomes than single-agent workflows at every stage of the development lifecycle.**

## Quick Start

### For a New Project

1. Create a new repository from this template on GitHub
2. Run `./scripts/setup.sh`
3. Edit `CLAUDE.md` with your project's details
4. Edit `AGENTS.md` if you want to customize agent stances
5. Start with `/new-feature` for your first feature

### For an Existing Project

1. Copy `docs/`, `scripts/`, `prompts/`, and `.claude/` directories into your project
2. Copy `CLAUDE.md` and `AGENTS.md` to your project root
3. Copy adapter directories for your platform (e.g., `.opencode/`, `.cursor/`) from `adapters/`
4. Run `./scripts/setup.sh`
5. Populate `CLAUDE.md` with your project's existing architecture and conventions

## Development Lifecycle

Three tiers match process weight to change size:

```
Light:    Feature Branch → Implement → PR
Standard: /new-feature → PRD → PRP → Plan Review → Implement → Code Review → PR
Full:     /new-feature → PRD → PRP → Plan Review → Implement → Code Review → Security Audit → PR
```

Default is **standard**. Use `--tier light` for small changes, `--tier full` for security-sensitive work. Each arrow is a gate — work does not proceed until the gate passes.

## Slash Commands

| Command | Tier | Input | Output | Stance |
|---------|------|-------|--------|--------|
| `/new-feature [--tier light\|standard\|full] <brief>` | All | Description | PRD (or branch only for light) | Proposer |
| `/generate-prp <prd>` | Standard, Full | PRD path | `docs/PRPs/<slug>.md` | Proposer |
| `/review-plan <prp>` | Standard, Full | PRP path | `docs/reviews/plans/<slug>.md` | Senior Training Partner |
| `/execute-prp <prp>` | Standard, Full | PRP path | Source code + tests | Proposer |
| `/review-code` | All (optional for Light) | Branch diff | `docs/reviews/code/<slug>.md` | Code Review Partner |
| `/security-audit` | Full (optional for Standard) | Branch diff | `docs/audits/<slug>.md` | Auditor + Client |
| `/update-claude-md` | All | — | Updated CLAUDE.md | Proposer |
| `/review-plan-team <prp>` | Standard, Full | PRP path | Revised PRP + plan review | Team: Proposer + Training Partner |
| `/execute-team <prp>` | Standard, Full | PRP path | Code + incremental review | Team: Proposer + Code Reviewer |
| `/security-audit-team` | Full | Branch diff | Audit report | Team: Auditor + Client |

## Supported Platforms

| Platform | Slash Commands | Agent Stances | Tool Enforcement | Agent Teams |
|----------|---------------|---------------|-----------------|-------------|
| Claude Code | Native | Native (tool-level) | Native | Native |
| OpenCode | Via prompt templates | Native (permission-level) | Native | No |
| Cursor | Via prompt templates | Rules (.mdc) | No (behavioral) | No |
| Windsurf | Via prompt templates | Rules (.md) | No (behavioral) | No |
| Cline | Via prompt templates | Rules (.clinerules) | No (behavioral) | No |
| Aider | Via prompt templates | Context loading | No (behavioral) | No |
| Other CLI | Via run-stage.sh | Context loading | No (behavioral) | No |

See `adapters/` for platform-specific setup and `prompts/` for portable workflow templates.

## Customization

- **Language/framework:** Replace `ruff`/`mypy`/`pytest` in hooks and validation gates with your stack's equivalents
- **Stances:** Edit `AGENTS.md` to adjust agent character and constraints, or customize `prompts/stances/` for portable definitions
- **Workflow tiers:** Use `--tier light` for small changes, `--tier full` for security-sensitive work. See [push-hands.md](push-hands.md) Section 4.1 for details.
- **Security audit frequency:** Per-feature for security-sensitive apps (use full tier), per-release for others
- **Platform adapters:** See `adapters/` for platform-specific configuration. Add new adapters by mapping `prompts/` templates into your platform's native format.
- **Team mode:** Run proposer and training partner stances concurrently using agent teams. Requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. See [docs/guides/agent-teams.md](docs/guides/agent-teams.md).

## Design Principles

1. **Tension is a practice, not a problem.** Every stage has a proposer and a training partner.
2. **Documents are the product.** Code is generated from documents. Invest in documents.
3. **Git is context engineering.** Branch structure and commit discipline manage agent context.
4. **Living over static.** CLAUDE.md, PRDs, and PRPs are updated by agents as they learn.
5. **Observable by default.** Every action logged, every decision has a paper trail.

See [push-hands.md](push-hands.md) for the complete PRD.

## License

MIT
