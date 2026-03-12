# Portable Prompt Templates

This directory contains the canonical, platform-agnostic definition of the Push Hands methodology. These templates are the source of truth — platform adapters (in `adapters/`) wire them into each platform's native format.

## Architecture

Push Hands uses a three-layer architecture:

1. **Shared Infrastructure** (already portable): `docs/`, `scripts/`, hooks, git workflow
2. **Portable Templates** (this directory): Stances and workflows expressed in platform-neutral language
3. **Platform Adapters** (`adapters/`): Thin wiring that maps templates into each platform's native format

## Stances (`stances/`)

Stances are behavioral definitions that shape how an agent approaches a task. They are context primes, not role descriptions — the framing shapes the model's attentional quality.

| Stance | Character | Used For |
|--------|-----------|----------|
| [Proposer](stances/proposer.md) | Thorough, systematic, completion-oriented | Feature creation, PRP generation, implementation |
| [Training Partner](stances/training-partner.md) | Patient, perceptive, structurally attuned | Plan review |
| [Code Reviewer](stances/code-reviewer.md) | Detail-oriented, convention-aware, quality-sensing | Code review |
| [Security Auditor](stances/security-auditor.md) | Exploit-minded, severity-calibrated (hard stance) | Security audit — attacker role |
| [Skeptical Client](stances/skeptical-client.md) | Budget-conscious, dubious, demands proof (hard stance) | Security audit — challenger role |

## Workflows (`workflows/`)

Workflows are step-by-step instructions for each pipeline stage. Each workflow includes a stance prime, input description, methodology instructions, artifact template, and validation commands.

| Workflow | Stage | Stance |
|----------|-------|--------|
| [new-feature](workflows/new-feature.md) | Feature brief to PRD | Proposer |
| [generate-prp](workflows/generate-prp.md) | PRD to implementation blueprint | Proposer |
| [review-plan](workflows/review-plan.md) | PRP review | Training Partner |
| [execute-prp](workflows/execute-prp.md) | PRP to source code | Proposer |
| [review-code](workflows/review-code.md) | Implementation review | Code Reviewer |
| [security-audit](workflows/security-audit.md) | Security assessment | Auditor + Client |
| [update-project-docs](workflows/update-project-docs.md) | Lesson capture | Proposer |

## How to Use These Templates

### With any AI coding agent (direct use)

1. Copy the relevant workflow template content into your agent's context or chat
2. Replace `{input}` with your actual input (e.g., feature description, PRP path)
3. Follow the instructions

### With a platform adapter

See `adapters/` for platform-specific setup:
- **Claude Code**: Native slash commands (`.claude/commands/`)
- **OpenCode**: Agent definitions with permission enforcement (`.opencode/agents/`)
- **Cursor**: Rules with conditional activation (`.cursor/rules/`)
- **Generic** (Windsurf, Cline, Aider, CLI agents): `run-stage.sh` helper

## What's NOT Here

- **Team mode**: Concurrent agent execution is Claude Code-exclusive. See `docs/guides/agent-teams.md`.
- **Tool-level enforcement**: Preventing agents from using specific tools (e.g., blocking file writes for reviewers) requires platform support. Only Claude Code and OpenCode enforce this at the tool level. Other platforms rely on behavioral guidance in the stance definitions.

For the complete methodology specification, see [push-hands.md](../push-hands.md).
