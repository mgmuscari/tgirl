# OpenCode Adapter

OpenCode is the closest platform to Claude Code — it supports named agent definitions with permission-based tool restrictions.

## Feature Comparison

| Feature | Claude Code | OpenCode |
|---------|-------------|----------|
| Slash commands | Native | No — use prompt templates from `prompts/workflows/` |
| Agent stances | Tool-level (`tools:` whitelist) | Permission-level (`permission:` allow/ask/deny) |
| Agent teams | Native (concurrent) | Subagent mode (`@agent-name`) — sequential only |
| Project context | `CLAUDE.md` | `AGENTS.md` and `CLAUDE.md` (both auto-read) |

## Setup

The `.opencode/agents/` directory is included in the template. No additional setup needed — OpenCode auto-discovers agent definitions.

## Running Workflows

OpenCode does not have slash commands. To run a workflow:

1. Load the prompt template from `prompts/workflows/<stage>.md`
2. Replace `{input}` with your actual input
3. Paste into the chat or load via `--read prompts/workflows/<stage>.md`
4. If the workflow uses a non-proposer stance, invoke the relevant agent with `@agent-name`

Example:
```
@training-partner Review the PRP at docs/PRPs/my-feature.md
```

## Agent Definitions

Five agents are defined in `.opencode/agents/`, matching the Push Hands stances:

| Agent | Mode | Edit | Bash | Key Constraint |
|-------|------|------|------|----------------|
| proposer | subagent | allow | allow all | Full access |
| training-partner | subagent | deny | ask (git/test/grep allowed) | Cannot modify files |
| code-reviewer | subagent | deny | ask (git/test/grep allowed) | Cannot modify files |
| security-auditor | subagent | deny | allow all | Bash for PoC testing |
| skeptical-client | subagent | deny | deny all | Pure analysis |

### Permissions Caveat

The bash glob patterns in the `permission:` field (e.g., `"git *": allow`) are best-effort mappings based on OpenCode documentation. If a permission pattern doesn't behave as expected, adjust the glob or switch to `ask` as the fallback.

The agent definitions prioritize safety — unmatched commands default to `ask` (for review agents) or `deny` (for skeptical-client), so failures prompt rather than silently allow.

### Design Note: `ask` vs `deny` Defaults

Review-only agents (training-partner, code-reviewer) default unmatched bash commands to `ask` rather than `deny`. This lets users approve unexpected but legitimate commands (e.g., `wc -l` for counting lines) while still preventing accidental writes. The skeptical-client uses `deny` because it should have no shell access at all.
