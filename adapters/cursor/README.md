# Cursor Adapter

Cursor supports rules with conditional activation via `.mdc` files. It cannot enforce tool restrictions — all stance guidance is behavioral.

## Feature Comparison

| Feature | Claude Code | Cursor |
|---------|-------------|--------|
| Slash commands | Native | No — use prompt templates from `prompts/workflows/` |
| Agent stances | Tool-level enforcement | Rules (.mdc) — behavioral only |
| Agent teams | Native (concurrent) | No |
| Project context | `CLAUDE.md` | `.cursor/rules/` + `AGENTS.md` (auto-read) |

## Setup

The `.cursor/rules/` directory is included in the template. Cursor auto-discovers rules on startup.

## Rules

### Context Rule (always active)

`00-push-hands-context.mdc` — Always applied. Contains a condensed overview of the Push Hands methodology: tier system, artifact paths, commit conventions. Equivalent to what `CLAUDE.md` provides for Claude Code.

### Stance Rules (agent-requested)

Stance rules are Agent-Requested — Cursor's AI reads the description and decides whether to include the rule based on what the user asks. They are NOT always applied.

- `01-proposer.mdc` — Activated when implementing features or generating documents
- `02-training-partner.mdc` — Activated when reviewing plans or PRPs
- `03-code-reviewer.mdc` — Activated when reviewing code
- `04-security-auditor.mdc` — Activated when performing security audits
- `05-skeptical-client.mdc` — Activated when challenging security findings

## Running Workflows

Cursor does not have slash commands. To run a workflow:

1. Copy the relevant `prompts/workflows/<stage>.md` content into the chat
2. Replace `{input}` with your actual input
3. The appropriate stance rule should auto-activate based on the context

## Limitations

- **No tool-level enforcement:** Stances are behavioral guidance. The training partner is told not to modify files but is not prevented from doing so.
- **No concurrent agents:** Each interaction is single-threaded.
- **No automated stage sequencing:** You manually invoke each workflow stage.
