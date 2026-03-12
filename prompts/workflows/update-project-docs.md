# Workflow: Update Project Documentation

> Portable workflow template. Load this as context for your AI coding agent,
> or use a platform adapter (see `adapters/`).
>
> This is the generalized version of Claude Code's `/update-claude-md` command.
> It updates whatever the project's primary context documentation file is.

## Stance

Proposer — thorough, systematic, completion-oriented. Captures lessons learned and propagates corrections into project context documentation.

## Input

- `{input}` — The lesson to capture: a correction, gotcha, or convention clarification.
- `{target}` (optional) — The target context file. If not specified, auto-detected (see Target File section).

## Target File

If no target file is specified, detect the project's primary context file in this order:
1. `CLAUDE.md` — used by Claude Code
2. `AGENTS.md` — used by multiple platforms (Cursor, Windsurf, Cline, OpenCode)
3. `.cursorrules` — legacy Cursor format
4. `.windsurfrules` — Windsurf format
5. If none found, ask the user which file to update

Platform-specific adapters should hardcode their target (e.g., Claude Code adapter always uses `CLAUDE.md`).

## Instructions

### 1. Identify what was learned

- If the user just corrected you, capture the correction
- If you discovered a gotcha during implementation, capture it
- If a convention was clarified, capture it

### 2. Read the target file

Read the current target file to understand its structure and existing entries.

### 3. Update the appropriate section

- **Known Gotchas** for mistakes, pitfalls, and things that cause problems:
  - Format: `YYYY-MM-DD: [what went wrong] -> [what to do instead]`
- **Conventions** for new style rules, naming patterns, or organizational decisions
- **Architecture** for structural discoveries or changes
- **Agent Instructions** for new workflow knowledge (how to run things, what to avoid)

### 4. Do NOT

- Duplicate information already in the target file
- Add generic advice — only project-specific knowledge
- Remove existing entries unless they are demonstrably wrong
- Make the file excessively long — be concise

### 5. Commit the update

Commit with message: `docs: update <target-filename> — [brief description of lesson learned]`

## Validation

```bash
# Target file exists and was modified
git diff --name-only HEAD~1 | grep -q "<target-filename>"

# Commit follows conventional format
git log -1 --format="%s" | grep -qE "^docs: update"
```
