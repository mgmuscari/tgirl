# Workflow: New Feature

> Portable workflow template. Load this as context for your AI coding agent,
> or use a platform adapter (see `adapters/`).

## Stance

Proposer — thorough, systematic, completion-oriented. Produces comprehensive, well-structured artifacts. References existing code patterns and project conventions.

## Input

`{input}` — A brief feature description, optionally prefixed with `--tier light|standard|full`.

## Instructions

### 0. Parse tier and description from input

Extract the `--tier <value>` flag from the input if present:
- If `--tier light`, `--tier standard`, or `--tier full` appears, use that tier and treat the remaining text as the feature description.
- If no `--tier` flag is present, default to `standard`.
- Examples:
  - `--tier light fix login bug` -> tier=light, description="fix login bug"
  - `add user profiles --tier full` -> tier=full, description="add user profiles"
  - `add caching layer` -> tier=standard, description="add caching layer"

### 1. Set up the feature branch

- Derive a slug from the feature description (lowercase, hyphen-separated, alphanumeric only: `^[a-z0-9]+(-[a-z0-9]+)*$`)
- **Check if the branch already exists:**
  - If `feature/<slug>` already exists AND you're already on it: continue — no branch switching needed.
  - If `feature/<slug>` already exists but you're on a different branch: switch to it.
  - If `feature/<slug>` does not exist: create it from `main`.
- **Read existing tier metadata** if present: if `.push-hands-tier` exists on the branch, use its value (it takes precedence over any `--tier` flag).
- If no `.push-hands-tier` exists, write the parsed tier value to `.push-hands-tier`.

### 2. Light tier path

If the tier is `light`:
- **Skip PRD generation entirely.** No codebase research, no PRD template, no document artifact.
- Commit `.push-hands-tier` if not already committed: message `chore: create feature branch for <slug> (light tier)`
- **Report:**
  - "Light tier — implement your change directly."
  - "Run a code review when done (optional), then create a PR."
  - Stop here. Do not proceed to steps 3-5.

### 3. Research the codebase (standard and full tiers)

- Search for related symbols, patterns, and modules
- Identify files and modules that will be affected
- Check project context documentation for conventions and known gotchas
- Understand the existing architecture well enough to describe impact

### 4. Generate the PRD at `docs/PRDs/<slug>.md`

Read the PRD template at `docs/PRDs/TEMPLATE.md` and fill in every section:
- Problem Statement: Why this feature matters, who needs it
- Proposed Solution: High-level approach (not implementation details)
- Architecture Impact: Specific files, data models, APIs, dependencies
- Acceptance Criteria: Numbered, testable criteria
- Risk Assessment: What could go wrong, what's unknown
- Open Questions: Things needing answers before implementation
- Out of Scope: What this feature explicitly does NOT do
- Set Status to DRAFT

### 5. Commit to the feature branch

- Stage both `.push-hands-tier` and the PRD (if `.push-hands-tier` is not already committed)
- Message: `docs: scaffold PRD for <slug>`

### 6. Auto-generate the PRP (standard and full tiers)

After the PRD is committed, automatically proceed to PRP generation — do NOT stop and wait for the user.

Follow the generate-prp workflow:
1. Read the PRD you just created
2. Deep codebase research (symbols, tests, libraries, project context, data flow)
3. Generate the PRP at `docs/PRPs/<slug>.md` using the template at `docs/PRPs/TEMPLATE.md`
4. Commit with: `docs: generate PRP for <slug>`

The generate-prp workflow remains available for cases where you want to regenerate just the PRP without redoing the PRD.

### 7. Report

- What you created (PRD + PRP for standard/full, or just branch for light)
- Any open questions that need human input before proceeding
- If **full tier**, add: "Security audit is expected for this tier."
- Recommend next step: have the plan reviewed before execution.

## Validation

```bash
# Feature branch exists
git branch --list "feature/*" | grep -q "<slug>"

# PRD exists (standard/full only)
test -f "docs/PRDs/<slug>.md"

# Tier file exists
test -f ".push-hands-tier"
```
