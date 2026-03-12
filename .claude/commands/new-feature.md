You are operating in the **Proposer** stance. Your goal is to produce a comprehensive, well-structured PRD for a new feature — or, for light tier, to set up the feature branch for direct implementation.

## Instructions

The user wants to create a new feature: $ARGUMENTS

### 0. Parse tier and description from arguments

Extract the `--tier <value>` flag from the arguments if present:
- If `--tier light`, `--tier standard`, or `--tier full` appears, use that tier and treat the remaining text as the feature description.
- If no `--tier` flag is present, default to `standard`.
- Examples:
  - `--tier light fix login bug` → tier=light, description="fix login bug"
  - `add user profiles --tier full` → tier=full, description="add user profiles"
  - `add caching layer` → tier=standard, description="add caching layer"

### 1. Set up the feature branch

- Derive a slug from the feature description (lowercase, hyphen-separated, alphanumeric only: `^[a-z0-9]+(-[a-z0-9]+)*$`)
- **Check if the branch already exists** (e.g., created by `scripts/new-feature.sh` or a prior invocation):
  - If `feature/<slug>` already exists AND you're already on it: continue — no branch switching needed.
  - If `feature/<slug>` already exists but you're on a different branch: switch to it with `git checkout feature/<slug>`.
  - If `feature/<slug>` does not exist: create it from `main` with `git checkout main && git checkout -b feature/<slug>`.
- **Read existing tier metadata** if present: if `.push-hands-tier` exists on the branch, use its value (it takes precedence over any `--tier` flag in the arguments, since the branch was already initialized with a tier).
- If no `.push-hands-tier` exists, write the parsed tier value to `.push-hands-tier`.

### 2. Light tier path

If the tier is `light`:
- **Skip PRD generation entirely.** No codebase research, no PRD template, no document artifact.
- Commit `.push-hands-tier` if not already committed: message `chore: create feature branch for <slug> (light tier)`
- **Report:**
  - "Light tier — implement your change directly."
  - "Run `/review-code` when done (optional), then create a PR."
  - Stop here. Do not proceed to steps 3-5.

### 3. Research the codebase (standard and full tiers)

- Search for related symbols, patterns, and modules
- Identify files and modules that will be affected
- Check CLAUDE.md for project conventions and known gotchas
- Understand the existing architecture well enough to describe impact

### 4. Generate the PRD at `docs/PRDs/<slug>.md` using the template at `docs/PRDs/TEMPLATE.md`

- Fill in every section with specifics from your research
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

Follow the `/generate-prp` workflow:
1. Read the PRD you just created
2. Deep codebase research (symbols, tests, libraries, CLAUDE.md, data flow)
3. Generate the PRP at `docs/PRPs/<slug>.md` using the template at `docs/PRPs/TEMPLATE.md`
4. Commit with: `docs: generate PRP for <slug>`

`/generate-prp` remains available for cases where you want to regenerate just the PRP without redoing the PRD.

### 7. Report

- What you created (PRD + PRP for standard/full, or just branch for light)
- Any open questions that need human input before proceeding
- If **full tier**, add: "Security audit is expected for this tier. Run `/security-audit` or `/security-audit-team` before creating the PR."
- Recommend next step: `/review-plan docs/PRPs/<slug>.md` or `/review-plan-team docs/PRPs/<slug>.md`
