# PRP: Tiered Workflow System

## Source PRD: docs/PRDs/tiered-workflow.md
## Date: 2026-02-10
## Confidence Score: 8/10

## 1. Context Summary

The push hands template requires 6 stage gates for every feature, which is too heavyweight for solo developers making small changes. We're adding three workflow tiers — light, standard, full — so the process weight matches the change size.

- **Light:** Feature branch + conventional commits + hooks. No PRD, no PRP, no plan review. Code review optional.
- **Standard:** Current full pipeline, unchanged. The default.
- **Full:** Standard + expected security audit + mandatory plan review. For auth/payments/PII.

Tier is a human decision stored in a `.push-hands-tier` file on the feature branch. The file must never reach `main` — `close-feature.sh` deletes it before the squash-merge.

## 2. Codebase Analysis

### Existing patterns and integration points

**Shell scripts follow a consistent pattern:**
- `scripts/new-feature.sh` (lines 1-52): `set -e`, validate slug with `grep -qE`, create branch, scaffold files, commit. Uses positional args (`$1` for feature name).
- `scripts/close-feature.sh` (lines 1-107): Same validation pattern. Checks for artifact files with `[ ! -f ... ]`, warns and prompts. Squash-merges to main.
- Both scripts use the slug regex `^[a-z0-9]+(-[a-z0-9]+)*$`.

**Slash commands use `$ARGUMENTS` for input:**
- `.claude/commands/new-feature.md` (line 5): `$ARGUMENTS` receives everything after the command name. The command currently expects a plain description. We need to parse `--tier <value>` from it.
- Other commands (`generate-prp.md`, `review-plan.md`, `execute-prp.md`, `review-code.md`, `security-audit.md`) don't need tier awareness — they're only invoked in the appropriate tier.

**CI workflow uses `env:` variables for untrusted input (per Known Gotcha):**
- `.github/workflows/push-hands-review.yml` (lines 18-19): `BRANCH: ${{ github.head_ref }}` passed via `env:`, not inline interpolation. All downstream steps use `$SLUG` from the env, not `${{ }}`. New steps must follow this pattern.

**PR template is static markdown with HTML comments for placeholders:**
- `.github/PULL_REQUEST_TEMPLATE.md`: Uses `<!-- slug -->` as fill-in-the-blank markers. Checklist is a flat list of `- [ ]` items. We need to add tier context and tier-appropriate checklist guidance.

**Documentation files (`push-hands.md`, `README.md`, `CLAUDE.md`):**
- `push-hands.md` Section 4.1 (lines 124-136): Contains the lifecycle diagram as an indented text block.
- `README.md` (lines 30-39): Contains a matching lifecycle diagram.
- `CLAUDE.md` (lines 23-31): Contains a compact pipeline diagram.
- All three need updated diagrams showing the tiered paths.

### Conventions to follow

- Commits: conventional commit format, max 72 chars first line.
- Slugs: `^[a-z0-9]+(-[a-z0-9]+)*$`.
- Known gotcha: GitHub Actions `${{ }}` — always use `env:` for untrusted values.
- Known gotcha: `grep -qE` checks all lines — use `head -1 | grep -qE` for single-line validation.
- Shell scripts: `set -e`, validate inputs early, echo status messages.

## 3. External Research

**Claude Code slash command arguments:**
- `$ARGUMENTS` receives the raw string after the command name. There's no built-in argument parsing — the slash command prompt must instruct the agent to parse flags from the argument string itself. This means `/new-feature --tier light fix login bug` would pass `--tier light fix login bug` as `$ARGUMENTS`, and the prompt must tell the agent to extract `--tier light` and treat the rest as the description.

**GitHub Actions — reading file content in workflow steps:**
- A `.push-hands-tier` file can be read with `cat .push-hands-tier 2>/dev/null || echo "standard"` to default gracefully if the file doesn't exist.
- The file check should happen after checkout in the workflow, since it's a committed file on the branch.

**No external libraries or dependencies are introduced by this change.**

## 4. Implementation Plan

### Task 1: Add tier support to `scripts/new-feature.sh`

**Files:** `scripts/new-feature.sh`
**Approach:**
1. Update usage line and help text:
   ```
   # Usage: ./scripts/new-feature.sh <feature-name> [--tier light|standard|full]
   # Example: ./scripts/new-feature.sh user-authentication --tier full
   ```
2. Parse arguments. Feature name is always the first positional arg (backwards compatible). The `--tier` flag can appear anywhere after it:
   ```bash
   FEATURE_NAME=""
   TIER="standard"

   while [ $# -gt 0 ]; do
       case "$1" in
           --tier)
               if [ -z "$2" ]; then
                   echo "Error: --tier requires a value (light, standard, full)"
                   exit 1
               fi
               TIER="$2"
               shift 2
               ;;
           -*)
               echo "Error: unknown flag: $1"
               exit 1
               ;;
           *)
               if [ -z "$FEATURE_NAME" ]; then
                   FEATURE_NAME="$1"
               else
                   echo "Error: unexpected argument: $1"
                   exit 1
               fi
               shift
               ;;
       esac
   done

   if [ -z "$FEATURE_NAME" ]; then
       echo "Usage: ./scripts/new-feature.sh <feature-name> [--tier light|standard|full]"
       exit 1
   fi
   ```
3. Validate tier value immediately after parsing:
   ```bash
   case "$TIER" in
       light|standard|full) ;;
       *)
           echo "Error: invalid tier '${TIER}' — must be light, standard, or full"
           exit 1
           ;;
   esac
   ```
4. After creating the branch, write the tier to `.push-hands-tier` and `git add` it.
5. For `light` tier: skip PRD template copy and `mkdir -p docs/PRPs`. Commit message: `chore: create feature branch for <slug> (light tier)`. Stage only `.push-hands-tier`.
6. For `standard`/`full` tier: keep existing PRD scaffolding behavior. Stage both the PRD and `.push-hands-tier`. Commit message stays `docs: scaffold PRD for <slug>`.
7. Update the "Next steps" echo to be tier-appropriate:
   - Light: "Implement your change, then run /review-code (optional) and close with close-feature.sh"
   - Standard: existing next steps
   - Full: existing next steps + "Security audit is expected for this tier"

**Tests:** Manual — run the script with each combination:
- `./scripts/new-feature.sh my-fix --tier light` — light tier, no PRD
- `./scripts/new-feature.sh my-feature` — defaults to standard, PRD scaffold created
- `./scripts/new-feature.sh --tier full auth-rework` — full tier, PRD scaffold created
- `./scripts/new-feature.sh my-fix --tier` — error: missing value
- `./scripts/new-feature.sh my-fix --tier bogus` — error: invalid tier
**Validation:** `bash -n scripts/new-feature.sh` (syntax check)

### Task 2: Add tier-aware artifact checking to `scripts/close-feature.sh`

**Files:** `scripts/close-feature.sh`
**Approach:**
1. After slug validation, read tier: `TIER=$(cat .push-hands-tier 2>/dev/null || echo "standard")`.
2. Validate tier value (light/standard/full), default to standard if unrecognized.
3. Replace the flat artifact-check block (lines 38-58) with tier-conditional checks:
   - **Light tier:** No artifact checks at all (no PRD, PRP, plan review, or code review required). Only hooks gate the merge.
   - **Standard tier:** Current behavior — warn on missing PRD, PRP, plan review, code review.
   - **Full tier:** Standard checks + warn on missing security audit at `docs/audits/${FEATURE_NAME}-audit.md`.
4. After the squash-merge stages changes (`git merge --squash`) but before the commit, unstage and remove `.push-hands-tier` so it never reaches main:
   ```bash
   git merge --squash "$BRANCH_NAME"
   # Remove tier metadata from the squash staging area
   git rm -f --cached .push-hands-tier 2>/dev/null || true
   rm -f .push-hands-tier
   ```
   This is cleaner than committing a removal on the branch — no extra commit, no branch history pollution. The squash stages the net diff; we simply remove the tier file from that staging before committing.
5. Update the squash-merge commit message body to be tier-appropriate:
   - Light: don't reference PRD/PRP (they don't exist).
   - Standard/Full: current message.
6. Echo the tier at the start: `echo "Tier: ${TIER}"`.

**Tests:** Manual — test close on each tier with various artifact states.
**Validation:** `bash -n scripts/close-feature.sh` (syntax check)

### Task 3: Update `/new-feature` slash command for tier argument

**Files:** `.claude/commands/new-feature.md`
**Approach:**
1. Add an argument parsing instruction block at the top of the Instructions section:
   - Parse `$ARGUMENTS` for a `--tier <value>` flag (light/standard/full). Default to `standard` if absent.
   - The remaining text after removing the `--tier <value>` flag is the feature description.
2. **Handle two entry modes** (addresses dual-path conflict):
   - **Fresh start (no existing branch):** Create the feature branch with `git checkout -b feature/<slug>`, write tier to `.push-hands-tier`, proceed normally.
   - **Existing branch (created by shell script or prior invocation):** Before attempting `git checkout -b`, check if the branch exists (`git branch --list "feature/<slug>"`). If it does, switch to it with `git checkout feature/<slug>` instead of creating. Read the existing `.push-hands-tier` file if present (it takes precedence over any `--tier` flag, since the branch was already initialized with a tier). If no `.push-hands-tier` exists on the branch, write one with the parsed tier value.
3. For **light tier**, replace the PRD generation steps with:
   - Skip PRD generation entirely.
   - Commit `.push-hands-tier` (if not already committed) with message `chore: create feature branch for <slug> (light tier)`.
   - Report: "Light tier — implement your change directly. Run `/review-code` when done (optional), then create a PR."
4. For **standard tier** (default): existing PRD generation behavior, plus commit `.push-hands-tier` alongside the PRD.
5. For **full tier**: same as standard, but report includes "Security audit is expected for this tier. Run `/security-audit` before creating the PR."
6. Update the final "Report" step to include tier-specific next steps.

**Tests:** Invoke `/new-feature --tier light fix a typo` and verify behavior. Also test: run `./scripts/new-feature.sh my-feature --tier standard` first, then invoke `/new-feature my-feature` — should detect existing branch and continue.
**Validation:** Read the file and verify the instruction structure is coherent.

### Task 4: Update `push-hands-review.yml` for tier-aware artifact checks

**Files:** `.github/workflows/push-hands-review.yml`
**Approach:**
1. Add a new step after "Extract feature slug from branch" that reads the tier:
   ```yaml
   - name: Read workflow tier
     if: steps.slug.outputs.skip != 'true'
     id: tier
     env:
       SLUG: ${{ steps.slug.outputs.slug }}
     run: |
       if [ -f ".push-hands-tier" ]; then
         TIER=$(cat .push-hands-tier)
         # Validate tier value
         case "$TIER" in
           light|standard|full) ;;
           *) TIER="standard" ;;
         esac
       else
         TIER="standard"
       fi
       echo "tier=${TIER}" >> "$GITHUB_OUTPUT"
       echo "Detected tier: ${TIER}"
   ```
2. Add conditions to existing artifact check steps:
   - PRD check: `if: steps.slug.outputs.skip != 'true' && steps.tier.outputs.tier != 'light'`
   - PRP check: same condition
   - Plan review check: same condition
   - Code review check: `if: steps.slug.outputs.skip != 'true'` (keep for all tiers, but could remain as-is since it's already a warning)
3. Add a new step for security audit check on `full` tier:
   ```yaml
   - name: Check for security audit (full tier)
     if: steps.slug.outputs.skip != 'true' && steps.tier.outputs.tier == 'full'
     env:
       SLUG: ${{ steps.slug.outputs.slug }}
     run: |
       if [ ! -f "docs/audits/${SLUG}-audit.md" ]; then
         echo "::warning::Full tier — missing security audit at docs/audits/${SLUG}-audit.md"
       fi
   ```
4. Add a step that warns if `.push-hands-tier` is present in a PR targeting main (safety net):
   ```yaml
   - name: Warn if tier file will reach main
     if: steps.slug.outputs.skip != 'true'
     run: |
       if git diff --name-only origin/main...HEAD | grep -q '^\.push-hands-tier$'; then
         echo "::warning::.push-hands-tier should be removed before merging to main. Run close-feature.sh or delete it manually."
       fi
   ```

**Tests:** This is a workflow file — tested by opening PRs with different tier files.
**Validation:** YAML syntax check: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/push-hands-review.yml'))"`

### Task 5: Update PR template with tier field

**Files:** `.github/PULL_REQUEST_TEMPLATE.md`
**Approach:**
1. Add a "Workflow Tier" field at the top, after Summary:
   ```markdown
   ## Workflow Tier

   <!-- Select one: Light | Standard | Full -->
   **Tier:** <!-- tier -->
   ```
2. Update the Linked Artifacts section with tier-conditional notes:
   ```markdown
   ## Linked Artifacts

   <!-- Light tier: no PRD/PRP/plan review artifacts expected -->
   <!-- Standard tier: PRD, PRP, plan review, code review -->
   <!-- Full tier: all of the above + security audit -->
   ```
3. Update the Checklist to show which items apply to which tier:
   ```markdown
   ## Checklist

   **All tiers:**
   - [ ] All tests pass
   - [ ] Lint and type checks pass
   - [ ] `.push-hands-tier` file removed before merge

   **Standard and Full tiers:**
   - [ ] PRD exists and is marked IMPLEMENTED
   - [ ] PRP exists with completed Uncertainty Log
   - [ ] Plan review completed (push hands)
   - [ ] Code review completed (push hands)
   - [ ] CLAUDE.md updated if new conventions or gotchas discovered

   **Full tier only:**
   - [ ] Security audit completed
   ```

**Tests:** Visual — open the template and verify it renders correctly in GitHub's PR creation UI.
**Validation:** Ensure valid markdown.

### Task 6: Update `push-hands.md` Section 4 with tiered lifecycle

**Files:** `push-hands.md`
**Approach:**
1. Replace the single lifecycle diagram in Section 4.1 (lines 126-136) with a tiered overview showing all three paths:
   ```
   Light Tier (bug fixes, config tweaks, docs):
       Feature Branch → Implement → /review-code (optional) → PR → Merge

   Standard Tier (features, multi-file changes):
       Feature Brief → /new-feature → PRD
           → /generate-prp → PRP
               → /review-plan → Plan Review
                   → /execute-prp → Implementation
                       → /review-code → Code Review
                           → PR → Merge

   Full Tier (security-sensitive, auth, payments, PII):
       Feature Brief → /new-feature → PRD
           → /generate-prp → PRP
               → /review-plan → Plan Review (mandatory)
                   → /execute-prp → Implementation
                       → /review-code → Code Review
                           → /security-audit → Security Audit
                               → PR → Merge
   ```
2. Add a new subsection **4.1.1 Choosing a Tier** with guidance and examples:
   - Light: "Fix the off-by-one in pagination", "Update the README", "Add a missing env var to config"
   - Standard: "Add user profile page", "Implement caching layer", "Refactor auth module"
   - Full: "Add payment processing", "Implement OAuth flow", "Handle PII export"
3. Add a note: "Tier is a human decision. If you're unsure, start with standard — you can always skip optional stages, but you can't retroactively add artifacts you didn't create."
4. Add a subsection **4.1.2 Escalating Mid-Flight** documenting how to change tiers after starting:
   - **Light → Standard:** Edit `.push-hands-tier` to `standard` and commit. Then run `/new-feature <description>` — it will detect the existing branch and generate the PRD. Continue with `/generate-prp`, etc.
   - **Light → Full:** Same as light→standard, then continue through the full pipeline including `/security-audit`.
   - **Standard → Full:** Edit `.push-hands-tier` to `full` and commit. All standard artifacts already exist. Run `/security-audit` before closing the feature.
   - **Downgrading (e.g., standard → light):** Not recommended — artifacts already created aren't removed. But you can edit `.push-hands-tier` to `light` and `close-feature.sh` will skip artifact checks accordingly.
5. Update Section 7 "Automated Branch Management" (lines 506-520) to reflect the `--tier` flag on `new-feature.sh` and tier-aware behavior in `close-feature.sh`.
6. Update Appendix A quick reference table (lines 676-687) to add a Tier column showing which tiers use each command.
7. Update Appendix B file naming conventions (lines 688-695) to add `.push-hands-tier` to the list.
8. Update Section 9 Customization Points (line 598) to replace the "Review intensity" bullet with a reference to the tier system.

**Tests:** Read the updated sections and verify coherence with the rest of the document.
**Validation:** Visual review.

### Task 7: Update `README.md` with tiered lifecycle

**Files:** `README.md`
**Approach:**
1. Replace the single lifecycle diagram (lines 30-39) with a compact tiered version:
   ```
   Light:    Feature Branch → Implement → PR
   Standard: /new-feature → PRD → PRP → Plan Review → Implement → Code Review → PR
   Full:     /new-feature → PRD → PRP → Plan Review → Implement → Code Review → Security Audit → PR
   ```
2. Update the Slash Commands table to add a "Tier" column showing which tiers use each command:
   | Command | Tier | Input | Output | Stance |
3. Update the Customization section: replace "Review intensity" bullet with mention of the tier system.

**Tests:** Visual review.
**Validation:** Ensure markdown renders correctly.

### Task 8: Update `CLAUDE.md` with tiered lifecycle

**Files:** `CLAUDE.md`
**Approach:**
1. Replace the single pipeline diagram (lines 23-31) with a tiered version matching README.
2. Add a brief note about tiers under "Key Design Decisions" or after the pipeline:
   - "Three workflow tiers (light/standard/full) match process weight to change size. Default is standard. Tier metadata is stored in `.push-hands-tier` on feature branches."
3. Add to Conventions section: `.push-hands-tier` file must never reach `main`.

**Tests:** Read and verify coherence.
**Validation:** Visual review.

## 5. Validation Gates

```bash
# Shell script syntax check
bash -n scripts/new-feature.sh
bash -n scripts/close-feature.sh

# YAML syntax check
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/push-hands-review.yml'))"
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

# Verify .push-hands-tier is NOT on main
git checkout main && test ! -f .push-hands-tier && echo "PASS: no tier file on main"
```

## 6. Rollback Plan

All changes are on the `feature/tiered-workflow` branch. If anything goes wrong:
1. `git checkout main` — main is untouched.
2. `git branch -D feature/tiered-workflow` — delete the branch entirely.

Since this is a template repo (no running application), there's no deployment to roll back. The rollback is simply "don't merge."

## 7. Uncertainty Log

1. **Slash command argument parsing.** Claude Code's `$ARGUMENTS` is a raw string. The `/new-feature` prompt instructs the agent to parse `--tier` from it, but I haven't tested whether models reliably parse flag-style arguments from natural language prompts. If this proves unreliable, a fallback would be to have the agent ask the user for the tier if not specified. **Confidence: medium.**

2. ~~**`.push-hands-tier` cleanup timing in `close-feature.sh`.**~~ **Resolved by plan review.** Cleanup now happens post-squash by unstaging the file before committing on main. No extra branch commit needed. **Confidence: high.**

3. **PR template with tier-conditional checklists.** GitHub PR templates are static markdown — there's no way to dynamically show/hide checklist items based on tier. The approach uses section headers ("All tiers" / "Standard and Full tiers" / "Full tier only") with all items visible. The developer is expected to ignore items not relevant to their tier. This is a UX compromise. **Confidence: high — it's the only option without a GitHub App.**

4. **The `origin/main...HEAD` diff in the workflow safety-net step.** In GitHub Actions context with `fetch-depth: 0`, `origin/main` will always exist since the workflow runs on GitHub's infrastructure. Adding `2>/dev/null || true` anyway for robustness. **Confidence: high.**

5. **Dual-path slash command detection.** The slash command now checks for an existing branch before creating one. The check uses `git branch --list` which is reliable. However, if the agent is already on the feature branch (e.g., user ran `new-feature.sh` in the same terminal session), `git checkout` is a no-op — the agent needs to detect it's already on the right branch and skip branch switching. The prompt instructions should cover this. **Confidence: medium.**
