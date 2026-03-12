# Plan Review: tiered-workflow

## Verdict: REQUESTS CHANGES
## Reviewer Stance: Senior Training Partner
## Date: 2026-02-10

## Yield Points Found

### 1. `.push-hands-tier` cleanup creates an unnecessary commit on the feature branch
**Severity:** Structural
**Evidence:** PRP Task 2, step 4 proposes this on the feature branch before switching to main:
```bash
if [ -f ".push-hands-tier" ]; then
    git rm .push-hands-tier
    git commit -m "chore: remove tier metadata before merge"
fi
```
Meanwhile, `close-feature.sh` lines 84-85 do:
```bash
git checkout main
git merge --squash "$BRANCH_NAME"
```
The squash-merge stages everything but doesn't auto-commit (line 94 does the commit).
**Pressure applied:** Why add a commit to the branch to remove the file, when you can simply unstage it after the squash? The squash-merge stages the net diff between main and branch HEAD. If `.push-hands-tier` exists at HEAD, it'll be staged. But between `git merge --squash` (line 85) and `git commit` (line 94), you can just remove it from the staging area: `git rm -f --cached .push-hands-tier 2>/dev/null; rm -f .push-hands-tier`. No extra branch commit, no history pollution, same result.
**Recommendation:** Replace the `git rm` + `git commit` approach in Task 2 with a post-squash unstage. This is simpler, cleaner, and doesn't modify the branch history. The cleanup happens on main's staging area, not on the branch.

### 2. Dual-path conflict: shell script then slash command will fail
**Severity:** Structural
**Evidence:** PRD Section 6, Resolved item 3: "developers can use either the shell script or the slash command to start a feature — both paths produce the same metadata." PRP Task 3 step 2: "Add a step after branch creation: write the tier to `.push-hands-tier`." The slash command (`new-feature.md` line 7-9) instructs the agent to create a feature branch with `git checkout -b`.
**Pressure applied:** What happens if a developer runs `./scripts/new-feature.sh my-feature --tier standard` and then later, in the same conversation, invokes `/new-feature my-feature`? The shell script already created the branch and committed the tier file. The slash command will try to `git checkout -b feature/my-feature`, which fails because the branch already exists. The PRP says the slash command "creates the `.push-hands-tier` file if it doesn't exist, or reads it if it does" — but it doesn't address the branch-already-exists case.
**Recommendation:** Task 3 needs to handle two entry modes explicitly:
- **Fresh start:** No branch exists yet. Create branch, write tier file, proceed as described.
- **Existing branch:** Branch already exists (created by shell script or a prior invocation). Detect this (`git branch --list "feature/<slug>"`), switch to it instead of creating, read the existing tier file, and proceed from there (skip branch creation, skip tier file creation).

### 3. Argument parsing in `new-feature.sh` is underspecified
**Severity:** Moderate
**Evidence:** PRP Task 1, step 2: "Parse arguments: loop through args to extract `--tier <value>` and the feature name. The feature name is the first positional arg that isn't a flag or flag value." PRD acceptance criteria show `my-fix --tier light` (flag after name) and `auth-rework --tier full` (also after). But "loop through args" with no pseudocode is ambiguous for bash.
**Pressure applied:** Does the parser handle `--tier light my-fix`, `my-fix --tier light`, and `--tier=light my-fix`? What about `--tier` with no value? What about `--tier invalid-value`? The PRP says "validate tier value" but doesn't specify the error message or exit code for invalid values. Bash argument parsing is the #1 source of bugs in shell scripts — "loop through args" is not enough detail for unambiguous one-pass execution.
**Recommendation:** Provide explicit pseudocode or a complete argument parsing implementation in Task 1. Pin down the argument order (recommend: `<feature-name> [--tier light|standard|full]`, feature name always first for backwards compatibility). Show the `while`/`case` loop or `getopts` equivalent. Specify error behavior for: missing tier value after `--tier`, unrecognized tier value, multiple positional arguments.

### 4. No escalation path described
**Severity:** Moderate
**Evidence:** PRD Section 2 (How tiers are selected): "the cost of choosing the wrong tier is low (you can always escalate mid-flight)." PRP has no task or documentation covering what "escalate mid-flight" means in practice.
**Pressure applied:** A developer starts with light tier. Halfway through, they realize the change is bigger than expected and needs a PRD + PRP. What do they do? Edit `.push-hands-tier` to say `standard` and run `/new-feature`? But that slash command tries to create a new branch. Run `/generate-prp` without a PRD? The PRP doesn't address this. If the documentation says "you can escalate," the documentation must say *how*.
**Recommendation:** Add a brief section to Task 6 (push-hands.md update) documenting tier escalation. The simplest approach: "To escalate, edit `.push-hands-tier` to the new tier and begin the pipeline from the current stage. For light→standard: create a PRD with `/new-feature` (it will detect the existing branch), then continue normally. For standard→full: the branch already has all standard artifacts; run `/security-audit` additionally before closing." This doesn't need a new task — it's 2-3 paragraphs in the tier documentation.

### 5. PRP misses several `push-hands.md` sections that describe the pipeline
**Severity:** Minor
**Evidence:** PRP Task 6 targets Section 4.1 (lifecycle diagram, lines 126-136) and Section 9 (customization, line 598). But:
- Section 7 "Automated Branch Management" (lines 506-520) describes `new-feature.sh` and `close-feature.sh` behavior with specific numbered steps that will change.
- Appendix A (lines 676-687) is a quick reference table identical to the README table — needs a Tier column.
- Appendix B (lines 688-695) lists file naming conventions — should add `.push-hands-tier`.
**Pressure applied:** Someone reading push-hands.md end-to-end will find Section 4.1 describes tiers, but Section 7 still describes `new-feature.sh` as `<name>` with no `--tier` flag, and Appendix A has no tier column. The document contradicts itself.
**Recommendation:** Expand Task 6 to also update Section 7 (new-feature.sh and close-feature.sh descriptions), Appendix A (add Tier column), and Appendix B (add `.push-hands-tier` to file naming conventions).

## What Holds Well

- **The tier design itself is sound.** Three tiers, human-selected, with standard as default, is the right level of granularity. The PRD's "not a risk: tier creep" statement holds — three is enough.
- **The additive approach is correct.** Standard tier = current behavior with no `--tier` flag is the right backwards-compatibility choice. Existing users see no change.
- **The `.push-hands-tier` file approach is simple and git-native.** Easy to read from shell scripts, CI, and agent prompts. No new dependencies. The "must never reach main" constraint is well-identified and the safety-net CI step is a good belt-and-suspenders measure.
- **Task ordering makes sense.** Scripts first (Task 1-2), then slash command (Task 3), then CI/GitHub (Task 4-5), then docs (Task 6-8). Each layer builds on the previous.
- **The Uncertainty Log is honest.** The slash command argument parsing concern (item 1) is real and correctly flagged as medium confidence.

## Summary

The plan's structure is solid but yields at five points. Two are structural: the tier file cleanup approach creates unnecessary git history noise (fix: unstage after squash instead of committing removal on the branch), and the dual-path shell-script-then-slash-command conflict will cause failures (fix: detect existing branch in the slash command). Two are moderate: argument parsing needs actual pseudocode for unambiguous execution, and the promised escalation path needs documentation. One is minor but important for document consistency: several push-hands.md sections beyond 4.1 and 9 reference the pipeline and need updating.

None of these are architectural — the design doesn't need to change. The PRP needs more specificity in Tasks 1-3 and broader coverage in Task 6. I recommend incorporating these changes and proceeding to execution.
