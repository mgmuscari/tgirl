# Workflow: Execute PRP

> Portable workflow template. Load this as context for your AI coding agent,
> or use a platform adapter (see `adapters/`).

## Stance

Proposer — thorough, systematic, completion-oriented. Executes the implementation plan faithfully, task by task, with validation at every step.

## Input

`{input}` — Path to the PRP file (e.g., `docs/PRPs/my-feature.md`).

## Instructions

### 1. Read the PRP

Read the PRP at the provided path and its associated plan review (if one exists at `docs/reviews/plans/<slug>-review.md`). Incorporate any feedback from the review.

### 2. Read project context

Read the project context documentation for conventions and known gotchas before writing any code.

### 3. Execute tasks in order using TDD

For each task in the Implementation Plan:

**a. RED — Write tests first:**
- Write test(s) that specify expected behavior for this task
- Run the test command from the PRP — verify tests FAIL
- If tests pass before implementation, they are too weak — rewrite them
- Tests define the contract; implementation fulfills it

**b. GREEN — Implement minimum code to pass:**
- Follow the approach described in the PRP
- Match existing code patterns and conventions
- Run the test command — verify tests PASS
- If tests still fail, fix the implementation (never weaken tests)
- Do NOT mock to make tests pass — fix the actual issue

**c. REFACTOR — Clean up if needed:**
- Simplify implementation while keeping tests green
- Run the full validation gate commands from the PRP
- If validation fails, fix before proceeding

**d. Commit** test + implementation together atomically:
- One task = one commit (test and implementation together)
- Use conventional commit format
- Reference the PRP in the commit body if helpful

**e. Log uncertainty:**
- If you had to guess about anything, add it to the PRP's Uncertainty Log
- If something doesn't match what the PRP expected, document the deviation

**TDD guard rails:**
- The PRP must include a `Test Command` field — if missing, ask before proceeding
- If no test framework exists yet, the first task MUST set one up
- Never skip the RED phase — untested code is unverified code
- Never weaken or delete tests to get GREEN — fix the implementation instead

### 4. If stuck

If stuck (3+ failed attempts on the same task):
- Stop and flag for human review
- Describe what you tried, what failed, and what you think the issue is
- Do NOT thrash — escalate

### 5. After all tasks complete

- Run the full validation gate suite one final time
- Update the PRD status to IMPLEMENTED
- Create a summary of what was implemented and any deviations from the plan

### 6. Recommend next step

Have the implementation reviewed — load the code review workflow.

## Validation

```bash
# All validation gates from the PRP pass
# (Run the project's lint, type check, and test commands)

# Each task has a corresponding commit
git log --oneline | head -20

# PRD status updated
grep -q "IMPLEMENTED" "docs/PRDs/<slug>.md"
```
