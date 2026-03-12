You are operating in the **Proposer** stance. Your goal is to execute the implementation plan faithfully, task by task, with validation at every step.

## Instructions

Execute the PRP at: $ARGUMENTS

1. **Read the PRP** and its associated plan review (if one exists at `docs/reviews/plans/<slug>-review.md`). Incorporate any feedback from the review.

2. **Read CLAUDE.md** for project conventions and known gotchas before writing any code.

3. **Execute tasks in order using TDD.** For each task in the Implementation Plan:

   a. **RED — Write tests first:**
      - Write test(s) that specify expected behavior for this task
      - Run the test command from the PRP — verify tests FAIL
      - If tests pass before implementation, they are too weak — rewrite them
      - Tests define the contract; implementation fulfills it

   b. **GREEN — Implement minimum code to pass:**
      - Follow the approach described in the PRP
      - Match existing code patterns and conventions
      - Run the test command — verify tests PASS
      - If tests still fail, fix the implementation (never weaken tests)
      - Do NOT mock to make tests pass — fix the actual issue

   c. **REFACTOR — Clean up if needed:**
      - Simplify implementation while keeping tests green
      - Run the full validation gate commands from the PRP
      - If validation fails, fix before proceeding

   d. **Commit** test + implementation together atomically:
      - One task = one commit (test and implementation together)
      - Use conventional commit format
      - Reference the PRP in the commit body if helpful

   e. **Log uncertainty:**
      - If you had to guess about anything, add it to the PRP's Uncertainty Log
      - If something doesn't match what the PRP expected, document the deviation

   **TDD guard rails:**
   - The PRP must include a `Test Command` field — if missing, ask the user before proceeding
   - If no test framework exists yet, the first task MUST set one up
   - Never skip the RED phase — untested code is unverified code
   - Never weaken or delete tests to get GREEN — fix the implementation instead

4. **If stuck** (3+ failed attempts on the same task):
   - Stop and flag for human review
   - Describe what you tried, what failed, and what you think the issue is
   - Do NOT thrash — escalate

5. **After all tasks complete:**
   - Run the full validation gate suite one final time
   - Update the PRD status to IMPLEMENTED
   - Create a summary of what was implemented and any deviations from the plan

6. **Recommend next step:** `/review-code` to have the Code Review Partner review the implementation.
