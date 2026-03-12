You are operating in the **Proposer** stance. Your goal is to produce a comprehensive PRP — a complete implementation blueprint that enables one-pass execution.

## Instructions

Generate a PRP from the PRD at: $ARGUMENTS

1. **Read the PRD** and verify its status is APPROVED (or confirm with the user if DRAFT).

2. **Deep codebase research:**
   - Find all related symbols, patterns, and conventions
   - Read existing tests to understand validation patterns
   - Check external documentation for libraries involved
   - Review CLAUDE.md for project-specific conventions and known gotchas
   - Trace data flow through affected modules
   - Identify integration points and potential conflicts

3. **Generate the PRP** at `docs/PRPs/<slug>.md` using the template at `docs/PRPs/TEMPLATE.md`:
   - **Context Summary:** Distilled from PRD — what and why
   - **Codebase Analysis:** Relevant patterns with file paths and line numbers, conventions to follow, integration points
   - **Implementation Plan:** Ordered task list where each task is:
     - Small enough to implement in one pass
     - Independently testable via TDD (RED → GREEN → REFACTOR)
     - Described with enough detail that an agent can execute without ambiguity
     - Includes specific file paths, approach, tests (written BEFORE implementation), and validation commands
     - Must include a **Test Command** field (e.g., `pytest tests/`, `npm test`, `go test ./...`)
   - **Validation Gates:** Actual commands to run (adapted to this project's stack)
   - **Rollback Plan:** _(Optional for standard tier, required for full tier.)_ How to undo if something breaks
   - **Uncertainty Log:** What you weren't sure about, what you guessed

4. **Commit** to the feature branch:
   - Message: `docs: generate PRP for <slug>`

5. **Recommend next step:** `/review-plan docs/PRPs/<slug>.md` to have the Senior Training Partner review the plan before execution.
