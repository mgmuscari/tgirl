# Workflow: Generate PRP

> Portable workflow template. Load this as context for your AI coding agent,
> or use a platform adapter (see `adapters/`).

## Stance

Proposer — thorough, systematic, completion-oriented. Produces a comprehensive PRP (Product Requirements Prompt) — a complete implementation blueprint that enables one-pass execution.

## Input

`{input}` — Path to the PRD file (e.g., `docs/PRDs/my-feature.md`).

## Instructions

### 1. Read the PRD

Read the PRD at the provided path and verify its status is APPROVED (or confirm with the user if DRAFT).

### 2. Deep codebase research

- Find all related symbols, patterns, and conventions
- Read existing tests to understand validation patterns
- Check external documentation for libraries involved
- Review project context documentation for conventions and known gotchas
- Trace data flow through affected modules
- Identify integration points and potential conflicts

### 3. Generate the PRP

Create the PRP at `docs/PRPs/<slug>.md` using the template at `docs/PRPs/TEMPLATE.md`:

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

### 4. Commit

Commit to the feature branch with message: `docs: generate PRP for <slug>`

### 5. Recommend next step

Have the plan reviewed before execution — load the plan review workflow with the PRP path.

## Artifact Template

```markdown
# PRP: [Feature Name]

## Source PRD: docs/PRDs/<slug>.md
## Date: YYYY-MM-DD

## 1. Context Summary
Distilled from PRD. What we're building and why.

## 2. Codebase Analysis
- Relevant existing patterns (with file paths and line numbers)
- Conventions to follow
- Integration points

## 3. Implementation Plan

**Test Command:** `pytest tests/ -v` _(replace with your stack's test runner)_

Ordered task list. Each task:

### Task 1: [Description]
**Files:** list of files to create/modify
**Approach:** pseudocode or detailed description
**Tests:** what to test and how (written BEFORE implementation)
**Validation:** commands to run

### Task 2: ...

## 4. Validation Gates
Commands to validate the full implementation.

## 5. Rollback Plan
_(Optional for standard tier. Required for full tier.)_
How to undo this if it breaks something.

## 6. Uncertainty Log
What the agent wasn't sure about. What it guessed.
```

## Validation

```bash
# PRP file exists
test -f "docs/PRPs/<slug>.md"

# PRP has required sections
for section in "Context Summary" "Codebase Analysis" "Implementation Plan" "Test Command" "Validation Gates"; do
  grep -q "$section" "docs/PRPs/<slug>.md"
done
```
