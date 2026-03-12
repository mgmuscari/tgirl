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

Ordered task list. Each task is:
- Small enough to implement in one pass
- Independently testable via TDD (RED → GREEN → REFACTOR)
- Described with enough detail that an agent can execute without ambiguity

### Task 1: [Description]
**Files:** list of files to create/modify
**Approach:** pseudocode or detailed description
**Tests:** what to test and how (written BEFORE implementation)
**Validation:** commands to run

### Task 2: [Description]
**Files:** ...
**Approach:** ...
**Tests:** ...
**Validation:** ...

## 4. Validation Gates
```bash
# Syntax/Style (replace with your stack's equivalents)
ruff check src/ --fix && mypy src/

# Unit Tests
pytest tests/ -v --cov=src

# Integration Tests (if applicable)
pytest tests/integration/ -v
```

## 5. Rollback Plan
_(Optional for standard tier. Required for full tier.)_
How to undo this if it breaks something.

## 6. Uncertainty Log
What the agent wasn't sure about. What it guessed. Where a human should double-check.
