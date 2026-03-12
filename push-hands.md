# Ontologi Push Hands

## Product Requirements Document v1.0

**Author:** Elaine Muscari / Ontologi LLC
**Date:** February 11, 2026
**Repository:** `ontologi/ontologi-push-hands`
**License:** MIT

---

## 1. Executive Summary

This template repository codifies **Ontologi Push Hands** — a structured, dialectical, AI-assisted software development lifecycle designed for solo developers and small teams building with Claude Code, OpenCode, Cursor, Windsurf, Cline, Aider, or other agentic coding tools.

The name comes from tui shou (推手), the t'ai chi practice where two partners test each other's structural integrity through continuous contact, sensing where balance breaks before force is applied. In push hands, you don't fight your partner — you *listen* through pressure. The same dynamic governs this methodology: agent pairs probe each other's work, finding where the structure is weak by applying sustained, calibrated pressure at every stage of development.

The core thesis: **structured tension between training partners produces better outcomes than single-agent workflows at every stage of the development lifecycle.** Features are isolated into PRDs (Product Requirements Documents), implementation is isolated into PRPs (Product Requirements Prompts), and push hands review gates separate each stage — ensuring that no code reaches `main` without surviving cross-examination from a training partner whose job is to sense where it yields.

This template is opinionated. It assumes:

- You're building with Claude Code or similar agentic coding tools
- You want deterministic, auditable workflows — not vibes
- You treat your AI agents as junior engineers who need structure, not oracles who need freedom
- You believe the highest-leverage work is *before* code gets written

### What This Is Not

This is not a prompt library, a CLAUDE.md tutorial, or a collection of slash commands. It's a complete development *practice* — in the martial arts sense — that happens to be implemented as a GitHub template repository. The repository structure, the documentation conventions, the git workflow, and the push hands review patterns are all load-bearing.

---

## 2. Background & Motivation

### The Problem

Most developers using AI coding tools operate in one of two failure modes:

1. **No structure** — "Just talk to Claude and see what happens." Works for scripts and throwaway code. Falls apart the moment you have multiple features, shared state, or anything that needs to work in six months.

2. **Cargo-cult structure** — Copy a CLAUDE.md template from a popular repo, set up `/generate-prp` and `/execute-prp` commands, assume the process will handle the rest. This is better, but it's still single-threaded and single-perspective. The agent that writes the plan is the same agent that executes it. Nobody is checking the work.

The missing piece is **push hands** — a second agent (or a second invocation with a different stance) whose job is to sense where the structure yields. Not to attack, but to test. This is how Anthropic's own engineering team uses Claude Code internally. It's how Ontologi runs security audits. It's a pattern that works across domains because it exploits the same mechanism: calibrated pressure between training partners reveals weaknesses that solo practice never finds.

### Prior Art

- **Cole Medin's context-engineering-intro** — Established the PRD → PRP → Execute pipeline for Claude Code. Good foundation, but single-threaded with no adversarial review step.
- **Anthropic's internal practices** (per Boris Cherny's team) — Parallel worktrees, adversarial plan review ("spin up a second Claude to review as a staff engineer"), hooks for automated guardrails, living CLAUDE.md.
- **Ontologi's security audit process** — Dueling agents (auditor vs. skeptical client) producing tighter, more defensible reports.
- **12-Factor Agents** (Dexa) — Principles for building reliable AI agent systems, particularly "own your prompts" and "tools are just structured output."

### Design Principles

1. **Tension is a practice, not a problem.** Every stage has a proposer and a training partner. They test each other by design. The output is neither agent's work — it's the synthesis that survives contact.
2. **Documents are the product.** Code is generated from documents. If the documents are wrong, the code will be wrong. Invest in documents.
3. **Git is context engineering.** Branch structure, commit discipline, and worktree organization are part of how you manage what the agent knows. Not separate from it.
4. **Living over static.** CLAUDE.md, PRDs, and PRPs are updated by agents as they learn. Corrections propagate. Documentation that doesn't evolve is documentation that lies.
5. **Observable by default.** Every agent action is logged. Every decision has a paper trail. You can reconstruct why any line of code exists.
6. **The name is load-bearing.** Naming the practice "push hands" isn't branding — it's a functional context prime. Models operating under a practice with lineage and discipline produce different (better) output than models executing a task list.

---

## 3. Repository Structure

```
project-root/
│
├── CLAUDE.md                          # Living project intelligence (auto-updated)
├── AGENTS.md                          # Agent role definitions and system prompts
├── .claude/
│   └── commands/
│       ├── new-feature.md             # PRD generation from feature brief
│       ├── generate-prp.md            # PRP generation from PRD
│       ├── review-plan.md             # Push hands plan review
│       ├── execute-prp.md             # PRP execution with validation
│       ├── review-code.md             # Push hands code review
│       ├── security-audit.md          # Security-focused audit (hard stance)
│       └── update-claude-md.md        # Self-update CLAUDE.md after corrections
│
├── docs/
│   ├── PRDs/                          # Product Requirements Documents
│   │   ├── TEMPLATE.md                # PRD template
│   │   └── *.md                       # One PRD per feature
│   ├── PRPs/                          # Product Requirements Prompts
│   │   ├── TEMPLATE.md                # PRP template
│   │   └── *.md                       # One PRP per implementation ticket
│   ├── reviews/                       # Push hands review artifacts
│   │   ├── plans/                     # Plan review results
│   │   └── code/                      # Code review results
│   ├── audits/                        # Security audit reports
│   │   └── TEMPLATE.md                # Audit report template
│   └── decisions/                     # Architecture Decision Records
│       └── TEMPLATE.md                # ADR template
│
├── src/                               # Application source code
├── tests/                             # Test suite
│
├── scripts/
│   ├── hooks/
│   │   ├── pre-commit                 # Lint + type check gate
│   │   ├── pre-push                   # Test suite gate
│   │   └── commit-msg                 # Conventional commit enforcement
│   ├── setup.sh                       # Install hooks + verify environment
│   ├── new-feature.sh                 # Create feature branch + PRD scaffold
│   ├── close-feature.sh              # Merge feature branch + cleanup
│   └── worktree-setup.sh             # Parallel worktree scaffolding
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── feature-request.md         # Maps to PRD generation
│   │   └── bug-report.md              # Maps to PRP generation (skip PRD)
│   ├── PULL_REQUEST_TEMPLATE.md       # Links to PRD, PRP, and review artifacts
│   └── workflows/
│       ├── ci.yml                     # Standard CI pipeline
│       └── push-hands-review.yml      # Triggered on PR, runs training partner
│
└── .gitignore
```

---

## 4. The Development Lifecycle

### 4.1 Overview

The development lifecycle has three tiers. Choose the tier that matches the size and sensitivity of the change:

```
Light Tier (bug fixes, config tweaks, docs):
    Feature Branch → Implement → /review-code (optional) → PR → Merge

Standard Tier (features, multi-file changes) — the default:
    Feature Brief → /new-feature → PRD + PRP (auto-generated)
        → /review-plan → Plan Review
            → /execute-prp (TDD) → Implementation
                → /review-code → Code Review
                    → PR → Merge

Full Tier (security-sensitive, auth, payments, PII):
    Feature Brief → /new-feature → PRD + PRP (auto-generated)
        → /review-plan → Plan Review (mandatory)
            → /execute-prp (TDD) → Implementation
                → /review-code → Code Review
                    → /security-audit → Security Audit
                        → PR → Merge
```

Each arrow is a gate. Work does not proceed until the gate passes. Gates are not rubber stamps — the training partner can reject, request changes, or flag structural weaknesses that send work back to a previous stage.

Tier metadata is stored in a `.push-hands-tier` file on the feature branch (containing `light`, `standard`, or `full`). This file must never reach `main` — `close-feature.sh` removes it before the squash-merge.

#### 4.1.1 Choosing a Tier

Tier is a human decision, not an automated classification. Use these heuristics:

| Tier | Use when... | Examples |
|------|-------------|----------|
| **Light** | Small, well-understood changes. Single file, no behavior change, low risk. | "Fix the off-by-one in pagination", "Update the README", "Add a missing env var to config" |
| **Standard** | New functionality, multiple files, or behavior changes. | "Add user profile page", "Implement caching layer", "Refactor auth module" |
| **Full** | Security-sensitive, touches auth/payments/PII, or public API changes. | "Add payment processing", "Implement OAuth flow", "Handle PII export" |

If you're unsure, start with standard — you can always skip optional stages, but you can't retroactively add artifacts you didn't create.

#### 4.1.2 Escalating Mid-Flight

If you realize mid-implementation that a change needs more process than its current tier provides:

- **Light → Standard:** Edit `.push-hands-tier` to `standard` and commit. Run `/new-feature <description>` — it will detect the existing branch and generate the PRD. Continue with `/generate-prp`, etc.
- **Light → Full:** Same as light→standard, then continue through the full pipeline including `/security-audit`.
- **Standard → Full:** Edit `.push-hands-tier` to `full` and commit. All standard artifacts already exist. Run `/security-audit` before closing the feature.
- **Downgrading (e.g., standard → light):** Not recommended — artifacts already created aren't removed. But you can edit `.push-hands-tier` to `light` and `close-feature.sh` will skip artifact checks accordingly.

#### 4.1.3 Team Mode (Agent Teams)

Team mode uses Claude Code agent teams to run stances as concurrent teammates rather than sequential invocations. Same artifact types at the same paths, with additional fields capturing the dialectical exchange through real-time peer messaging.

| Team Command | Replaces | Agents | Coordination |
|-------------|----------|--------|--------------|
| `/review-plan-team <prp>` | `/review-plan` | Proposer + Training Partner | Simultaneous start, message exchange |
| `/execute-team <prp>` | `/execute-prp` + `/review-code` | Proposer + Code Reviewer | Message-gated: Proposer waits for review after each commit |
| `/security-audit-team` | `/security-audit` | Auditor + Skeptical Client | Simultaneous start, finding-by-finding challenge |

Team mode is opt-in and available only for standard and full tiers. It requires Claude Code agent teams to be enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`). Team commands can be mixed freely with sequential commands in the same pipeline.

Agent definitions in `.claude/agents/` enforce stance constraints at the tool level — the training partner literally cannot write files, not just prompt-instructed not to. See `docs/guides/agent-teams.md` for setup, cost implications, known bugs, workarounds, and usage guidance.

**Known team bugs and hardening:**
- Agent definitions don't load for team members (bug #24316) — team commands inline stance definitions in spawn prompts
- Model inheritance is unreliable (bug #32368) — all team spawns require explicit `model: "opus"`
- Team commands include a health check: if teammates don't start within 90 seconds, the team shuts down and the user is notified. Teams never silently fall back to solo implementation.

#### 4.1.4 Multi-Platform Support

The Push Hands methodology uses a three-layer architecture to support multiple AI coding platforms:

1. **Shared Infrastructure** (already portable): `docs/`, `scripts/`, hooks, git workflow — these work with any platform
2. **Portable Prompt Templates** (`prompts/`): Canonical, platform-neutral stance and workflow definitions
3. **Platform Adapters** (`adapters/`): Thin wiring that maps portable templates into each platform's native format

| Platform | Slash Commands | Agent Stances | Tool Enforcement | Agent Teams |
|----------|---------------|---------------|-----------------|-------------|
| Claude Code | Native | Native (tool-level) | Native | Native |
| OpenCode | Via prompt templates | Native (permission-level) | Native | No |
| Cursor | Via prompt templates | Rules (.mdc) | No (behavioral) | No |
| Windsurf | Via prompt templates | Rules (.md) | No (behavioral) | No |
| Cline | Via prompt templates | Rules (.clinerules) | No (behavioral) | No |
| Aider | Via prompt templates | Context loading | No (behavioral) | No |
| Other CLI | Via run-stage.sh | Context loading | No (behavioral) | No |

Portable templates are the canonical methodology reference. Claude Code commands are the reference adapter implementation. When methodology changes, update `prompts/workflows/` first, then sync platform-specific files. A drift-detection script (`scripts/check-prompt-sync.sh`) verifies alignment.

See `adapters/` for platform-specific setup instructions and `prompts/README.md` for direct template usage.

### 4.2 Stage 1: Feature Isolation → PRD

**Command:** `/new-feature <brief description>`

**What happens:**

1. Creates a feature branch: `feature/<slug>` from `main`
2. Generates `docs/PRDs/<slug>.md` from the template
3. The agent researches the codebase (using code search, symbol lookup, dependency analysis) and fills in:
   - How the feature interacts with existing architecture
   - Which files/modules will be affected
   - What data models need to change
   - What the user-facing behavior should be
   - What success looks like (acceptance criteria)
   - What could go wrong (risk assessment)
4. The PRD is committed to the feature branch

**PRD Template Structure:**

```markdown
# PRD: [Feature Name]

## Status: DRAFT | IN REVIEW | APPROVED | IMPLEMENTED
## Author: [agent/human]
## Date: YYYY-MM-DD
## Branch: feature/<slug>

## 1. Problem Statement
What problem does this solve? Who has this problem? Why now?

## 2. Proposed Solution
High-level description. Not implementation details — those go in the PRP.

## 3. Architecture Impact
- Files/modules affected
- Data model changes
- API changes
- Dependency additions

## 4. Acceptance Criteria
Numbered list. Each criterion is testable.

## 5. Risk Assessment
What could go wrong? What are the unknowns?

## 6. Open Questions
Things that need answers before implementation begins.

## 7. Out of Scope
Explicitly: what this feature does NOT do.
```

**Why PRDs matter:** The PRD forces the agent to think about the feature at the product level before dropping into implementation. It's where you catch "wait, this conflicts with feature X" or "the data model can't support this without a migration." PRDs are cheap. Rewrites are expensive.

### 4.3 Stage 2: Ticket Isolation → PRP

**Command:** `/generate-prp docs/PRDs/<slug>.md` _(or auto-generated by `/new-feature`)_

**Note:** `/new-feature` now auto-generates the PRP after the PRD for standard/full tiers. `/generate-prp` remains available for regenerating just the PRP without redoing the PRD.

**What happens:**

1. Reads the approved PRD
2. Researches the codebase in depth:
   - Finds all related symbols, patterns, and conventions
   - Reads existing tests to understand validation patterns
   - Checks external documentation for libraries involved
   - Reviews CLAUDE.md for project-specific conventions
3. Generates `docs/PRPs/<slug>.md` — a comprehensive implementation blueprint
4. The PRP is committed to the feature branch

**PRP Template Structure:**

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

Ordered task list. Each task is:
- Small enough to implement in one pass
- Independently testable via TDD (RED → GREEN → REFACTOR)
- Described with enough detail that an agent can execute without ambiguity

### Task 1: [Description]
**Files:** list of files to create/modify
**Approach:** pseudocode or detailed description
**Tests:** what to test and how (written BEFORE implementation)
**Validation:** commands to run

### Task 2: ...

## 4. Validation Gates
```bash
# Syntax/Style
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
```

**Key design decision:** The PRP is not a prompt. It's a complete implementation blueprint that contains everything the executing agent needs in one document — code patterns, library docs, test strategies, validation commands. The goal is **one-pass implementation.** If the PRP is good enough, the executing agent should be able to implement the feature without asking questions or searching the internet.

### 4.4 Stage 3: Push Hands Plan Review

**Command:** `/review-plan docs/PRPs/<slug>.md`

**What happens:**

1. A **second agent invocation** reads the PRP with a different stance — the "Senior Training Partner" role (defined in AGENTS.md)
2. The training partner's job is to sense where the structure yields:
   - Missing edge cases
   - Incorrect assumptions about existing code
   - Underspecified tasks
   - Security implications
   - Performance concerns
   - Unnecessary complexity
3. The review is saved to `docs/reviews/plans/<slug>-review.md`
4. The review either **APPROVES** (with optional notes) or **REQUESTS CHANGES** (with specific pressure points)
5. If changes requested, the PRP is regenerated incorporating the feedback

**Training Partner Stance (from AGENTS.md):**

The training partner is explicitly instructed to probe for structural weakness, to assume the plan has at least three yield points, and to check every file path, symbol reference, and API assumption against the actual codebase. The training partner does not write code — it only tests the plan's balance.

**Why this works:** The proposer optimizes for completeness and coherence. The training partner optimizes for correctness and resilience. These are genuinely different optimization targets, and the tension between them reveals problems that either stance alone would miss. Like in tui shou — you can't find your own blind spots. You need contact with a partner who's listening for them.

### 4.5 Stage 4: PRP Execution

**Command:** `/execute-prp docs/PRPs/<slug>.md`

**What happens:**

1. Reads the approved PRP
2. Executes tasks in order, committing after each task
3. Runs validation gates after each task (lint, type check, tests)
4. If validation fails, fixes before proceeding — does NOT mock to make tests pass
5. Logs uncertainty (anything it had to guess about) to the PRP's Uncertainty Log
6. Creates a summary commit message referencing the PRP

**Execution Rules (TDD):**

All implementation follows a strict TDD loop:

1. **RED:** Write test(s) that specify expected behavior. Run the test command — verify tests FAIL. If tests pass before implementation, they are too weak — rewrite them.
2. **GREEN:** Implement minimum code to make tests pass. Run the test command — verify tests PASS. If tests still fail, fix implementation (never weaken tests). Never mock to make tests pass.
3. **REFACTOR:** Clean up if needed. Tests must still pass.
4. **COMMIT:** Test + implementation together, one atomic commit with conventional format.

Additional rules:
- The PRP must include a `Test Command` field
- If no test framework exists yet, the first task must set one up
- If a task requires changing the plan, the change is documented in the PRP before proceeding
- If the agent gets stuck (3+ failed attempts on same task), it stops and flags for human review rather than thrashing

### 4.6 Stage 5: Push Hands Code Review

**Command:** `/review-code`

**What happens:**

1. A second agent invocation reviews all changes on the feature branch against `main`
2. Uses the "Code Review Partner" stance (defined in AGENTS.md)
3. Tests for:
   - Does the code match the PRP specification?
   - Are there security vulnerabilities? (injection, auth bypass, data exposure)
   - Are there performance issues? (N+1 queries, unbounded loops, memory leaks)
   - Are tests meaningful or are they testing implementation details?
   - Does the code follow project conventions (from CLAUDE.md)?
   - Is there dead code, commented-out code, or TODO items that should be resolved?
4. Produces `docs/reviews/code/<slug>-review.md`
5. Either APPROVES or REQUESTS CHANGES

### 4.7 Stage 6: Security Audit (Hard Stance)

**Command:** `/security-audit`

**What happens:**

This is where push hands shifts to a harder frame. The security audit deliberately drops the cooperative training partner dynamic and adopts an adversarial stance — because security testing requires a model that's trying to break things, not just sense weakness.

1. Two agents with opposing stances:
   - **Auditor Agent** — Finds vulnerabilities, rates severity, provides PoC exploits
   - **Skeptical Client Agent** — Challenges severity ratings, questions false positives, demands proof
2. The tension between them produces a report that is both thorough AND defensible
3. Produces `docs/audits/<slug>-audit.md`
4. Severity ratings: CRITICAL / HIGH / MEDIUM / LOW / INFO
5. Each finding includes: description, affected code, PoC or explanation, remediation, effort estimate

**Audit Report Template:**

```markdown
# Security Audit: [Feature/Component]

## Scope
What was examined. What was NOT examined.

## Methodology
Tools and techniques used. Agent roles and interaction pattern.

## Findings Summary
| # | Severity | Category | Description | Effort |
|---|----------|----------|-------------|--------|

## Detailed Findings
### Finding 1: [Title]
**Severity:** CRITICAL | HIGH | MEDIUM | LOW | INFO
**Category:** Auth | Input Validation | Data Exposure | Config | ...
**Affected Code:** file:line
**Description:** ...
**Proof of Concept:** ...
**Remediation:** ...
**Effort Estimate:** XS | S | M | L | XL

## What This Audit Did NOT Find
Explicit statement of limitations.

## Remediation Priority
Ordered list with effort estimates.
```

---

## 5. CLAUDE.md — The Living Project Brain

CLAUDE.md is not a static file you write once. It's a living document that agents update as they learn about the project.

### Structure

```markdown
# [Project Name]

## Project Overview
One paragraph. What this is, who it's for, what it does.

## Architecture
High-level architecture description. Key patterns. Data flow.

## Tech Stack
Languages, frameworks, key libraries with versions.

## Conventions
- Coding style rules
- Naming conventions
- File organization rules
- Testing conventions
- Commit message format

## Known Gotchas
Things that have caused problems before. Updated by agents after corrections.
Format: "YYYY-MM-DD: [what went wrong] → [what to do instead]"

## Agent Instructions
- How to run the dev server
- How to run tests
- How to lint
- What NOT to do (e.g., "never mock auth in integration tests")

## Active Work
Links to in-progress PRDs and PRPs. Updated as features move through the pipeline.
```

### Self-Update Protocol

**Command:** `/update-claude-md`

After any correction from the developer — "no, don't do it that way, do it this way" — the agent should update CLAUDE.md's Known Gotchas section so the mistake doesn't repeat. This is triggered manually or can be hooked into the post-correction flow.

The update is committed with the message: `docs: update CLAUDE.md — [brief description of lesson learned]`

---

## 6. AGENTS.md — Stance Definitions

This file defines the stances — system prompts and behavioral constraints — for each agent role used in the push hands workflow. These aren't just configuration; they're context primes. The name, the framing, and the metaphor all shape the model's output character.

```markdown
# Push Hands Stance Definitions

## Proposer (Default Stance)
Used by: /new-feature, /generate-prp, /execute-prp
Character: Thorough, systematic, completion-oriented
Goal: Produce comprehensive, well-structured artifacts
Constraints: Must reference existing code patterns, must follow CLAUDE.md conventions

## Senior Training Partner
Used by: /review-plan
Character: Patient, perceptive, structurally attuned
Goal: Sense where the plan yields under pressure — before implementation exposes it
Constraints: Cannot write code, only test the plan's balance. Must cite specific yield points with evidence.
Key behaviors:
- Assumes the plan has at least 3 structural weaknesses
- Checks every file path and symbol reference against actual codebase
- Senses security implications even if not explicitly a security review
- Tests necessity: "Does this need to exist? What's the simpler structure?"
- Probes for hidden coupling and unstated assumptions

## Code Review Partner
Used by: /review-code
Character: Detail-oriented, convention-aware, quality-sensing
Goal: Ensure code matches spec, is resilient, and follows project standards
Constraints: Reviews diff only. Does not rewrite — only identifies where balance breaks.
Key behaviors:
- Compares implementation against PRP task by task
- Tests for common vulnerability patterns
- Validates test quality (not just coverage)
- Flags convention violations per CLAUDE.md

## Security Auditor (Hard Stance)
Used by: /security-audit (attacker role)
Character: Thorough, exploit-minded, severity-calibrated
Goal: Find real vulnerabilities with actionable remediation
Note: This is a deliberate shift from push hands to a harder adversarial frame.
Constraints: Must provide proof of concept or clear exploitation path for HIGH+ findings.

## Skeptical Client (Hard Stance)
Used by: /security-audit (challenger role)
Character: Budget-conscious, dubious, demands proof
Goal: Challenge inflated severity, catch false positives, ensure report is defensible
Constraints: Cannot dismiss findings without technical justification.
```

### Stance vs. Role

The distinction matters. A "role" is a job description — it tells the model what to do. A "stance" is a way of being — it tells the model how to attend. "Staff Engineer Reviewer" produces output that looks like a code review checklist. "Senior Training Partner" produces output that reads like someone who's been listening carefully and found the three things you didn't realize you were assuming. Same mechanism, different attentional quality.

---

## 7. Git Workflow

### Branch Strategy

```
main                    ← Production-ready code only
  └── feature/<slug>    ← One branch per feature (from PRD)
        └── (worktrees) ← Optional parallel implementation tracks
```

### Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/) enforced by pre-commit hook:

```
feat: add user authentication flow
fix: prevent IDOR in claim-analysis endpoint
docs: update CLAUDE.md — never use raw SQL in handlers
test: add integration tests for payment processing
refactor: extract context assembly into dedicated module
chore: update dependencies
```

### Automated Branch Management

**`scripts/new-feature.sh <name> [--tier light|standard|full]`**
1. `git checkout main && git pull`
2. `git checkout -b feature/<name>`
3. Writes tier to `.push-hands-tier` (default: `standard`)
4. For standard/full: creates `docs/PRDs/<name>.md` from template; for light: skips PRD
5. Initial commit: `docs: scaffold PRD for <name>` (standard/full) or `chore: create feature branch for <name> (light tier)` (light)

**`scripts/close-feature.sh <name>`**
1. Reads `.push-hands-tier` to determine tier
2. Verifies review artifacts exist (tier-appropriate: light skips all, standard checks PRD/PRP/reviews, full additionally checks audit)
3. Verifies all tests pass
4. Squash-merges feature branch to main, removing `.push-hands-tier` from staging before commit
5. Cleans up branch

### Parallel Worktrees (Advanced)

**`scripts/worktree-setup.sh <feature> <track-name>`**
For complex features that can be parallelized (e.g., backend + frontend simultaneously):
1. Creates a named worktree: `git worktree add ../project-<track-name> feature/<feature>`
2. Each worktree gets its own Claude Code session
3. Worktrees merge back to the feature branch, not directly to main

---

## 8. Hooks & Automation

### Git Hooks (scripts/hooks/)

**pre-commit:**
```bash
#!/bin/bash
# Lint and type check — fast gate, runs on every commit
ruff check src/ --fix
mypy src/ --ignore-missing-imports
```

**pre-push:**
```bash
#!/bin/bash
# Full test suite — slower gate, runs before push
pytest tests/ -v --tb=short
```

**commit-msg:**
```bash
#!/bin/bash
# Enforce conventional commits
PATTERN="^(feat|fix|docs|test|refactor|chore|style|perf|ci|build|revert)(\(.+\))?: .{1,72}"
if ! grep -qE "$PATTERN" "$1"; then
    echo "ERROR: Commit message must follow Conventional Commits format"
    echo "Example: feat: add user authentication"
    exit 1
fi
```

### Setup

The template includes a setup script that installs hooks automatically:

```bash
# Run after cloning the template
./scripts/setup.sh
```

This copies hooks to `.git/hooks/`, installs dependencies, and verifies the development environment.

---

## 9. Adapting the Template

### For a New Project

1. Create a new repository from this template on GitHub
2. Run `./scripts/setup.sh`
3. Edit CLAUDE.md with your project's details
4. Edit AGENTS.md if you want to customize agent roles
5. Start with `/new-feature` for your first feature

### For an Existing Project

1. Copy the `docs/`, `scripts/`, and `.claude/` directories into your project
2. Copy CLAUDE.md and AGENTS.md to your project root
3. Run `./scripts/setup.sh`
4. Populate CLAUDE.md with your project's existing architecture and conventions
5. Retroactively create PRDs for major existing features (optional but recommended)

### Customization Points

- **Language/framework:** Replace `ruff`/`mypy`/`pytest` in hooks and validation gates with your stack's equivalents
- **Stances:** Edit AGENTS.md to adjust stance character and constraints
- **Workflow tiers:** Use `--tier light` for small changes, `--tier full` for security-sensitive work. Default is `standard`. See Section 4.1 for details.
- **Security audit frequency:** Per-feature for security-sensitive apps (use full tier), per-release for others
- **Worktrees:** Optional. Only needed if you're running parallel Claude Code sessions.

---

## 10. Why This Works (The Theory)

### Push Hands as Agent Orchestration

Single-agent workflows have a fundamental problem: the agent that proposes a solution is incentivized to believe its solution is good. It will rationalize gaps, downplay risks, and optimize for completion over correctness. This isn't a flaw in the model — it's a structural property of having one perspective. You cannot find your own blind spots through introspection. You need contact.

Adding a second agent with a different stance changes the optimization landscape. The proposer now knows its work will be tested, which (via the training dynamics of instruction-tuned models) produces more careful initial work. The training partner, freed from the need to be constructive or comprehensive, can focus purely on sensing where the structure gives.

This is the same dynamic that makes:

- **Tui shou (push hands)** effective in martial arts — you learn your structural weaknesses through contact, not forms practice
- **Red team/blue team** exercises effective in security
- **Peer review** effective in academia
- **Cross-examination** effective in law
- **Devil's advocate** effective in strategic planning

The template formalizes this dynamic and makes it a repeatable, automatable part of the development process.

The critical insight: **the name matters.** "Adversarial review" primes a model toward attack and fault-finding. "Push hands review" primes toward sensing and structural testing. The second frame produces output with different attentional qualities — more patient, more attuned to subtle imbalances, less focused on surface-level catches. The stance definitions in AGENTS.md aren't just system prompts; they're context primes that shape the character of the model's attention. This is a known property of LLM behavior (role priming / narrative framing) applied deliberately as an engineering tool.

### Documents as Context Engineering

Every document in this system (CLAUDE.md, PRDs, PRPs, reviews) is a piece of the context engineering puzzle. When an agent runs `/execute-prp`, the PRP IS the context — it contains everything the agent needs to know. When an agent runs `/review-plan`, the PRP plus the project's CLAUDE.md IS the context.

By structuring documents carefully and keeping them current, you're solving the context engineering problem at the process level, not the prompt level. You never need to figure out "what should I put in my prompt?" because the process generates the context automatically.

### Git as Memory

AI agents don't have memory across sessions. But git does. Every commit, every branch, every PR description is persistent context that a future agent invocation can access. The git workflow in this template is designed to make that context maximally useful — conventional commits are searchable, feature branches isolate context, review artifacts explain *why* decisions were made.

---

## 11. Success Metrics

For Ontologi's own use and for clients adopting this template:

- **First-pass implementation rate:** % of PRPs that result in working code without human intervention. Target: 80%+.
- **Defect escape rate:** % of bugs found in production that were NOT caught by push hands review. Target: <10%.
- **Time to feature:** Calendar time from feature brief to merged PR. Should decrease over time as CLAUDE.md accumulates project knowledge.
- **CLAUDE.md growth rate:** Number of "Known Gotchas" entries over time. A healthy project adds 2-3 per week initially, tapering as the agent learns the codebase.
- **Review rejection rate:** % of plan/code reviews that request changes. If this is 0%, the reviews aren't strict enough. If it's >50%, the PRPs aren't detailed enough. Target: 15-30%.

---

## 12. Roadmap

### v1.0 — Template Release
- Complete repository structure with all templates
- Slash commands for full lifecycle
- Git hooks and automation scripts
- Setup script
- README with usage guide
- Example PRD → PRP → Review → Implementation walkthrough

### v1.1 — CI/CD Integration
- GitHub Actions workflow for automated push hands review on PR
- Status checks that block merge without review artifacts
- Automated CLAUDE.md staleness detection

### v1.2 — Analytics
- Token usage tracking per stage
- Time tracking per stage
- Review effectiveness metrics
- Dashboard for project health

### v1.3 — Multi-Agent Orchestration ✓
- Agent team mode for concurrent proposer/training partner workflows
- Custom agent definitions in `.claude/agents/` with tool-level stance enforcement
- Team-mode slash commands: `/review-plan-team`, `/execute-team`, `/security-audit-team`
- User guide at `docs/guides/agent-teams.md`

---

## Appendix A: Quick Reference

| Stage | Command | Tier | Input | Output | Stance |
|-------|---------|------|-------|--------|--------|
| Feature Brief → PRD + PRP | `/new-feature <brief>` | Standard, Full | Description | `docs/PRDs/<slug>.md` + `docs/PRPs/<slug>.md` | Proposer |
| Feature Branch (no PRD) | `/new-feature --tier light <brief>` | Light | Description | Feature branch only | Proposer |
| Regenerate PRP | `/generate-prp <prd>` | Standard, Full | PRD path | `docs/PRPs/<slug>.md` | Proposer |
| Plan Review | `/review-plan <prp>` | Standard, Full | PRP path | `docs/reviews/plans/<slug>.md` | Senior Training Partner |
| Implementation | `/execute-prp <prp>` | Standard, Full | PRP path | Source code + tests | Proposer |
| Code Review | `/review-code` | All (optional for Light) | Branch diff | `docs/reviews/code/<slug>.md` | Code Review Partner |
| Security Audit | `/security-audit` | Full (optional for Standard) | Branch diff | `docs/audits/<slug>.md` | Auditor + Client (Hard) |
| Learn from Correction | `/update-claude-md` | All | — | Updated CLAUDE.md | Proposer |
| Concurrent Plan Review | `/review-plan-team <prp>` | Standard, Full | PRP path | Revised PRP + `docs/reviews/plans/<slug>.md` | Team: Proposer + Training Partner |
| Parallel Implementation | `/execute-team <prp>` | Standard, Full | PRP path | Source code + tests + `docs/reviews/code/<slug>.md` | Team: Proposer + Code Reviewer |
| Dual-Agent Security Audit | `/security-audit-team` | Full | Branch diff | `docs/audits/<slug>.md` | Team: Auditor + Client |

## Appendix B: File Naming Conventions

- PRDs: `docs/PRDs/<feature-slug>.md` — lowercase, hyphen-separated
- PRPs: `docs/PRPs/<feature-slug>.md` — matches PRD slug
- Reviews: `docs/reviews/{plans,code}/<feature-slug>-review.md`
- Audits: `docs/audits/<feature-slug>-audit.md`
- Tier metadata: `.push-hands-tier` — single word (`light`, `standard`, or `full`), lives on feature branches only, never on `main`
- Branches: `feature/<feature-slug>`
- Commits: Conventional Commits format, max 72 chars first line
