# PRP: Multi-Agent Platform Parity

## Source PRD: docs/PRDs/multi-agent-parity.md
## Date: 2026-02-11
## Confidence Score: 8/10

## 1. Context Summary

Ontologi Push Hands is a methodology-as-code repository currently locked to Claude Code. The methodology itself — PRD/PRP pipelines, stance-based dialectical review, tiered workflows — is agent-agnostic. The implementation layer (slash commands, agent definitions with tool restrictions, agent teams) is not.

This PRP implements a three-layer decoupling:

1. **Shared Infrastructure** (already portable): `docs/`, `scripts/`, hooks, git workflow
2. **Portable Prompt Templates** (new `prompts/`): canonical methodology definitions — stances and workflows — expressed in platform-neutral language
3. **Platform Adapters** (new `adapters/` + platform-specific directories): thin wiring that maps portable templates into each platform's native format

The goal is additive — Claude Code functionality must remain identical. Other platforms get sequential workflow support with prompt-level stance enforcement. Team mode stays Claude Code-exclusive.

### Key Research Finding: AGENTS.md Is Context, Not Agent Definitions

The AGENTS.md convention (stewarded by the Agentic AI Foundation under the Linux Foundation) is a **plain markdown context/instructions document** — not an agent definition format. It does not support personas, tool restrictions, or structured metadata. It's read by 16+ tools as project context.

Only two platforms support **named agent definitions with tool/permission restrictions**:
- **Claude Code**: `.claude/agents/<name>.md` with YAML frontmatter (`tools:` field)
- **OpenCode**: `.opencode/agents/<name>.md` with YAML frontmatter (`permission:` field)

Other platforms (Cursor, Windsurf, Cline, Aider) only support rules/instructions — behavioral guidance, not enforced constraints.

## 2. Codebase Analysis

### Current Structure (Claude Code-Specific)

```
.claude/
  commands/          # 10 slash commands (7 sequential + 3 team)
    new-feature.md
    generate-prp.md
    review-plan.md
    execute-prp.md
    review-code.md
    security-audit.md
    update-claude-md.md
    review-plan-team.md      # Team mode only
    execute-team.md          # Team mode only
    security-audit-team.md   # Team mode only
  agents/            # 5 stance definitions with tool restrictions
    proposer.md              # tools: Read, Write, Edit, Bash, Grep, Glob
    training-partner.md      # tools: Read, Grep, Glob, Bash (NO Write/Edit)
    code-reviewer.md         # tools: Read, Grep, Glob, Bash (NO Write/Edit)
    security-auditor.md      # tools: Read, Grep, Glob, Bash (NO Write/Edit)
    skeptical-client.md      # tools: Read, Grep, Glob (NO Write/Edit/Bash)
```

### Content Analysis: What Is Methodology vs. What Is Wiring

Each `.claude/commands/*.md` file interleaves:

**Methodology (portable):**
- Stance prime (character, constraints, key behaviors)
- Artifact template (section structure, required fields)
- Review criteria (yield points, severity scales, verdict logic)
- Process rules (commit discipline, validation gates, escalation)
- Convergence protocols (max rounds, termination conditions)

**Claude Code wiring (platform-specific):**
- `$ARGUMENTS` variable injection
- Tool names: `Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob`
- Team primitives: `TeamCreate`, `SendMessage`, `TaskCreate`, `TaskUpdate`, `TaskList`, `TeamDelete`
- `run_in_background: true` for concurrent agent spawning
- `subagent_type:` references to `.claude/agents/` definitions

### Agent Definition Comparison: Claude Code vs. OpenCode

| Feature | Claude Code (`.claude/agents/`) | OpenCode (`.opencode/agents/`) |
|---|---|---|
| Frontmatter | `name`, `tools`, `model`, `description` | `description`, `mode`, `model`, `temperature`, `permission`, `steps`, etc. |
| Tool restriction | `tools: Read, Grep, Glob, Bash` (whitelist) | `permission: { edit: deny, bash: { "*": ask } }` (per-tool allow/ask/deny) |
| Model override | `model: inherit` or explicit | `model: provider/model-id` |
| Body | System prompt markdown | System prompt markdown |

### Platform-Specific Formats

| Platform | Config Location | Format | Agent Support |
|---|---|---|---|
| Claude Code | `.claude/commands/`, `.claude/agents/` | MD with YAML frontmatter | Full (definitions + teams) |
| OpenCode | `.opencode/agents/`, `AGENTS.md` | MD with YAML frontmatter | Definitions + subagents (no teams) |
| Cursor | `.cursor/rules/*.mdc` | MDC (frontmatter: description, globs, alwaysApply) | Rules only, no agents |
| Windsurf | `.windsurf/rules/*.md` | Plain MD (4 activation modes) | Rules only, no agents |
| Cline | `.clinerules/*.md` | MD (optional `paths:` frontmatter) | Rules only, no agents |
| Aider | `.aider.conf.yml` + `--read` files | YAML config + MD files | Context loading only |

### Conventions to Follow

- Slugs: `^[a-z0-9]+(-[a-z0-9]+)*$` (from CLAUDE.md)
- Commits: Conventional Commits, max 72 chars first line
- Each PRP task = one atomic commit
- File paths for artifacts: `docs/PRDs/`, `docs/PRPs/`, `docs/reviews/{plans,code}/`, `docs/audits/`

### Key Constraint: Claude Code Commands Are Self-Contained

Claude Code `.claude/commands/*.md` cannot import or reference external files via includes. They are loaded as-is. The "import from `prompts/`" approach described in the PRD must be implemented as:

1. Portable templates in `prompts/` are the **canonical reference**
2. Claude Code commands remain self-contained for reliability
3. The relationship is documented: commands implement the methodology defined in `prompts/`
4. A sync-check script (`scripts/check-prompt-sync.sh`) detects drift between portable templates and Claude Code commands by comparing section headers

This avoids a build step and keeps Claude Code commands working without modification.

**PRD AC #3 deviation:** The PRD specifies that Claude Code commands should be "refactored to delegate to `prompts/workflows/`." This is not technically feasible — Claude Code commands cannot import external files. Instead, commands remain self-contained and a drift-detection script (Task 10) ensures the portable templates and Claude Code commands stay aligned. The PRD's HIGH-risk mitigation ("Lint/CI check") is addressed by this script. This is a deliberate trade-off: reliability of self-contained commands over DRY purity, with automated drift detection as the safety net.

## 3. External Research

### AGENTS.md Convention (Agentic AI Foundation)

- **Repository:** github.com/agentsmd/agents.md (transferred from openai/agents.md)
- **Format:** Plain markdown, no required schema, no frontmatter
- **Common sections:** Project Overview, Build & Test, Code Style, Architecture, Security, Git Conventions
- **Adoption:** 60,000+ open-source projects, 16+ tools
- **Key insight:** AGENTS.md is project context, not agent configuration. It supplements but does not replace tool-specific config.

### OpenCode Agent Definitions

- **Docs:** opencode.ai/docs/agents/
- **Permission system:** `edit: deny`, `bash: { "git diff": allow, "*": ask }`, `webfetch: deny`
- **Modes:** `primary` (direct interaction), `subagent` (invoked via `@name`), `all` (both)
- **Deprecated:** `tools:` field replaced by `permission:` field
- **Also reads:** `AGENTS.md` and `CLAUDE.md` from project root

### Cursor Rules (.mdc format)

- **Docs:** cursor.com documentation, design.dev/guides/cursor-rules/
- **Activation types:** Always Applied, Auto-Attached (globs), Agent-Requested (description-based), Manual (@ruleName)
- **File naming:** Numeric prefixes for load order (e.g., `00-project-context.mdc`)
- **Also reads:** `AGENTS.md` from project root

### Cline Rules

- **Docs:** docs.cline.bot/features/cline-rules/
- **Priority:** `.clinerules/` folder > `.clinerules` file > `AGENTS.md` > `.cursor/rules/` > `.windsurf/rules`
- **Scoping:** `paths:` frontmatter with glob patterns
- **Key:** Cline also reads `.cursor/rules/` and `.windsurf/rules` as fallbacks

### Claude Code ↔ AGENTS.md Interop

Claude Code reads `CLAUDE.md`, not `AGENTS.md`. The recommended interop pattern is a symlink: `ln -s AGENTS.md CLAUDE.md`. However, Push Hands needs both files (AGENTS.md for stances, CLAUDE.md for project-specific context), so a symlink is not appropriate here.

## 4. Implementation Plan

### Task 1: Create portable stance definitions

**Files:**
- `prompts/stances/proposer.md` (create)
- `prompts/stances/training-partner.md` (create)
- `prompts/stances/code-reviewer.md` (create)
- `prompts/stances/security-auditor.md` (create)
- `prompts/stances/skeptical-client.md` (create)

**Approach:**

Extract the behavioral/methodological content from `.claude/agents/*.md`, stripping all Claude Code-specific elements:

- Remove YAML frontmatter (`name`, `tools`, `model`, `description`)
- Remove "Team Communication" sections (SendMessage, TaskUpdate references)
- Remove tool-specific references ("you have no Write or Edit tools" → "you must not modify files")
- Keep: stance character, constraints, key behaviors, output format
- Add a header noting this is the portable stance definition and pointing to platform adapters for enforcement details

**Template for each stance file:**
```markdown
# [Stance Name]

> Portable stance definition. For platform-specific enforcement (tool restrictions,
> agent configurations), see `adapters/` for your platform.

## Character
[from .claude/agents/*.md]

## Constraints
[reworded to be platform-neutral]

## Key Behaviors
[from .claude/agents/*.md]

## Output Format
[from .claude/agents/*.md]
```

Specific content per stance:

**proposer.md**: Character (thorough, systematic, completion-oriented), constraints (reference project context docs, follow conventions, log uncertainty, run validation gates), no tool restriction note needed (proposer has full access everywhere).

**training-partner.md**: Character (patient, perceptive, structurally attuned), constraints (must not modify files — analysis and review only, cite specific yield points with evidence, check file paths and symbol references against codebase, assume at least 3 structural weaknesses), key behaviors (missing edge cases, incorrect assumptions, underspecified tasks, security implications, performance concerns, unnecessary complexity, hidden coupling, test gaps), output format (APPROVES or REQUESTS CHANGES with yield points).

**code-reviewer.md**: Character (detail-oriented, convention-aware, quality-sensing), constraints (must not modify files — reviews only, compare implementation against PRP task by task), key behaviors (spec mismatch, security patterns, performance issues, test quality, convention violations, dead code/TODOs), severity levels (Blocking, Significant, Minor, Nit), output format (APPROVES or REQUESTS CHANGES).

**security-auditor.md**: Character (thorough, exploit-minded, severity-calibrated — hard stance), constraints (must not modify files except for PoC testing via shell commands, must provide PoC or clear exploitation path for HIGH+ findings, must state scope limitations), severity ratings (CRITICAL/HIGH/MEDIUM/LOW/INFO), output format (finding with description, affected code, PoC, remediation, effort estimate).

**skeptical-client.md**: Character (budget-conscious, dubious, demands proof — hard stance), constraints (must not modify files or run shell commands — pure analysis, cannot dismiss findings without technical justification, must challenge every HIGH+ finding), output format (severity assessment with justification per finding).

**Tests:** Verify all 5 files exist, each contains Character/Constraints sections with non-trivial content, none contain Claude Code tool names (`Read`, `Write`, `Edit`, `Grep`, `Glob`, `SendMessage`, `TaskUpdate`, `TeamCreate`).

**Validation:**
```bash
# All 5 stances exist
for stance in proposer training-partner code-reviewer security-auditor skeptical-client; do
  test -f "prompts/stances/${stance}.md" && echo "✓ ${stance}" || echo "✗ ${stance} MISSING"
done

# No Claude Code tool references in portable stances
! grep -rEl '(SendMessage|TaskUpdate|TeamCreate|TaskList|TaskDelete)' prompts/stances/

# Each stance has required sections with non-trivial content (>2 lines after header)
for stance in proposer training-partner code-reviewer security-auditor skeptical-client; do
  file="prompts/stances/${stance}.md"
  grep -q '## Character' "$file" && echo "✓ ${stance} Character" || echo "✗ ${stance} missing Character"
  grep -q '## Constraints' "$file" && echo "✓ ${stance} Constraints" || echo "✗ ${stance} missing Constraints"
done
```

---

### Task 2: Create portable workflow templates

**Files:**
- `prompts/workflows/new-feature.md` (create)
- `prompts/workflows/generate-prp.md` (create)
- `prompts/workflows/review-plan.md` (create)
- `prompts/workflows/execute-prp.md` (create)
- `prompts/workflows/review-code.md` (create)
- `prompts/workflows/security-audit.md` (create)
- `prompts/workflows/update-project-docs.md` (create)

**Approach:**

Extract methodology from `.claude/commands/*.md` (the 7 sequential commands), stripping Claude Code wiring:

- Replace `$ARGUMENTS` with `{input}` placeholder and a "## Input" section describing what the user provides
- Replace tool names with generic verbs: "Read file X" → "Read the file at X", "Run `git diff`" → "Run: `git diff`"
- Remove slash command framing ("You are operating in the **X** stance" stays — that's methodology)
- Remove agent team references (those stay in the Claude Code adapter)
- Add a header: "Portable workflow template. To use: load this file as context for your AI coding agent."
- Include the relevant stance prime inline (reference `prompts/stances/X.md` and include a condensed version)
- Preserve artifact templates exactly (section structure, required fields)
- Preserve validation gate commands exactly (they're already language-agnostic placeholders)

**Template structure for each workflow:**
```markdown
# Workflow: [Stage Name]

> Portable workflow template. Load this as context for your AI coding agent,
> or use a platform adapter (see `adapters/`).

## Stance
[Condensed stance prime — 3-4 sentences from prompts/stances/X.md]

## Input
[What the user provides — file path, description, etc.]

## Instructions
[Step-by-step methodology, using generic verbs]

## Artifact Template
[The output format/template to produce]

## Validation
[Commands to run after completion]
```

Specific notes per workflow:

**new-feature.md**: Tier parsing logic, branch creation, PRD scaffolding. Replace "Read the template at `docs/PRDs/TEMPLATE.md`" tool call with "Read the PRD template at `docs/PRDs/TEMPLATE.md`". Keep the tier decision table. Keep the light-tier short-circuit.

**generate-prp.md**: Deep codebase research procedure, PRP structure, task decomposition rules, confidence scoring. This is the most methodology-dense workflow — nearly all of it is portable.

**review-plan.md**: Training Partner stance, yield point framework, pressure testing categories, verdict logic. Key constraint: "You must not modify files — only analyze and review."

**execute-prp.md**: Task execution sequence, per-task validation, commit discipline, escalation rules. Key rule: "Each task = one atomic commit."

**review-code.md**: Code Review Partner stance, diff analysis, PRP compliance check, severity levels, artifact format.

**security-audit.md**: Two-phase approach (Auditor → Skeptical Client), vulnerability categories, PoC requirements, severity calibration. Note: In the portable version, this is sequential — one agent plays both roles. The "stance shift" mid-audit is the key innovation (same agent, different prime).

**update-project-docs.md**: Generalized from `/update-claude-md`. Two inputs: `{input}` is the lesson to capture (correction, gotcha, convention clarification). `{target}` is the target context file — defaults to `CLAUDE.md` if not specified. The workflow includes a target detection heuristic: check for `CLAUDE.md` first, then `AGENTS.md`, then `.cursorrules`, then `.windsurfrules`, then fall back to asking the user. The lesson capture protocol is identical to `/update-claude-md`: identify what was learned, read the target file's structure, update the appropriate section (Known Gotchas, Conventions, Architecture), commit with `docs: update <target> — <lesson>`. The generalization adds a "## Target File" section specifying the detection order and a note that platform-specific adapters should hardcode their target (Claude Code adapter always uses CLAUDE.md).

**Tests:** Verify all 7 files exist, each contains required template sections (Stance, Input, Instructions, Validation), none contain `$ARGUMENTS` or Claude Code-specific tool names in instructions.

**Validation:**
```bash
# All 7 workflows exist
for wf in new-feature generate-prp review-plan execute-prp review-code security-audit update-project-docs; do
  test -f "prompts/workflows/${wf}.md" && echo "✓ ${wf}" || echo "✗ ${wf} MISSING"
done

# No $ARGUMENTS references
! grep -rl '\$ARGUMENTS' prompts/workflows/

# Each workflow has required template sections
for wf in new-feature generate-prp review-plan execute-prp review-code security-audit update-project-docs; do
  file="prompts/workflows/${wf}.md"
  for section in "## Stance" "## Input" "## Instructions" "## Validation"; do
    grep -q "$section" "$file" && echo "✓ ${wf} ${section}" || echo "✗ ${wf} missing ${section}"
  done
done

# No Claude Code team primitives
! grep -rEl '(TeamCreate|SendMessage|TaskCreate|subagent_type)' prompts/workflows/
```

---

### Task 3: Create prompts/README.md

**Files:**
- `prompts/README.md` (create)

**Approach:**

Write a README that explains:

1. **What this directory is:** The canonical, platform-agnostic definition of the Push Hands methodology
2. **Three-layer architecture:** Shared infra → Portable templates → Platform adapters
3. **How to use these templates directly:** Copy-paste into any AI agent's context window
4. **How stances work:** Behavioral definitions that shape how the agent approaches the task
5. **How workflows work:** Step-by-step instructions for each pipeline stage
6. **Relationship to platform adapters:** These templates are the source of truth; adapters wire them into native platform features
7. **What's NOT here:** Team mode (Claude Code-exclusive), tool-level enforcement (platform-specific)

Keep it concise — under 100 lines. Reference the main `push-hands.md` spec for full details.

**Tests:** File exists, contains sections for stances and workflows.

**Validation:**
```bash
test -f prompts/README.md
```

---

### Task 4: Create Claude Code adapter documentation

**Files:**
- `adapters/claude-code/README.md` (create)

**Approach:**

Document how Claude Code integrates with Push Hands — this is the reference adapter that other platforms compare against:

1. **Feature matrix:** What Claude Code supports (slash commands, agent definitions with tool enforcement, agent teams with concurrent execution, permission sandboxing)
2. **How it works:** `.claude/commands/*.md` = workflow execution, `.claude/agents/*.md` = stance enforcement with tool restrictions, `.claude/settings.local.json` = permission sandboxing
3. **Relationship to portable templates:** Claude Code commands are self-contained implementations of the methodology defined in `prompts/workflows/`. They add Claude Code-specific wiring (tool names, `$ARGUMENTS`, team spawning). The portable templates are the canonical reference.
4. **Team mode:** Only available in Claude Code. Requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. Points to `docs/guides/agent-teams.md`.
5. **Sync convention:** When methodology changes, update `prompts/workflows/` first (canonical), then update `.claude/commands/` to match.
6. **Setup:** `./scripts/setup.sh` handles everything. No additional steps needed.

**Tests:** File exists, mentions sync convention.

**Validation:**
```bash
test -f adapters/claude-code/README.md
```

---

### Task 5: Create OpenCode adapter

**Files:**
- `adapters/opencode/README.md` (create)
- `.opencode/agents/proposer.md` (create)
- `.opencode/agents/training-partner.md` (create)
- `.opencode/agents/code-reviewer.md` (create)
- `.opencode/agents/security-auditor.md` (create)
- `.opencode/agents/skeptical-client.md` (create)

**Approach:**

OpenCode is the closest platform to Claude Code — it supports named agent definitions with permission-based tool restrictions.

**README.md** covers:
1. What OpenCode supports (agent definitions with permissions, subagent mode, AGENTS.md context)
2. Setup: Copy `.opencode/agents/` into your project (already in template)
3. How to run workflows: Load the prompt template from `prompts/workflows/` via `--read` or paste into chat, then invoke with the relevant agent via `@agent-name`
4. Feature comparison with Claude Code (no teams, no slash commands — use prompt templates directly)
5. The permission system maps to Claude Code's tool restrictions but with finer granularity
6. **Permissions caveat:** Bash glob patterns in the `permission:` field are best-effort mappings based on OpenCode documentation. If a permission pattern doesn't behave as expected, adjust the glob or switch to `ask` as the fallback. The agent definitions prioritize safety (defaulting to `ask` or `deny` for unmatched commands) so failures should prompt rather than silently allow.

**Agent definitions** use OpenCode's YAML frontmatter format with `permission:` field:

```yaml
# .opencode/agents/training-partner.md
---
description: Senior Training Partner — senses where plans yield under pressure
mode: subagent
permission:
  edit: deny
  bash:
    "git *": allow
    "test -f *": allow
    "grep *": allow
    "*": ask
  webfetch: deny
---
[Body from prompts/stances/training-partner.md]
```

Map each Claude Code tool restriction to OpenCode permissions:

| Claude Code Agent | Claude Code Tools | OpenCode Permissions |
|---|---|---|
| proposer | Read, Write, Edit, Bash, Grep, Glob | `edit: allow`, `bash: { "*": allow }` |
| training-partner | Read, Grep, Glob, Bash | `edit: deny`, `bash: { "git *": allow, "test *": allow, "grep *": allow, "*": ask }` |
| code-reviewer | Read, Grep, Glob, Bash | `edit: deny`, `bash: { "git *": allow, "test *": allow, "grep *": allow, "*": ask }` |
| security-auditor | Read, Grep, Glob, Bash | `edit: deny`, `bash: { "*": allow }` (needs shell for PoC) |
| skeptical-client | Read, Grep, Glob | `edit: deny`, `bash: { "*": deny }`, `webfetch: deny` |

**Design note:** The default for unmatched bash commands on review-only agents (training-partner, code-reviewer) is `ask` rather than `deny`. This lets users approve unexpected but legitimate commands (e.g., `wc -l` for counting lines) while still preventing accidental writes. The skeptical-client uses `deny` because it should have no shell access at all.

Each OpenCode agent file's body content comes from the corresponding `prompts/stances/*.md` file (the portable stance definition).

**Tests:** All 6 files exist. Each `.opencode/agents/*.md` has valid YAML frontmatter with `permission:` field. No Claude Code tool names in the body. The YAML example and the mapping table use consistent permission defaults.

**Validation:**
```bash
# All OpenCode agents exist
for agent in proposer training-partner code-reviewer security-auditor skeptical-client; do
  test -f ".opencode/agents/${agent}.md" && echo "✓ ${agent}" || echo "✗ ${agent} MISSING"
done

test -f adapters/opencode/README.md

# Check frontmatter contains permission field
grep -l 'permission:' .opencode/agents/*.md | wc -l  # Should be 5
```

---

### Task 6: Create Cursor adapter

**Files:**
- `adapters/cursor/README.md` (create)
- `.cursor/rules/00-push-hands-context.mdc` (create)
- `.cursor/rules/01-proposer.mdc` (create)
- `.cursor/rules/02-training-partner.mdc` (create)
- `.cursor/rules/03-code-reviewer.mdc` (create)
- `.cursor/rules/04-security-auditor.mdc` (create)
- `.cursor/rules/05-skeptical-client.mdc` (create)

**Approach:**

Cursor uses `.mdc` files (Modular Document with Context) with frontmatter for activation rules. Cursor cannot enforce tool restrictions — all rules are behavioral guidance.

**README.md** covers:
1. What Cursor supports (rules with conditional activation, AGENTS.md context, no agent definitions, no teams)
2. Setup: `.cursor/rules/` directory is included in the template
3. How to run workflows: Copy the relevant `prompts/workflows/*.md` content into the chat, or use `@rule-name` to activate a stance rule
4. Limitations: No tool-level enforcement (stances are behavioral, not architectural), no concurrent agents

**Rule files** use `.mdc` format:

```
---
description: Push Hands Training Partner stance — activate when reviewing plans or PRPs
globs:
alwaysApply: false
---

[Content from prompts/stances/training-partner.md]
```

The context rule (`00-push-hands-context.mdc`) is `alwaysApply: true` and contains a condensed version of the Push Hands methodology (tier system, artifact paths, commit conventions) — equivalent to what CLAUDE.md provides for Claude Code.

Stance rules (`01-05`) are Agent-Requested (description set, no globs, `alwaysApply: false`) — Cursor's AI reads the description and decides whether to include the rule based on what the user asks.

**Tests:** All 7 files exist. Each `.mdc` file has valid frontmatter with `description:` field.

**Validation:**
```bash
# All Cursor rules exist
test -f ".cursor/rules/00-push-hands-context.mdc"
for i in 01-proposer 02-training-partner 03-code-reviewer 04-security-auditor 05-skeptical-client; do
  test -f ".cursor/rules/${i}.mdc" && echo "✓ ${i}" || echo "✗ ${i} MISSING"
done

test -f adapters/cursor/README.md
```

---

### Task 7: Create generic adapter

**Files:**
- `adapters/generic/README.md` (create)
- `adapters/generic/run-stage.sh` (create)

**Approach:**

The generic adapter serves Windsurf, Cline, Aider, and any CLI-based AI coding agent. It provides:

**README.md** covers:
1. How to use Push Hands with any AI coding agent
2. Platform-specific quick-start subsections:
   - **Windsurf:** Create `.windsurf/rules/push-hands.md` pointing to stances, or rely on `AGENTS.md` (auto-read)
   - **Cline:** Create `.clinerules/` folder with stance files, or rely on `AGENTS.md` (auto-read). Cline also reads `.cursor/rules/` as fallback.
   - **Aider:** Add `read: [AGENTS.md, prompts/stances/proposer.md]` to `.aider.conf.yml`
   - **Other CLI agents:** Use `run-stage.sh` to load and display prompt templates
3. Running a workflow: Load the relevant `prompts/workflows/*.md` as context, then follow the instructions
4. Limitations: No tool-level enforcement, no concurrent agents, no automated stage sequencing

**run-stage.sh** is a minimal shell wrapper:
```bash
#!/bin/bash
# Usage: ./adapters/generic/run-stage.sh <stage-name> [arguments]
# Loads a portable workflow template and outputs it for piping to any CLI agent.
#
# Examples:
#   ./adapters/generic/run-stage.sh review-plan docs/PRPs/my-feature.md
#   ./adapters/generic/run-stage.sh new-feature "add user profiles"
#   cat <(./adapters/generic/run-stage.sh review-plan docs/PRPs/my-feature.md) | aider --message -

STAGE="$1"
shift
ARGS="$*"
TEMPLATE="prompts/workflows/${STAGE}.md"

if [ ! -f "$TEMPLATE" ]; then
    echo "Error: Unknown stage '${STAGE}'. Available stages:" >&2
    ls prompts/workflows/ | sed 's/\.md$//' >&2
    exit 1
fi

# Output the template with {input} replaced by the actual arguments.
# Uses awk to avoid sed delimiter collision and metacharacter injection
# when ARGS contains |, &, \, or other sed-special characters.
awk -v input="$ARGS" '{gsub(/\{input\}/, input)}1' "$TEMPLATE"
```

The script is intentionally minimal — it loads and parameterizes a template. It does NOT run an agent.

**Tests:** Both files exist. `run-stage.sh` is executable, passes `bash -n` syntax check, outputs template content when given a valid stage name, and handles pathological inputs (paths containing `|`, `&`, `\`) without injection.

**Validation:**
```bash
test -f adapters/generic/README.md
test -f adapters/generic/run-stage.sh
test -x adapters/generic/run-stage.sh
bash -n adapters/generic/run-stage.sh
# Verify pathological input doesn't corrupt output
echo '{input}' > /tmp/test-template.md
TEMPLATE=/tmp/test-template.md awk -v input='docs/foo|bar&baz\qux' '{gsub(/\{input\}/, input)}1' /tmp/test-template.md | grep -q 'docs/foo|bar&baz\\qux'
```

---

### Task 8: Refactor AGENTS.md for cross-platform use

**Files:**
- `AGENTS.md` (modify)

**Approach:**

Currently `AGENTS.md` mixes portable stance definitions with Claude Code-specific tool restriction tables. Refactor to:

1. **Keep** the opening section explaining stances vs. roles (lines 1-6 — this is core methodology)
2. **Keep** Tool Commands and Code Style Guidelines sections (lines 10-72 — already language-agnostic placeholders)
3. **Keep** all 5 stance definitions (Proposer, Training Partner, Code Review Partner, Security Auditor, Skeptical Client) with their Character, Constraints, Key Behaviors, and Output Format subsections
4. **Modify** the stance Constraints to use platform-neutral language:
   - "Cannot write code — you have no Write or Edit tools" → "Must not modify files — analysis and review only. See your platform's adapter for enforcement details."
5. **Replace** the "Agent Definitions for Team Mode" section (lines 158-172) with a "Platform-Specific Enforcement" section that:
   - Explains the two levels of enforcement (prompt-level = portable, tool-level = platform-specific)
   - Points to `adapters/` for platform-specific setup
   - Notes which platforms support tool-level enforcement (Claude Code, OpenCode)
   - Keeps the tool restriction table but labels it as "Claude Code / OpenCode reference"
6. **Add** a "Cross-Platform Support" section at the top (after the opening) noting that stances are portable, enforcement varies by platform, and canonical definitions live in `prompts/stances/`
7. **Keep** the reference to `docs/guides/agent-teams.md` for team mode

The tone should acknowledge that `AGENTS.md` is also read directly by many platforms (Cursor, Windsurf, Cline, OpenCode) as project context — so it should be useful as a standalone document.

**Tests:** File contains all 5 stance definitions. File contains "Platform-Specific Enforcement" section. File references `prompts/stances/`. File does not contain "you have no Write or Edit tools" (replaced with platform-neutral language).

**Validation:**
```bash
# All stances present
for stance in "Proposer" "Senior Training Partner" "Code Review Partner" "Security Auditor" "Skeptical Client"; do
  grep -q "$stance" AGENTS.md && echo "✓ $stance" || echo "✗ $stance MISSING"
done

# Platform-neutral language
grep -q "Platform-Specific Enforcement" AGENTS.md
! grep -q "you have no Write or Edit tools" AGENTS.md
```

---

### Task 9: Update core documentation

**Files:**
- `push-hands.md` (modify — Section 4.1.3 area and Section 10/Architecture)
- `CLAUDE.md` (modify — add Multi-Platform section)
- `README.md` (modify — add Supported Platforms section, update Quick Start)

**Approach:**

**push-hands.md changes:**
1. After the existing Section 4.1.3 (Team Mode), add a new subsection **4.1.4 Multi-Platform Support** that describes:
   - The three-layer architecture (shared infra, portable templates, platform adapters)
   - Which platforms are supported and at what level
   - How portable templates relate to platform-specific commands
   - Where to find adapter setup instructions
2. In the executive summary (Section 1, around line 14), update "designed for solo developers and small teams building with Claude Code (or comparable agentic coding tools)" to explicitly list supported platforms.

**CLAUDE.md changes:**
1. After the "Team Mode" subsection (line 43), add a **Multi-Platform Support** subsection:
   - "Portable workflow templates live in `prompts/`. Platform adapters live in `adapters/`."
   - "Claude Code commands in `.claude/commands/` are the reference implementation."
   - "When methodology changes, update `prompts/workflows/` first, then sync platform-specific files."
2. Add to Conventions section:
   - `Portable stances: prompts/stances/<stance>.md`
   - `Portable workflows: prompts/workflows/<stage>.md`
   - `Platform adapters: adapters/<platform>/`

**README.md changes:**
1. Update the opening line to list supported platforms explicitly instead of "(or comparable agentic coding tools)"
2. After "Quick Start" section, add a **Supported Platforms** section:

```markdown
## Supported Platforms

| Platform | Slash Commands | Agent Stances | Tool Enforcement | Agent Teams |
|----------|---------------|---------------|-----------------|-------------|
| Claude Code | Native | Native (tool-level) | Native | Native |
| OpenCode | Via prompt templates | Native (permission-level) | Native | No |
| Cursor | Via prompt templates | Rules (.mdc) | No (behavioral) | No |
| Windsurf | Via prompt templates | Rules (.md) | No (behavioral) | No |
| Cline | Via prompt templates | Rules (.clinerules) | No (behavioral) | No |
| Aider | Via prompt templates | Context loading | No (behavioral) | No |
| Other CLI | Via run-stage.sh | Context loading | No (behavioral) | No |
```

3. Update "For an Existing Project" to mention adapter directories
4. Update "Customization" to mention platform adapters

**Tests:** All three files modified. README contains "Supported Platforms" table. CLAUDE.md contains "Multi-Platform Support" section. push-hands.md contains "Multi-Platform Support" subsection.

**Validation:**
```bash
grep -q "Supported Platforms" README.md
grep -q "Multi-Platform Support" CLAUDE.md
grep -q "Multi-Platform Support" push-hands.md
```

---

### Task 10: Update scripts/setup.sh and create drift-detection script

**Files:**
- `scripts/setup.sh` (modify)
- `scripts/check-prompt-sync.sh` (create)

**Approach:**

**Part A: `scripts/check-prompt-sync.sh`**

Create a drift-detection script that compares section headers between portable workflow templates (`prompts/workflows/*.md`) and Claude Code commands (`.claude/commands/*.md`). This addresses PRD Risk #1 ("Prompt template drift", severity HIGH) and PRD AC #3's intent.

The script:
1. For each portable workflow, finds the matching Claude Code command (mapping `update-project-docs` → `update-claude-md`)
2. Extracts `## ` headers from both files
3. Compares the methodology-relevant headers (Stance, Instructions, Artifact Template, Validation)
4. Reports any drift — missing sections, renamed sections, or structural divergence
5. Exits non-zero if drift is detected (suitable for CI)

```bash
#!/bin/bash
# Usage: ./scripts/check-prompt-sync.sh
# Compares section structure between portable templates and Claude Code commands.
# Exit 0 = in sync, Exit 1 = drift detected.

DRIFT=0
WORKFLOW_DIR="prompts/workflows"
COMMAND_DIR=".claude/commands"

# Mapping: portable workflow name → Claude Code command name
declare -A CMD_MAP=(
  ["new-feature"]="new-feature"
  ["generate-prp"]="generate-prp"
  ["review-plan"]="review-plan"
  ["execute-prp"]="execute-prp"
  ["review-code"]="review-code"
  ["security-audit"]="security-audit"
  ["update-project-docs"]="update-claude-md"
)

for workflow in "${!CMD_MAP[@]}"; do
  command="${CMD_MAP[$workflow]}"
  wf_file="${WORKFLOW_DIR}/${workflow}.md"
  cmd_file="${COMMAND_DIR}/${command}.md"

  if [ ! -f "$wf_file" ]; then
    echo "✗ Missing portable workflow: ${wf_file}"
    DRIFT=1
    continue
  fi
  if [ ! -f "$cmd_file" ]; then
    echo "✗ Missing Claude Code command: ${cmd_file}"
    DRIFT=1
    continue
  fi

  # Extract ## headers from portable template (methodology sections)
  wf_headers=$(grep '^## ' "$wf_file" | sort)
  # Check that key methodology sections exist in the command file
  for section in "Stance" "Validation"; do
    if echo "$wf_headers" | grep -q "$section"; then
      if ! grep -q "## .*${section}" "$cmd_file"; then
        echo "⚠ Drift: ${workflow} has '${section}' section missing from ${cmd_file}"
        DRIFT=1
      fi
    fi
  done
  echo "✓ ${workflow} ↔ ${command}"
done

if [ "$DRIFT" -eq 0 ]; then
  echo "All portable templates in sync with Claude Code commands."
else
  echo "Drift detected — review the warnings above."
  exit 1
fi
```

**Part B: `scripts/setup.sh` updates**

Add a platform detection section at the end of `setup.sh` (after existing checks). This section:

1. Checks which platform adapter directories exist and reports them:
   ```
   Checking platform adapters...
     ✓ Claude Code (.claude/commands/, .claude/agents/)
     ✓ OpenCode (.opencode/agents/)
     ✓ Cursor (.cursor/rules/)
     ✗ Windsurf (.windsurf/rules/) — not configured
     ✗ Cline (.clinerules/) — not configured
   ```

2. Checks for portable prompt templates:
   ```
   Checking portable templates...
     ✓ prompts/stances/ (5 stances)
     ✓ prompts/workflows/ (7 workflows)
   ```

3. Updates the "Next steps" section to mention platform adapters:
   ```
   4. See adapters/ for multi-platform setup
   ```

4. Also add `prompts/stances` and `prompts/workflows` to the `mkdir -p` directory creation section.

Do NOT create directories for platforms that don't have adapters yet (Windsurf, Cline). Only create directories for platforms included in this feature (Claude Code, OpenCode, Cursor, generic).

**Tests:** Scripts pass `bash -n` syntax check. Running setup creates the expected directories and reports platform status. Sync-check script detects drift when section headers diverge.

**Validation:**
```bash
bash -n scripts/setup.sh
bash -n scripts/check-prompt-sync.sh
test -x scripts/check-prompt-sync.sh
```

---

## 5. Validation Gates

Since this feature is entirely documentation and prompt engineering (plus one shell script), validation is structural:

```bash
# 1. All portable stances exist (5)
for stance in proposer training-partner code-reviewer security-auditor skeptical-client; do
  test -f "prompts/stances/${stance}.md" || exit 1
done

# 2. All portable workflows exist (7)
for wf in new-feature generate-prp review-plan execute-prp review-code security-audit update-project-docs; do
  test -f "prompts/workflows/${wf}.md" || exit 1
done

# 3. Prompts README exists
test -f prompts/README.md || exit 1

# 4. All adapter READMEs exist
for adapter in claude-code opencode cursor generic; do
  test -f "adapters/${adapter}/README.md" || exit 1
done

# 5. OpenCode agent definitions exist (5)
for agent in proposer training-partner code-reviewer security-auditor skeptical-client; do
  test -f ".opencode/agents/${agent}.md" || exit 1
done

# 6. Cursor rules exist (6)
test -f ".cursor/rules/00-push-hands-context.mdc" || exit 1
for i in 01-proposer 02-training-partner 03-code-reviewer 04-security-auditor 05-skeptical-client; do
  test -f ".cursor/rules/${i}.mdc" || exit 1
done

# 7. Generic adapter shell script is valid
bash -n adapters/generic/run-stage.sh || exit 1

# 8. No Claude Code team primitives leaked into portable layer
! grep -rEl '(TeamCreate|SendMessage|TaskCreate|TaskUpdate|TaskList|TaskDelete)' prompts/

# 9. No $ARGUMENTS in portable workflows
! grep -rl '\$ARGUMENTS' prompts/workflows/

# 10. Existing Claude Code commands unchanged
for cmd in new-feature generate-prp review-plan execute-prp review-code security-audit update-claude-md review-plan-team execute-team security-audit-team; do
  test -f ".claude/commands/${cmd}.md" || exit 1
done

# 11. AGENTS.md still has all stances
for stance in "Proposer" "Senior Training Partner" "Code Review Partner" "Security Auditor" "Skeptical Client"; do
  grep -q "$stance" AGENTS.md || exit 1
done

# 12. Updated docs have new sections
grep -q "Supported Platforms" README.md || exit 1
grep -q "Multi-Platform Support" CLAUDE.md || exit 1

# 13. Drift-detection script exists and is valid
test -f scripts/check-prompt-sync.sh || exit 1
bash -n scripts/check-prompt-sync.sh || exit 1

echo "All validation gates passed."
```

## 6. Rollback Plan

This feature is purely additive — it creates new files and modifies existing documentation. No existing Claude Code functionality is changed.

**To rollback:**
```bash
git revert <commit-range>
# Or more specifically:
rm -rf prompts/ adapters/ .opencode/ .cursor/
git checkout main -- AGENTS.md CLAUDE.md README.md push-hands.md scripts/setup.sh
```

Since each task is one atomic commit, partial rollback is straightforward — revert specific commits while keeping others.

## 7. Uncertainty Log

1. **OpenCode permission granularity** — I mapped Claude Code tool restrictions to OpenCode's `permission:` field based on documentation research. The exact behavior of glob patterns in the `bash:` permission (e.g., `"git *": allow`) needs testing against a real OpenCode instance. I chose conservative defaults. **Confidence: 7/10.**

2. **Cursor .mdc format stability** — Cursor has changed its rules format multiple times. The `.mdc` format with `description`/`globs`/`alwaysApply` frontmatter is current as of February 2026 research, but may evolve. The adapter is thin enough that updating is cheap. **Confidence: 8/10.**

3. **Claude Code command self-containment** — I assumed Claude Code commands cannot import external files and must be self-contained. This is based on observed behavior, not explicit documentation. If Claude Code adds an include mechanism in the future, the sync convention could be replaced with actual imports. **Confidence: 9/10.**

4. **Cline and Windsurf as "generic" rather than first-class adapters** — The PRD listed four adapters (claude-code, opencode, cursor, generic). I folded Windsurf and Cline into the generic adapter because their rules systems are simple enough that a few paragraphs of guidance suffice. If these platforms add agent definition support, they'd warrant their own adapter directories. **Confidence: 8/10.**

5. **AGENTS.md refactoring scope** — The current AGENTS.md also contains Tool Commands and Code Style Guidelines sections that are project-specific (Python defaults). These are already language-agnostic placeholders. I'm leaving them as-is since they're orthogonal to the platform parity work. **Confidence: 9/10.**

6. **Portable workflow template fidelity** — Extracting methodology from interleaved command files requires judgment about what's methodology vs. wiring. Some instructions are borderline (e.g., "commit to the feature branch" — is `git commit` a tool-specific operation or a generic instruction?). I erred on the side of keeping generic git operations in the portable layer since every platform supports shell/terminal access. **Confidence: 8/10.**

7. **PRD AC #3 deviation: commands not refactored to delegate** — The PRD specifies Claude Code commands should "delegate to `prompts/workflows/`" but commands cannot import external files. We keep commands self-contained and use `scripts/check-prompt-sync.sh` for drift detection instead. This is a deliberate trade-off (reliability over DRY), not an oversight. If Claude Code gains an include mechanism, this should be revisited. **Confidence: 9/10.**

8. **Additional PRD architecture divergences (intentional improvements based on research):**
   - **OpenCode path:** PRD shows `.agents/opencode/` — PRP uses `.opencode/agents/` which is the actual OpenCode convention per documentation research. PRD shows a single `agents.md` file — PRP creates 5 separate agent files matching OpenCode's per-agent file convention.
   - **Prompts directory structure:** PRD shows a flat `prompts/` layout — PRP organizes into `prompts/stances/` and `prompts/workflows/` subdirectories for clarity.
   - **Workflow count:** PRD AC #1 says "six sequential workflow stages" — PRP creates 7, adding `update-project-docs` per PRD AC #12's requirement to generalize `update-claude-md`.

   All three are improvements over the PRD's initial sketch based on research findings. **Confidence: 9/10.**
