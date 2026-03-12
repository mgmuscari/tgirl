# Security Audit: agent-teams

## Scope

**Examined:** All 19 files changed between `main` and `feature/agent-teams`:
- 5 agent definitions: `.claude/agents/{proposer,training-partner,code-reviewer,security-auditor,skeptical-client}.md`
- 3 team slash commands: `.claude/commands/{review-plan-team,execute-team,security-audit-team}.md`
- 1 shell script modification: `scripts/setup.sh`
- 10 documentation/config files: `CLAUDE.md`, `AGENTS.md`, `push-hands.md`, `README.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `docs/guides/agent-teams.md`, `docs/PRDs/agent-teams.md`, `docs/PRPs/agent-teams.md`, `docs/reviews/code/agent-teams-review.md`, `docs/reviews/plans/agent-teams-review.md`
- Related unchanged files for context: `.github/workflows/push-hands-review.yml`, `.github/workflows/ci.yml`, `scripts/hooks/commit-msg`, `scripts/hooks/pre-commit`, existing sequential commands

**Categories audited:** Authentication/authorization, input validation (injection, path traversal), data exposure, configuration (secrets, insecure defaults), business logic (race conditions, abuse), dependencies (CVEs), cryptography, prompt injection

**NOT examined:**
1. Runtime behavior of Claude Code's agent teams engine (audited configuration only, not the execution platform)
2. Claude Code's actual auto-delegation algorithm internals (treated as black-box)
3. Token cost amplification / DoS scenarios
4. Interaction between git hooks and team-mode agent commits at runtime
5. Network exfiltration potential from agents with Bash access (inherent to all Claude Code sessions, not specific to this feature)
6. Content of existing sequential commands (verified unchanged via `git diff`)

## Methodology

Dual-agent team audit — Security Auditor and Skeptical Client operating as separate agents with direct peer messaging. Findings were challenged in real time, producing severity ratings that survived adversarial scrutiny. Each finding followed a fixed protocol: auditor reports, client challenges, auditor defends or acknowledges, finding is closed with final severity determined.

## Findings Summary

| # | Severity (final) | Category | Description | Effort |
|---|-------------------|----------|-------------|--------|
| 1 | LOW | Configuration / Business Logic | Bash tool undermines documented "tool-level read-only enforcement" for critic agents | XS |
| 2 | LOW | Configuration / Business Logic | Auto-delegation prevention relies on LLM prompt compliance, not technical constraint | XS |

## Detailed Findings

### Finding 1: Bash Tool Undermines "Tool-Level Read-Only Enforcement" for Critic Agents

**Initial Severity:** MEDIUM
**Final Severity:** LOW
**Category:** Configuration / Business Logic
**Affected Code:** `.claude/agents/training-partner.md:4`, `.claude/agents/code-reviewer.md:4`, `.claude/agents/security-auditor.md:4`

**Description:** Three of four critic agents (training-partner, code-reviewer, security-auditor) include `Bash` in their tool allowlist while being documented as unable to modify files. Documentation in CLAUDE.md:64, AGENTS.md:160, and docs/guides/agent-teams.md:89 claims "tool-level enforcement" prevents file modification, but Bash provides equivalent file modification capability via shell commands (`echo >`, `sed -i`, `rm`, `git commit`, etc.). The only properly constrained critic is skeptical-client (`tools: Read, Grep, Glob`).

**Proof of Concept:** Code-reviewer could execute `echo "content" >> src/main.py && git add -A && git commit -m "fix: something"` via Bash, potentially breaking the message-gated protocol in `/execute-team` that assumes only the proposer modifies the working tree. Training-partner doesn't need Bash at all for its documented operations (file path checking, symbol verification).

**Client Challenge:** The PoC requires an AI agent to go rogue — not a realistic external attack vector. Claude Code has user permission gating on Bash commands, so this isn't a silent bypass. The tradeoff was already a known, documented design decision flagged in the code review artifact.

**Auditor Defense:** Accepted the downgrade to LOW. Conceded that permission gating is a mitigating control and the rogue-agent scenario is low-probability. Maintained that the documentation inaccuracy matters because "tool-level enforcement" specifically contrasts with "prompt-level enforcement" — users are told this is architectural, which it isn't for 3 of 4 critic agents. The claim appears in 3 separate locations creating consistent but misleading assurance.

**Resolution:** Severity downgraded from MEDIUM to LOW. The documentation inaccuracy is real but the practical exploitation risk is minimal given permission gating and the rogue-agent attack model.

**Remediation:**
1. Remove `Bash` from training-partner tools: `tools: Read, Grep, Glob` (doesn't need it)
2. For code-reviewer and security-auditor: keep Bash but update documentation in CLAUDE.md, AGENTS.md, and user guide to accurately describe the constraint
3. Update AGENTS.md table "Key Constraint" column

**Effort Estimate:** XS

### Finding 2: Auto-Delegation Prevention Relies on LLM Prompt Compliance

**Initial Severity:** LOW
**Final Severity:** LOW
**Category:** Configuration / Business Logic
**Affected Code:** All five `.claude/agents/*.md` `description:` fields

**Description:** All five agent definitions use "Push Hands team-mode agent. Do not invoke directly" descriptions to prevent Claude Code's auto-delegation from routing tasks to team-mode agents during sequential command execution. This is an LLM-interpreted hint, not a hard technical constraint. If Claude Code's routing heuristics change, agents could be invoked during sequential commands, potentially routing users to agents with different tool sets than expected.

**Proof of Concept:** Theoretical — no concrete exploit. Based on the probabilistic nature of LLM-based routing. The PRP already identified this risk and the current descriptions are reasonable mitigations. Failure mode would be immediately obvious (agent would fail due to missing tools — noisy, not silent).

**Client Challenge:** LOW is appropriate. Nothing actionable to fix beyond what's already done. Suggested adding a debugging note to the user guide.

**Auditor Defense:** No defense needed — client accepted at initial severity.

**Resolution:** Severity maintained at LOW. Acceptable risk with current mitigation.

**Remediation:**
1. Acceptable as-is for current risk level
2. Adopt `auto_delegate: false` frontmatter if Claude Code adds it in a future version
3. Add a debugging note to the user guide about checking `.claude/agents/` descriptions if unexpected agent behavior occurs during sequential commands

**Effort Estimate:** XS

## What This Audit Did NOT Find

- No injection vulnerabilities (shell script changes use hardcoded values with no user input)
- No secrets or credentials in the diff
- No dependency vulnerabilities (no new dependencies added)
- No cryptographic issues (no cryptographic operations)
- No authentication/authorization bypasses (no auth boundaries in scope)
- No data exposure risks (no data processing or storage)
- Runtime behavior of the Claude Code agent teams engine was not audited — only configuration files were examined
- Token cost amplification and DoS scenarios were not evaluated
- Interaction between git hooks and team-mode agent commits at runtime was not tested

## Remediation Priority

1. **Finding 1 — Remove Bash from training-partner** (XS effort, immediate win, zero risk)
2. **Finding 1 — Update documentation accuracy** (XS effort, fixes misleading security claims in 3 locations)
3. **Finding 2 — Add user guide debugging note** (XS effort, defense-in-depth, opportunistic)
