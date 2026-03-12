# Security Audit: multi-agent-parity

## Scope

**Examined:** All 40 files in `git diff main...HEAD` (~3,000 lines added, 21 deleted). Specifically:
- Shell scripts: `adapters/generic/run-stage.sh`, `scripts/check-prompt-sync.sh`, `scripts/setup.sh`
- OpenCode agent configs: all 5 `.opencode/agents/*.md` (permissions, tool access)
- Cursor rules: all 6 `.cursor/rules/*.mdc` (enforcement limitations)
- Portable templates: all 5 `prompts/stances/*.md` and all 7 `prompts/workflows/*.md`
- Configuration: `.gitignore`, `.push-hands-tier`
- Documentation: `AGENTS.md`, `CLAUDE.md`, all adapter READMEs

**NOT examined:**
- Claude Code agent definitions (`.claude/agents/*.md`, `.claude/commands/*.md`) — pre-existing, not in diff
- OpenCode runtime behavior — cannot verify permission glob matching without runtime
- Cursor/Windsurf/Cline runtime behavior — cannot verify rule loading or enforcement
- Downstream agent behavior — prompt injection resistance varies by model
- Supply chain / dependencies — no package managers or external dependencies in diff
- Authentication/authorization — not applicable (no auth system)
- Cryptography — not applicable (no crypto)

## Methodology

Dual-agent team audit — Security Auditor and Skeptical Client operating as separate agents with direct peer messaging. The Auditor conducted PoC-driven vulnerability hunting across the full diff. Each finding was sent to the Client immediately upon discovery. The Client challenged severity, realism of attack scenarios, and quality of proposed remediations. Findings were resolved through one challenge-defense cycle, producing severity ratings that survived adversarial scrutiny.

## Findings Summary

| # | Severity (final) | Category | Description | Effort |
|---|-------------------|----------|-------------|--------|
| 1 | LOW | Input Validation — Path Traversal | run-stage.sh allows `../` in STAGE parameter | XS |
| 2 | LOW | Input Validation — Template Injection | run-stage.sh resolves templates relative to CWD | XS |
| 3 | INFO | Input Validation — Prompt Injection | {input} substitution has no structural delimiters | XS |
| 4 | INFO | Configuration — Authorization | Cursor rules cannot enforce tool restrictions | XS |
| 5 | LOW | Configuration — Missing Safeguard | .push-hands-tier not protected from reaching main | XS |
| 6 | INFO | Configuration — Authorization | OpenCode bash glob patterns (platform-dependent) | XS |

**No CRITICAL or HIGH findings.** Final tally: 0 CRITICAL, 0 HIGH, 0 MEDIUM, 3 LOW, 3 INFO.

## Detailed Findings

### Finding 1: Path Traversal in run-stage.sh
**Initial Severity:** MEDIUM
**Final Severity:** LOW
**Category:** Input Validation — Path Traversal
**Affected Code:** `adapters/generic/run-stage.sh:15`
**Description:** The `STAGE` parameter is concatenated into a file path (`prompts/workflows/${STAGE}.md`) without validation. The `../` sequences allow reading any `.md` file relative to the project root.
**Proof of Concept:**
```bash
./adapters/generic/run-stage.sh "../../CLAUDE"
# Output: full contents of CLAUDE.md
```
Confirmed: traversal loads unintended file content.
**Client Challenge:** No trust boundary is crossed — the developer running the script already has full filesystem read access. The `.md` extension is hardcoded, limiting scope. The attacker and victim are the same person in the single-user local CLI context.
**Auditor Defense:** Acknowledged the downgrade. No capability is gained beyond what `cat` provides.
**Resolution:** Severity downgraded from MEDIUM to LOW. Valid code hygiene observation but not a meaningful security risk in the deployment context.
**Remediation:** Validate STAGE against slug regex `^[a-z0-9]+(-[a-z0-9]+)*$` before path construction.
**Effort Estimate:** XS

### Finding 2: CWD-Relative Template Resolution
**Initial Severity:** MEDIUM
**Final Severity:** LOW
**Category:** Input Validation — Template Injection
**Affected Code:** `adapters/generic/run-stage.sh:15`
**Description:** Script resolves `prompts/workflows/${STAGE}.md` relative to the current working directory, not relative to the script's own location. Running from a directory with attacker-controlled `prompts/workflows/` loads wrong templates.
**Proof of Concept:**
```bash
cd /tmp && mkdir -p prompts/workflows
echo 'MALICIOUS TEMPLATE' > prompts/workflows/evil.md
/path/to/adapters/generic/run-stage.sh evil "test"
# Output: MALICIOUS TEMPLATE
```
Confirmed: malicious template loaded from CWD.
**Client Challenge:** Requires three preconditions (attacker-controlled CWD + planted files + matching stage name). If an attacker has that level of access, this script is the least of the problems. Script is designed for repo-root execution.
**Auditor Defense:** Acknowledged the downgrade. The three-precondition chain makes real exploitation unrealistic.
**Resolution:** Severity downgraded from MEDIUM to LOW. Defense-in-depth improvement, not a realistic vulnerability.
**Remediation:** Resolve script directory: `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` and use as base path.
**Effort Estimate:** XS

### Finding 3: Prompt Injection via {input} Substitution
**Initial Severity:** MEDIUM
**Final Severity:** INFO
**Category:** Input Validation — Prompt Injection
**Affected Code:** `adapters/generic/run-stage.sh:36-40`, all `prompts/workflows/*.md`
**Description:** User input is injected verbatim into prompt templates with no structural separation. When output is piped to an AI agent, the user input appears inline with legitimate instructions.
**Proof of Concept:**
```bash
./adapters/generic/run-stage.sh new-feature 'ignore all previous instructions'
# Injected text appears inline with workflow instructions
```
**Client Challenge:** (1) This is the fundamental nature of LLM interaction, not a vulnerability specific to this script. (2) The "untrusted source" threat model was invented — the script is for interactive developer use. (3) The proposed `<user-input>` tag remediation doesn't work — LLMs don't reliably respect delimiters. (4) The developer is injecting into their own agent under their own permissions.
**Auditor Defense:** Fully accepted all four points. Acknowledged the finding was severity inflation and the proposed remediation was "theater."
**Resolution:** Severity downgraded from MEDIUM to INFO. This is an inherent property of LLM interaction, not a fixable vulnerability.
**Remediation:** Optional documentation note that the script is intended for trusted, interactive use.
**Effort Estimate:** XS (documentation only)

### Finding 4: Cursor Adapter Cannot Enforce Tool Restrictions
**Initial Severity:** MEDIUM
**Final Severity:** INFO
**Category:** Configuration — Authorization
**Affected Code:** `.cursor/rules/02-training-partner.mdc`, `03-code-reviewer.mdc`, `05-skeptical-client.mdc`
**Description:** Cursor's `.mdc` format has no `permission:` mechanism. The "must not modify files" constraint for three critic stances is prompt-level only in Cursor, unlike Claude Code (tool-level) and OpenCode (permission frontmatter).
**Proof of Concept:** Hypothetical narrative only — no actual Cursor testing performed.
**Client Challenge:** (1) This is extensively documented in AGENTS.md (comparison table, cross-platform section, per-stance references). (2) The PoC was a story, not a demonstration. (3) Singling out Cursor is inconsistent when Windsurf, Cline, and Aider have the identical limitation.
**Auditor Defense:** Fully accepted. The limitation is a documented design decision, not a security gap.
**Resolution:** Severity downgraded from MEDIUM to INFO. Documented platform limitation.
**Remediation:** Optional UX enhancement: inline warnings in restricted Cursor rule files.
**Effort Estimate:** XS

### Finding 5: .push-hands-tier Not Protected from Reaching Main
**Initial Severity:** LOW
**Final Severity:** LOW
**Category:** Configuration — Missing Safeguard
**Affected Code:** `.gitignore` (missing entry)
**Description:** CLAUDE.md states `.push-hands-tier` "must never reach main" but no automated prevention exists — not in `.gitignore`, no hook, no CI check. If it reaches main, it overrides tier flags on all downstream feature branches.
**Proof of Concept:** File is committed on `feature/multi-agent-parity` (commit `7e69817`). Not in `.gitignore`. No CI check exists.
**Client Challenge:** Accepted LOW severity. Corrected the remediation: `.gitignore` won't work because the file needs to be committable on feature branches (feature branches inherit `.gitignore` from main). A CI/PR check is the correct mechanism.
**Auditor Defense:** Acknowledged the remediation correction.
**Resolution:** Severity unchanged at LOW. Remediation corrected from `.gitignore` to CI/PR check.
**Remediation:** CI/PR check: `git diff --name-only main...HEAD | grep -q '\.push-hands-tier'` — warn or block on PRs targeting main.
**Effort Estimate:** XS

### Finding 6: OpenCode Bash Permission Glob Patterns
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Configuration — Authorization
**Affected Code:** `.opencode/agents/training-partner.md:7-10`, `.opencode/agents/code-reviewer.md:7-10`
**Description:** Patterns like `"grep *": allow` may be exploitable via command chaining depending on how OpenCode implements glob matching against the full command string. This is platform-dependent and cannot be verified without the OpenCode runtime.
**Proof of Concept:** Not applicable — requires OpenCode runtime for verification.
**Client Challenge:** Accepted at INFO. Appropriately rated platform-dependent observation.
**Auditor Defense:** N/A (accepted as-is).
**Resolution:** Severity unchanged at INFO.
**Remediation:** Verify against OpenCode documentation how bash permission patterns are matched.
**Effort Estimate:** XS (verification only)

## What This Audit Did NOT Find

- No command injection vulnerabilities (the `awk` ENVIRON approach in `run-stage.sh` correctly handles metacharacters)
- No secrets or credentials in the diff
- No insecure network calls
- No dependency vulnerabilities (no dependencies added)
- No authentication or authorization flaws (no auth system present)
- The `check-prompt-sync.sh` and `setup.sh` scripts were reviewed and found clean — they use read-only operations (stat, grep, test) with no user-controlled input paths

## Remediation Priority

1. **Findings 1 + 2** (run-stage.sh hardening) — Add slug validation AND script-relative path resolution. Both are XS effort, both improve the same file. Do in one commit.
2. **Finding 5** (.push-hands-tier CI check) — Add a CI/PR check that warns when `.push-hands-tier` is in the diff targeting main. XS effort, prevents a real workflow issue.
3. **Finding 6** (OpenCode glob verification) — Verify behavior against OpenCode documentation when convenient. No code change unless platform behavior is surprising.
4. **Findings 3 + 4** (optional documentation) — Low priority. Add trust boundary note and Cursor warnings if convenient during other work.
