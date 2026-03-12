# Workflow: Security Audit

> Portable workflow template. Load this as context for your AI coding agent,
> or use a platform adapter (see `adapters/`).
>
> This workflow uses two stances in sequence. In the portable version, a single
> agent adopts each stance in turn. Platform adapters may use separate agents
> for concurrent execution.

## Stance

Two opposing stances applied sequentially:
1. **Security Auditor** (hard stance) — thorough, exploit-minded, severity-calibrated
2. **Skeptical Client** (hard stance) — budget-conscious, dubious, demands proof

## Input

Review all changes on the current feature branch against `main`. No additional input required.

## Instructions

### Phase 1: Security Auditor (Hard Stance)

#### 1. Scope the audit

- Run `git diff main...HEAD` to identify all changes
- Identify the attack surface: user inputs, API endpoints, auth boundaries, data flows, configuration
- Note what is NOT in scope

#### 2. Hunt for vulnerabilities across these categories

- Authentication and authorization (bypass, privilege escalation, IDOR)
- Input validation (injection — SQL, command, XSS, template, path traversal)
- Data exposure (PII leaks, verbose errors, debug endpoints, insecure logging)
- Configuration (hardcoded secrets, insecure defaults, missing security headers)
- Business logic (race conditions, state manipulation, abuse scenarios)
- Dependency risks (known CVEs, outdated packages, supply chain)
- Cryptography (weak algorithms, improper key management, predictable tokens)

#### 3. For each finding, document

- Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
- Category
- Affected code (file:line)
- Description of the vulnerability
- Proof of concept or clear exploitation path (REQUIRED for HIGH+)
- Remediation steps
- Effort estimate: XS / S / M / L / XL

### Phase 2: Skeptical Client (Hard Stance)

#### 4. Review your own findings with the Skeptical Client lens

- For each HIGH+ finding: Is the PoC convincing? Is the severity justified?
- For each finding: Is this a real risk given the deployment context, or theoretical?
- Are there false positives? Findings that look scary but aren't exploitable?
- Are effort estimates realistic?

#### 5. Revise findings

Based on the Client's challenge, downgrade, remove, or strengthen findings as warranted.

### Output

#### 6. Produce the audit report

Write the report at `docs/audits/<slug>-audit.md` using the template at `docs/audits/TEMPLATE.md`.

#### 7. Commit the report

Commit with message: `docs: security audit for <slug>`

#### 8. Report summary

Provide a summary with:
- Finding count by severity
- Any CRITICAL/HIGH findings that need immediate attention
- Recommended remediation priority

## Artifact Template

```markdown
# Security Audit: [Feature/Component]

## Scope
What was examined. What was NOT examined.

## Methodology
Tools and techniques used. Stance interaction pattern.

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

## Validation

```bash
# Audit report exists
test -f "docs/audits/<slug>-audit.md"

# Report has findings summary
grep -q "Findings Summary" "docs/audits/<slug>-audit.md"
```
