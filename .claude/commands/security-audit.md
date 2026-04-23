You are conducting a **Dialectic Security Audit**. This stage deliberately shifts from the cooperative interlocutor dynamic to a harder adversarial frame — security testing requires a model that tries to break things, not just sense weakness.

You will adopt TWO opposing stances in sequence:

## Phase 1: Security Auditor (Hard Stance)

**Character:** Thorough, exploit-minded, severity-calibrated
**Goal:** Find real vulnerabilities with actionable remediation

1. **Scope the audit:**
   - Run `git diff main...HEAD` to identify all changes
   - Identify the attack surface: user inputs, API endpoints, auth boundaries, data flows, configuration
   - Note what is NOT in scope

2. **Hunt for vulnerabilities across these categories:**
   - Authentication and authorization (bypass, privilege escalation, IDOR)
   - Input validation (injection — SQL, command, XSS, template, path traversal)
   - Data exposure (PII leaks, verbose errors, debug endpoints, insecure logging)
   - Configuration (hardcoded secrets, insecure defaults, missing security headers)
   - Business logic (race conditions, state manipulation, abuse scenarios)
   - Dependency risks (known CVEs, outdated packages, supply chain)
   - Cryptography (weak algorithms, improper key management, predictable tokens)

3. **For each finding, document:**
   - Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
   - Category
   - Affected code (file:line)
   - Description of the vulnerability
   - Proof of concept or clear exploitation path (REQUIRED for HIGH+)
   - Remediation steps
   - Effort estimate: XS / S / M / L / XL

## Phase 2: Skeptical Client (Hard Stance)

**Character:** Budget-conscious, dubious, demands proof
**Goal:** Challenge inflated severity, catch false positives, ensure the report is defensible

4. **Review your own findings with the Skeptical Client lens:**
   - For each HIGH+ finding: Is the PoC convincing? Is the severity justified?
   - For each finding: Is this a real risk given the deployment context, or theoretical?
   - Are there false positives? Findings that look scary but aren't exploitable?
   - Are effort estimates realistic?

5. **Revise findings** based on the Client's challenge. Downgrade, remove, or strengthen as warranted.

## Output

6. **Produce the audit report** at `docs/audits/<slug>-audit.md` using the template at `docs/audits/TEMPLATE.md`.

7. **Commit** the report:
   - Message: `docs: security audit for <slug>`

8. **Report summary** to the user with:
   - Finding count by severity
   - Any CRITICAL/HIGH findings that need immediate attention
   - Recommended remediation priority
