# Code Review: plugin-architecture security-fix interstitial

## Verdict: APPROVED for merge with tracked follow-ups
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: 2026-04-26
## Mode: Agent Team (message-gated incremental review)

## Context

This review covers the security-fix interstitial commits between PRP Tasks 7 and 8 — six commits that remediate the CRITICAL findings from `docs/audits/plugin-architecture-audit.md` (3 CRITICAL + 6 MEDIUM in-scope). The audit verdict was REJECT-for-merge; this interstitial fixes #1, #2, #4, #5, #6, #7, #8, #12 and defers #3, #9, #10, #11 per the audit's own scoping recommendation.

## Commit ledger

| # | SHA | Subject | Verdict | Audit findings closed |
|---|---|---|---|---|
| 1 | `7004513` | Sandbox B `__builtins__` substitution on plugin module exec | LGTM (deviations addressed in commit 4) | #4 CRITICAL |
| 2 | `9bb808b` | Ban tgirl-internal modules + consolidate Gate 1 classifier | initially BLOCKING; remediated in commit 4 | #2 partial |
| 3 | `83d45a7` | Tighten AST scan against chr-built reflection chains | LGTM with 2 Significant items documented as residual | #5 (AST), #8 |
| 4 | `baeb6c6` | Close `from`-form import bypass + Sandbox B test contract + Sig items | LGTM (closes BLOCKING) | #2 complete |
| 5 | `fc042f1` | Tool-name validator + Lark string-terminal escape + TOML module path | LGTM | #6, #7 |
| 6 | `4fa9c35` | Phase B grant propagation + #12 doubling-rule test battery | LGTM with 1 Significant test-isolation issue | #1, #12 |
| 7 | `683e236` | Reviewer-followup hardening across commits 3/5/6 (post-review) | Closes 2 of 3 tracked follow-ups | tightens #3, #5 residual |

## PRP / Audit Compliance (per finding)

| # | Severity | Closed by | Status | PoC verified |
|---|---|---|---|---|
| #1 Phase B unimplemented | MEDIUM | `4fa9c35` | Closed | Yes — granted plugin returns "ok" |
| #2 cap-dict mutation | CRITICAL | `9bb808b` + `baeb6c6` | Closed | Yes — all 4 import shapes + wildcard rejected |
| #3 sys.modules concurrency | MEDIUM | not addressed | Surface widened by #1 keeping sys.modules pattern; deferred | Test-ordering regression below |
| #4 raw `open()` bypass | CRITICAL | `7004513` | Closed | Yes — NameError; marker not written |
| #5 getattr+__globals__ chain | CRITICAL | `83d45a7` AST + `7004513` runtime sink | Primary closed; chr-Name-binding workaround documented as acceptable residual | Yes — chr-built reflection rejected |
| #6 Lark grammar injection | MEDIUM | `fc042f1` | Closed | Yes — ValueError at registration |
| #7 TOML path traversal | MEDIUM | `fc042f1` | Closed | Yes — InvalidPluginConfigError at parse |
| #8 missing breakpoint/input | MEDIUM | `83d45a7` | Closed | Yes — forbidden_name at Gate 1 |
| #9 private API access | LOW | not in scope | Deferred per audit §Remediation Priority | n/a |
| #10 env_proxy `__contains__` | LOW | not in scope | Deferred per audit §Remediation Priority | n/a |
| #11 fs_write_proxy.rename annotation | INFO | not in scope | Deferred per audit §Remediation Priority | n/a |
| #12 test cluster lacks invocation | MEDIUM | `4fa9c35` | Closed (6-way parametrized doubling-rule battery: NETWORK/CLOCK/RANDOM/ENV/FS_READ/SUBPROCESS) | n/a |

**Summary:** 3/3 CRITICAL closed, 5/6 in-scope MEDIUM closed (#3 deferred with surface widened; documented). 7/7 audit PoCs verifiably fail to exploit at HEAD.

## Issues Found

### 1. Phase B kept sys.modules mutation rather than clean-rewrite
**Category:** Test isolation / future-proofing
**Severity:** Significant (deferred)
**Location:** `src/tgirl/plugins/guard.py` (`guard_scope` semantics); `tests/test_plugins_capabilities.py` and `tests/test_plugins_security.py` (test ordering)
**Details:** Audit Finding #1's recommended remediation included an option to rewrite `guard_scope` to use a thread-local meta_path lookup, eliminating the global `sys.modules` mutation that surfaced as audit Finding #3. Commit `4fa9c35` kept the existing pattern. Result: six tests in `test_plugins_capabilities.py` fail when ordered after `test_plugins_security.py`. Default pytest order hides the regression; `pytest-randomly` or any reordering would expose it.
**Resolution:** Open. Production is not exposed (proxy modules are only accessed inside `guard_scope` by plugin code). The audit had already deferred #3 to MEDIUM with a v1.1 hot-reload structural-blocker tag; this is the same surface re-emerging. **Filed as a tracked follow-up; not a security regression.** Three fix options were left for the proposer ranging from rewrite-per-audit to test-fixture cleanup.

### 2. chr-binding workaround for AST Subscript handler (commit 3)
**Category:** Sandbox / defense-in-depth
**Severity:** Significant (documented as acceptable residual)
**Location:** `src/tgirl/plugins/ast_scan.py`
**Details:** The Subscript handler walks `BinOp` chains for chr-constructed dunder strings, but accepts the documented workaround `key = chr(95)+chr(95)+...; b[key]` — Name binding short-circuits the BinOp check.
**Resolution:** Documented as residual at commit time. Runtime sink (Gate 2 wrap + Gate 3 deny via `__builtins__` substitution) closes the chain regardless — this is documented defense-in-depth, not a live exploit. Follow-up recommendation: add `chr`/`ord` to FORBIDDEN_NAMES, which closes the entire string-construction class (also handles `str.join`, f-string variants). Not blocking.

### 3. Minor / nit items
**Category:** Code quality
**Severity:** Minor / Nit
**Location:** Various — `_grant_scoped.__name__` reassignment (use `functools.wraps`); duplicate `pathlib` import; `_ = CapabilityGrant` dead code; `test_tool_decorator_rejects_bad_function_name` is a zero-assertion non-test.
**Resolution:** Open. Bundle into the next cleanup commit or address inline during PRP Tasks 8–14 work.

## What's Done Well

- **All three CRITICAL findings empirically closed** with the original audit PoCs verified to fail at HEAD.
- **Test cluster doubling rule (#12) implemented as parametrized battery** across all six capabilities — the structural test gap that hid Finding #1 is closed at the pattern level, not just for the specific case.
- **Single source of truth for import classification** (commit 2's Gate 1 consolidation) eliminates the divergence between `_check_one_import` and `is_allowed_for_grant` that enabled the original cap-dict mutation bypass.
- **`from X import Y` and wildcard `from X import *` bypasses both closed** (commit 4) — the proposer was held by the reviewer's blocking flag and resolved cleanly rather than papering over.
- **Sandbox B safe-builtins introspection encoded as a test contract** — future agents trying to "fix" by adding `__import__` to the safe list get a failing test that explains the deviation.
- **Audit PoCs preserved as RED tests** — every audit finding has a corresponding test that exploits the bug on the parent commit and fails to exploit on the fix commit. The audit artifact remains the spec.
- **mypy 0 → 0 maintained** across all six commits. Test count grew from 1611 to 1702 (+91 tests, including the doubling-rule battery).

## Summary

The security-fix interstitial structurally closes all seven audit-PoC attack chains and the dev-readiness gap from Finding #12. The two open Significants are documented as deferred (#1's sys.modules choice surfacing as #3's test-ordering regression) or acceptable residual (chr-binding bypass collapsed by the runtime sink). Neither blocks merge.

**Verdict: APPROVED for merge.** Of the three tracked follow-ups, two were closed by the proposer in commit `683e236` after this review was filed:

- (a) ~~Phase B clean-rewrite~~ — partially addressed: `guard_scope` now also walks parent-package `__dict__` and removes cached `CapabilityScopedModule` attributes on exit, hardening the cleanup boundary that surfaced as the test-ordering regression in Issue #1 above. The structural rewrite eliminating sys.modules mutation is still deferred (Task #3 in the running task list, originally audit Finding #3).
- (b) ~~`chr`/`ord` to FORBIDDEN_NAMES~~ **CLOSED** — added in `683e236`, collapsing the entire string-construction class (Name-binding workaround, `str.join` Call form, f-string `JoinedStr` form) at the AST scan layer. The acceptable residual from Issue #2 is now hardened in depth.
- (c) Gate 2 deny-by-default for unmapped modules during plugin context — still open (tracked as Task #3 in the team task list).

Test count post-followup: 1682 (the +91 figure cited above was at `4fa9c35`; `683e236` replaced one no-assertion test and added 3 grant-counterparts, net +3 from a different baseline). All 7 audit PoCs remain verifiably closed at HEAD. mypy 0 → 0 maintained.

The original audit's REJECT verdict is overturned by this interstitial. The branch is ready to resume PRP Tasks 8–14 once these commits merge or are confirmed as acceptable on the working branch.
