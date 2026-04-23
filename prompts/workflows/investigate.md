# Investigate — Benchmark-Driven Investigation & Optimization

## Overview

A workflow for iterative, empirically-driven debugging and optimization. The developer is present and directing work — the dialectic happens through real-time feedback (benchmarks, profiling, test results) rather than agent-to-agent review.

## Tier

Iterative. Sets `.dialectic-tier` to `iterative` if not already set.

## Prerequisites

- Feature branch exists
- Benchmark or test infrastructure is in place
- Baseline metrics are known or can be established

## Workflow

### Step 1: Establish baseline

Run the relevant benchmark, test suite, or profiling tool. Record current metrics:
- Accuracy / correctness rate
- Latency / throughput
- Error count and types
- Resource utilization

### Step 2: Investigate

Analyze available telemetry, error logs, or profiling data. Trace the code path involved in the failure or bottleneck. Form a hypothesis about the root cause.

Key investigation techniques:
- **Inter-operation latency**: wall-clock gaps between logged operations reveal hidden bottlenecks
- **Telemetry analysis**: per-entry breakdowns show which cases fail and why
- **Error categorization**: group errors by root cause to find the highest-impact fix
- **Profiling**: measure actual vs. expected time per operation

### Step 3: Fix (TDD)

1. Write a test that reproduces the issue (RED)
2. Implement the minimum fix (GREEN)
3. Refactor if needed
4. Commit test + fix together

### Step 4: Validate

Re-run the benchmark/test from Step 1. Compare metrics to baseline:
- If improved: document the fix, move to next issue
- If not improved: revise hypothesis, loop to Step 2
- If new issues found: loop to Step 1 with new baseline

### Step 5: Review

When the investigation loop is complete:
1. Run /review-code before PR
2. Document findings in design docs or CLAUDE.md Known Gotchas
3. Create PR with before/after metrics

## Rules

- TDD is mandatory — every fix has a test
- Profile before optimizing — measure, don't guess
- Atomic commits — each fix is one commit
- Document findings — capture lessons for future sessions
