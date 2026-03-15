You are operating in the **Proposer** stance for benchmark-driven investigation and optimization.

## Workflow

This skill sets the working mode to **iterative tier** — direct implementation with TDD, driven by empirical feedback loops.

### Step 1: Establish baseline
- Run the relevant benchmark, test suite, or profiling tool
- Record current metrics (accuracy, latency, error rate, throughput)
- Identify the specific failure or performance issue to investigate

### Step 2: Investigate
- Analyze telemetry, error logs, or profiling data
- Trace the code path involved
- Form a hypothesis about the root cause

### Step 3: Fix (TDD)
- Write a test that reproduces the issue (RED)
- Implement the minimum fix (GREEN)
- Refactor if needed
- Commit test + fix together

### Step 4: Validate
- Re-run the benchmark/test from Step 1
- Compare metrics to baseline
- If improved: document the fix and move to next issue
- If not improved or new issues found: loop back to Step 2

### Step 5: Review
- When the investigation loop is complete, run /review-code
- Document findings in appropriate design docs

## Rules

- **TDD is mandatory** — every fix has a test
- **Profile before optimizing** — use telemetry and measurement, not intuition
- **Atomic commits** — each fix is one commit (test + implementation)
- **Document findings** — update CLAUDE.md Known Gotchas if the issue could recur
- **All tensor/matrix math** must use accelerated libraries (MLX or PyTorch), never Python list comprehensions on tensor data
- **Never convert between linalg libraries** unless there is absolutely no alternative — exhaust the library's own API first

## Tier

If `.push-hands-tier` is not already set, set it to `iterative`:
```bash
echo "iterative" > .push-hands-tier
```

This enables direct implementation without team mode, while preserving TDD requirements and code review before PR.
