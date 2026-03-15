#!/bin/bash
# Push Hands methodology enforcement hook.
#
# Fires on UserPromptSubmit to inject a methodology reminder before
# the agent starts working. This counteracts system-level nudges (e.g.
# TaskCreate reminders) that push toward solo-implementation patterns.
#
# The reminder is injected as context into the agent's conversation,
# not shown to the user.
#
# Why this exists: Claude Code's built-in task management nudges
# ("consider using TaskCreate...") can override project-specific
# operating principles in CLAUDE.md, steering toward solo-implementation
# when the methodology requires team-based dialectical execution.

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // ""')

# Skip monitoring/status messages — these are noise during iterative work
if echo "$PROMPT" | grep -qiE '^(check|progress|status|how|what are|show|report)'; then
  exit 0
fi

# Detect implementation-intent keywords in the user's prompt
if echo "$PROMPT" | grep -qiE '(implement|execute|build|create|add|fix|refactor|write|develop|make)'; then
  cat <<'REMINDER'
<push-hands-reminder>
STOP. Before implementing, check CLAUDE.md operating principles:

1. Default stance is TEAM LEAD. Do NOT solo-implement for standard/full tier.
2. For changes touching >3 files or >1 module: use /execute-team (proposer + code-reviewer).
3. For changes touching 1-3 files in one module: light tier allows direct implementation.
4. Iterative tier (benchmark/debug loops): direct implementation allowed with TDD.
5. The dialectic IS the product. Bypassing it is practicing alone.
6. TaskCreate is for tracking within teams, NOT a substitute for /execute-team.

Determine the tier FIRST, then act accordingly:
- Light (1-3 files, 1 module): implement directly
- Iterative (benchmark/debug/perf loops): implement directly, TDD mandatory
- Standard (multi-file, multi-module): /execute-team or /execute-prp
- Full (security-sensitive): /execute-team + /security-audit-team (or sequential equivalents)

TDD is mandatory for all tiers:
- RED: Write tests first, verify they FAIL
- GREEN: Implement minimum code to pass
- REFACTOR: Clean up, tests must still pass
- COMMIT: Test + implementation together
</push-hands-reminder>
REMINDER
fi

exit 0
