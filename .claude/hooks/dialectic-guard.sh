#!/bin/bash
# Dialectic methodology enforcement hook.
#
# Fires on UserPromptSubmit and unconditionally injects a brief methodology
# reminder. The reminder is scoped by its own text ("skip if informational")
# so the language model filters at reasoning time — we do NOT classify intent
# with regex in a shell hook, because intent classification is exactly the
# kind of work the model is good at.
#
# Load-bearing job: land before Claude Code's built-in TaskCreate system
# reminders so methodology guidance isn't overridden by solo-implementation
# nudges. The reminder is deliberately short to conserve context tokens given
# it fires on every turn.

# Consume the JSON payload on stdin (even though we don't inspect it) so the
# caller's pipe closes cleanly.
cat >/dev/null

cat <<'REMINDER'
<dialectic-reminder>
Dialectical methodology applies here. Before implementing, pick a tier:
- Light (1-3 files) or Iterative (benchmark/debug loops): implement directly with TDD.
- Standard or Full: use /execute-team (or sequential /execute-prp + /review-code). Don't solo-implement.

TDD is mandatory: RED → GREEN → REFACTOR → COMMIT. Never weaken tests.

Skip this reminder if the request is purely informational (no file writes intended).
</dialectic-reminder>
REMINDER

exit 0
