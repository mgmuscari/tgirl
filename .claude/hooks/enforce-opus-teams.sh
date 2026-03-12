#!/bin/bash
# Enforce model: "opus" on all team agent spawns and warn for regular subagent spawns.
#
# Fires on PreToolUse for the Agent tool.
#
# Team spawns (team_name set):
#   - If model is not "opus", DENY the call.
#
# Regular subagent spawns (no team_name):
#   - If model is not explicitly set, allow but inject a warning.
#
# Why: Team agents spawned without an explicit model inherit the
# parent's model setting, which may be a cheaper/faster model.
# This causes agents to go idle or produce poor results (bug #32368).
# Opus is required for all team-based work.
#
# Hook output: JSON with permissionDecision: "deny" to block,
# or exit 0 to allow.

set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""')

# Only gate Agent tool calls
if [ "$TOOL_NAME" != "Agent" ]; then
  exit 0
fi

# Check if this is a team spawn (has team_name)
TEAM_NAME=$(echo "$INPUT" | jq -r '.tool_input.team_name // ""')
MODEL=$(echo "$INPUT" | jq -r '.tool_input.model // ""')

if [ -n "$TEAM_NAME" ]; then
  # Team spawn — model: "opus" is REQUIRED
  if [ "$MODEL" = "opus" ]; then
    exit 0
  fi

  cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "BLOCKED: Team agent spawn without model: \"opus\". You specified model: \"${MODEL:-<not set>}\" for team \"${TEAM_NAME}\". ALL team agents MUST use model: \"opus\" to avoid bug #32368 (model inheritance). Re-call the Agent tool with model: \"opus\" explicitly set."
  }
}
EOF
  exit 0
fi

# Regular subagent spawn — warn if model not explicitly set
if [ -z "$MODEL" ]; then
  echo "NOTE: Subagent spawned without explicit model. Consider setting model: \"opus\" for best results. Team spawns REQUIRE model: \"opus\"."
fi

exit 0
