#!/bin/bash
# Block solo implementation on standard/full tier feature branches.
#
# Fires on PreToolUse for Edit and Write tools. Checks if:
#   1. We're on a feature/* branch
#   2. .push-hands-tier is "standard" or "full"
#   3. The target file is under src/ or tests/
#
# If all conditions are met, exits with code 2 to BLOCK the tool call.
# Team agents (proposer) are exempt — detected via PUSH_HANDS_TEAM_AGENT=1
# env var set by scripts/claude-teammate-wrapper.sh.
#
# Exit codes:
#   0 = allow (not on feature branch, or light tier, or non-src file)
#   2 = block (standard/full tier, editing src/tests without a team)

set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""')

# Only gate Edit and Write tools
case "$TOOL_NAME" in
  Edit|Write) ;;
  *) exit 0 ;;
esac

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""')

# Only gate src/ and tests/ files (allow docs, config, hooks, etc.)
case "$FILE_PATH" in
  */src/*|*/tests/*) ;;
  *) exit 0 ;;
esac

# Check if we're on a feature branch
BRANCH=$(git branch --show-current 2>/dev/null || echo "")
case "$BRANCH" in
  feature/*) ;;
  *) exit 0 ;;
esac

# Allow team agents — detected via env var (set by claude-teammate-wrapper.sh)
# or by checking if an active execute team exists for this branch
if [ "${PUSH_HANDS_TEAM_AGENT:-}" = "1" ]; then
  exit 0
fi

# Also allow if an active execute team exists (spawned agents don't get env var)
SLUG="${BRANCH#feature/}"
TEAM_CONFIG="$HOME/.claude/teams/execute-${SLUG}/config.json"
if [ -f "$TEAM_CONFIG" ]; then
  exit 0
fi

# Check tier
TIER_FILE="$(git rev-parse --show-toplevel 2>/dev/null)/.push-hands-tier"
if [ ! -f "$TIER_FILE" ]; then
  exit 0
fi
TIER=$(cat "$TIER_FILE" | tr -d '[:space:]')
case "$TIER" in
  standard|full) ;;
  # Light and iterative tiers allow direct implementation
  *) exit 0 ;;
esac

# BLOCK: standard/full tier, editing src/tests
cat <<'BLOCK'
BLOCKED: Solo implementation detected on a standard/full tier feature branch.

You are the TEAM LEAD. You must NOT directly edit src/ or tests/ files.

Options:
  - Team mode: /execute-team <prp-path> (proposer + code-reviewer team)
  - Sequential: /execute-prp <prp-path> then /review-code

If teammates previously failed:
  - Check claude --version (v2.1.46+ may fix team bugs)
  - ALWAYS spawn agents with model: "opus"
  - Clean up the old team first, then restart
  - Do NOT fall back to solo implementation

The dialectic IS the product. Bypassing it is practicing alone.
BLOCK
exit 2
