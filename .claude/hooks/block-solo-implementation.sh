#!/bin/bash
# Block solo implementation on standard/full tier feature branches.
#
# Fires on PreToolUse for Edit and Write tools. Checks if:
#   1. We're on a feature/* branch
#   2. .dialectic-tier is "standard" or "full"
#   3. The target file is under src/ or tests/
#
# If all conditions are met, exits with code 2 to BLOCK the tool call.
# Team agents (proposer) are exempt — detected via DIALECTIC_TEAM_AGENT=1
# env var set by scripts/claude-teammate-wrapper.sh. When that env var is
# not inherited (Claude Code bug #32368 can cascade), a fallback consults
# ~/.claude/teams/<prefix>-<slug>/config.json for a matching cwd across
# the known team-command prefixes (execute, review, audit, plan-review).
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

# Allow team agents (proposer) — they're spawned via claude-teammate-wrapper.sh
# which sets this env var to signal they're not the team lead
if [ "${DIALECTIC_TEAM_AGENT:-}" = "1" ]; then
  exit 0
fi

# Fallback: for any known dialectic team type, if a team config exists for
# this branch AND at least one member's cwd matches this repo, exempt the
# caller. Closes false positive when team agents don't inherit
# DIALECTIC_TEAM_AGENT. Cross-repo slug collision is blocked by the cwd
# equality check — ~/.claude/teams/ is user-global.
#
# All four team commands must be covered: /execute-team (execute-{slug}),
# /review-plan-team (plan-review-{slug}), /security-audit-team (audit-{slug}),
# /review-code-team (review-{slug}). Any new team command must add its prefix
# here — otherwise its teammates get falsely blocked on src/tests edits.
DIALECTIC_TEAM_PREFIXES=("execute" "review" "audit" "plan-review")

SLUG="${BRANCH#feature/}"
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -n "$REPO_ROOT" ]; then
  for prefix in "${DIALECTIC_TEAM_PREFIXES[@]}"; do
    TEAM_CONFIG="$HOME/.claude/teams/${prefix}-${SLUG}/config.json"
    if [ -f "$TEAM_CONFIG" ] && \
       jq -e --arg root "$REPO_ROOT" \
            '.members[] | select(.cwd == $root)' \
            "$TEAM_CONFIG" >/dev/null 2>&1; then
      exit 0
    fi
  done
fi

# Check tier
TIER_FILE="$(git rev-parse --show-toplevel 2>/dev/null)/.dialectic-tier"
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
  - Team execute:     /execute-team <prp-path>
  - Team review:      /review-code-team (after implementation commits)
  - Sequential:       /execute-prp <prp-path> then /review-code

If teammates previously failed:
  - Check claude --version (v2.1.46+ may fix team bugs)
  - ALWAYS spawn agents with model: "opus"
  - Clean up the old team first, then restart
  - Do NOT fall back to solo implementation

The dialectic IS the product. Bypassing it is practicing alone.
BLOCK
exit 2
