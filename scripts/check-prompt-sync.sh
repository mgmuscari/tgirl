#!/bin/bash
# Usage: ./scripts/check-prompt-sync.sh
# Compares section structure between portable templates and Claude Code commands.
# Exit 0 = in sync, Exit 1 = drift detected.
#
# This script addresses the drift risk between portable workflow templates
# (prompts/workflows/) and Claude Code commands (.claude/commands/). Since
# Claude Code commands cannot import external files, they must be kept in
# sync manually. This script detects structural divergence.

DRIFT=0
WORKFLOW_DIR="prompts/workflows"
COMMAND_DIR=".claude/commands"

# Mapping: portable workflow name:Claude Code command name
# Using a simple list instead of associative arrays for bash 3.x compatibility (macOS)
MAPPINGS="
new-feature:new-feature
generate-prp:generate-prp
review-plan:review-plan
execute-prp:execute-prp
review-code:review-code
security-audit:security-audit
update-project-docs:update-claude-md
"

for mapping in $MAPPINGS; do
  workflow="${mapping%%:*}"
  command="${mapping##*:}"
  wf_file="${WORKFLOW_DIR}/${workflow}.md"
  cmd_file="${COMMAND_DIR}/${command}.md"

  if [ ! -f "$wf_file" ]; then
    echo "✗ Missing portable workflow: ${wf_file}"
    DRIFT=1
    continue
  fi
  if [ ! -f "$cmd_file" ]; then
    echo "✗ Missing Claude Code command: ${cmd_file}"
    DRIFT=1
    continue
  fi

  # Extract ## headers from portable template (methodology sections)
  wf_headers=$(grep '^## ' "$wf_file" | sort)
  # Check that key methodology sections exist in the command file
  for section in "Stance" "Validation"; do
    if echo "$wf_headers" | grep -q "$section"; then
      if ! grep -q "## .*${section}" "$cmd_file"; then
        echo "⚠ Drift: ${workflow} has '${section}' section missing from ${cmd_file}"
        DRIFT=1
      fi
    fi
  done
  echo "✓ ${workflow} ↔ ${command}"
done

if [ "$DRIFT" -eq 0 ]; then
  echo ""
  echo "All portable templates in sync with Claude Code commands."
else
  echo ""
  echo "Drift detected — review the warnings above."
  exit 1
fi
