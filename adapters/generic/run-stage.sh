#!/bin/bash
# Usage: ./adapters/generic/run-stage.sh <stage-name> [arguments]
# Loads a portable workflow template and outputs it for piping to any CLI agent.
#
# Examples:
#   ./adapters/generic/run-stage.sh review-plan docs/PRPs/my-feature.md
#   ./adapters/generic/run-stage.sh new-feature "add user profiles"
#   ./adapters/generic/run-stage.sh review-plan docs/PRPs/my-feature.md | aider --message -
#
# The script only loads and parameterizes templates — it does NOT run an agent.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STAGE="$1"
shift
ARGS="$*"
TEMPLATE="${REPO_ROOT}/prompts/workflows/${STAGE}.md"

if [ -z "$STAGE" ]; then
    echo "Usage: $0 <stage-name> [arguments]" >&2
    echo "" >&2
    echo "Available stages:" >&2
    ls "${REPO_ROOT}/prompts/workflows/" 2>/dev/null | sed 's/\.md$//' >&2
    exit 1
fi

# Validate stage name matches slug format (prevents path traversal)
if ! echo "$STAGE" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
    echo "Error: Invalid stage name '${STAGE}'. Stage must be a lowercase slug (e.g., 'review-plan')." >&2
    exit 1
fi

if [ ! -f "$TEMPLATE" ]; then
    echo "Error: Unknown stage '${STAGE}'. Available stages:" >&2
    ls "${REPO_ROOT}/prompts/workflows/" 2>/dev/null | sed 's/\.md$//' >&2
    exit 1
fi

# Output the template with {input} replaced by the actual arguments.
# Uses awk with ENVIRON instead of -v to avoid metacharacter interpretation.
# awk -v treats \ as escape and & as matched-text in gsub replacements.
# ENVIRON + index/substr performs literal string replacement with no escaping.
export PUSH_HANDS_INPUT="$ARGS"
awk '{
    while ((idx = index($0, "{input}")) > 0) {
        $0 = substr($0, 1, idx-1) ENVIRON["PUSH_HANDS_INPUT"] substr($0, idx+7)
    }
    print
}' "$TEMPLATE"
