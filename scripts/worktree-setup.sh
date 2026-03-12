#!/bin/bash
# Create a parallel worktree for concurrent implementation tracks
#
# Usage: ./scripts/worktree-setup.sh <feature-name> <track-name>
# Example: ./scripts/worktree-setup.sh user-auth backend-api
#
# This creates a worktree at ../project-<track-name> on the feature branch.
# Each worktree can run its own Claude Code session for parallel work.
# Worktrees merge back to the feature branch, not directly to main.

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./scripts/worktree-setup.sh <feature-name> <track-name>"
    echo "Example: ./scripts/worktree-setup.sh user-auth backend-api"
    exit 1
fi

FEATURE_NAME="$1"
TRACK_NAME="$2"

# Validate slug format: lowercase, hyphen-separated, alphanumeric
if ! echo "$FEATURE_NAME" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
    echo "Error: feature name must be lowercase, hyphen-separated (e.g., user-auth)"
    echo "  Got: ${FEATURE_NAME}"
    exit 1
fi

if ! echo "$TRACK_NAME" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
    echo "Error: track name must be lowercase, hyphen-separated (e.g., backend-api)"
    echo "  Got: ${TRACK_NAME}"
    exit 1
fi

BRANCH_NAME="feature/${FEATURE_NAME}"
PROJECT_NAME=$(basename "$(pwd)")
WORKTREE_PATH="../${PROJECT_NAME}-${TRACK_NAME}"

# Verify feature branch exists
if ! git rev-parse --verify "$BRANCH_NAME" &>/dev/null; then
    echo "Error: Branch ${BRANCH_NAME} does not exist."
    echo "Create it first with: ./scripts/new-feature.sh ${FEATURE_NAME}"
    exit 1
fi

# Create worktree
git worktree add "$WORKTREE_PATH" "$BRANCH_NAME"

echo ""
echo "Worktree created:"
echo "  Path: ${WORKTREE_PATH}"
echo "  Branch: ${BRANCH_NAME}"
echo "  Track: ${TRACK_NAME}"
echo ""
echo "Start a Claude Code session in the worktree:"
echo "  cd ${WORKTREE_PATH} && claude"
echo ""
echo "When done, merge changes back and remove the worktree:"
echo "  git worktree remove ${WORKTREE_PATH}"
