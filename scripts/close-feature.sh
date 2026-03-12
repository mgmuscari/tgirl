#!/bin/bash
# Close a feature branch: verify artifacts, merge to main, clean up
#
# Usage: ./scripts/close-feature.sh <feature-name>
# Example: ./scripts/close-feature.sh user-authentication

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/close-feature.sh <feature-name>"
    echo "Example: ./scripts/close-feature.sh user-authentication"
    exit 1
fi

FEATURE_NAME="$1"

# Validate slug format: lowercase, hyphen-separated, alphanumeric
if ! echo "$FEATURE_NAME" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
    echo "Error: feature name must be lowercase, hyphen-separated (e.g., user-authentication)"
    echo "  Got: ${FEATURE_NAME}"
    exit 1
fi

BRANCH_NAME="feature/${FEATURE_NAME}"

# Verify we're on the feature branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$BRANCH_NAME" ]; then
    echo "Switching to ${BRANCH_NAME}..."
    git checkout "$BRANCH_NAME"
fi

# Read tier metadata
TIER=$(cat .push-hands-tier 2>/dev/null || echo "standard")
case "$TIER" in
    light|standard|full) ;;
    *) TIER="standard" ;;
esac

echo "Closing feature: ${FEATURE_NAME}"
echo "Branch: ${BRANCH_NAME}"
echo "Tier: ${TIER}"
echo ""

# Verify review artifacts exist (tier-conditional)
MISSING_ARTIFACTS=0

if [ "$TIER" = "standard" ] || [ "$TIER" = "full" ]; then
    if [ ! -f "docs/PRDs/${FEATURE_NAME}.md" ]; then
        echo "WARNING: Missing PRD at docs/PRDs/${FEATURE_NAME}.md"
        MISSING_ARTIFACTS=1
    fi

    if [ ! -f "docs/PRPs/${FEATURE_NAME}.md" ]; then
        echo "WARNING: Missing PRP at docs/PRPs/${FEATURE_NAME}.md"
        MISSING_ARTIFACTS=1
    fi

    if [ ! -f "docs/reviews/plans/${FEATURE_NAME}-review.md" ]; then
        echo "WARNING: Missing plan review at docs/reviews/plans/${FEATURE_NAME}-review.md"
        MISSING_ARTIFACTS=1
    fi

    if [ ! -f "docs/reviews/code/${FEATURE_NAME}-review.md" ]; then
        echo "WARNING: Missing code review at docs/reviews/code/${FEATURE_NAME}-review.md"
        MISSING_ARTIFACTS=1
    fi
fi

if [ "$TIER" = "full" ]; then
    if [ ! -f "docs/audits/${FEATURE_NAME}-audit.md" ]; then
        echo "WARNING: Missing security audit at docs/audits/${FEATURE_NAME}-audit.md"
        MISSING_ARTIFACTS=1
    fi
fi

if [ "$MISSING_ARTIFACTS" -eq 1 ]; then
    echo ""
    read -p "Review artifacts are missing. Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Run tests
echo "Running test suite..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short || {
        echo "Tests failed. Fix before closing feature."
        exit 1
    }
else
    echo "WARNING: No test runner detected. Skipping tests."
fi

# Squash-merge to main
echo ""
echo "Merging ${BRANCH_NAME} to main..."
git checkout main
git merge --squash "$BRANCH_NAME"

# Remove tier metadata from the squash staging area so it never reaches main
git rm -f --cached .push-hands-tier 2>/dev/null || true
rm -f .push-hands-tier

# Determine commit type from branch convention (default: feat)
COMMIT_TYPE="feat"
echo "Commit type defaults to 'feat'. Override? (feat/fix/refactor/chore) [feat]: "
read -r REPLY
if [ -n "$REPLY" ]; then
    COMMIT_TYPE="$REPLY"
fi

# Tier-appropriate commit message
if [ "$TIER" = "light" ]; then
    git commit -m "${COMMIT_TYPE}: ${FEATURE_NAME}

Squash merge of ${BRANCH_NAME} (light tier)."
else
    git commit -m "${COMMIT_TYPE}: ${FEATURE_NAME}

Squash merge of ${BRANCH_NAME}.
See docs/PRDs/${FEATURE_NAME}.md for requirements.
See docs/PRPs/${FEATURE_NAME}.md for implementation plan."
fi

# Clean up branch
git branch -d "$BRANCH_NAME"

echo ""
echo "Feature ${FEATURE_NAME} merged to main."
echo "Branch ${BRANCH_NAME} deleted."
echo ""
echo "Don't forget to update CLAUDE.md Active Work section."
