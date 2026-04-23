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

# Pre-merge: verify the feature branch HEAD has .dialectic-tier.
# Without this check, a missing tier file is masked by a silent fallback to
# "standard", so the user thinks they're closing tier X but the script
# proceeds as standard. Hard error, with a recovery path.
if ! git show "$BRANCH_NAME:.dialectic-tier" >/dev/null 2>&1; then
    echo "ERROR: .dialectic-tier is missing from $BRANCH_NAME HEAD." >&2
    echo "       Tier metadata is required to determine artifact expectations." >&2
    echo "       Recovery: either" >&2
    echo "         (a) git checkout $BRANCH_NAME && echo standard > .dialectic-tier && \\" >&2
    echo "             git add .dialectic-tier && git commit -m 'chore: restore tier metadata'" >&2
    echo "         (b) re-create the branch via ./scripts/new-feature.sh" >&2
    exit 1
fi

# Read tier metadata. The pre-merge check guarantees the file exists in
# BRANCH_NAME's HEAD. If the working tree is out of sync (e.g. script was
# invoked from main without a prior checkout), restore from that HEAD.
if [ ! -f .dialectic-tier ]; then
    git show "$BRANCH_NAME:.dialectic-tier" > .dialectic-tier
fi
TIER=$(cat .dialectic-tier)
case "$TIER" in
    light|iterative|standard|full) ;;
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

# Remove tier metadata from the squash staging area so it never reaches main.
# The squash merge pulls .dialectic-tier into the index; unstage + delete it
# explicitly and verify absence. Loud error if it survives.
if git diff --cached --name-only | grep -q '^\.dialectic-tier$'; then
    git restore --staged .dialectic-tier
    rm -f .dialectic-tier
    echo "Removed .dialectic-tier from squash staging area."
fi

# Defense in depth: verify the file is absent from the index after unstage.
if git diff --cached --name-only | grep -q '^\.dialectic-tier$'; then
    echo "ERROR: .dialectic-tier is still in the index after cleanup." >&2
    echo "       Manual intervention required before the squash commit." >&2
    exit 1
fi

# Also verify no .dialectic-tier in the working tree.
if [ -f .dialectic-tier ]; then
    rm -f .dialectic-tier
fi

# Determine commit type from branch convention (default: feat)
COMMIT_TYPE="feat"
echo "Commit type defaults to 'feat'. Override? (feat/fix/refactor/chore) [feat]: "
read -r REPLY
if [ -n "$REPLY" ]; then
    COMMIT_TYPE="$REPLY"
fi

# Tier-appropriate commit message. Light and iterative tiers produce no
# PRD/PRP, so their commit messages don't reference those artifact paths.
if [ "$TIER" = "light" ] || [ "$TIER" = "iterative" ]; then
    git commit -m "${COMMIT_TYPE}: ${FEATURE_NAME}

Squash merge of ${BRANCH_NAME} (${TIER} tier)."
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
