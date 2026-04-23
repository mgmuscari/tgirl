#!/bin/bash
# Create a new feature branch and scaffold the PRD
#
# Usage: ./scripts/new-feature.sh <feature-name> [--tier light|standard|full]
# Example: ./scripts/new-feature.sh user-authentication
# Example: ./scripts/new-feature.sh fix-typo --tier light
# Example: ./scripts/new-feature.sh auth-rework --tier full

set -e

# Parse arguments
FEATURE_NAME=""
TIER="standard"

while [ $# -gt 0 ]; do
    case "$1" in
        --tier)
            if [ -z "$2" ]; then
                echo "Error: --tier requires a value (light, standard, full)"
                exit 1
            fi
            TIER="$2"
            shift 2
            ;;
        -*)
            echo "Error: unknown flag: $1"
            exit 1
            ;;
        *)
            if [ -z "$FEATURE_NAME" ]; then
                FEATURE_NAME="$1"
            else
                echo "Error: unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$FEATURE_NAME" ]; then
    echo "Usage: ./scripts/new-feature.sh <feature-name> [--tier light|iterative|standard|full]"
    echo "Example: ./scripts/new-feature.sh user-authentication"
    echo "Example: ./scripts/new-feature.sh fix-typo --tier light"
    echo "Example: ./scripts/new-feature.sh perf-bottleneck --tier iterative"
    exit 1
fi

# Validate tier value
case "$TIER" in
    light|iterative|standard|full) ;;
    *)
        echo "Error: invalid tier '${TIER}' — must be light, iterative, standard, or full"
        exit 1
        ;;
esac

# Validate slug format: lowercase, hyphen-separated, alphanumeric
if ! echo "$FEATURE_NAME" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
    echo "Error: feature name must be lowercase, hyphen-separated (e.g., user-authentication)"
    echo "  Got: ${FEATURE_NAME}"
    exit 1
fi

BRANCH_NAME="feature/${FEATURE_NAME}"
PRD_PATH="docs/PRDs/${FEATURE_NAME}.md"

# Ensure we're starting from an up-to-date main
git checkout main
git pull --ff-only 2>/dev/null || true

# Create feature branch
git checkout -b "$BRANCH_NAME"

# Write tier metadata
echo "$TIER" > .dialectic-tier

if [ "$TIER" = "light" ] || [ "$TIER" = "iterative" ]; then
    # Light/iterative tier: no PRD scaffold, just tier file
    git add .dialectic-tier
    git commit -m "chore: create feature branch for ${FEATURE_NAME} (${TIER} tier)"

    echo ""
    echo "Feature branch created: ${BRANCH_NAME}"
    echo "Tier: ${TIER}"
    echo ""
    if [ "$TIER" = "iterative" ]; then
        echo "Next steps:"
        echo "  1. Implement with TDD (mandatory)"
        echo "  2. Benchmark/test → fix → re-benchmark (iterate)"
        echo "  3. Run /review-code before PR"
        echo "  4. Close with ./scripts/close-feature.sh ${FEATURE_NAME}"
    else
        echo "Next steps:"
        echo "  1. Implement your change directly"
        echo "  2. Run /review-code (optional)"
        echo "  3. Close with ./scripts/close-feature.sh ${FEATURE_NAME}"
    fi
else
    # Standard/Full tier: scaffold PRD
    cp docs/PRDs/TEMPLATE.md "$PRD_PATH"
    mkdir -p docs/PRPs

    git add .dialectic-tier "$PRD_PATH"
    git commit -m "docs: scaffold PRD for ${FEATURE_NAME}"

    echo ""
    echo "Feature branch created: ${BRANCH_NAME}"
    echo "Tier: ${TIER}"
    echo "PRD scaffold created: ${PRD_PATH}"
    echo ""
    echo "Next steps:"
    echo "  1. Fill in the PRD (or use /new-feature to have an agent do it)"
    echo "  2. Get PRD approved"
    echo "  3. Run /generate-prp ${PRD_PATH}"
    if [ "$TIER" = "full" ]; then
        echo ""
        echo "  NOTE: Security audit is expected for this tier."
        echo "  Run /security-audit before creating the PR."
    fi
fi
