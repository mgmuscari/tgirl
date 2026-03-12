#!/bin/bash
# Push Hands template setup script
# Run this after creating a new project from the template or after cloning.

set -e

echo "Setting up Push Hands development environment..."
echo ""

# Install git hooks
HOOKS_DIR=".git/hooks"
SCRIPTS_HOOKS_DIR="scripts/hooks"

if [ -d "$SCRIPTS_HOOKS_DIR" ]; then
    echo "Installing git hooks..."
    for hook in "$SCRIPTS_HOOKS_DIR"/*; do
        hook_name=$(basename "$hook")
        cp "$hook" "${HOOKS_DIR}/${hook_name}"
        chmod +x "${HOOKS_DIR}/${hook_name}"
        echo "  Installed: ${hook_name}"
    done
    echo ""
else
    echo "WARNING: scripts/hooks/ directory not found. Skipping hook installation."
fi

# Create directory structure if missing
echo "Ensuring directory structure..."
mkdir -p docs/PRDs
mkdir -p docs/PRPs
mkdir -p docs/reviews/plans
mkdir -p docs/reviews/code
mkdir -p docs/audits
mkdir -p docs/decisions
mkdir -p src
mkdir -p tests
mkdir -p docs/guides
mkdir -p .claude/agents
mkdir -p prompts/stances
mkdir -p prompts/workflows
echo "  Directories created."
echo ""

# Verify key files exist
echo "Checking key files..."
MISSING=0

for file in CLAUDE.md AGENTS.md push-hands.md; do
    if [ -f "$file" ]; then
        echo "  ✓ ${file}"
    else
        echo "  ✗ ${file} — MISSING"
        MISSING=1
    fi
done

for file in docs/PRDs/TEMPLATE.md docs/PRPs/TEMPLATE.md docs/audits/TEMPLATE.md docs/decisions/TEMPLATE.md; do
    if [ -f "$file" ]; then
        echo "  ✓ ${file}"
    else
        echo "  ✗ ${file} — MISSING"
        MISSING=1
    fi
done
echo ""

if [ "$MISSING" -eq 1 ]; then
    echo "WARNING: Some template files are missing. The template may be incomplete."
else
    echo "All template files present."
fi

# Check methodology enforcement hooks
echo "Checking methodology enforcement hooks..."
for hook in push-hands-guard.sh block-solo-implementation.sh enforce-opus-teams.sh; do
    if [ -f ".claude/hooks/${hook}" ]; then
        echo "  ✓ .claude/hooks/${hook}"
        if [ ! -x ".claude/hooks/${hook}" ]; then
            chmod +x ".claude/hooks/${hook}"
            echo "    (made executable)"
        fi
    else
        echo "  ✗ .claude/hooks/${hook} — MISSING"
    fi
done

if [ -f ".claude/settings.json" ]; then
    echo "  ✓ .claude/settings.json (hook configuration)"
else
    echo "  ✗ .claude/settings.json — MISSING (hooks will not fire)"
fi
echo ""

# Check agent definitions (optional — needed for team mode)
echo "Checking agent definitions (for team mode)..."
AGENTS_MISSING=0
for agent in proposer training-partner code-reviewer security-auditor skeptical-client; do
    if [ -f ".claude/agents/${agent}.md" ]; then
        echo "  ✓ .claude/agents/${agent}.md"
    else
        echo "  ✗ .claude/agents/${agent}.md — MISSING"
        AGENTS_MISSING=1
    fi
done

if [ "$AGENTS_MISSING" -eq 1 ]; then
    echo "  Some agent definitions are missing. Team mode commands may not work."
    echo "  Agent definitions are optional — sequential commands work without them."
fi
echo ""

# Check portable prompt templates
echo "Checking portable templates..."
STANCES_COUNT=0
for stance in proposer training-partner code-reviewer security-auditor skeptical-client; do
    if [ -f "prompts/stances/${stance}.md" ]; then
        STANCES_COUNT=$((STANCES_COUNT + 1))
    fi
done

WORKFLOWS_COUNT=0
for wf in new-feature generate-prp review-plan execute-prp review-code security-audit update-project-docs; do
    if [ -f "prompts/workflows/${wf}.md" ]; then
        WORKFLOWS_COUNT=$((WORKFLOWS_COUNT + 1))
    fi
done

if [ "$STANCES_COUNT" -gt 0 ]; then
    echo "  ✓ prompts/stances/ (${STANCES_COUNT} stances)"
else
    echo "  ✗ prompts/stances/ — no stances found"
fi

if [ "$WORKFLOWS_COUNT" -gt 0 ]; then
    echo "  ✓ prompts/workflows/ (${WORKFLOWS_COUNT} workflows)"
else
    echo "  ✗ prompts/workflows/ — no workflows found"
fi
echo ""

# Check platform adapters
echo "Checking platform adapters..."
if [ -d ".claude/commands" ] && [ -d ".claude/agents" ]; then
    echo "  ✓ Claude Code (.claude/commands/, .claude/agents/)"
else
    echo "  ✗ Claude Code (.claude/commands/, .claude/agents/) — not configured"
fi

if [ -d ".opencode/agents" ]; then
    echo "  ✓ OpenCode (.opencode/agents/)"
else
    echo "  ✗ OpenCode (.opencode/agents/) — not configured"
fi

if [ -d ".cursor/rules" ]; then
    echo "  ✓ Cursor (.cursor/rules/)"
else
    echo "  ✗ Cursor (.cursor/rules/) — not configured"
fi

if [ -d ".windsurf/rules" ]; then
    echo "  ✓ Windsurf (.windsurf/rules/)"
else
    echo "  ✗ Windsurf (.windsurf/rules/) — not configured"
fi

if [ -d ".clinerules" ]; then
    echo "  ✓ Cline (.clinerules/)"
else
    echo "  ✗ Cline (.clinerules/) — not configured"
fi
echo ""

# Check teammate wrapper (optional — for team mode model inheritance bug)
echo "Checking teammate wrapper (optional)..."
if [ -f "scripts/claude-teammate-wrapper.sh" ]; then
    echo "  ✓ scripts/claude-teammate-wrapper.sh"
    if [ ! -x "scripts/claude-teammate-wrapper.sh" ]; then
        chmod +x scripts/claude-teammate-wrapper.sh
        echo "    (made executable)"
    fi
    echo "  To use: export CLAUDE_CODE_TEAMMATE_COMMAND=\"./scripts/claude-teammate-wrapper.sh\""
    echo "  Or add to .claude/settings.json env block."
    echo "  This works around Claude Code bug #32368 (model inheritance)."
else
    echo "  ✗ scripts/claude-teammate-wrapper.sh — not present (optional)"
fi
echo ""

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Edit CLAUDE.md with your project's details"
echo "  2. Edit AGENTS.md if you want to customize agent stances"
echo "  3. Replace lint/test commands in scripts/hooks/ with your stack's tools"
echo "  4. See adapters/ for multi-platform setup"
echo "  5. Start your first feature: /new-feature <description>"
