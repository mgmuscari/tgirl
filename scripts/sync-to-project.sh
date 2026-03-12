#!/bin/bash
# Sync Push Hands methodology files from this template repo to a downstream project.
#
# Usage:
#   ./scripts/sync-to-project.sh <target-project-dir>
#   ./scripts/sync-to-project.sh --all <parent-dir>   # sync all Push Hands projects under parent
#   ./scripts/sync-to-project.sh --dry-run <target>    # show what would change, don't copy
#   ./scripts/sync-to-project.sh --with-docs <target>  # also update CLAUDE.md/AGENTS.md via claude
#   ./scripts/sync-to-project.sh --list <parent-dir>   # list detected Push Hands projects
#
# What syncs (template-owned methodology files):
#   .claude/commands/*.md        — workflow definitions
#   .claude/agents/*.md          — agent stance definitions
#   .claude/hooks/*.sh           — methodology enforcement hooks
#   prompts/stances/*.md         — portable stance definitions
#   prompts/workflows/*.md       — portable workflow templates
#   docs/PRDs/TEMPLATE.md        — PRD template
#   docs/PRPs/TEMPLATE.md        — PRP template
#   docs/audits/TEMPLATE.md      — audit template
#   docs/decisions/TEMPLATE.md   — ADR template
#   scripts/setup.sh             — setup script
#   scripts/claude-teammate-wrapper.sh — teammate wrapper
#   scripts/hooks/*              — git hooks
#
# What does NOT sync (project-owned):
#   CLAUDE.md, AGENTS.md         — have project-specific content
#   push-hands.md                — template's own PRD
#   src/, tests/, docs/PRDs/*.md (non-template), etc.
#
# After syncing, review changes with `git diff` in the target project.

set -euo pipefail

TEMPLATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=false
WITH_DOCS=false
MODE="single"  # single, all, list

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --with-docs)
            WITH_DOCS=true
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --list)
            MODE="list"
            shift
            ;;
        -*)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--dry-run] [--with-docs] [--all|--list] <target-dir>" >&2
            exit 1
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

if [ -z "${TARGET:-}" ]; then
    echo "Usage: $0 [--dry-run] [--with-docs] [--all|--list] <target-dir>" >&2
    exit 1
fi

# Files and directories to sync (relative to repo root)
# Each entry is either a file or a directory (directories sync all contents)
SYNC_DIRS=(
    ".claude/commands"
    ".claude/agents"
    ".claude/hooks"
    "prompts/stances"
    "prompts/workflows"
    "scripts/hooks"
)

SYNC_FILES=(
    "docs/PRDs/TEMPLATE.md"
    "docs/PRPs/TEMPLATE.md"
    "docs/audits/TEMPLATE.md"
    "docs/decisions/TEMPLATE.md"
    "scripts/setup.sh"
    "scripts/claude-teammate-wrapper.sh"
)

# Files to flag for manual review (have both methodology and project content)
REVIEW_FILES=(
    "CLAUDE.md"
    "AGENTS.md"
)

# Detect if a directory is a Push Hands project
is_push_hands_project() {
    local dir="$1"
    # Must be a git repo
    [ -d "$dir/.git" ] || return 1
    # Must not be the template repo itself
    [ "$(cd "$dir" && pwd)" != "$TEMPLATE_DIR" ] || return 1
    # Must have at least one Push Hands indicator
    [ -d "$dir/.claude/commands" ] || [ -d "$dir/prompts/workflows" ] || [ -f "$dir/.claude/hooks/push-hands-guard.sh" ] || return 1
    return 0
}

# Find all Push Hands projects under a parent directory
find_projects() {
    local parent="$1"
    local found=0
    for dir in "$parent"/*/; do
        [ -d "$dir" ] || continue
        if is_push_hands_project "$dir"; then
            echo "$dir"
            found=1
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "No Push Hands projects found under $parent" >&2
        return 1
    fi
}

# Sync a single file from template to target
sync_file() {
    local src="$TEMPLATE_DIR/$1"
    local dst="$2/$1"

    if [ ! -f "$src" ]; then
        return 0
    fi

    # Create parent directory if needed
    local dst_dir
    dst_dir="$(dirname "$dst")"

    if [ "$DRY_RUN" = true ]; then
        if [ -f "$dst" ]; then
            if ! diff -q "$src" "$dst" >/dev/null 2>&1; then
                echo "  CHANGED: $1"
                diff --unified=3 "$dst" "$src" | head -20 || true
                echo ""
            fi
        else
            echo "  NEW:     $1"
        fi
    else
        mkdir -p "$dst_dir"
        cp "$src" "$dst"
        # Preserve executable bit
        if [ -x "$src" ]; then
            chmod +x "$dst"
        fi
    fi
}

# Update CLAUDE.md and AGENTS.md via a single claude invocation
update_docs() {
    local target="$1"
    local project_name
    project_name="$(basename "$target")"

    # Check if claude CLI is available
    if ! command -v claude >/dev/null 2>&1; then
        echo "  SKIP docs update: 'claude' CLI not found in PATH"
        return 1
    fi

    echo "  Updating CLAUDE.md and AGENTS.md via claude..."

    # Build prompt in a temp file to avoid quoting issues with heredocs
    local prompt_file
    prompt_file="$(mktemp)"
    trap "rm -f '$prompt_file'" RETURN

    cat > "$prompt_file" <<'HEADER'
You are updating a downstream project's CLAUDE.md and AGENTS.md to incorporate methodology changes from the Push Hands template.

RULES:
- Update ONLY methodology sections. Preserve ALL project-specific content exactly as-is.
- If a methodology section doesn't exist in the project file, add it in the appropriate location.
- If the project file has extra sections not in the template, leave them untouched.
- Do not rewrite project-specific descriptions, architecture notes, tech stack, or project-specific Known Gotchas.
- Match the template's structure for methodology sections but keep the project's voice and content for everything else.

TEMPLATE CLAUDE.md (the source of truth for methodology sections):
HEADER
    cat "$TEMPLATE_DIR/CLAUDE.md" >> "$prompt_file"

    cat >> "$prompt_file" <<'MIDDLE'

---

TEMPLATE AGENTS.md (the source of truth for methodology sections):
MIDDLE
    cat "$TEMPLATE_DIR/AGENTS.md" >> "$prompt_file"

    cat >> "$prompt_file" <<'FOOTER'

---

Now read this project's CLAUDE.md and AGENTS.md, and update the methodology sections to match the template while preserving all project-specific content.

Methodology sections in CLAUDE.md to update:
- Operating Principles
- Development Lifecycle Pipeline (including Team Mode subsection)
- Agent Stances
- Key Design Decisions
- Conventions (merge: keep project-specific conventions, add any missing methodology conventions)
- Execution Rules
- Methodology Enforcement Hooks (was "Methodology Enforcement Hook" — update name and content)
- Known Gotchas (MERGE: keep project-specific gotchas, add new methodology gotchas from template dated 2026-03-09)

Methodology sections in AGENTS.md to update:
- Cross-Platform Support (intro)
- Stance definitions (Proposer, Senior Training Partner, Code Review Partner, Security Auditor, Skeptical Client)
- Platform-Specific Enforcement (including Team Mode subsection with bug notes)

DO NOT touch: Project Overview, Architecture, Tech Stack, Setup (if project-specific commands differ), project-specific Known Gotchas, project-specific Agent Instructions, or any sections not listed above.

After editing both files, report what you changed.
FOOTER

    # Run claude in the target project directory, non-interactive
    # --verbose shows tool calls as they happen; while-read avoids pipe buffering
    (cd "$target" && claude -p "$(cat "$prompt_file")" --dangerously-skip-permissions --allowedTools 'Read,Edit' --verbose 2>&1) | while IFS= read -r line; do echo "    $line"; done

    rm -f "$prompt_file"
}

# Sync one project
sync_project() {
    local target="$1"
    target="$(cd "$target" && pwd)"  # resolve to absolute path

    if [ ! -d "$target/.git" ]; then
        echo "ERROR: $target is not a git repository" >&2
        return 1
    fi

    if [ "$target" = "$TEMPLATE_DIR" ]; then
        echo "SKIP: $target is the template repo itself" >&2
        return 0
    fi

    local project_name
    project_name="$(basename "$target")"

    if [ "$DRY_RUN" = true ]; then
        echo "=== DRY RUN: $project_name ($target) ==="
    else
        echo "=== Syncing: $project_name ($target) ==="
    fi

    local changed=0

    # Sync directories
    for dir in "${SYNC_DIRS[@]}"; do
        if [ -d "$TEMPLATE_DIR/$dir" ]; then
            for file in "$TEMPLATE_DIR/$dir"/*; do
                [ -f "$file" ] || continue
                local relpath="${file#$TEMPLATE_DIR/}"
                if [ "$DRY_RUN" = true ]; then
                    if [ -f "$target/$relpath" ]; then
                        if ! diff -q "$file" "$target/$relpath" >/dev/null 2>&1; then
                            echo "  CHANGED: $relpath"
                            changed=1
                        fi
                    else
                        echo "  NEW:     $relpath"
                        changed=1
                    fi
                else
                    sync_file "$relpath" "$target"
                fi
            done
        fi
    done

    # Sync individual files
    for file in "${SYNC_FILES[@]}"; do
        if [ "$DRY_RUN" = true ]; then
            if [ -f "$TEMPLATE_DIR/$file" ]; then
                if [ -f "$target/$file" ]; then
                    if ! diff -q "$TEMPLATE_DIR/$file" "$target/$file" >/dev/null 2>&1; then
                        echo "  CHANGED: $file"
                        changed=1
                    fi
                else
                    echo "  NEW:     $file"
                    changed=1
                fi
            fi
        else
            sync_file "$file" "$target"
        fi
    done

    # Handle CLAUDE.md and AGENTS.md
    local has_doc_changes=false
    for file in "${REVIEW_FILES[@]}"; do
        if [ -f "$target/$file" ] && [ -f "$TEMPLATE_DIR/$file" ]; then
            if ! diff -q "$TEMPLATE_DIR/$file" "$target/$file" >/dev/null 2>&1; then
                has_doc_changes=true
                changed=1
            fi
        fi
    done

    if [ "$has_doc_changes" = true ]; then
        if [ "$WITH_DOCS" = true ] && [ "$DRY_RUN" = false ]; then
            update_docs "$target"
        else
            for file in "${REVIEW_FILES[@]}"; do
                if [ -f "$target/$file" ] && [ -f "$TEMPLATE_DIR/$file" ]; then
                    if ! diff -q "$TEMPLATE_DIR/$file" "$target/$file" >/dev/null 2>&1; then
                        echo "  REVIEW:  $file has upstream methodology changes — use --with-docs to auto-update"
                    fi
                fi
            done
        fi
    fi

    if [ "$DRY_RUN" = true ]; then
        if [ "$changed" -eq 0 ]; then
            echo "  (up to date)"
        fi
    else
        echo "  Done. Run 'cd $target && git diff' to review changes."
    fi
    echo ""
}

# Main
case "$MODE" in
    single)
        if [ ! -d "$TARGET" ]; then
            echo "ERROR: $TARGET is not a directory" >&2
            exit 1
        fi
        sync_project "$TARGET"
        ;;
    list)
        if [ ! -d "$TARGET" ]; then
            echo "ERROR: $TARGET is not a directory" >&2
            exit 1
        fi
        find_projects "$TARGET"
        ;;
    all)
        if [ ! -d "$TARGET" ]; then
            echo "ERROR: $TARGET is not a directory" >&2
            exit 1
        fi
        projects=$(find_projects "$TARGET") || exit 1
        echo "$projects" | while read -r project; do
            sync_project "$project"
        done
        ;;
esac
