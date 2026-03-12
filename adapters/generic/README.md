# Generic Adapter

For Windsurf, Cline, Aider, and any CLI-based AI coding agent that doesn't have a dedicated adapter.

## Platform-Specific Quick Start

### Windsurf

Windsurf reads `.windsurf/rules/*.md` files. Create a rules file pointing to the Push Hands stances:

```bash
mkdir -p .windsurf/rules
cp prompts/stances/*.md .windsurf/rules/
```

Or rely on `AGENTS.md` which Windsurf auto-reads from the project root.

### Cline

Cline reads `.clinerules/` folder files. Create rules files with stance content:

```bash
mkdir -p .clinerules
cp prompts/stances/*.md .clinerules/
```

Cline also reads `AGENTS.md`, `.cursor/rules/`, and `.windsurf/rules/` as fallbacks — if you've set up the Cursor adapter, Cline will pick up those rules automatically.

### Aider

Add the relevant context files to your `.aider.conf.yml`:

```yaml
read:
  - AGENTS.md
  - prompts/stances/proposer.md
```

Swap the stance file for the relevant one when doing reviews:

```bash
aider --read prompts/stances/training-partner.md --read prompts/workflows/review-plan.md
```

### Other CLI Agents

Use `run-stage.sh` to load and parameterize workflow templates:

```bash
./adapters/generic/run-stage.sh review-plan docs/PRPs/my-feature.md
```

Pipe to your agent:

```bash
./adapters/generic/run-stage.sh review-plan docs/PRPs/my-feature.md | your-agent --stdin
```

## Running a Workflow

1. Load the relevant `prompts/workflows/<stage>.md` as context for your agent
2. Replace `{input}` with your actual input (or use `run-stage.sh` to do this automatically)
3. Follow the instructions in the template

## run-stage.sh

A minimal shell wrapper that loads a portable workflow template and replaces `{input}` with your arguments:

```bash
# Usage
./adapters/generic/run-stage.sh <stage-name> [arguments]

# Examples
./adapters/generic/run-stage.sh review-plan docs/PRPs/my-feature.md
./adapters/generic/run-stage.sh new-feature "add user profiles"

# List available stages
./adapters/generic/run-stage.sh
```

The script only loads and parameterizes templates — it does NOT run an agent.

**Trust boundary:** `run-stage.sh` is intended for trusted, interactive use. Input is passed verbatim into prompt templates with no structural separation — do not wire it to untrusted input sources without additional controls.

## Limitations

- **No tool-level enforcement:** Stances are behavioral guidance only
- **No concurrent agents:** Each interaction is single-threaded
- **No automated stage sequencing:** You manually invoke each workflow stage
- **No slash commands:** Use prompt templates directly or via `run-stage.sh`
