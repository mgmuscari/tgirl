#!/bin/bash
# Claude Code teammate wrapper script.
#
# Workaround for Claude Code bug #32368 (model inheritance).
# When set as CLAUDE_CODE_TEAMMATE_COMMAND, this wrapper ensures
# all teammates spawn with the correct model configuration.
#
# Usage:
#   export CLAUDE_CODE_TEAMMATE_COMMAND="./scripts/claude-teammate-wrapper.sh"
#
# Or add to .claude/settings.json:
#   {
#     "env": {
#       "CLAUDE_CODE_TEAMMATE_COMMAND": "./scripts/claude-teammate-wrapper.sh"
#     }
#   }
#
# Based on community workaround by trhinehart-attentive.
# See: https://github.com/anthropics/claude-code/issues/32368

# Signal to hooks that this process is a team agent, not the team lead.
# block-solo-implementation.sh checks this to allow proposer writes.
export PUSH_HANDS_TEAM_AGENT=1

# Pass through all arguments to claude, ensuring model is set
exec claude "$@"
