You are operating in the **Proposer** stance. Your goal is to update CLAUDE.md with lessons learned from a recent correction or discovery.

## Instructions

$ARGUMENTS

1. **Identify what was learned:**
   - If the user just corrected you, capture the correction
   - If you discovered a gotcha during implementation, capture it
   - If a convention was clarified, capture it

2. **Read the current CLAUDE.md** to understand its structure and existing entries.

3. **Update the appropriate section:**
   - **Known Gotchas** for mistakes, pitfalls, and things that cause problems:
     - Format: `YYYY-MM-DD: [what went wrong] → [what to do instead]`
   - **Conventions** for new style rules, naming patterns, or organizational decisions
   - **Architecture** for structural discoveries or changes
   - **Agent Instructions** for new workflow knowledge (how to run things, what to avoid)

4. **Do NOT:**
   - Duplicate information already in CLAUDE.md
   - Add generic advice — only project-specific knowledge
   - Remove existing entries unless they are demonstrably wrong
   - Make the file excessively long — be concise

5. **Commit** the update:
   - Message: `docs: update CLAUDE.md — [brief description of lesson learned]`
