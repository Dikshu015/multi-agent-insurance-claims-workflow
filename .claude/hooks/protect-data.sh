#!/bin/bash
# Hook: PreToolUse (Bash)
# Block accidental deletion of audit logs and Postgres connection strings.
# We use Neon Postgres (no local .db files) -- see docs/DECISIONS.md.
# Reads JSON from stdin with tool_input.command

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

# Block direct deletion of audit log directory or its contents
# Audit logs are retained 7 years for IRDAI/insurance compliance
if echo "$COMMAND" | grep -qE '(rm|del).*audit_log'; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: audit logs are retained for 7 years for insurance compliance. Use python scripts/clean_data.py instead."}}'
  exit 0
fi

# Block accidental wiping of the data directory entirely
if echo "$COMMAND" | grep -qE 'rm\s+-rf\s+.*data/'; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: do not delete the data/ directory directly. Use python scripts/clean_data.py instead."}}'
  exit 0
fi

exit 0\