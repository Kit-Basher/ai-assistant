# Managed Action Reliability Standard

Core rule: any action the assistant takes must either complete cleanly or recover cleanly.

This standard applies to assistant-managed mutating actions, including local service setup, model acquisition, provider configuration, Telegram setup, external pack lifecycle actions, registry hygiene, autoconfig/self-heal, and future file operations. It does not make Docker, SearXNG, llama.cpp, Ollama, Telegram, or any external pack required.

## Required Pattern

Every mutating managed action must have:

1. preflight of the target, prerequisites, and allowed operation
2. user-visible preview before mutation
3. explicit confirmation
4. action journal before mutation
5. ownership tracking for resources created or changed by this action
6. verification after mutation
7. rollback of resources created by this action if verification fails
8. no silent mutation of pre-existing user resources
9. clear user-facing failure message
10. safe next step that names the next specific action or manual cleanup
11. no stale or expired confirmation execution
12. no generic "try again" after failed actions

## Rollback Boundary

Rollback must be scoped to owned changes only:

- rollback owned changes only
- never silently delete, stop, overwrite, or reconfigure pre-existing user resources
- never silently leave surprise background services running after a failed setup
- never use arbitrary cleanup commands
- never use `shell=True` for managed cleanup
- report partial rollback failure and what remains

If rollback is impossible or unsafe, the assistant must say what remains and require a separate confirmation or manual operator step.

## Confirmation Reliability

Confirmations are safety boundaries:

- valid confirmations may continue exactly one pending action
- expired confirmations must not execute
- consumed confirmations must never replay
- stale confirmations must not be matched to a different action
- a "yes" without an active valid action must not mutate state

If a confirmation expired, the user-facing response should say it expired and that no changes were made.

## User-Facing Failure Standard

Failures must be specific enough for a normal user:

- what was attempted
- what changed, if anything
- what was rolled back
- what remains, if anything
- whether anything is still running
- what the next safe step is

Avoid vague fallback text such as "try again" or "I could not complete that" unless paired with a named reason and next safe step.

## Test Expectations

Every mutating flow should eventually have tests for:

- preflight rejects invalid targets
- preview does not mutate
- confirmation is required
- expired confirmation does not execute
- consumed confirmation does not replay
- action journal records planned and executed steps
- verification failure triggers owned rollback
- rollback never mutates pre-existing resources
- partial rollback reports what remains
- user-facing failure text names the reason and next step

