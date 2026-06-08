# Managed Action Recovery

Managed actions are native runtime operations that can mutate local state: service setup, service cleanup, pack import, pack approval, model downloads, provider config writes, and future bounded file operations. They must be treated as transactions, not best-effort scripts.

## Lifecycle

Every mutating managed action should follow this sequence:

1. Preflight the exact target and allowed operation.
2. Show a user-facing preview.
3. Require explicit confirmation.
4. Create an action journal before mutation.
5. Record planned steps.
6. Record each actual step and whether it succeeded.
7. Record created or changed resources.
8. Verify success.
9. Roll back only resources created or changed by this action if verification fails.
10. Report what happened, what was cleaned up, and what remains.

Confirmation gates remain required. A confirmation for one action does not authorize a later action.

## Action Journal

The journal should record:

- action id
- action type
- target resource
- planned steps
- executed steps
- created resources
- changed resources
- verification result
- rollback steps
- rollback result

The journal is not permission to touch arbitrary state. It is evidence for owned changes and bounded cleanup.

Persistent storage is designed separately in
`docs/design/PERSISTENT_MANAGED_ACTION_JOURNAL.md`. A minimal SQLite helper now
exists for redacted journal rows and read-only incomplete-action status, but
existing flows are not converted yet. Do not claim crash/restart recovery is
complete until each flow persists status transitions and has restart/status
tests.

## Rollback Rules

Rollback must be conservative:

- rollback owned changes only
- never silently mutate pre-existing user resources
- never delete a resource only because its name is similar
- never run arbitrary cleanup commands
- use fixed arguments with `shell=False`
- report partial rollback failures plainly

If rollback cannot fully complete, the assistant must say what remains and require a separate confirmed cleanup or manual inspection.

## SearXNG Managed Service Setup

SearXNG setup is the first implemented managed-action recovery user.

The setup action records:

- approved image pull
- approved container creation/start
- approved managed volume path
- health-check result
- rollback stop/remove steps if health check fails

If health check fails after the runtime created `personal-agent-searxng`, the runtime stops and removes only that owned container. If a container existed before the action, the runtime does not remove it automatically.

Successful setup must report that a background service is running and reachable only locally. It should also tell the user they can ask the assistant to stop web search.

## Future Coverage

Future mutating flows should use the same pattern:

- external pack fetch/import/approval/enable/configuration
- managed adapter permission/config writes
- model downloads and imports
- provider configuration writes
- local file operations
- managed local service setup, stop, restart, and removal

Each future flow needs tests proving owned rollback, pre-existing resource protection, stale confirmation rejection, and clear user-facing recovery messages.
