# Cleanup Preview

Cleanup preview is a read-only operator lane for identifying old or oversized
Personal Agent artifacts. It estimates recoverable space, but it does not
delete, move, truncate, rewrite, compress, or stop anything.

## User Flow

User-facing prompts:

```text
clean old runtime files
clean old backup files
clean old backups
```

The assistant returns a Plan Mode v2 preview with `action_type=operator.cleanup`
and `executor_status=preview_only`. The response must say:

```text
I did not delete anything.
```

Confirming the plan returns `executor_not_enabled`, `mutated=false`, and a
registry journal id. Actual deletion is intentionally not implemented in this
lane.

## Candidate Classes

The preview scans only approved Personal Agent locations and classifies
candidates as:

- `oversized backup artifact`: Backup v1 artifact that exceeds the current
  backup size cap or contains legacy oversized summary data.
- `old backup artifact`: valid Backup v1 artifact older than 7 days and not
  the latest valid backup.
- `old support bundle artifact`: support bundle temp artifact older than 24
  hours.
- `old runtime release`: runtime release older than 14 days and not the active
  `runtime/current` target.
- `unknown/unsafe candidate`: backup-like artifact that cannot be validated.
  These are never marked safe to delete later.

Each candidate includes a redacted path, size, age, reason, and
`safe_to_delete_later`.

## Protected Paths

Cleanup preview must not suggest deleting:

- the active `runtime/current` target
- the latest valid Backup v1 artifact
- the local secret store
- active user service files
- arbitrary home directory data

The preview also does not scan model caches, virtual environments, broad home
directories, or raw secret files.

## Proof

Run:

```bash
python scripts/cleanup_preview_smoke.py
```

The installed-product smoke proves:

- cleanup preview uses Plan Mode v2
- the response is read-only and says nothing was deleted
- oversized backup candidates are detected when present
- the latest valid backup and active runtime are protected
- unknown candidates are not marked safe
- confirmation returns `executor_not_enabled`, `mutated=false`
- git status remains unchanged

