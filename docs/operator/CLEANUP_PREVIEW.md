# Cleanup Preview And Execution

Cleanup preview identifies old or oversized Personal Agent artifacts. Cleanup
execution is available only after Plan Mode v2 confirmation and only for the
exact candidates from the preview that still pass revalidation.

## User Flow

User-facing prompts:

```text
clean old runtime files
clean old backup files
clean old backups
```

The assistant returns a Plan Mode v2 preview with `action_type=operator.cleanup`
and `executor_status=enabled`. The preview response must say:

```text
I did not delete anything.
```

Confirming the plan executes `operator.cleanup.v1` through the Executor
Registry. The executor revalidates every candidate before deletion, skips
anything that changed or became protected, and journals a bounded redacted
result.

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

## Execution Boundary

Cleanup can delete only these owned artifact classes:

- old or oversized Backup v1 directories under the approved Personal Agent
  backup root
- old support bundle temp directories with the approved Personal Agent support
  bundle prefix
- old runtime release directories under the approved Personal Agent runtime
  releases root

The executor rejects path traversal, symlink candidates, mount points, unknown
classes, unapproved roots, candidates that changed after preview, the latest
valid backup, the active runtime, secret stores, active service files, and
arbitrary user paths.

Cleanup deletion is not automatically reversible. The result must report
deleted, skipped, protected, and failed candidates truthfully.

## Proof

Run:

```bash
python scripts/cleanup_preview_smoke.py
python scripts/cleanup_execution_smoke.py
```

The installed-product smoke proves:

- cleanup preview uses Plan Mode v2
- the response is read-only and says nothing was deleted
- oversized backup candidates are detected when present
- the latest valid backup and active runtime are protected
- unknown candidates are not marked safe
- cancellation leaves the installed daily-driver cleanup plan unexecuted
- git status remains unchanged

`cleanup_execution_smoke.py` proves actual deletion against an isolated fixture
through the Executor Registry, and verifies the installed product exposes the
enabled cleanup plan without deleting existing daily-driver artifacts.
