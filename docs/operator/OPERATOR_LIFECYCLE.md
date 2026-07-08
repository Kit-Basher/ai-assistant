# Operator Lifecycle

Personal Agent should answer normal operator questions without requiring the
user to know systemd, runtime bundle paths, or internal state files.

This lane is safe by default:

- status and storage questions are read-only
- repair, backup, restore, update, cleanup, uninstall, and support-bundle
  requests produce Plan Mode-style previews
- destructive actions require explicit confirmation
- restore executes only validated Backup v1 allowlisted non-secret state after
  confirmation
- update executes only bounded Update v1 outcomes after confirmation:
  isolated staged-release proof, verified live no-op, or a structured
  no-mutation blocker when live promotion is not rollback-safe
- uninstall is extra cautious and must not run from an ambiguous chat request
- support bundle wording must say secrets are redacted

## Current Product Flow

These prompts are deterministic through the installed `/chat` API:

- `is the assistant healthy?`
- `what is broken?`
- `how much space is this using?`
- `repair the assistant`
- `back up the assistant`
- `restore from backup`
- `update the assistant`
- `clean old runtime files`
- `uninstall the assistant`
- `make a support bundle`

`how much space is this using?` reports a read-only estimate for the main local
Personal Agent paths and does not delete anything.

The mutating prompts return a preview with:

- resources that may be created, changed, or deleted
- rollback scope
- confirmation requirement
- no mutation during preview

For this checkpoint, lifecycle execution is still partial but no longer purely
preview-only. Backup, support bundle, cleanup, restore, and update have bounded
Executor Registry paths. Update v1 is intentionally conservative for the live
installed runtime: dirty checkouts and non-no-op live promotion are blocked
unless a rollback-safe staged-release handoff is available. Uninstall remains
preview-only.

## Proof

Run:

```bash
python scripts/operator_lifecycle_smoke.py
```

The smoke talks to `http://127.0.0.1:8765` and verifies:

- health and broken-status questions produce understandable local status
- storage usage is read-only
- backup/restore/update/cleanup/uninstall/support-bundle requests preview safely
- support-bundle text includes redaction language
- destructive confirmation can be cancelled
- stale confirmation does not execute uninstall
- the repo working tree status is unchanged by the smoke

Run `python scripts/update_execution_smoke.py` for the isolated Update v1
execution proof. It does not update the real daily-driver install.

## Boundaries

Do not claim this lane proves full lifecycle completion. It proves the
user-facing installed product no longer falls into generic chat for lifecycle
questions and does not silently mutate destructive operator actions. It does
not claim arbitrary live self-update or uninstall completion.
