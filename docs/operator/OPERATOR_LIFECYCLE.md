# Operator Lifecycle

Personal Agent should answer normal operator questions without requiring the
user to know systemd, runtime bundle paths, or internal state files.

This lane is safe by default:

- status and storage questions are read-only
- repair, backup, restore, update, cleanup, uninstall, and support-bundle
  requests produce Plan Mode-style previews
- destructive actions require explicit confirmation
- restore defaults to dry-run/preview
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

For this checkpoint, the lifecycle executor behind these previews is still
partial. A confirmation proves the preview is bounded and does not silently
mutate unsupported lifecycle actions. Full backup/restore/update/cleanup/
uninstall execution remains a later operator-lifecycle hardening task.

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

## Boundaries

Do not claim this lane proves full lifecycle completion. It proves the
user-facing installed product no longer falls into generic chat for lifecycle
questions and does not silently mutate destructive operator actions.
