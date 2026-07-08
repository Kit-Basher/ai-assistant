# Executor Registry v1

Executor Registry v1 is the narrow apply boundary behind Plan Mode v2. Plan
Mode still owns preview, user confirmation, expiry, and thread/session binding.
The registry owns executor lookup, refusal for non-enabled executors, canonical
result fields, and an append-only redacted mutation journal.

## Result Schema

Every registry result includes:

- `ok`
- `mutated`
- `executor_id`
- `plan_id`
- `action_type`
- `target`
- `started_at`
- `finished_at`
- `resources_touched`
- `journal_id`
- `rollback_available`
- `rollback_hint`
- `error_code`
- `user_message`

## Enforcement

- `executor_status=preview_only` returns `executor_not_enabled` and
  `mutated=false`.
- `executor_status=unavailable` returns `executor_unavailable` and
  `mutated=false`.
- `executor_status=enabled` can run only when the confirmed Plan Mode action
  reaches the registry with the exact matching `plan_id`.
- High-risk/destructive actions stay behind Plan Mode confirmation and are not
  enabled unless a bounded executor exists.
- Registry refusals are journaled too, so attempted preview-only execution is
  auditable.

## Journal

The registry writes append-only JSONL records under runtime state:

`executor_registry_journal.jsonl`

The journal redacts token, secret, password, bearer, API key, and confirmation
token fields. It should not contain raw secrets, raw support logs, or private
unredacted payloads.

## Enabled Executors

Current enabled executors:

- `operator.support_bundle` via `operator.support_bundle.v1`
- `operator.backup` via `operator.backup.v1`
- `operator.cleanup` via `operator.cleanup.v1`
- `operator.restore` via `operator.restore.v1`
- `operator.update` via `operator.update.v1`
- `operator.uninstall` via `operator.uninstall.v1`

`operator.support_bundle` creates a Support Bundle v2 diagnostics package in a
temporary directory. It is additive and returns a `journal_id`. Rollback is
limited to removing the newly created temporary support bundle directory. See
`docs/operator/SUPPORT_BUNDLE.md`.

`operator.backup` creates a Backup v1 directory under the approved Personal
Agent local backup path. It is additive and returns a `journal_id`. Rollback is
limited to removing the newly created backup directory. See
`docs/operator/BACKUP_V1.md`.

`operator.cleanup` deletes only preview-identified, revalidated, owned
Personal Agent artifacts such as generated cleanup fixtures, old backups, old
support bundles, and old runtime releases. It preserves current runtime, active
service files, secret stores, latest valid backup, and arbitrary user files.

`operator.restore` restores only validated Backup v1 allowlisted non-secret
preference values for system-resource baselines/context. It stages content,
creates a pre-restore safety snapshot, verifies live state after apply, and
attempts rollback from the snapshot on post-mutation failure. It does not
restore secrets, logs, arbitrary files, model caches, runtime releases, or
untrusted executable/pack content.

`operator.update` is enabled for bounded Update v1 outcomes: isolated
staged-release fixture promotion/rollback proof through Host Lifecycle Runner
v1, verified live no-op, and structured live-promotion blockers when
preconditions are not rollback-safe.
It rejects arbitrary repositories, branches, commits, scripts, URLs, dirty
working trees, and target drift after preview. See
`docs/operator/UPDATE_EXECUTOR_V1.md`.

`operator.uninstall` is enabled for Uninstall v1 preserve-data execution against
approved isolated fixtures through Host Lifecycle Runner v1 and for guarded
live daily-driver refusal. The executor requires a final safety backup, exact
owned-resource allowlists, target-snapshot validation, and a durable uninstall
receipt. Live daily-driver uninstall remains blocked unless the target is an
approved isolated fixture; it does not stop services or remove runtime files
from ordinary installed-product proofs. See
`docs/operator/UNINSTALL_EXECUTOR_V1.md`.

## Preview-Only Lanes

These remain preview-only in v1:

- memory delete/export/redact/dedupe/control executors
- repair lifecycle executors unless a separate bounded executor is already
  used by an existing managed-service path

Confirming one of these plans must return `executor_not_enabled`,
`mutated=false`, and a registry `journal_id`.

Cleanup, restore, update, and uninstall are no longer preview-only. They remain
Plan Mode gated and Executor Registry dispatched. Update v1 live promotion
remains guarded unless the action is a verified no-op or an approved
staged-release runner input. Uninstall v1 live daily-driver removal remains
guarded unless the action targets an approved isolated fixture.

## Proof

Run:

```bash
python scripts/executor_registry_smoke.py
python scripts/support_bundle_v2_smoke.py
python scripts/backup_v1_smoke.py
python scripts/cleanup_execution_smoke.py
python scripts/restore_execution_smoke.py
python scripts/host_lifecycle_runner_smoke.py
python scripts/host_lifecycle_systemd_smoke.py
python scripts/update_execution_smoke.py
python scripts/uninstall_execution_smoke.py
```

The smoke talks to the installed `/chat` API and proves:

- preview-only memory delete cannot execute
- live daily-driver uninstall confirmation is guarded with `mutated=false`
- cleanup has an enabled executor plan and can be cancelled without mutation
- support bundle has an enabled executor and returns a journal id
- support bundle creates only a redacted temporary diagnostics artifact
- Support Bundle v2 writes a manifest and bounded summary files
- backup has an enabled executor and returns a journal id
- Backup v1 writes a manifest and bounded redacted summary files
- Restore v1 Validator is read-only
- Restore v1 execution is proven against isolated fixture state with staging,
  safety snapshot, allowlisted preference apply, duplicate confirmation safety,
  and rollback on forced post-apply verification failure
- Update v1 execution is proven against isolated fixture releases with staged
  promotion, rollback checkpoint, forced post-promotion rollback, dirty-tree
  refusal, target-drift refusal, and live no-op behavior
- Uninstall v1 execution is proven against isolated fixture installs with a
  final safety backup, exact runtime/service removal, preserve-data behavior,
  durable receipt, idempotency, symlink-escape rejection, live guard, and
  truthful partial-failure reporting
- cleanup preview classifies candidates; cleanup execution is proven against
  an isolated generated fixture by `cleanup_execution_smoke.py`
- stale confirmation after API restart does not execute
- unrelated thread/session cannot execute the pending plan
- obvious secret markers are not present in executor results
