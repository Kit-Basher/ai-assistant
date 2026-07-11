# Executor Registry v1

Executor Registry v1 is the narrow apply boundary behind Plan Mode v2. Plan
Mode still owns preview, user confirmation, expiry, and thread/session binding.
The registry owns executor lookup, refusal for non-enabled executors, canonical
result fields, and an append-only redacted mutation journal.

Capability Policy v1 adds central authorization metadata for migrated
executors. Universal Plan Mode v1 adds a shared Mutation Plan security object,
fingerprint, and receipt metadata for the migrated mutating set. See
`docs/operator/CAPABILITY_POLICY_V1.md` and
`docs/operator/UNIVERSAL_PLAN_MODE_V1.md`.

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
- `capability_id`
- `policy_schema_version`
- `authorization_mode`
- `risk_level`
- `plan_fingerprint`
- `target_fingerprint`
- `authorization_decision_id`
- `confirmation_timestamp`
- `mutation_plan_schema_version`

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
- Migrated mutating executors call the central capability gate before mutation.
  A denied decision returns `mutated=false` and is journaled.
- Executor capability ids come from trusted registration code, not user text.
- Migrated mutating executors require a valid Universal Mutation Plan before
  mutation. If a legacy canonical plan reaches the registry for a migrated
  action, the registry synthesizes and validates the v1 Mutation Plan before
  authorization.

## Journal

The registry writes append-only JSONL records under runtime state:

`executor_registry_journal.jsonl`

The journal redacts token, secret, password, bearer, API key, and confirmation
token fields. It should not contain raw secrets, raw support logs, or private
unredacted payloads.

## Enabled Executors

Current enabled executors:

- `package.install` via `operator.package.install.v1`
- `operator.support_bundle` via `operator.support_bundle.v1`
- `operator.backup` via `operator.backup.v1`
- `operator.cleanup` via `operator.cleanup.v1`
- `operator.restore` via `operator.restore.v1`
- `operator.update` via `operator.update.v1`
- `operator.uninstall` via `operator.uninstall.v1`
- `operator.file.create` via `operator.file.create.v1`
- `operator.file.modify` via `operator.file.modify.v1`
- `operator.file.delete` via `operator.file.delete.v1`
- `operator.git.commit` via `operator.git.commit.v1`
- `operator.git.push` via `operator.git.push.v1`
- `operator.service.restart` via `operator.service.restart.v1`

`package.install` installs an allowlisted Debian package through the shell
package primitive only after Plan Mode confirmation, capability authorization,
and trusted invocation context creation. Already-installed packages return a
verified no-op receipt. It is bound to `system.package.install`.

`operator.support_bundle` creates a Support Bundle v2 diagnostics package in a
temporary directory. It is additive and returns a `journal_id`. Rollback is
limited to removing the newly created temporary support bundle directory. See
`docs/operator/SUPPORT_BUNDLE.md`. It is bound to `support_bundle.create` and
requires trusted invocation context before the helper writes an artifact.

`operator.backup` creates a Backup v1 directory under the approved Personal
Agent local backup path. It is additive and returns a `journal_id`. Rollback is
limited to removing the newly created backup directory. See
`docs/operator/BACKUP_V1.md`. It is bound to `backup.create` and requires
trusted invocation context before the helper writes an artifact.

`operator.cleanup` deletes only preview-identified, revalidated, owned
Personal Agent artifacts such as generated cleanup fixtures, old backups, old
support bundles, and old runtime releases. It preserves current runtime, active
service files, secret stores, latest valid backup, and arbitrary user files.
It is bound to `cleanup.execute`.

`operator.restore` restores only validated Backup v1 allowlisted non-secret
preference values for system-resource baselines/context. It stages content,
creates a pre-restore safety snapshot, verifies live state after apply, and
attempts rollback from the snapshot on post-mutation failure. It does not
restore secrets, logs, arbitrary files, model caches, runtime releases, or
untrusted executable/pack content. It is bound to `restore.execute` and
requires trusted invocation context before the helper creates restore locks,
staging directories, snapshots, or state changes.

`operator.update` is enabled for bounded Update v1 outcomes: isolated
staged-release fixture promotion/rollback proof through Host Lifecycle Runner
v1, verified live no-op, and structured live-promotion blockers when
preconditions are not rollback-safe.
It rejects arbitrary repositories, branches, commits, scripts, URLs, dirty
working trees, and target drift after preview. See
`docs/operator/UPDATE_EXECUTOR_V1.md`. It is bound to `system.update`.

`operator.uninstall` is enabled for Uninstall v1 preserve-data execution against
approved isolated fixtures through Host Lifecycle Runner v1 and for guarded
live daily-driver refusal. The executor requires a final safety backup, exact
owned-resource allowlists, target-snapshot validation, and a durable uninstall
receipt. Live daily-driver uninstall remains blocked unless the target is an
approved isolated fixture; it does not stop services or remove runtime files
from ordinary installed-product proofs. See
`docs/operator/UNINSTALL_EXECUTOR_V1.md`. It is bound to `system.uninstall`.

`operator.file.create`, `operator.file.modify`, and `operator.file.delete`
perform bounded local-file fixture mutations only after Universal Plan
authorization. They require canonical paths under approved roots, reject
symlink targets and pseudo-filesystems, enforce expected hashes for modify and
delete, write through temporary siblings, preserve rollback copies for
overwrites, and stage deletions instead of permanent unlink. They are bound to
`files.create`, `files.modify`, and `files.delete`.

`operator.git.commit` commits an already staged diff in an approved repository
after the staged diff fingerprint is revalidated. It does not stage files
implicitly and does not accept arbitrary Git options. `operator.git.push`
records the external-side-effect policy boundary and denies force push; actual
remote push execution remains deferred until a dedicated remote proof exists.
They are bound to `git.commit` and `git.push`.

`operator.service.restart` is limited to exact allowlisted fixture services and
fixture state roots. It does not restart, stop, disable, or mutate the primary
Personal Agent service during proofs. It is bound to
`system.service.restart`.

## Preview-Only Lanes

These remain preview-only in v1:

- memory delete/export/redact/dedupe/control executors remain preview-only
  where no destructive executor is implemented, but mutation lanes are now
  classified as `memory.forget`, `memory.export`, `memory.redact`, or
  `memory.compact`
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
python scripts/capability_policy_smoke.py
python scripts/capability_policy_audit.py
python scripts/universal_plan_mode_smoke.py
python scripts/universal_plan_mode_audit.py
python scripts/executor_authorization_migration_smoke.py
python scripts/files_git_service_migration_smoke.py
python scripts/executor_registry_smoke.py
python scripts/support_bundle_v2_smoke.py
python scripts/backup_v1_smoke.py
python scripts/cleanup_execution_smoke.py
python scripts/restore_execution_smoke.py
python scripts/host_lifecycle_runner_smoke.py
python scripts/host_lifecycle_systemd_smoke.py
python scripts/active_host_enablement_smoke.py
python scripts/update_execution_smoke.py
python scripts/uninstall_execution_smoke.py
```

The smoke talks to the installed `/chat` API and proves:

- preview-only memory delete cannot execute
- live daily-driver uninstall confirmation is guarded with `mutated=false`
- cleanup has an enabled executor plan and can be cancelled without mutation
- support bundle has an enabled executor and returns a journal id
- support bundle creates only a redacted temporary diagnostics artifact
- support bundle, backup, and restore helpers block direct calls without
  trusted invocation context
- Support Bundle v2 writes a manifest and bounded summary files
- backup has an enabled executor and returns a journal id
- package install is an enabled executor path, while direct shell package
  mutation remains blocked without trusted context
- bounded file create/modify/delete use Universal Plan metadata, approved
  roots, symlink rejection, rollback copies, deletion staging, and receipt
  metadata
- Git commit uses Universal Plan metadata and staged diff fingerprinting;
  direct shell Git mutation is blocked and force push is denied
- service restart uses an approved fixture service only; direct shell
  `systemctl` mutation is blocked
- notification local-send, Telegram fixture send, mark-read, and prune use
  Universal Plan metadata; direct notification provider delivery is blocked
  without trusted invocation context, and real email/calendar providers are not
  implemented in this checkpoint
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
