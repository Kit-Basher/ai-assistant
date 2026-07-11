# Executor Authorization Migration v1

Checkpoint base:

- Tag: `v0.2.2-universal-plan-mode-v1`
- Commit: `401b6e5e46b1581b02c8697cc1b3480d5af20361`

This checkpoint migrates the next low-to-medium-risk executor group onto the
central capability policy and Universal Mutation Plan v1 contract.

## Migrated Capabilities

Read-only inspection remains frictionless and must return `mutated=false`:

- `backup.inspect`
- `backup.validate`
- `restore.inspect`
- `support_bundle.inspect`
- `memory.inspect`

Mutating lanes now have explicit capability ids:

- `backup.create`
- `restore.execute`
- `support_bundle.create`
- `memory.forget`
- `memory.export`
- `memory.redact`
- `memory.compact`

Backup, restore, and support-bundle creation execute through Executor Registry
specs with trusted capability ids. Current memory lifecycle execution remains
preview-only where no destructive executor is implemented, but the mutation
lanes are now classified and mapped to Universal Plan capabilities.

## Backup v1

`operator.backup.v1` is bound to `backup.create`.

The Plan records the Backup v1 format, destination root, generated artifact
class, included summary categories, excluded secret/private categories, source
fingerprint inputs, expected filesystem writes, validation steps, and receipt
behavior.

Backup creation writes only bounded redacted summaries. Raw secrets,
unrestricted logs, arbitrary home files, model caches, and untrusted pack source
text remain excluded. Backup validation and listing remain read-only.

## Restore v1

`operator.restore.v1` is bound to `restore.execute`.

Restore remains limited to validated Backup v1 artifacts and allowlisted
non-secret preferences. Confirmation revalidates the backup path, manifest,
checksums, fingerprint, selected categories, current target state, rollback
snapshot destination, policy, and operation conflicts. Restore creates a
pre-restore safety snapshot before applying state and verifies the result.

Duplicate confirmation must not apply a restore twice. Changed backup manifests
or changed target fingerprints require a new Plan.

## Support Bundle

`operator.support_bundle.v1` is bound to `support_bundle.create`.

Support-bundle creation is a local filesystem mutation because it writes an
artifact. The default bundle includes version metadata, bounded redacted status
summaries, bounded receipt/journal summaries, and configuration presence/status.
It excludes raw tokens, secret-store contents, unrestricted environment,
arbitrary private files, raw conversation history, unrestricted database dumps,
and broad home-directory listings.

This checkpoint does not upload or externally send support bundles.

## Memory Lifecycle

Memory mutation lanes are split by capability:

- `memory.forget` for destructive forgetting/deletion;
- `memory.export` for writing a redacted memory export artifact;
- `memory.redact` for sensitive memory redaction;
- `memory.compact` for compaction into a shorter derived representation.

Read-only memory inspection remains immediate. Destructive memory execution is
not broadened in this checkpoint; preview-only lanes still fail closed with
`mutated=false` unless a trusted executor is explicitly implemented and
registered.

## Runtime Revalidation

Migrated mutations use Universal Mutation Plan v1 metadata and revalidate
runtime truth before execution:

- backup destination and source summaries;
- restore manifest, checksum/fingerprint, target state, and rollback snapshot;
- support-bundle destination and redaction boundaries;
- memory target scope and fingerprint where an executor exists.

If a target changes after preview, confirmation fails closed and requires a new
Plan.

## Bypass Prevention

Backup, restore, and support-bundle mutation helpers now require a trusted
invocation context issued by the Executor Registry. Direct calls without that
context return `generic_bypass_blocked` with `mutated=false` before creating
artifacts, locks, staging directories, or restore snapshots.

Trusted context fields are not accepted from user text and are not reusable
authorization tokens.

## Conflict Matrix

- Restore conflicts with another restore and with lifecycle operations that
  could change runtime/state boundaries.
- Backup may run during ordinary use, but not during uninstall finalization or
  state migration.
- Support-bundle creation may run during ordinary read-only activity, but its
  receipt summaries are bounded and redacted.
- Memory import/reset/forget/redact/compact conflict with other memory
  mutations. Memory inspection remains allowed.

## Receipts

Migrated Executor Registry results include common authorization and Plan
metadata:

- capability id;
- executor id;
- policy schema version;
- Mutation Plan schema version;
- Plan id and fingerprint;
- authorization mode and risk;
- target fingerprint;
- authorization decision id;
- confirmation timestamp;
- execution outcome.

Executor-specific details include backup artifact paths, restore categories and
snapshots, support-bundle artifact paths and redaction policy, or memory
mutation scope where implemented. Raw secrets and full memory contents are not
stored in receipts.

## Audit State

`scripts/capability_policy_audit.py` and
`scripts/universal_plan_mode_audit.py` no longer report backup, restore,
support bundle, or memory lifecycle as legacy migration warnings.

Expected remaining warnings:

- files;
- Git;
- service control;
- communications;
- broader skill-pack mutation paths.

The later Files, Git, and Service Mutation Migration v1 checkpoint supersedes
the first three warnings. The active expected warnings are communications and
broader skill-pack mutation paths.

## Proof

Focused proof:

```bash
python scripts/executor_authorization_migration_smoke.py
```

The smoke uses isolated temp fixtures for backup/restore/support-bundle and
does not overwrite live memory or live Personal Agent state.

Historical expected shape for this checkpoint:

```text
PASS=<n> WARN=5 FAIL=0
```

The five warnings are the explicitly remaining legacy mutation areas listed
above.

## Limitations

This is not final v0.2.2 authorization maturity. Remaining batches are:

1. Files/Git/Service Mutation Migration.
2. Communications Mutation Migration.
3. Generic Tool Bypass Hardening.
4. Full Adversarial Authorization Proof.
5. Final v0.2.2 or v0.3.0 decision.
