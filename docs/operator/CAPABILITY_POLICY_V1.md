# Capability Policy v1

Checkpoint truth:

- Tag: `v0.2.2-executor-authorization-migration-v1`
- Commit: `e5e097b48761d1218c218bca079933ce92ad3f5e`

Capability Policy v1 is the first foundation checkpoint for centrally enforced
assistant tool authorization. It does not migrate every executor. It defines the
schema, registry, decision model, representative executor bindings, receipt
metadata, and audit surface needed for the broader authorization roadmap.

## Schema

Capability definitions live in `agent/capability_policy.py` and use
`schema_version=1`.

Required fields:

- `capability_id`
- `title`
- `effect`: `read_only` or `mutating`
- `scope`: `local_process`, `local_filesystem`, `local_host`, or
  `external_service`
- `reversibility`: `reversible`, `conditionally_reversible`, or
  `irreversible`
- `risk_level`: `low`, `medium`, `high`, or `critical`
- `authorization_mode`: `allow`, `plan`, `plan_and_confirm`,
  `local_activation_and_confirm`, or `deny`
- `receipt_required`
- `runtime_revalidation_required`
- `target_binding_required`
- `external_side_effect`
- `generic_bypass_forbidden`
- `implementation_status`

Capability ids are stable, lowercase, dot-separated identifiers. Unknown enum
values, duplicate ids, malformed ids, critical weak authorization, and
read-only receipt inconsistencies fail validation.

## Registry

`build_default_capability_registry()` is the authoritative v1 registry. The
base policy is code-defined and version-controlled. Local overrides are not yet
implemented; future overrides may only make defaults stricter.

Migrated read-only capabilities:

- `system.package.inspect`
- `system.service.inspect`
- `system.lifecycle.status`
- `backup.inspect`
- `backup.validate`
- `restore.inspect`
- `support_bundle.inspect`
- `memory.inspect`

Migrated mutating capabilities:

- `system.package.install`
- `cleanup.execute`
- `system.update`
- `system.uninstall`
- `backup.create`
- `restore.execute`
- `support_bundle.create`
- `memory.forget`
- `memory.export`
- `memory.redact`
- `memory.compact`
- `files.create`
- `files.modify`
- `files.delete`
- `git.commit`
- `git.push`
- `system.service.restart`

Legacy/unmigrated actions remain visible in audit output and are not claimed as
centrally enforced.

## Authorization Gate

`authorize_capability()` returns a structured decision with machine-readable
reason codes:

- `allowed`
- `read_only_allowed`
- `plan_required`
- `confirmation_required`
- `local_activation_required`
- `capability_denied`
- `capability_unknown`
- `capability_unimplemented`
- `stale_plan`
- `target_changed`
- `policy_changed`
- `activation_invalid`
- `conflicting_operation`
- `generic_bypass_blocked`

The gate does not mutate. It evaluates capability existence, implementation
status, authorization mode, Plan fingerprint, target fingerprint, confirmation
state, policy version, and local activation where required.

## Executor Binding

Migrated executor definitions declare trusted capability ids:

- `operator.cleanup.v1` -> `cleanup.execute`
- `operator.update.v1` -> `system.update`
- `operator.uninstall.v1` -> `system.uninstall`
- `operator.package.install.v1` -> `system.package.install`
- `operator.backup.v1` -> `backup.create`
- `operator.restore.v1` -> `restore.execute`
- `operator.support_bundle.v1` -> `support_bundle.create`
- memory lifecycle preview lanes -> `memory.forget`, `memory.export`,
  `memory.redact`, or `memory.compact`
- `operator.file.create.v1` -> `files.create`
- `operator.file.modify.v1` -> `files.modify`
- `operator.file.delete.v1` -> `files.delete`
- `operator.git.commit.v1` -> `git.commit`
- `operator.git.push.v1` -> `git.push`
- `operator.service.restart.v1` -> `system.service.restart`

Package install is now an Executor Registry executor. The confirmed Plan Mode
path gates `system.package.install`, creates trusted invocation context, and
then calls the shell install primitive through `operator.package.install.v1`.

Executor capability ids are trusted registration metadata. User text cannot
choose or replace them.

## Plan Mode

Canonical Plan Mode records now include capability metadata for migrated
actions:

- capability id and title;
- policy schema version;
- authorization mode;
- scope and reversibility;
- risk;
- target fingerprint;
- Plan fingerprint;
- Mutation Plan schema version for Universal Plan Mode migrated actions;
- receipt and runtime revalidation requirements;
- local activation requirement where applicable.

Confirmation fails closed when the policy version, target fingerprint, Plan
fingerprint, thread/session binding, expiry, or activation state no longer
matches.

## Lifecycle Mapping

`system.update` remains high-risk, Plan-and-confirm, runtime-revalidated,
receipt-required, and conditionally reversible through the existing Host
Lifecycle Runner and rollback protections.

`system.uninstall` remains critical, local-activation-and-confirm, preserve-data
only, receipt-required, and runtime-revalidated. The central gate does not
replace the strict primary uninstall marker validator. Purge remains
unsupported.

Isolated fixture and production-shaped proofs can still exercise the uninstall
executor without enabling active primary uninstall. Active primary uninstall
still requires the local marker policy.

## Generic Bypass Prevention

The shell package-install primitive and migrated backup, restore,
support-bundle, bounded file, Git, and service fixture mutation helpers now
require a trusted invocation context before actual mutation. A direct
lower-level call without that context returns
`generic_bypass_blocked` with `mutated=false`.

The v1 trusted invocation context contains:

- `capability_id`
- `executor_id`
- `authorization_decision_id`
- `plan_fingerprint`
- `operation_id`
- `policy_version`

The context is created by trusted orchestrator/registry code after the central
gate allows the confirmed Plan. It is not accepted from user text.

## Receipts

Migrated Executor Registry results and journals include:

- capability id;
- policy schema version;
- authorization mode;
- risk level;
- Plan fingerprint;
- target fingerprint;
- authorization decision id;
- confirmation timestamp.

Universal Plan Mode migrated receipts also include Mutation Plan schema version,
Plan id, Plan fingerprint, target fingerprint, authorization decision id, and
execution outcome. Update and uninstall host-lifecycle handoff records include a
bounded `capability_policy` summary. Uninstall receipts carry the same summary.
Raw prompts, secrets, tokens, and bearer strings remain redacted.

## Skill-Pack Boundary

External skill packs cannot define or downgrade core capability policy in v1.
Mutating pack tools without a recognized capability remain outside the central
enforcement claim and are audit-visible. Read-only pack behavior remains
subject to the existing sandbox and pack-governance rules.

## Audit Commands

Run:

```bash
python scripts/capability_policy_smoke.py
python scripts/capability_policy_audit.py
python scripts/universal_plan_mode_smoke.py
python scripts/universal_plan_mode_audit.py
```

The smoke proves registry load, schema validation, read-only allow, Plan and
confirmation requirements, local activation requirement, stale/changed Plan
blocking, shell bypass blocking, receipt metadata, status categories, and
unmigrated action reporting.

The audit reports registered capabilities, migrated executor bindings, policy
gaps, receipt requirements, bypass requirements, and legacy/unmigrated
warnings.

## Limitations

- Universal Plan Mode enforcement covers the currently migrated set, not every
  mutating assistant action.
- Implemented notification communications are migrated; email and calendar
  providers are not implemented and remain unsupported rather than routed
  through generic transports. Broad skill-pack mutation paths are not fully
  migrated.
  Unsupported destructive file, Git, and service-control variants remain
  denied or deferred until dedicated bounded executors exist.
- Generic bypass hardening is proven for migrated package install, lifecycle
  operations, backup, restore, support-bundle, bounded file, Git, service
  fixture, and notification helpers in this checkpoint; broader shell/
  subprocess brokers remain next-phase work.

Recommended next checkpoint:

`v0.2.2-universal-plan-mode-v1`
