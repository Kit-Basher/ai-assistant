# Skill-Pack Permission Boundary v1

Audit 3 UX rule: present each permission first as its plain-language consequence
(what the pack may read, change, contact, or send), with the exact technical
scope available as detail. Installation, review approval, enablement, and
permission grant remain separate; external packs start with zero permissions
and cannot approve or grant themselves.

Checkpoint truth:

- Tag: `v0.2.2-communications-migration-v1`
- Commit: `24bfca5436d2e2916c02ddad181397083a8979d3`

Skill-Pack Permission Boundary v1 prevents installed skill packs from receiving
privileged mutation authority merely because they are installed or invoked.
Skill-pack requests for Personal Agent platform APIs now pass through identity,
manifest, grant, capability, Universal Plan, trusted-context, and receipt
checks.

## Architecture Found

Native skills are discovered by `agent/skills_loader.py`. A native skill can
load a `manifest.json` and `handler.py`; handler modules are imported
in-process. External packs are tracked by the pack store and existing pack
policy, and managed adapters expose bounded dry-run style surfaces.

This checkpoint enforces the Personal Agent platform API boundary. It does not
claim sandbox isolation against arbitrary malicious Python loaded in-process.
External untrusted code execution remains outside the supported guarantee.

## Identity Model

`agent/skill_pack_permissions.py` defines `SkillPackIdentity`:

- `skill_pack_id`
- `publisher_id`
- `package_name`
- `version`
- `manifest_version`
- `install_source`
- `install_path`
- `content_fingerprint`
- `signature_status`
- `bundled_or_external`
- `enabled`

Security decisions use the stable id, publisher, version, and content
fingerprint from trusted runtime state. Display names are not security
identities. Symlinked install roots are rejected.

## Manifest Schema

The v1 manifest validator requires:

- `schema_version: 1`
- `skill_pack_id`
- `publisher_id`
- `name`
- `version`
- `declared_permissions`
- optional `entrypoints`, `read_only_surfaces`, `network_domains`,
  `filesystem_roots`, `provider_accounts`, `background_tasks`, and
  `configuration_schema`

The validator rejects unknown permissions, duplicate permissions, wildcard
permissions, wildcard domains, broad filesystem roots such as `/` or the whole
home directory, undeclared background-task permissions, mismatched skill ids,
malformed schemas, and symlinked install roots.

## Permission Registry

The permission registry maps narrow skill-pack permissions to fixed platform
capabilities and executors where mutation exists. Current permissions include:

- `read.notifications.inspect`
- `read.files.inspect`
- `read.git.inspect`
- `invoke.notification.local.send`
- `invoke.notification.external.send`
- `invoke.files.create`
- `invoke.backup.create`
- `invoke.git.commit`

There is no `shell.execute`, `http.post`, `network.all`, `filesystem.all`,
`system.full_access`, or raw secret permission.

## Declared, Granted, Effective

Skill-pack authority has three layers:

- Declared permissions: what the manifest requests.
- Granted permissions: durable operator grants for that exact identity,
  version, content fingerprint, permission, and target scope.
- Effective permissions: declared intersected with granted, platform policy,
  runtime availability, enabled state, and target-scope checks.

Unknown, undeclared, ungranted, expired, revoked, version-mismatched,
fingerprint-mismatched, or scope-expanded permissions fail closed.

## Grant Storage

`SkillGrantStore` stores JSON grants atomically with schema version 1. Grants
include:

- `grant_id`
- skill id and publisher
- version and content fingerprint
- permission id
- target scope
- grant time and grant source
- optional expiry
- optional revocation timestamp
- reason

Grant mutation itself is capability-gated through:

- `skill_pack.permission.grant`
- `skill_pack.permission.revoke`

Fixture executors `operator.skill_pack.permission.grant.v1` and
`operator.skill_pack.permission.revoke.v1` prove the flow using stores under
`/tmp`; production grant UX remains bounded by the same Plan contract.

## Invocation Broker

`SkillPackInvocationBroker` is the platform API entry point for skill-pack
privileged actions. It:

- resolves trusted skill identity;
- validates the manifest and fingerprint;
- checks enabled state;
- checks declaration and effective grant;
- maps permission to a fixed capability/action/executor;
- creates a Universal Mutation Plan for mutation;
- injects skill-pack provenance into trusted invocation context;
- dispatches through the Executor Registry;
- returns bounded denial or result fields.

Skill payloads cannot select raw capabilities, executors, or trusted context
fields.

## Read-Only Behavior

Read-only skill-pack inspection remains frictionless after declaration and
grant checks. It returns `mutated=false`, is size-bounded by the caller surface,
and must not perform hidden provider writes, repairs, refreshes that change
state, or durable mutation.

## Mutation Behavior

Skill-pack-triggered mutation requires:

- declared permission;
- active grant;
- target-scope match;
- central capability gate;
- Universal Mutation Plan;
- confirmation where the mapped capability requires it;
- runtime revalidation;
- trusted invocation context;
- fixed executor dispatch;
- durable receipt.

Plans and results identify the requesting skill pack and grant. Duplicate
confirmation is bounded by operation id, Plan fingerprint, target fingerprint,
executor, skill identity, and grant. Completed duplicates return prior state or
a bounded no-op rather than repeating mutation.

## Target-Scoped Grants

Grants can narrow targets, including:

- file roots and maximum bytes;
- fixed notification destinations;
- exact repositories or services where supported;
- backup-only without restore;
- read-only memory or export-only memory.

Scope expansion after update requires a new grant. New permissions after update
are reported as `newly_requested` and are not inherited automatically.

## Background Tasks

Background tasks must be declared in the manifest and reference a declared
permission. Background mutation is denied unless the permission explicitly
allows it and a dedicated grant exists. No background task can retain reusable
confirmation tokens.

## Boundaries

Secrets: skill packs do not receive raw secret-store access through platform
APIs. Provider adapters return safe ids/status, not tokens or authorization
headers.

Filesystem/import: platform APIs reject broad roots and symlinked install
roots. This does not isolate arbitrary in-process Python imports; that is a
future process-isolation problem.

Network: no arbitrary HTTP mutation permission exists. Provider-style mutation
must use configured adapters and fixed endpoint classes.

Shell/subprocess: no shell permission exists. Fixed executors may use
subprocess internally only through trusted invocation context.

## Receipts And Audit

Skill-pack mutation receipts include skill id, publisher, version, content
fingerprint, permission id, grant id, capability, executor, Plan metadata,
target summary, result, and mutation truth. Receipts must not include raw
secrets, authorization headers, reusable trusted context, or full sensitive
payloads.

The audit view is based on registry, grant, Plan, operation, and receipt truth,
not chat memory.

## Generic Bypass Hardening

Generic Mutation Bypass Hardening v1 extends the same boundary below the broker:
trusted invocation context now binds caller provenance, operation id, target
fingerprint, expiry, and single-use state. Direct platform helper calls,
arbitrary shell, arbitrary HTTP mutation, raw domain DB mutation, raw secret
read, copied contexts, and registry mutation after freeze are denied through
supported platform APIs.

## Proof

Run:

```bash
python scripts/skill_pack_permission_boundary_smoke.py
python scripts/capability_policy_audit.py
python scripts/universal_plan_mode_audit.py
python scripts/capability_policy_smoke.py
python scripts/universal_plan_mode_smoke.py
python scripts/full_adversarial_authorization_proof.py
```

The smoke uses fixture skill packs only. It does not load untrusted external
code into the primary runtime, does not grant real external account access, and
does not enable primary uninstall.

Expected smoke classification:

- `FAIL=0`
- `WARN=1` for the documented in-process Python isolation limitation

Full Adversarial Authorization Proof v1 adds cross-skill and cross-grant
attacks: undeclared permissions, ungranted permissions, revoked grants, scope
expansion, update-requested permissions, copied authority fields, and direct
primitive access all fail closed through supported platform APIs.

## Remaining Limitations

- This phase closes Personal Agent platform API permission bypasses for
  skill-pack-triggered privileged actions.
- It does not provide OS/process sandboxing for arbitrary malicious in-process
  Python.
- External untrusted skill-pack code execution remains unsupported.
- Future work should add process isolation.

Recommended checkpoint tag:

`v0.2.2-skill-pack-permission-boundary-v1`

## Audit v2B per-invocation continuation

Mutation requests now stop at a persisted preview. Confirmation binds actor,
thread, session, pack identity/version/content fingerprint, permission, grant,
capability/executor, arguments fingerprint, target fingerprint, and expiry.
Apply re-reads the manifest and effective grant, rejects drift, and uses the
Executor Registry once. Cancel is scope-bound. Read-only inspection remains
immediate; external packs have zero grants by default and executable foreign
packs remain unsupported.
