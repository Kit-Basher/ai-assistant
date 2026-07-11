# Universal Plan Mode v1

Universal Plan Mode v1 is the shared mutation contract for newly migrated
mutating executors. It sits above executor-specific validation and below the
normal chat preview UX.

## Scope

Migrated in this checkpoint:

- `system.package.install` through `operator.package.install.v1`
- `cleanup.execute` through `operator.cleanup.v1`
- `system.update` through `operator.update.v1`
- `system.uninstall` through `operator.uninstall.v1`
- `backup.create` through `operator.backup.v1`
- `restore.execute` through `operator.restore.v1`
- `support_bundle.create` through `operator.support_bundle.v1`
- memory lifecycle preview lanes through `memory.forget`, `memory.export`,
  `memory.redact`, and `memory.compact`
- `files.create`, `files.modify`, and `files.delete` through bounded file
  executors
- `git.commit` through `operator.git.commit.v1`
- `git.push` through `operator.git.push.v1` as an external-side-effect-aware
  bounded policy lane; force push remains denied
- `system.service.restart` through `operator.service.restart.v1` for approved
  fixture services

Implemented notification communications now use Universal Plan metadata.
Email/calendar providers are unsupported in this checkpoint. The remaining
legacy/unmigrated mutation area remains broader skill-pack mutation paths.
Unsupported destructive file, Git, service, and provider variants remain
denied or deferred rather than silently allowed.

## Mutation Plan Schema

The security object is a structured `MutationPlan` with schema version `1`.
It includes:

- Plan id, capability id, executor id, policy version, authorization mode, risk,
  scope, and reversibility
- creation and expiry timestamps
- thread, session, and actor binding
- normalized target snapshot and target fingerprint
- mutation inventory, preserved resources, expected side effects, and recovery
  truth
- activation-policy fingerprint where required
- confirmation requirement
- receipt and runtime-revalidation requirements
- deterministic Plan fingerprint

Rendered preview text is not the authorization object. Display formatting can
change without changing the security fingerprint.

## Fingerprints

Fingerprints are calculated from canonical JSON with stable key ordering,
normalized timestamps, normalized paths, and no raw prompts or secrets.

The Plan fingerprint changes when any security-relevant field changes,
including capability, executor, policy version, target snapshot, mutation
inventory, preserved boundary, recovery truth, activation fingerprint,
thread/session, expiry, or confirmation mode.

## Plan Store

The Plan store supports save, load, cancellation, expiry, atomic status
transition, duplicate handling, bounded pruning, and optional JSON persistence
under runtime state. Status values are:

`pending`, `confirmed`, `executing`, `completed`, `failed`, `cancelled`,
`expired`, and `invalidated`.

## Confirmation

Confirmation is bound to the Plan id, Plan fingerprint, capability id, executor
id, thread/session/actor, expiry, policy version, target fingerprint, mutation
inventory, and activation fingerprint where required.

Failures return `mutated=false`. Examples include missing Plan, expired Plan,
cancelled Plan, thread/session mismatch, target change, policy change,
activation change, and fingerprint mismatch.

## Runtime Revalidation

Preview never guarantees later execution. Confirmation revalidates runtime truth
before mutation:

- package install refreshes package state and package-manager availability
- cleanup refreshes inventory and protected-resource exclusions
- update rechecks target commit, source, working tree, staged release, and
  runner readiness
- uninstall rechecks local activation, preserve-data inventory, final backup
  readiness, and runner readiness
- backup rechecks destination and source summary boundaries
- restore rechecks the Backup v1 manifest, checksum/fingerprint, selected
  categories, target state, and safety snapshot destination
- support bundle rechecks local artifact creation and redaction boundaries
- memory mutation lanes recheck target scope and fingerprints where an executor
  is implemented
- file mutation rechecks canonical path, approved root, regular-file type,
  symlink status, expected content hash, and staged rollback/deletion behavior
- Git commit rechecks repository root, staged diff fingerprint, HEAD, and
  command class; Git push rechecks repository and refuses force/external push
  mutation in this checkpoint
- service restart rechecks the exact allowlisted fixture service and writes
  only fixture state

Lifecycle-specific operation records remain separate from the conversational
Mutation Plan.

## Backup, Restore, Support Bundle, And Memory

Executor Authorization Migration v1 extends this contract to Backup v1, Restore
v1, support-bundle creation, and memory lifecycle mutation classification.
Backup, restore, and support-bundle execution now require a trusted invocation
context issued by the Executor Registry. Direct lower-level calls fail with
`generic_bypass_blocked` and `mutated=false`.

Memory lifecycle destructive execution remains preview-only where no executor
exists, but the mutation lanes now have stable capabilities and Universal Plan
metadata.

## Files, Git, And Services

Files, Git, and Service Mutation Migration v1 extends this contract to bounded
local-host mutation fixtures:

- file create/modify/delete use exact paths, approved roots, symlink rejection,
  expected hashes, rollback copies, and deletion staging;
- Git commit uses a repository target model and staged diff fingerprint;
- Git push is classified as an external side effect and force push is denied in
  this checkpoint;
- service restart is limited to approved fixture services and does not stop,
  disable, or restart the primary Personal Agent service.

Direct lower-level shell `git` and `systemctl` mutation attempts remain blocked.

## Package Executor

Package install is now an Executor Registry executor:

`package.install` -> `operator.package.install.v1` -> `system.package.install`

Inspection remains read-only. Preview creates a Universal Mutation Plan with the
normalized package name, current package state, dependency warning, capability
metadata, and receipt behavior. Confirmation dispatches through the Executor
Registry and uses the shell package primitive only with trusted invocation
context. Already-installed packages are treated as no-op verified completions.

The shell primitive rejects direct package mutation without trusted context.

## Receipts

Migrated mutating receipts include common policy metadata:

- capability id
- executor id
- policy schema version
- Mutation Plan schema version
- Plan id and Plan fingerprint
- authorization mode and risk
- target fingerprint
- authorization decision id
- confirmation timestamp
- execution outcome

Executor-specific receipt content remains allowed. Historical receipts are not
rewritten.

## Cancellation, Expiry, And Duplicates

Cancellation marks the Plan cancelled and prevents later confirmation. Expired
Plans require a fresh preview. Duplicate confirmation returns current/prior
status and must not repeat mutation.

Multiple pending Plans are resolved within the same thread/session. Ambiguous
confirmation must not guess.

## Proof

Run:

```bash
python scripts/universal_plan_mode_smoke.py
python scripts/universal_plan_mode_audit.py
python scripts/executor_authorization_migration_smoke.py
python scripts/capability_policy_smoke.py
python scripts/capability_policy_audit.py
python scripts/plan_mode_v2_smoke.py
```

The Universal Plan smoke is non-destructive. It does not enable primary
uninstall, does not uninstall the primary installation, and does not install a
new package.

Recommended checkpoint tag:

`v0.2.2-universal-plan-mode-v1`
