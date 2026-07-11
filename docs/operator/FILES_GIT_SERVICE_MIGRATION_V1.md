# Files, Git, And Service Mutation Migration v1

Checkpoint truth:

- Tag: `v0.2.2-executor-authorization-migration-v1`
- Commit: `e5e097b48761d1218c218bca079933ce92ad3f5e`

This checkpoint migrates bounded local-host mutation areas into Capability
Policy v1 and Universal Plan Mode v1. It does not grant arbitrary filesystem,
Git, shell, or service administration.

## Migrated Capabilities

Read-only inspection remains frictionless:

- `files.inspect`, `files.list`, `files.diff`
- `git.inspect`, `git.status`, `git.diff`, `git.log`
- `system.service.inspect`, `system.service.logs.inspect`

Mutating lanes now have central policy:

- `files.create`
- `files.modify`
- `files.delete`
- `git.commit`
- `git.push`
- `system.service.restart`

Denied or deferred high-risk lanes include force push, Git reset/clean, and
service disablement.

## File Policy

The migrated file executors are bounded:

- `operator.file.create.v1`
- `operator.file.modify.v1`
- `operator.file.delete.v1`

Targets must resolve under approved roots supplied by trusted code. The helper
rejects path traversal, pseudo-filesystems, symlink targets or parents, and
non-regular file mutation. Modify and delete revalidate the expected content
hash. Writes use a temporary sibling and atomic replace. Overwrites preserve a
rollback copy. Delete moves the exact file into a staging directory rather than
permanently unlinking it.

Recursive delete, broad glob expansion, chmod/chown, archive extraction,
arbitrary home-directory mutation, and secret/state-root mutation are not part
of this checkpoint.

## Git Policy

The migrated Git executor is:

- `operator.git.commit.v1`

It commits an already staged diff in an approved repository. The Plan binds the
repository root and staged diff fingerprint, and confirmation revalidates the
fingerprint before `git commit` runs. The executor does not stage extra files,
does not accept arbitrary Git options, and records old/new HEAD and diff
metadata in the receipt.

`operator.git.push.v1` is registered and classified as an external side-effect
lane, but remote push execution remains disabled in this checkpoint. Force push
is denied. Reset, clean, rebase, merge, branch delete, remote mutation, and tag
mutation remain deferred unless a later batch adds bounded executors.

## Service Policy

The migrated service executor is:

- `operator.service.restart.v1`

It is limited to exact allowlisted fixture services and fixture state roots.
The proof does not stop, disable, restart, or uninstall the primary Personal
Agent API service. Unknown services, protected services, and arbitrary
`systemctl` shell calls are blocked.

## Universal Plan Behavior

All migrated mutations require a Universal Mutation Plan with:

- capability id;
- executor id;
- policy schema version;
- target snapshot and fingerprint;
- mutation inventory;
- recovery truth;
- Plan fingerprint;
- confirmation binding;
- runtime revalidation;
- receipt metadata.

Inspection and previews return `mutated=false`.

## Bypass Protection

Dangerous lower-level file, Git, and service helpers require trusted invocation
context issued by the Executor Registry after central authorization. Generic
shell `git` and `systemctl` mutation paths are blocked. User text cannot supply
or replace capability ids, executor ids, Plan fingerprints, or trusted context.

## Conflict Matrix

- File mutation conflicts with changed hashes, symlink replacement, protected
  roots, and pseudo-filesystem targets.
- Git commit conflicts with changed staged diff, changed repository target, and
  absent staged changes.
- Git push is treated as an external side effect and remains non-mutating in
  this checkpoint.
- Service restart conflicts with unknown services, protected services, and
  non-fixture service roots.
- Read-only file, Git, and service inspection remains allowed.

## Receipts And Status

Migrated receipts include capability id, executor id, policy schema version,
Mutation Plan schema version, Plan id/fingerprint, target fingerprint,
authorization decision id, confirmation timestamp, outcome, and
executor-specific before/after summaries. Receipts do not include raw secrets or
reusable authorization context.

## Proof

Run:

```bash
python scripts/files_git_service_migration_smoke.py
python scripts/universal_plan_mode_audit.py
python scripts/capability_policy_audit.py
python scripts/executor_registry_smoke.py
```

The smoke uses isolated temporary files, a temporary Git repository, and a
fixture service state root. It does not mutate real repository history, does not
run a real push, and does not stop or disable the primary Personal Agent
service.

## Remaining Legacy Areas

The expected remaining authorization migration warnings are:

- communications;
- broader skill-pack mutation paths.

Recommended checkpoint tag:

`v0.2.2-files-git-service-migration-v1`
