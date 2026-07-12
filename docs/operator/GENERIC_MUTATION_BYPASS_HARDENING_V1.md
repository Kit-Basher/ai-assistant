# Generic Mutation Bypass Hardening v1

Checkpoint truth:

- Tag: `v0.2.2-skill-pack-permission-boundary-v1`
- Commit: `71e25f3199b542acacabf529ce6c1ad7ccca9382`

Generic Mutation Bypass Hardening v1 reviews supported mutation surfaces after
first-party and skill-pack mutation migration. The goal is to prove that
supported host or external mutation cannot occur outside central capability
policy, Universal Mutation Plans, trusted invocation context, Executor Registry
bindings, runtime revalidation, and durable receipt/status truth.

This is not process isolation for arbitrary malicious in-process Python.

## Mutation Inventory

Reviewed mutation-capable surfaces:

- `agent/executor_registry.py`: migrated executor primitives, trusted-context
  issuance, receipt persistence, direct helper denial.
- `agent/capability_policy.py`: capability registry, authorization decisions,
  trusted context schema/validation.
- `agent/mutation_boundary.py`: primitive classification and common mutation
  assertion/denial helpers.
- `agent/mutation_plan.py`: Universal Plan persistence, expiry, cancellation,
  and fingerprinting.
- `agent/skill_pack_permissions.py`: manifest validation, grant store, and
  brokered dispatch.
- `agent/llm/notify_delivery.py`: provider delivery adapters requiring trusted
  context.
- `agent/api_server.py`: API/control-plane handlers, notification autopilot,
  managed service routes, and model/support routes.
- `agent/services/managed_local_services.py`: managed SearXNG setup/repair
  with fixed commands and loopback policy.
- `agent/host_lifecycle.py` and lifecycle runner scripts: update/uninstall
  operation records and runner handoff.
- `agent/primary_uninstall_policy.py`: strict local activation marker
  management.
- memory, semantic-memory, action-ledger, logging, and modelops stores:
  domain/internal persistence; user-facing destructive lanes are Plan-gated,
  preview-only, or unsupported.
- scripts under `scripts/`: operator/test/fixture proof surfaces, not runtime
  API entrypoints.

Static scanner output records suspicious subprocess, filesystem, SQL, HTTP,
systemctl, Git, secret, executor-registry, and trusted-context sites and fails
on unreviewed critical findings or runtime `shell=True`.

## Primitive Policy

`agent/mutation_boundary.py` defines primitive classes:

- `READ_ONLY`
- `INTERNAL_STATE_MUTATION`
- `LOCAL_FILESYSTEM_MUTATION`
- `HOST_CONTROL_MUTATION`
- `EXTERNAL_MUTATION`
- `SECURITY_SENSITIVE_MUTATION`
- `DENIED_PRIMITIVE`

Each registered primitive records allowed caller types, trusted-context
requirement, capability, executor, target binding, receipt requirement, and
whether direct use is prohibited.

Denied primitives include generic shell mutation, generic HTTP mutation, raw
domain DB mutation, and raw secret read.

## Trusted Context

Trusted invocation context now binds:

- capability id;
- executor id;
- authorization decision id;
- operation id;
- Plan fingerprint;
- target fingerprint;
- policy version;
- caller type and caller id;
- source module and source surface;
- issued and expiry timestamps;
- single-use/consumed state;
- parent operation id;
- skill-pack id/version/fingerprint, permission id, and grant id where
  applicable.

Validation rejects missing context, invalid caller types, fixture contexts in
production mode, wrong capability/executor/operation/target, stale Plan
fingerprint, consumed context, expired context, and policy-version drift.

## Boundaries

Shell/subprocess: direct package mutation remains blocked without trusted
context. Runtime code must use fixed executables, fixed argument schemas, no
arbitrary shell strings, and no user-controlled environment/cwd for privileged
commands. Static audit fails runtime `shell=True`.

Git: direct commit/push/reset/clean style mutation is blocked outside bounded
executors. Git commit is Plan-bound to repository and staged diff. Git push is
classified and force push remains denied.

Filesystem: bounded file create/modify/delete executors require approved roots,
target fingerprints, symlink rejection, staged writes/deletes, and receipts.
Internal receipt/Plan/grant/log persistence is classified as internal
bookkeeping tied to a parent operation or system event.

Database: user-visible domain mutation must go through authorized domain
services or migrated executors. Raw SQL mutation is not exposed through public
API or skill payloads. Internal stores keep fixed schemas and paths.

Secrets: raw secret reads are denied through platform APIs. Provider adapters
may use secrets internally but expose only presence/status and redacted errors.
Secret mutation is not exposed as a generic capability.

Network/HTTP: arbitrary HTTP mutation is denied. Provider mutation uses fixed
adapters/executors. Read-only network probes are bounded, timeout-limited, and
domain/loopback constrained where applicable.

API/control plane: untrusted request payloads cannot supply capability ids,
executor ids, trusted contexts, caller types, grant ids, or fixture flags as
authorization. Confirmation remains Plan-bound.

Background tasks: background code cannot fabricate confirmations or retain
reusable trusted contexts. Background mutation without a safe policy remains
unsupported.

Debug/fixture: fixture contexts are rejected in production-mode validation.
Fixture roots are isolated in smoke/proof scripts.

Registries: Executor Registry now rejects duplicate action types, duplicate
executor ids, and registration after freeze. Orchestrator freezes the runtime
registry after trusted startup registration.

## Static Audit

Run:

```bash
python scripts/generic_mutation_bypass_audit.py
```

The audit scans runtime and script Python for subprocess, shell, filesystem,
SQL, HTTP mutation, provider, systemctl, Git, secret, registry, and trusted
context patterns. It reports reviewed categories and fails on unreviewed
critical mutation sites.

Regex findings are not treated as proof by themselves; reviewed inventory and
dynamic smoke must also pass.

## Dynamic Smoke

Run:

```bash
python scripts/generic_mutation_bypass_smoke.py
```

The smoke uses isolated fixtures and proves:

- direct file/Git/service helpers deny mutation;
- direct provider send denies mutation;
- shell package mutation denies without trusted context;
- raw DB, raw secret, generic HTTP, and generic shell primitives deny;
- expired, consumed, fixture, wrong-target, and wrong-operation contexts deny;
- registry mutation after freeze denies;
- API-supplied capability/executor/context override is ignored when the
  installed API is reachable;
- denials return `mutated=false`;
- process-isolation limitation is reported honestly.

## Remaining Risks

- Arbitrary malicious in-process Python is not isolated.
- Static scanner reviewed baseline must be maintained when new mutation-capable
  files are added.
- Some older operator/control-plane flows remain supported legacy operator
  surfaces; they are not exposed as generic assistant mutation APIs.
- Unsupported destructive variants stay denied or deferred until bounded
  capability/executor/proof exists.

Recommended checkpoint tag:

`v0.2.2-generic-mutation-bypass-hardening-v1`
