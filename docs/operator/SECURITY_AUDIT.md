# Security Audit

Date: 2026-06-20
Checkpoint scope: `v0.2.0-chat-reliability-harness` plus release-readiness hardening pass.

This is an operator-focused security boundary audit. It records what is enforced
now, what remains release-blocking, and what must stay out of normal assistant
behavior.

## Current Rating

Yellow for controlled local trial. No new public release claim is made until the
fresh Debian VM install proof passes.

Architecture and Safety Audit v2 adds an important qualification: central
authorization coverage is incomplete. Migrated Executor Registry actions now
require an existing Universal Mutation Plan and durable, scope-bound,
expiring, single-use confirmation metadata. After v2E, the 94-surface inventory
records ten `legacy_unmigrated` and seven older domain-confirmed pack/search
surfaces; see `ARCHITECTURE_SAFETY_AUDIT_V2E.md`. These remaining surfaces must
not be represented as covered by the central capability/executor contract.

## Enforced Boundaries

- Secrets/tokens: doctor, status, support bundle, Telegram setup, provider/API
  key setup, and managed-action journals are expected to redact raw tokens,
  API keys, private paths where applicable, hostile pack text, and SearXNG
  `server.secret_key`.
- Plan Mode: known mutators require plan, preview, confirmation token, expiry,
  apply validation, journaling, verification, and scoped rollback where
  supported. Unknown operations default to mutating unless explicitly
  classified read-only.
- Search: SearXNG search is metadata-only. The assistant must not fetch pages,
  run JavaScript, download files, import packs, or treat result metadata as
  trusted fact. Search remains disabled until a trusted endpoint or managed
  loopback SearXNG setup is explicitly configured.
- External packs and skill packs: text packs are quarantined, normalized,
  reviewed, approved, enabled, and permission-gated. Skill-Pack Permission
  Boundary v1 enforces skill identity, manifest permissions, durable grants,
  target scopes, brokered Executor Registry dispatch, and trusted invocation
  context for Personal Agent platform APIs. External pack code execution is not
  supported. Search results cannot install/import packs directly.
- Managed SearXNG: the approved container image is
  `docker.io/searxng/searxng:latest`; bind is loopback-only; config mount is
  owned and seeded; empty config mounts and arbitrary volume paths are rejected;
  Docker is fallback-only; rootless Podman is preferred. Verified setup may
  persist only a loopback SearXNG runtime config; non-loopback persisted search
  config is rejected as untrusted and does not enable search.
- Shell/package/system mutations: normal chat may preview bounded package or
  managed-service actions only. It must not expose arbitrary shell, arbitrary
  Podman/Docker, host networking, broad filesystem writes, or hidden sudo.
  Capability Policy v1 centrally gates package install, cleanup, update, and
  uninstall. Universal Plan Mode v1 adds a shared Mutation Plan schema,
  fingerprint, confirmation binding, cancellation/expiry handling, and receipt
  metadata for the same migrated set. The package-install shell primitive
  rejects direct mutation without a trusted invocation context from a confirmed
  Executor Registry dispatch.
- Generic mutation bypass hardening: trusted invocation context now binds
  caller provenance, operation id, Plan fingerprint, target fingerprint, expiry,
  single-use/consumed state, and skill grant metadata where applicable.
  Executor Registry registration is frozen after trusted startup registration.
  Generic shell, generic HTTP mutation, raw domain DB mutation, raw secret read,
  and direct low-level helper calls are denied through platform APIs.
- Telegram: optional by default. Inactive optional Telegram must not fail core
  readiness. Raw tokens and token-bearing Telegram Bot API URLs must not appear
  in status, chat, doctor, journald, support-bundle, audit, or diagnostic
  output.
- Backup/restore/support: backup paths include sensitive local state. Support
  bundles are redacted artifacts, not raw state exports.
- Local exposure: runtime and web UI are intended for loopback/local user
  operation. Public network exposure is outside the current release boundary.

## Cheap Safety Checks Now Covered

- `python scripts/external_pack_safety_smoke.py` covers hostile pack intake,
  lifecycle boundaries, managed-service tamper rejection, rollback scope, and
  support redaction.
- `python scripts/chat_eval.py` covers deterministic semantic routing,
  mutation-preview boundaries, stale-context escape, public lookup/search
  routing, no-search suppression, malformed input, and mixed prompt invariants.
- `python scripts/llm_behavior_eval.py` covers full orchestrator response
  invariants with mocked tools/status/search: no mutation without confirmation,
  no token/secret markers, no raw Podman/shell advice for managed services, no
  irrelevant pack hijack, no stale clarification loop, and no page
  fetch/browser/download/import claims.
- `python scripts/prove_ready.py` runs the current single-command readiness
  gate and distinguishes release-blocking failures from optional runtime
  warnings such as isolated proof search being disabled.
- `python scripts/capability_policy_smoke.py` and
  `python scripts/capability_policy_audit.py` cover the Capability Policy v1
  schema, central gate, migrated executor bindings, receipt metadata, stale
  Plan blocking, local activation requirement, and generic package-install
  bypass blocking. Remaining warnings must identify unsupported/deferred
  variants or documented process-isolation limits, not silently accepted
  first-party mutation bypasses.
- `python scripts/universal_plan_mode_smoke.py` and
  `python scripts/universal_plan_mode_audit.py` cover the Universal Mutation
  Plan schema, package registry dispatch, migrated executor Plan metadata,
  direct shell package bypass blocking, cancellation, expiry, duplicate
  handling, changed-target rejection, uninstall activation blocking, receipt
  metadata, and documented legacy visibility.
- `python scripts/executor_authorization_migration_smoke.py` covers Backup v1,
  Restore v1, support-bundle creation, and memory lifecycle mutation
  classification under central capability policy and Universal Plan metadata.
  It uses isolated fixtures and proves backup/support/restore direct helper
  bypasses fail with `mutated=false`.
- `python scripts/files_git_service_migration_smoke.py` covers bounded file
  create/modify/delete, Git commit/push policy boundaries, service restart
  fixtures, direct shell Git/systemctl blocking, symlink/path traversal
  rejection, force-push denial, and receipt metadata. It uses isolated files,
  an isolated temporary Git repository, and fixture service state only.
- `python scripts/communications_migration_smoke.py` covers implemented
  notification communications, fake Telegram delivery, local notification
  records, mark-read/prune history mutation, unsupported email/calendar
  providers, active-channel exception scope, secret-content blocking, and
  direct provider-client bypass blocking.
- `python scripts/skill_pack_permission_boundary_smoke.py` covers skill-pack
  manifest validation, declared/granted/effective permission handling,
  target-scope enforcement, identity/version/fingerprint binding, brokered
  Universal Plan dispatch, direct helper blocking, arbitrary shell/HTTP/secret
  platform API denial, grant revocation, update permission diffs, receipt
  metadata, and the documented in-process Python isolation limitation.
- `python scripts/generic_mutation_bypass_audit.py` and
  `python scripts/generic_mutation_bypass_smoke.py` cover repository-wide
  reviewed mutation-surface inventory plus dynamic denial of direct
  file/Git/service/provider/shell helpers, raw DB/secret/HTTP/shell primitives,
  expired/consumed/copied contexts, registry mutation after freeze, and API
  authorization override attempts.
- `python scripts/full_adversarial_authorization_proof.py` covers the
  end-to-end request-to-receipt authorization chain with forged
  capability/executor attempts, trusted-context replay, Plan tampering,
  confirmation replay, target drift, cross-scope skill/grant reuse, direct
  primitive access, partial and uncertain result truth, receipt/status truth,
  and fixture isolation. It writes machine-readable evidence to `/tmp` and
  reports the in-process Python isolation limitation as a documented
  non-blocking warning.

## Release Blockers

- Public network hardening is not claimed. Do not expose the API/web UI or
  SearXNG beyond loopback for this release track.
- A release is blocked if capability, Universal Plan, generic bypass,
  adversarial authorization, latency closure, version consistency, artifact,
  docs truth, or final release audit gates report unresolved failures.

## Non-Blocking Gaps To Track

- Fresh Debian VM install proof remains a high-cost environmental proof. The
  final v0.2.2 audit uses clean-checkout and installed-product local proofs;
  do not describe that as proof of every fresh machine.
- Real local-LLM behavior fuzzing is not part of the release gate yet; the
  second-tier eval is deterministic/mocked by default.
- Broader managed-action journal rollout is intentionally paused.
- Universal capability migration covers the implemented first-party mutation
  lanes and skill-pack platform API boundary. Email/calendar providers remain
  unsupported. Unsupported destructive file, Git, service-control, provider,
  and process-isolation variants remain denied or deferred unless a bounded
  policy already controls them.
- Malicious arbitrary in-process Python skill code is not isolated in this
  checkpoint. Do not enable untrusted external code execution until a real
  process isolation boundary exists.
- Startup auto-recovery that mutates state is intentionally absent.
- Direct llama.cpp binary/library management is absent.
- MCP/tool runtime execution is absent.
- Semantic memory remains off by default and release-gated.
- Assistant memory policy refuses secret-like values and directs operators to
  the secret store or a password manager instead of durable chat memory.

## Operator Rule

If any chat/status/doc output appears to imply arbitrary shell, pack code
execution, full browser/page fetching, public web exposure, or silent install,
treat it as a security bug and add a regression before fixing it.

## Audit v2B authorization status

Managed skill-pack mutation confirmation is now scoped and single-use, with
revalidation of pack content/version, grant, permission, arguments, target,
actor, thread, session, and expiry. Read-only skill inspection stays immediate
and foreign-code execution remains denied.

Universal authorization is not closed. Sixteen runtime files retain an
explicit release-blocking migration disposition; the route inventory reports
47 legacy mutations and seven legacy Plan/apply paths.

## Audit v2C transactional and internal authority

Validated confirmation is durably reserved with `BEGIN IMMEDIATE`, then marked
executing before an executor can mutate. Terminal results are persisted before
the append-only receipt. A crash before execution is fail-closed; a crash after
the execution boundary is indeterminate and cannot be retried automatically.

Internal writer authority is nonce-backed and in-process only, with exact
writer/capability/operation/resource/target/trigger/mode binding and a durable
redacted journal. Public structured boundaries reject claimed internal
identities. This does not defend against malicious Python already in the
process and is not process isolation.

## Provider/model/secret boundary (Audit v2D)

Provider URLs reject credentials, non-HTTP schemes, undeclared loopback, and
literal private/link-local/reserved addresses before a Plan exists. Secret
Plans contain only namespace plus keyed opaque current/proposed versions;
plaintext is excluded from Plans, confirmations, receipts, journals, and CLI
output. DNS rebinding and malicious in-process Python remain limitations.
Secret delete/import/repair and bulk configuration import/reset are not public
surfaces and are explicitly recorded as unimplemented/denied rather than
silently treated as authorized functionality.

## Audit v2E additions

Organization and memory Plans store keyed opaque content fingerprints, not
plaintext. Semantic ingestion rejects out-of-root paths, symlink components,
unsupported types, and files over 8 MiB. Execution opens each path relative to
an allowed-root directory descriptor with no-follow semantics, verifies the
regular-file identity before and after a bounded read, and ingests those exact
verified bytes as untrusted data. `/done` and document apply reject target
drift. The 37 assistant mutations have unique operation/executor bindings and
seven narrow capability families rather than one generic organization grant. Notification public
effects require durable confirmation. The scheduled delivery-before-receipt
crash interval remains an explicit risk; exactly-once scheduled delivery is not
claimed. Malicious Python already inside the process remains out of scope.

## Audit v2F closure boundary

All inventoried public mutation surfaces now have a central authorization or
explicit-denial disposition. Pack-source configuration grants no trust;
permission changes cannot use wildcard or unknown capabilities; local external
pack lifecycle stages are distinct and stale-state bound. Arbitrary remote pack
fetch is denied pending connection-time destination validation and a separate
digest-bound quarantine stage. This avoids a false DNS-rebinding claim.

Scheduled notification sends persist redacted intent before transport I/O.
Interrupted sends become indeterminate and cannot be automatically retried.
Malicious Python already inside the trusted process remains outside this
authorization threat boundary.

## Audit 3 user-visible safety semantics

Public copy now matches this boundary: source allowlisting permits untrusted
metadata queries only, arbitrary remote pack acquisition is unavailable, and a
registered pack removal is distinct from denied arbitrary deletion. Approval
copy is derived from the canonical Plan but hides capability/executor IDs and
fingerprints from the primary explanation. Indeterminate execution is not
reported as success or failure and is not blindly retried. Controlled Mode is
not described as unrestricted. Exactly-once notification delivery, immutable
SearXNG image pinning, and process isolation from trusted in-process Python are
not claimed.
