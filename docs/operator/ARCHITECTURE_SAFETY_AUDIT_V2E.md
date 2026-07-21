# Architecture and Safety Audit v2E

Status: implemented in the working tree; not committed or released.

## Scope and result

Audit v2E moves user-directed organization, memory, semantic-memory, and
notification mutations onto the durable central authorization stack. The
machine inventory is
`ORGANIZATION_MEMORY_AUTHORIZATION_INVENTORY_V2E.json` and regenerates with:

```bash
python scripts/organization_memory_authorization_audit.py --check
```

Forty-four explicit operations are centrally authorized: 37 command-specific
assistant sub-operations plus memory reset, semantic ingest/rebuild/repair, and
notification test/mark-read/prune. The compatibility entry
`assistant.mutate` is only a resolver and has no capability or executor of its
own. Read/list/search/status/recall operations remain immediate.

## Canonical path

```text
Web/API or ordinary assistant/Telegram message
  -> AgentOrchestrator (one assistant front door)
  -> OrganizationMemoryAuthorizationService.preview
  -> Universal Mutation Plan (private content is opaque)
  -> durable scoped confirmation transaction
  -> Capability Policy
  -> operation-bound Executor Registry entry
  -> existing bounded runtime/orchestrator implementation
  -> durable result and redacted receipt
```

`route_inference()` remains the only inference orchestrator. Authorized
assistant writes re-enter the same command handler through a private one-shot
in-process sentinel; no transport payload can supply it. This inherits the
documented limitation that malicious Python already executing in the trusted
process is outside the threat model.

Each assistant command declares an argument schema, resource types, target
tables, stale-state fingerprint, rollback limit, audit description, narrow
capability family, and unique executor. Unknown commands and aliases, extra or
nested fields, multiple commands in one request, and caller-supplied batch
payloads fail closed. Create operations also carry a durable, namespace-keyed
idempotency identity so repeat delivery in the same actor/thread/session scope
does not create a second record.

## Persistence separation

- Current conversation events, pending-state bookkeeping, audit records, and
  scheduler cursors remain bounded internal persistence.
- User requests to remember, forget, create, complete, label, merge, reset,
  ingest, repair, send, mark read, or prune require a Plan and confirmation.
- Retrieved memory, document text, note text, and model output are untrusted
  advisory data. They cannot become policy or authorization metadata.
- Model-generated candidates do not receive automatic durable authority.

Scheduled persistence leaves now enforce exact registered identities for
`llm_notifications`, `llm_model_discovery_policy`, `model_watch`, and
`model_watch_hf`. Public and manual entry points do not receive those internal
identities.

## Target and privacy binding

Plans bind actor, thread, session, capability, executor, runtime policy, target
fingerprint, and expiry. `/done` binds the exact task ID and current record
fingerprint. Other command families bind their relevant table set; changes
between preview and apply fail closed. Private content appears in Plans only as
a keyed opaque fingerprint and byte count. Executor journal keys named
`private_content`, `memory_content`, `notification_content`, or `document_text`
are redacted.

Semantic file ingestion opens paths component-by-component relative to a
declared allowed-root directory descriptor with `O_NOFOLLOW`, verifies a
regular file before and after the bounded read, and binds device, inode, size,
mtime, type, parser, count, and SHA-256. The verified bytes are passed directly
to the text ingester; the path is not reopened. Files are limited to 8 MiB and
`.txt`, `.md`, `.markdown`, `.json`, or `.csv`. Inline and file content remains
provenance-labelled untrusted data. Remote URL ingestion, macros, foreign code,
embedded execution, package installation, and shell execution are unavailable.

## Notification disposition

Public test/send effects, mark-read, and prune are centrally authorized.
Scheduled internal authority records delivery-result metadata only; it cannot
choose a recipient, enable Telegram, or invoke a public mutation executor.
Telegram configuration and enablement were not changed.

One delivery limitation remains: the legacy scheduled notification path can
deliver before its metadata receipt is durably appended. A process crash in
that narrow interval can make the outcome uncertain; automatic resend must not
be presented as proven exactly-once until v2F either reserves the scheduled
delivery before transport I/O or disables that retry path. Public confirmed
notification tests are protected by the durable confirmation transaction.

## Inventory delta and release disposition

Global public inventory before v2E: 19 `legacy_unmigrated`, seven
`plan_confirm_gated`. After v2E: ten `legacy_unmigrated`, seven
`plan_confirm_gated`. The remaining groups are pack sources, pack lifecycle,
pack permissions, managed-search setup compatibility, and general permission
cleanup assigned to v2F. Universal authorization is not yet complete.

Supported-but-absent CRUD variants are not invented here. Notification
retry/resend and delivery-policy writes, semantic remote ingestion, arbitrary
memory merge/restore aliases, and unsupported project/task CRUD verbs have no
public executor and remain explicitly unavailable rather than bypassing policy.

## Verification

The v2E focused suite covers opaque content, boolean-confirmation rejection,
single-use/concurrent confirmation, direct executor bypass, assistant
preview/confirm, stale `/done`, root containment, symlink rejection, and changed
document invalidation. Final working-tree verification passed: 632 focused
domain/security tests, 74 canonical release-smoke tests, all capability/Plan/
internal-writer/adversarial/transaction/pack proofs, and the full suite at
2,563 passed with 22 named environment-dependent skips. Deterministic inventory
checks and `git diff --check` also passed.

## v2F follow-through

The ten deferred legacy and seven Plan-confirm pack/search/permission groups
were dispositioned in v2F. Notification delivery bookkeeping now persists an
operation identity before I/O and reconciles interrupted execution as
indeterminate without automatic resend.
