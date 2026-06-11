# Persistent Managed-Action Journal

Status: design plus minimal shared skeleton, with preference reset/clear,
support bundle creation, provider/API key config, Telegram token/service setup,
default model/temporary chat override, and model acquisition/import as converted
reference flows. This is not yet a claim that every managed action survives
crash/restart with complete recovery.

## Problem

Many managed actions now create an in-request `ManagedActionJournal`, verify
their writes, and roll back scoped owned resources when verification fails. That
protects the current request, but it loses evidence if the process crashes after
planning or mutation and before verification/rollback reporting.

Persistent storage closes that evidence gap. It must not weaken confirmation
gates, store secrets, archive private memory, or automatically mutate on restart.

## Decision

Use SQLite, not JSONL.

Reasons:

- managed actions need one current row per `action_id` with status transitions;
- recovery/status views need indexed reads for incomplete actions;
- SQLite is already a project dependency and fits local single-user state;
- JSON columns keep this compatible with the current data-only journal shape
  while avoiding a large migration across all flows.

The minimal skeleton lives in `agent/actions/persistent_journal.py` and writes a
dedicated state database at:

```text
~/.local/share/personal-agent/managed_actions.db
```

Preference reset/clear, support bundle creation, provider/API key config,
Telegram token/service setup, default model/temporary chat override, and model
acquisition/import now opt into the store at existing journal creation/update
points. These are converted reference flows. Other managed-action callers
remain follow-up work.

## Schema

Table: `managed_action_journals`

| Field | Purpose |
|---|---|
| `action_id` | Stable action id, primary key. |
| `action_type` | Bounded action class, such as `provider_api_key_config`. |
| `target_redacted` | Redacted target label for status display. |
| `status` | One of `planned`, `running`, `verified`, `rolled_back`, `failed`, `recovery_needed`. |
| `created_at` | Journal creation timestamp from the in-request journal when available. |
| `updated_at` | Last persisted transition timestamp. |
| `completed_at` | Set for terminal non-recovery states. |
| `recovery_needed` | Boolean read index for incomplete/recovery-needed actions. |
| `recovery_hint` | Safe next-step text, never a command to run silently. |
| `journal_json` | Redacted full current journal snapshot. |
| `owned_resources_json` | Redacted `created_resources` and `changed_resources` subset. |
| `verification_json` | Redacted verification result. |
| `rollback_json` | Redacted rollback result. |

Indexes:

- `(status, updated_at)` for recent status reads;
- `(recovery_needed, updated_at)` for recovery-needed reads.

## Status Lifecycle

Expected transitions:

1. `planned`: target and planned steps recorded before mutation.
2. `running`: at least one mutating or externally visible step started.
3. `verified`: verification passed; no recovery needed.
4. `rolled_back`: verification failed and owned rollback completed.
5. `failed`: action failed before a recoverable state or failed with no owned
   recovery path.
6. `recovery_needed`: process cannot prove the action is complete or cleaned up.

`recovery_needed` is intentionally non-terminal from the product perspective:
the status surface should show it and propose a safe next step, but the runtime
must not repair or clean up automatically on restart.

## Redaction And Privacy

Persistent journal storage must be stricter than in-memory request results.

Never store:

- API keys, bot tokens, passwords, bearer headers, auth URLs, or raw secret
  values;
- raw private memory, preference values, semantic chunks, prompts, message
  bodies, notification bodies, or support-bundle payload text;
- hostile imported pack text, raw pack files, raw README/SKILL content, or
  quarantine source bodies;
- full user document contents or unbounded command output.

Allowed fields:

- action ids, action types, statuses, timestamps;
- redacted target labels;
- resource kind plus stable non-secret identifiers where needed for ownership;
- hashes, counts, sizes, booleans, provider/model ids, and bounded error kinds;
- verification and rollback summaries after redaction.

The skeleton redacts sensitive keys such as `api_key`, `token`, `secret`,
`password`, `authorization`, `target`, `path`, `identifier`, `deleted_keys`,
`value`, `content`, `text`, `body`, `prompt`, `memory`, and `raw`; preserves
explicit `_hash` and `_hashes` fields; scrubs common token patterns; and
truncates long strings. Flow owners should still pass hashes/counts instead of
raw content.

## Owned Resource Tracking

Every persistent journal row should distinguish:

- resources created by this action;
- pre-existing resources changed by this action;
- whether rollback is supported;
- what snapshot/hash/previous metadata exists for rollback;
- whether a resource is append-only and cannot be rolled back.

Ownership evidence is required before cleanup. A matching name is not ownership.
Examples:

- SearXNG cleanup may target only the known managed service/container and must
  preserve a pre-existing container.
- Failed model acquisition may remove owned temp files, but not Ollama cache or
  model data until ownership markers prove the model was created by this action.
- Pack removal may restore metadata/tombstones, but quarantine/source artifacts
  need path ownership before deletion.
- Notification delivery and action-ledger rows are append-only; recovery can
  verify/read/report, not unsend or erase history.

## Read-Only Status Surface

Add an operator/read-only surface after these reference flow conversions:

- recent actions: newest journal rows with action id, type, status, timestamps,
  redacted target, resource summary, verification result, rollback result, and
  safe next step;
- incomplete actions: rows in `planned`, `running`, or `recovery_needed`;
- recovery-needed actions: explicit subset where cleanup/repair may be useful
  but requires confirmation;
- redacted details: never show raw secret/private/pack/user content.

This can appear in `/runtime`, an operations/admin endpoint, or a CLI command.
It must be read-only. It should not trigger verification, rollback, service
restart, file deletion, model deletion, source cleanup, or repair.

## Restart And Recovery Behavior

Startup behavior:

1. Open the journal store read-only or initialize schema if needed.
2. Read incomplete/recovery-needed rows.
3. Surface a warning/status item with redacted details and a safe next step.
4. Do not mutate.

Repair behavior:

1. User/operator asks for repair or cleanup.
2. Runtime shows a preview from the persisted journal plus current-state
   revalidation.
3. Runtime requires explicit confirmation unless an existing flow-specific
   policy already permits the exact action.
4. Runtime performs only the scoped repair/cleanup allowed by the original flow
   and current ownership proof.
5. Runtime writes the resulting status, verification, and rollback/repair
   result to the persistent journal.

If current state is ambiguous, the safe next step is manual inspection or a
read-only diagnostic, not automated cleanup.

## Flow Adoption Plan

Adopt in this order:

1. memory/preference reset and clear: converted for global, user, thread, and
   approved `memory_runtime:` prefix clears.
2. support bundle creation: converted for redacted doctor bundle artifacts and
   owned incomplete `agent-support-*` cleanup.
3. provider/API key writes and config update: converted for provider secret
   save, save-and-test, provider config update, and the provider config/secret
   portions of OpenRouter setup.
4. Telegram token/service setup: converted for token save, enablement drop-in
   writes, and `personal-agent-telegram.service` enable/disable state
   management. Persisted rows keep only redacted token metadata, service/drop-in
   ownership metadata, verification summaries, and rollback results. Online
   Telegram `getMe` remains optional, and no startup auto-recovery is added.
5. default model changes and temporary chat model override: converted for
   persisted default chat model changes, temporary override assignment, and
   clearing/restoring temporary overrides. Persisted rows keep previous/requested
   provider/model ids, changed setting names, readback/effective routing
   summaries, and verification/rollback status without prompts, raw chat text,
   provider response bodies, or secrets.
6. model acquisition/import: converted for approved Ollama pulls, Hugging Face
   local GGUF download/import into Ollama, download-only artifact markers, and
   direct Ollama GGUF import through an existing Modelfile. Persisted rows keep
   provider/runtime name, model id/name, source type, artifact ids or basenames,
   verification summaries, and cleanup/rollback status. They do not store raw
   prompts, private paths, provider bodies, subprocess stdout/stderr, tokens, API
   keys, raw GGUF contents, or unbounded file contents. Rollback removes only
   owned generated temp files such as Personal Agent Modelfiles/markers; it does
   not delete unrelated Ollama models, Ollama cache data, or user-provided GGUF
   or Modelfile paths.
7. managed local services/SearXNG;
8. pack lifecycle/removal/source deletion and registry maintenance;
9. semantic-memory ingest/repair;
10. notifications and action ledger status reads.

For each flow:

- persist `planned` before the first mutation;
- persist `running` after each owned mutation or externally visible step;
- persist `verified`, `rolled_back`, `failed`, or `recovery_needed` before
  returning;
- add tests for redaction, successful completion, failed verification, scoped
  rollback, and simulated restart/status read.

## Retention

Default retention target:

- keep terminal rows for 30 days or the newest 500 rows, whichever keeps more
  recent diagnostic value without turning the journal into a long-term archive;
- keep `planned`, `running`, and `recovery_needed` rows until they are explicitly
  resolved or manually pruned with confirmation;
- prune only terminal rows and only via an operator-confirmed maintenance action;
- never prune as a hidden startup side effect.

Prune operations are themselves managed actions and should be journaled.

## Current Implementation Boundary

Implemented now:

- SQLite schema creation;
- redaction helper;
- `upsert`, `get`, `recent`, and `incomplete` helpers;
- preference reset/clear persistent status transitions for global, user,
  thread, `/prefs_reset`, `/prefs_thread_reset`, and approved `memory_runtime:`
  prefix clear paths;
- support bundle creation persistent status transitions for planned/running,
  verified, rolled-back cleanup, and recovery-needed cleanup failure states;
- provider/API key config persistent status transitions for provider secret
  save, save-and-test, provider config update, and OpenRouter provider
  config/secret setup rollback states;
- Telegram token/service setup persistent status transitions for token save,
  enablement drop-in writes, service enable/disable verification, scoped
  token/drop-in rollback, and recovery-needed cleanup failure states;
- default model and temporary chat override persistent status transitions for
  planned/running, verified, rolled-back verification failure, recovery-needed
  default restore failure, and pre-mutation failed preflight states;
- model acquisition/import persistent status transitions for approved Ollama
  pulls, Hugging Face local GGUF download/import, download-only markers, and
  direct Ollama GGUF import. Verification reads back model inventory or artifact
  marker state; failed verification records rolled_back when owned generated
  cleanup succeeds and recovery_needed when cleanup cannot complete;
- focused tests for redaction, read-only recovery-needed reads, verified
  preference reset rows, support bundle verified rows, provider secret/config
  rows, Telegram token/service rows, default model/temporary override rows,
  model acquisition/import rows, rollback rows, recovery-needed rows, and
  preview-only non-mutation.

Not implemented now:

- conversion of other managed-action flows to persistent writes;
- startup recovery surfacing;
- repair/cleanup commands;
- complete product-wide crash/restart recovery tests.

Therefore the correct claim after this pass is: persistent journal storage
infrastructure exists, preference reset/clear persists journal transitions,
support bundle creation persists journal transitions, and provider/API key
config, Telegram token/service setup, default model/temporary chat override, and
model acquisition/import persist journal transitions. Product crash/restart
recovery is not complete until the remaining flows are converted and
restart/status surfacing is tested.
