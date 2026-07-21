# Memory Audit

Date: 2026-06-20

This audit describes the current Personal Agent memory surfaces before the
fresh Debian VM proof. It is an operator map, not a release claim.

## Architecture

Primary durable state lives in the SQLite database configured by `AGENT_DB_PATH`
or the active runtime config.

Key tables in `memory/schema.sql`:

- `preferences`: legacy/global local-state preferences.
- `user_prefs`: user-scoped preferences and internal continuity records.
- `thread_prefs`: explicit per-thread preference overrides.
- `thread_anchors`, `thread_labels`: thread continuity labels/anchors.
- `graph_*`: thread-scoped graph memory and relation constraints.
- `projects`, `tasks`, `notes`, `open_loops`, `reminders`: user-visible planner memory.
- `audit_log`, `activity_log`: local operational/audit history.

Additional optional stores:

- Memory v2 tables are initialized only when `memory_v2_enabled` is enabled.
- Semantic memory tables are initialized only when semantic memory is enabled.
- Managed-action journals are stored in `managed_actions.db`.
- Search runtime config is a small loopback-only `search_runtime_config.json`
  next to the runtime DB after verified managed SearXNG setup.
- Secrets are stored through `agent.secret_store.SecretStore`, not memory tables.

## Scopes

- Global: `preferences`.
- User: `user_prefs`, open loops, projects, notes, reminders.
- Thread: `thread_prefs`, anchors, labels, graph state.
- Temporary conversation state: pending confirmations/follow-ups under
  `memory_runtime:<user_id>:pending_items` in `user_prefs`.
- Working memory: `memory_runtime:<user_id>:working_memory_state` in
  `user_prefs`.
- Pack state: external pack metadata and grants are stored by the pack store,
  not as general chat memory.
- Project/local repo context: supplied by current chat/runtime surfaces; it is
  not a license to blend unrelated thread memory.

## Reads And Writes

Reads:

- `/memory/status` and memory chat prompts inspect continuity health and saved
  context.
- Chat may select bounded working-memory context for generic LLM turns.
- Deterministic status/setup/search/package routes should not need LLM memory
  context.

Writes:

- Working-memory hot turns are appended around chat turns unless the user uses
  the current-turn no-memory override.
- Explicit preference writes use known keys and managed-action reliability.
- Preference reset/clear operations are scoped, verified, journaled, and
  rollback-aware.
- Destructive memory reset uses preview/confirmation through `/memory/reset`.

Deletion/clear:

- `new`, `cancel`, `start over`, and `forget that` clear pending transient chat
  state. They do not erase durable preferences or history.
- Bulk memory reset is destructive and requires explicit confirmation.
- Prefix clears for internal `memory_runtime:` state are bounded and journaled.

## Privacy And Secret Boundaries

Current protections:

- Obvious Telegram tokens, API keys, bearer tokens, and password assignments are
  redacted before normal working-memory and last-request/action persistence.
- Managed-action journals redact private values, tokens, private paths, and raw
  hostile pack text.
- Doctor/support bundles run through redaction helpers before writing output.
- External pack tombstones retain audit metadata but redact raw skill text.

Risks:

- Free-form notes, open loops, and explicit remembered text are local private
  data. Treat DB backups and release bundles as sensitive.
- Redaction is pattern-based and cannot prove every possible secret shape.
- Existing historical memory may predate the latest redaction hardening.

## Stale Context Risks

Transient pending state is separate from durable memory:

- Pending confirmations/follow-ups can be aborted without deleting preferences.
- Fresh deterministic intents such as runtime status, Telegram status, search
  status, and package previews must beat stale pending state.
- Current user instructions should override old memory. If a saved memory
  conflicts with the current message, prefer the current message or ask one
  concise clarification.

The current-turn no-memory override is supported for phrases such as
`do not use memory ...`, `do not use old context ...`, `without memory ...`,
and `/nomem ...`. This suppresses memory use and persistence for that turn only;
it does not delete durable memory.

Known gap: there is not yet a full user-facing scoped memory editor for
arbitrary “forget X” beyond existing reset/clear/status surfaces. Do not fake
precise deletion of arbitrary concepts until that exists.

## Support Bundles And Logs

Support bundles are redacted and written to owned temp directories. They may
include status summaries, journal metadata, and runtime facts, but should not
include raw secrets or hostile imported pack text.

Operator logs and runtime DBs still contain local operational history. Treat
them as sensitive during backup, restore, sharing, and deletion.

## Storage Growth

Potentially growing surfaces:

- SQLite runtime DB: preferences, working memory, planner data, graph state,
  audit/activity logs, optional memory v2 and semantic memory.
- Managed-action journal DB.
- External pack cache/normalized artifacts/tombstones.
- Support bundles in temporary directories when explicitly requested.
- Runtime logs from systemd/user services.
- Release/build bundles under `dist/` or release output paths.

Bounded or cleanup-aware surfaces:

- Working memory compacts hot turns into summaries with token budgets.
- Pack lifecycle removal keeps redacted tombstones but removes installed records.
- Failed managed SearXNG setup removes only owned containers and restores runtime
  search config.

Known gap: there is no single `storage_status` CLI that summarizes all growth
surfaces yet. Use `python -m agent doctor`, `/memory/status`, `/packs/state`,
and operator filesystem checks until that read-only report exists.

## Operator Checks

Useful commands:

- `GET /memory/status`
- `python scripts/perf_smoke.py`
- `python scripts/chat_eval.py`
- `python scripts/llm_behavior_eval.py`
- `python scripts/prove_ready.py`

Before sharing logs or bundles, run the existing redaction path and still review
the output as sensitive local data.

## Authorization update (Audit v2E)

Explicit remember/forget, organization commands, graph/anchor/preference
changes, and destructive reset now preview before mutation on shipping runtime
paths. Retrieved memory and remembered instructions are untrusted context.
Plans contain opaque content fingerprints; stale record/table state invalidates
apply. Ordinary conversation-event persistence remains bounded bookkeeping.
