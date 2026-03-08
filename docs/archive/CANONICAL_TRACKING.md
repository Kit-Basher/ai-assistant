# Archived Project Tracking Snapshot (Frozen As Of 63caa73076bd5d0aeda43333cf841bdbe4bdcccf)

> Status: historical snapshot only.  
> Do not use this file as current truth.  
> Authoritative current-branch status lives at repo root: `PROJECT_STATUS.md`.

## Project Identity + Non-Negotiables
Personal Agent is a local-first, Telegram-first assistant that treats Telegram input as untrusted and enforces safe, auditable, reversible, bounded behavior; it separates factual reports from optional narration or opinion and forbids surprise actions, with audit hard-fail semantics and explicit confirmation gates for sensitive operations.

## Current Repo Status
### Git
- Branch: `phase-2-opinions`
- Latest commit: `63caa73076bd5d0aeda43333cf841bdbe4bdcccf` — `fix: make reminders idempotent`

### Key Directories / Files
```
agent/
skills/
memory/
telegram_adapter/
tests/
```
- `agent/`: orchestration, policy, routing, LLM routing, disk diff/anomaly logic, scheduled snapshots.
- `skills/`: manifested skills (core CRUD, storage/resource/network governors, reflection, recall/opinion).
- `memory/`: SQLite schema + DB access layer.
- `telegram_adapter/`: Telegram bot integration and scheduled jobs.
- `tests/`: unit tests for routing, snapshots, reflection, runtime status, config.

### Tests
```
pytest -q
```
- Status: `65 passed in 0.36s`

## Implemented Capabilities
- Intent routing + pending clarification
  - `agent/intent_router.py`
  - `agent/orchestrator.py`
  - `memory/schema.sql` (`pending_clarifications`)
  - `memory/db.py`
- Orchestrator flow + logging or audit behaviors
  - `agent/orchestrator.py`
  - `agent/logging_utils.py`
  - `memory/db.py`
  - `skills/*/handler.py`
- Disk snapshot, diff, and anomaly reporting
  - `skills/storage_governor/collector.py`
  - `skills/storage_governor/handler.py`
  - `agent/disk_diff.py`
  - `agent/disk_anomalies.py`
  - `agent/orchestrator.py` (`/disk_changes`, `/disk_baseline`, `/disk_grow`, `/disk_digest`)
  - `agent/scheduled_snapshots.py`
- Weekly reflection observe-only layer
  - `skills/reflection/handler.py`
  - `skills/reflection/manifest.json`
  - `tests/test_weekly_reflection.py`
- Model routing and provider switching
  - `agent/llm_router.py` (tiered local or cloud candidate routing)
  - Historical note: this snapshot predates removal of the old broker and direct-provider narration stack.
  - `agent/config.py` (provider config, OpenRouter settings)

## Data + Persistence
### SQLite Tables (memory, audit, clarifications, snapshots)
- `projects`, `tasks`, `notes`, `reminders`, `preferences`
- `activity_log`
- `audit_log`
- `disk_baselines`, `disk_snapshots`, `dir_size_samples`, `storage_scan_stats`
- `resource_snapshots`, `resource_process_samples`, `resource_scan_stats`
- `network_snapshots`, `network_interfaces`, `network_nameservers`
- `pending_clarifications`

### DB Location
- Default path: `memory/agent.db` (from `AGENT_DB_PATH`)

### Migration Approach
- Schema created via `memory/schema.sql` at startup (`MemoryDB.init_schema`).
- Minimal inline migration: `MemoryDB._ensure_reminder_columns()` adds missing reminder columns.

## Safety / Action-Gating
- Allowed by default: read-only operations and observe-only snapshots or reports.
- Blocked or disabled:
  - Action proposals disabled (`agent/action_gate.py`).
  - Runner is unconfigured and returns `runner_not_configured` (`agent/runner.py`).
  - Advice requests are refused in `/ask` and `/ask_opinion`.
- Confirmation store behavior:
  - `agent/confirmations.py` stores pending actions and requires `/confirm`.
  - `agent/policy.py` requires confirmation for delete or overwrite action types.
- Hard-fails:
  - Audit log failures abort operations with `Audit logging failed. Operation aborted.`

## Known Gaps / Deferred Items
- Telegram UX polish (minimal interface in `telegram_adapter/bot.py`).
- Natural language chat over stored knowledge limited to `/ask` and `/ask_opinion`.
- Stronger model switching escalation exists but is not fully wired into all flows.
- Placeholders in orchestrator for:
  - next-best-task suggestions
  - daily planning
  - weekly review (distinct from weekly reflection)
- LLM intent parsing is present but explicitly “not wired yet.”
- Disk scheduled snapshot uses stubbed disk snapshot (`skills/disk_report/safe_disk.py` returns `{}`).

## Next Milestones (Strictly Next Steps)
1. Goal: Implement next-best-task and daily plan results from stored tasks.
   Acceptance criteria: `/next` and `/plan` return task-based suggestions or plans, not placeholders.
   Affected files: `skills/core/handler.py`, `agent/orchestrator.py`, `agent/intent_router.py`.
   Test additions required: new unit tests in `tests/` validating populated suggestions or plans.
2. Goal: Wire weekly review command to weekly reflection skill output.
   Acceptance criteria: `/weekly` and weekly review intent return reflection output instead of “coming soon.”
   Affected files: `agent/orchestrator.py`, `telegram_adapter/bot.py`.
   Test additions required: tests ensuring `/weekly` routes to reflection and returns report.
3. Goal: Make disk scheduled snapshot collect real data.
   Acceptance criteria: scheduled snapshot writes non-empty snapshot content and diffs work.
   Affected files: `skills/disk_report/safe_disk.py`, `agent/scheduled_snapshots.py`.
   Test additions required: tests validating non-empty snapshot payloads and diff behavior.

## How To Reproduce Locally
```
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
TELEGRAM_BOT_TOKEN=... .venv/bin/python -m telegram_adapter
```
