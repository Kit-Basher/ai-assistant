# Project Status

Last updated: 2026-02-14
Repo path: `~/personal-agent`
Branch: `brief-v0.2-clean`
HEAD: `6f65804`

## 1) What this project is
Personal Agent is a local-first, Telegram-first assistant built to be predictable and auditable.

Core runtime flow:
Telegram input -> orchestrator -> manifested skill handler -> SQLite + audit -> deterministic report/cards -> optional narration.

## 2) Current architecture (live code)
- Telegram entrypoint: `telegram_adapter/bot.py`
- Core behavior and routing: `agent/orchestrator.py`
- Skill discovery/loading: `agent/skills_loader.py`
- Persistence layer: `memory/db.py` + `memory/schema.sql`
- Ops control plane: `ops/supervisor.py` + `agent/ops/supervisor_client.py`
- Health checks: `scripts/doctor.py` + `agent/doctor.py`

## 3) User-facing capability surface today
### Telegram slash commands currently registered
- `/remind`
- `/status`
- `/runtime_status`
- `/disk_grow`
- `/audit`
- `/storage_snapshot`
- `/storage_report`
- `/resource_report`
- `/brief`
- `/network_report`
- `/weekly_reflection`
- `/today`
- `/open_loops`
- `/health`
- `/daily_brief_status`
- `/ask`
- `/ask_opinion`

### Natural language support (high level)
- Observe/report prompts (disk, resource, network, service health)
- Preference updates (daily brief config, display preferences)
- Open loop add/done/list
- Daily brief status explanation
- Basic conversational fallback/chitchat

## 4) Active manifested skills
Only folders with both `manifest.json` and `handler.py` are loaded.

Active skills (15):
- `core`
- `disk_pressure_report`
- `git`
- `knowledge_query`
- `network_governor`
- `observe_now`
- `opinion`
- `opinion_on_report`
- `ops_supervisor`
- `recall`
- `reflection`
- `resource_governor`
- `runtime_status`
- `service_health_report`
- `storage_governor`

Note: `skills/` also contains non-manifest directories that are not part of runtime skill loading.

## 5) Data model and persistence
Primary DB entities include:
- core memory: `projects`, `tasks`, `notes`, `reminders`, `preferences`
- workflow/safety: `pending_clarifications`, `audit_log`, `activity_log`
- observation data: disk/resource/network snapshots + scan stats
- anomaly tracking: `anomaly_events`
- unified brief/delta snapshots: `system_facts_snapshots`

Versioning:
- App version file: `VERSION` (currently `0.2.0`)
- Doctor expects schema minor to match DB schema version.

## 6) Safety and control boundaries
- Telegram input is treated as untrusted.
- Policy layer enforces permission checks and confirmation requirements for sensitive actions.
- `action_gate` is effectively disabled in this build.
- `runner` is not configured for real command execution (`runner_not_configured` path).
- Ops supervisor uses HMAC signature validation + nonce replay protection + timestamp skew checks.
- Allowed supervisor ops are constrained (`restart`, `status`, `logs`).

## 7) Systemd/ops state in repo
Unit files present under `ops/systemd/`:
- `personal-agent.service`
- `personal-agent-supervisor.service`
- `personal-agent-observe.service`
- `personal-agent-observe.timer`
- `personal-agent-daily-brief.service` (in-progress)
- `personal-agent-daily-brief.timer` (in-progress)

Installer:
- `ops/install.sh` supports system and `--user` mode installation.

Doctor checks:
- DB/schema readability
- observe timer/service health proof
- version/schema compatibility
- observe-now dry-run
- daily brief timer/service check (in-progress changes in working tree)

## 8) Test status
Current full suite run on this branch:
- `pytest -q` -> `162 passed` (2026-02-14)

## 9) Work in progress right now
Goal in progress: stable public daily brief scheduler entrypoint + user timer at 07:00.

Uncommitted files currently in progress:
- `agent/scheduled_daily_brief.py` (new public entrypoint)
- `ops/systemd/personal-agent-daily-brief.service` (new)
- `ops/systemd/personal-agent-daily-brief.timer` (new, `OnCalendar=*-*-* 07:00:00`)
- `ops/install.sh` (install/enable wiring)
- `agent/doctor.py` (daily brief timer health check)
- `tests/test_scheduled_daily_brief_entrypoint.py` (new)
- `tests/test_doctor.py` (updated)
- `tests/test_ops_install.py` (updated)

Current target behavior for this workstream:
- Entry point invokable as `python -m agent.scheduled_daily_brief`
- Must not call private `_scheduled_daily_brief`
- Exit `0` on disabled/outside-time-window skip paths
- Log decisions to stdout for journald
- Support env from `~/.config/personal-agent/agent.env` via systemd `EnvironmentFile`

## 10) Known gaps / cleanup backlog
- Some docs are stale vs current branch reality (notably frozen tracking docs and older README sections).
- Command surface mismatch: orchestrator has additional slash command logic not all exposed by Telegram command handlers.
- Placeholder responses remain for some flows (`/next`, `/plan`, `/weekly`, `/done`).
- Runtime scaffolding directories in `skills/` can create confusion (not loaded unless manifest+handler exist).

## 11) Immediate next steps
1. Commit the in-progress daily-brief scheduler/timer changes.
2. Add short docs update in `LOOKHERE.md`/`README.md` for the new daily brief timer usage.
3. Optionally align Telegram command registration with orchestrator-supported commands.
4. Refresh stale project tracking docs after merge to avoid future context drift.

## 12) How to refresh this status file
Use this quick checklist before updating:
1. `git branch --show-current && git rev-parse --short HEAD`
2. `git status --short`
3. `pytest -q`
4. Verify active skills by checking manifest+handler pairs in `skills/`
5. Reconcile docs with actual command handlers in `telegram_adapter/bot.py`
