# PERSONAL AGENT - CURRENT PROJECT STATUS

Last updated: 2026-02-18

This report is code-grounded (current repo state), not a roadmap.

**Authority Note**
- `PROJECT_STATUS.md` is current-branch truth for command surface and test counts.
- `README.md` is setup/overview and may drift from code.

## 1) Repo Snapshot (Authoritative)

- Branch: `brief-v0.2-clean`
- HEAD: `b186249` - `feat(modelops): add permissions UI, audit view, and operator docs`
- Git working tree: **dirty** (`git status --porcelain`)
  - `22` changed paths total
  - `19 modified`, `1 deleted`, `2 untracked`
  - Current touched areas include `agent/orchestrator.py`, `agent/api_server.py`, `telegram_adapter/bot.py`, `memory/schema.sql`, `memory/db.py`, `agent/perception/`, `tests/test_perception.py`
- Last 10 commits (`git log --oneline -10`):
  1. `b186249 feat(modelops): add permissions UI, audit view, and operator docs`
  2. `fab683d feat(modelops): add planner/executor and API permission gates`
  3. `da8abfb feat(modelops): add permissions policy and append-only audit log`
  4. `e0e1d8c feat(model-scout): add API endpoints and web UI management tab`
  5. `de60cb2 feat(model-scout): add telegram notify flow and /scout commands`
  6. `ace507f feat(model-scout): add deterministic scout engine and storage`
  7. `27090a3 chore(webui): clarify model refresh label in provider setup`
  8. `d8e9eb1 feat(webui): unify add-provider flow for openai-compatible APIs`
  9. `8de9297 feat(api): support generic provider test and model management flows`
  10. `b55aae9 feat(webui): show provider/model health and recommendations`

## 2) Tests + Version

- `pytest -q`: `391 passed in 7.32s`
- Version source: `VERSION`
  - File exists and contains: `0.2.0`
  - Runtime reads `VERSION` in `agent/api_server.py:86`

## 3) Runtime Entry Points + Surfaces

- Telegram runtime entry: `telegram_adapter/bot.py:752` (`main()`), app builder at `telegram_adapter/bot.py:658`
- API runtime entry: `agent/api_server.py:1839` (`main()`), server start at `agent/api_server.py:1806`

Telegram commands currently registered (`telegram_adapter/bot.py:702` to `telegram_adapter/bot.py:727`):
- `/remind`
- `/status`
- `/runtime_status`
- `/disk_grow`
- `/audit`
- `/permissions`
- `/storage_snapshot`
- `/storage_report`
- `/resource_report`
- `/brief`
- `/network_report`
- `/sys_metrics_snapshot`
- `/sys_health_report`
- `/sys_inventory_summary`
- `/weekly_reflection`
- `/today`
- `/task_add`
- `/done`
- `/open_loops`
- `/health`
- `/daily_brief_status`
- `/ask`
- `/ask_opinion`
- `/scout`
- `/scout_dismiss`
- `/scout_installed`

API endpoints currently wired (`agent/api_server.py:1562` to `agent/api_server.py:1798`):
- `GET /health`
- `GET /version`
- `GET /telegram/status`
- `GET /models`
- `GET /config`
- `GET /defaults`
- `GET /providers`
- `GET /permissions`
- `GET /audit`
- `GET /model_scout/status`
- `GET /model_scout/suggestions`
- `POST /chat`
- `POST /providers`
- `POST /providers/test` (backward-compatible)
- `POST /models/refresh`
- `POST /telegram/secret`
- `POST /telegram/test`
- `POST /model_scout/run`
- `POST /modelops/plan`
- `POST /modelops/execute`
- `POST /providers/{id}/secret`
- `POST /providers/{id}/models`
- `POST /providers/{id}/test`
- `POST /providers/{id}/models/refresh`
- `POST /model_scout/suggestions/{id}/dismiss`
- `POST /model_scout/suggestions/{id}/mark_installed`
- `PUT /config`
- `PUT /defaults`
- `PUT /permissions`
- `PUT /providers/{id}`
- `DELETE /providers/{id}`
- Static/UI serving: `GET /`, `GET /assets/*` (`agent/api_server.py:1611`)

## 4) State + Persistence

- DB default path:
  - Main config: `AGENT_DB_PATH` else `<repo>/memory/agent.db` (`agent/config.py:86`)
  - Observe config uses same fallback (`agent/config.py:65` to `agent/config.py:70`)
- Key tables from live SQLite schema query (`sqlite3 memory/agent.db "select name from sqlite_master where type='table' order by name;"`):
  - Core: `projects`, `tasks`, `notes`, `open_loops`, `reminders`, `preferences`, `user_prefs`
  - Thread/graph: `topic_threads`, `thread_entries`, `thread_anchors`, `thread_labels`, `thread_focus`, `graph_nodes`, `graph_edges`, `graph_aliases`, `graph_relation_types`, `graph_relation_mode`, `graph_relation_constraints`
  - Reporting/system: `disk_baselines`, `disk_snapshots`, `dir_size_samples`, `storage_scan_stats`, `resource_snapshots`, `resource_process_samples`, `resource_scan_stats`, `network_snapshots`, `network_interfaces`, `network_nameservers`, `system_facts_snapshots`, `last_report_registry`, `report_history`
  - Model/autonomy: `model_scout_runs`, `model_scout_suggestions`, `model_scout_baselines`, `audit_log`, `anomaly_events`
  - Perception: `metrics_snapshot`, `events`
  - Runtime legacy/auxiliary present in DB: `conversation_state`, `conversation_events`, `sensory_events`, `memory_jobs`, `pending_opinions`, `last_reports`, `long_term_notes`
- Newly-added tables since prior status (from `git diff -- memory/schema.sql`):
  - `metrics_snapshot` + `idx_metrics_snapshot_ts`
  - `events` + `idx_events_ts`

## 5) What Exists In Code Today

- Orchestration/routing modules:
  - `agent/orchestrator.py` (command handling + skill invocation)
  - `agent/intent_router.py` (rule routing)
  - `agent/nl_router.py` (NL route/cards)
  - `agent/llm/router.py` (provider/model routing)
- Memory system state:
  - SQLite adapter in `memory/db.py`
  - Base schema in `memory/schema.sql`
  - Runtime schema hardening via `_ensure_*` in `memory/db.py:92`
- Scheduling model:
  - systemd timers/services under `ops/systemd/` and `systemd/`
  - optional in-process scheduled snapshots behind `ENABLE_SCHEDULED_SNAPSHOTS` in `telegram_adapter/bot.py:743`
- New subsystem now present:
  - Perception package: `agent/perception/collector.py`, `agent/perception/diagnostics.py`, `agent/perception/inventory.py`
  - Routed commands in orchestrator: `agent/orchestrator.py:477`, `agent/orchestrator.py:494`, `agent/orchestrator.py:513`

## 5.5) Subsystem Status Matrix (Code-grounded)

| Subsystem | Status | Wiring | Persistence | Tests | Notes |
|---|---|---|---|---|---|
| Telegram runtime | stable | wired in Telegram | SQLite | yes (`tests/test_telegram_commands.py`, `tests/test_telegram_token_loading.py`) | Runtime entry and command handlers are in `telegram_adapter/bot.py:658` and `telegram_adapter/bot.py:702`. |
| API + Web UI | stable | wired in API | JSON | yes (`tests/test_api_server.py`) | API handler table is in `agent/api_server.py:1562`; UI static serving is `agent/api_server.py:1611`. |
| Orchestrator routing | stable | wired in Telegram | SQLite | yes (`tests/test_orchestrator.py`, `tests/test_intent_router.py`, `tests/test_nl_router_cards.py`) | Telegram routes through `Orchestrator.handle_message` in `agent/orchestrator.py:1359`. |
| Skills system | stable | wired in Telegram | none | yes (`tests/test_skill_loader.py`, `tests/test_git_skill.py`) | Dynamic skill load is via `SkillLoader.load_all` in `agent/skills_loader.py:31`. |
| Memory graph (typed relations + acyclic constraints) | stable | wired in Telegram | SQLite | yes (`tests/test_memory_graph.py`, `tests/test_typed_relations.py`, `tests/test_graph_constraints.py`) | Typed relation and acyclic enforcement are in `memory/db.py:942` and `memory/db.py:1010`, called by orchestrator checks at `agent/orchestrator.py:1781`. |
| Task system | stable | wired in Telegram | SQLite | yes (`tests/test_db.py`, `tests/test_open_loops.py`, `tests/test_orchestrator.py`) | Task add/done/open_loops routes are in `agent/orchestrator.py:1433`, `agent/orchestrator.py:1587`, `agent/orchestrator.py:3185`. |
| Scheduling (systemd timers vs in-process jobs) | beta | internal only | none | yes (`tests/test_daily_brief_scheduler.py`, `tests/test_scheduled_daily_brief_entrypoint.py`) | systemd timers exist in `ops/systemd/personal-agent-observe.timer:1` and `ops/systemd/personal-agent-daily-brief.timer:1`; optional in-process jobs are at `telegram_adapter/bot.py:743`. |
| Reporting (storage/resource/network) | stable | wired in Telegram | SQLite | yes (`tests/test_storage_governor.py`, `tests/test_resource_governor.py`, `tests/test_network_governor.py`) | Command routes are `agent/orchestrator.py:3130`, `agent/orchestrator.py:3141`, `agent/orchestrator.py:3152`; snapshot tables are in `memory/schema.sql:213` and `memory/schema.sql:261`. |
| Perception (metrics_snapshot/events + diagnostics) | experimental | wired in Telegram | SQLite | yes (`tests/test_perception.py`) | New commands route in `agent/orchestrator.py:3163`; persistence is in `memory/schema.sql:378` and `memory/db.py:1864`. |
| Model Scout | beta | wired in API | SQLite | yes (`tests/test_model_scout.py`, `tests/test_scheduled_model_scout.py`, `tests/test_api_server.py`) | API endpoints are wired at `agent/api_server.py:1598`; Telegram commands also exist at `telegram_adapter/bot.py:725`; store supports SQLite with JSON fallback in `agent/model_scout.py:67`. |
| ModelOps (plan/execute + permissions + audit) | beta | wired in API | JSON | yes (`tests/test_modelops_planner.py`, `tests/test_permissions.py`, `tests/test_api_server.py`, `tests/test_audit_log.py`) | Permission decision and execution gates are in `agent/api_server.py:1240` and `agent/api_server.py:1351`; policy is `agent/permissions.py:166`; append log writer is `agent/audit_log.py:62`. |

## 6) Guardrails

- ModelOps action constraints allowlisted in code: `agent/permissions.py:12`
- Restrictive default permission document (`manual_confirm`, actions disabled by default): `agent/permissions.py:23`
- Permission policy enforcement: `agent/permissions.py:166`
- Audit logging with redaction and append writes: `agent/audit_log.py:62` to `agent/audit_log.py:93`
- API exposes policy/audit surfaces: `GET/PUT /permissions`, `GET /audit` (`agent/api_server.py:1586`, `agent/api_server.py:1589`, `agent/api_server.py:1773`)
- Write toggle exists and is off by default:
  - parse env in `agent/config.py:113`
  - injected into runtime config at `agent/config.py:260`
  - consumed by orchestrator as `self.enable_writes` in `agent/orchestrator.py:132`

## 6.5) Epistemic Contract Status (What is enforced vs aspirational)

- `“I’m not sure” + 1 question when uncertain`: `Partially enforced` (`agent/orchestrator.py:1361` always applies epistemic gate; intercept prefix and one-question formatting are in `agent/epistemics/gate.py:16` and `agent/epistemics/gate.py:47`; shape is tested in `tests/test_epistemics_gate.py:36`).
- `No confident guessing / no thread stitching`: `Partially enforced` (cross-thread risk detectors and hard intercepts are in `agent/epistemics/detectors.py:154` and `agent/epistemics/detectors.py:205`; unsupported-claim and provenance constraints are in `agent/epistemics/contract.py:177`; contract behavior is covered in `tests/test_epistemics_gate.py:63` and `tests/test_epistemics_contract.py:21`).
- `Authoritative domains must use tools (if implemented)`: `Enforced in code` (deterministic classifier is in `agent/orchestrator.py:181`; `/ask` gate wiring is in `agent/orchestrator.py:3283` via `agent/orchestrator.py:761`; API `/chat` server-side enforcement is in `agent/api_server.py:1154` with observation collection in `agent/api_server.py:972`; coverage is in `tests/test_authoritative_domain_gate.py`).
- `Auditability / append-only logs for operational actions`: `Partially enforced` (`agent/audit_log.py:62` appends JSONL records with fsync; SQLite audit rows are mutable via `memory/db.py:2035`).
- `Permission gating for ModelOps`: `Enforced in code` (`agent/permissions.py:166` evaluates policy; API plan/execute enforce deny/confirm in `agent/api_server.py:1317` and `agent/api_server.py:1399`; endpoint behavior is tested in `tests/test_api_server.py:646`).

## 7) Known Gaps / Follow-ups (Repo-visible Only)

- README Telegram command list is stale versus actual bot registration:
  - docs list: `README.md:233` to `README.md:257`
  - code list includes 3 additional commands: `telegram_adapter/bot.py:713` to `telegram_adapter/bot.py:715`
- README test count line is stale:
  - `README.md:277` says `350 passed`
  - current run is `391 passed`

## 8) Immediate Next Work (Evidence-based, max 3)

1. Sync `README.md` Telegram command list with `telegram_adapter/bot.py` registrations.
2. Update or remove hard-coded README pass-count line (`README.md:277`) to avoid drift.
