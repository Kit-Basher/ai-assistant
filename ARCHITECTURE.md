# Architecture (Current Branch)

This file describes the structural shape of the project as implemented on `brief-v0.2-clean`.

## Runtime Entry Points

- Telegram bot: `telegram_adapter/bot.py`
- Local API server: `agent/api_server.py`

Both runtimes depend on shared core modules (`agent/orchestrator.py`, skills, memory DB, llm router).

## Core Request Flow

Telegram/API input (untrusted)
  -> command/intent handling
  -> orchestrator
  -> skill or domain module
  -> persistence/audit/memory
  -> response formatting
  -> transport response (Telegram/API)

## Main Components

- Orchestration and routing: `agent/orchestrator.py`, `agent/intent_router.py`, `agent/nl_router.py`
- Skill system: `skills/*`, loaded by `agent/skills_loader.py`
- Persistence: `memory/db.py` + schema in `memory/schema.sql`
- LLM abstraction: `agent/llm/*`, registry in `llm_registry.json`
- ModelOps controls: `agent/modelops/*`, `agent/permissions.py`, `agent/audit_log.py`
- Scheduling entrypoints:
  - observe: `agent/scheduled_observe.py`
  - daily brief: `agent/scheduled_daily_brief.py`
  - optional in-process snapshots in Telegram runtime when enabled

## Data/State Boundaries

- Primary structured state: SQLite (`memory/agent.db` by default)
- Operational logs: JSONL (`logs/agent.jsonl` by default)
- Secrets: OS keyring where available, encrypted file fallback
- Provider/model registry: JSON (`llm_registry.json`)

## Design Constraints

- Input is untrusted until routed and validated.
- Side effects are explicit and auditable.
- Model-management autonomy is constrained to whitelisted actions.
- Local operation is first-class; remote providers are optional.

## Change Rules

- New behavior should go through orchestrator/skill boundaries (not ad hoc handler logic).
- New persistence should be schema-backed and migration-safe.
- New operational actions should include explicit permission and audit paths.
- New modules should ship with tests in `tests/`.
