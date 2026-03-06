# Architecture (Current Branch)

This file describes the structural shape of the project as implemented on `brief-v0.2-clean`.

## Runtime Entry Points

- Telegram bot: `telegram_adapter/bot.py`
- Local API server: `agent/api_server.py`
- Unified operator CLI: `python -m agent` (`agent/cli.py`, `agent/__main__.py`)

Both runtimes depend on shared core modules (`agent/orchestrator.py`, skills, memory DB, llm router).

## Source Of Truth

- Core agent + LLM runtime state is authoritative (`agent/api_server.py`, `agent/llm/*`).
- User-facing transports (Telegram/CLI) should surface runtime facts and avoid speculative state.
- Operational checks/fixes are centralized under `python -m agent doctor`.

## Core Request Flow

Telegram/API input (untrusted)
  -> command/intent handling
  -> orchestrator
  -> skill or domain module
  -> persistence/audit/memory
  -> response formatting
  -> transport response (Telegram/API)

## Golden Path Runtime Flow

Service startup
  -> `agent/startup_checks.py` (fast PASS/WARN/FAIL checks)
  -> runtime init
  -> API listen + embedded Telegram runner
  -> user sends plain text on Telegram
  -> deterministic pre-routing (doctor/status/health/brief/setup/help)
  -> orchestrator + LLM chat path
  -> deterministic fallback/bootstrapping only when LLM is unavailable

## Startup Checks

- Shared startup checks are implemented in `agent/startup_checks.py`.
- Both API and Telegram startup paths run them.
- Failures are deterministic and include:
  - `trace_id`
  - `component`
  - `failure_code`
  - `next_action`

## Unified Error UX

- Deterministic error blocks use `agent/error_response_ux.py::deterministic_error_message`.
- User-facing failures provide exactly one next action.
- Telegram send path retries safely on parse errors and logs a definitive `telegram.out` on success.

## Surface Routing

- CLI (`python -m agent`): operator workflows (`doctor/status/health/brief/logs/version`).
- Telegram: natural-language command mapping + orchestrator forwarding.
- API: contract-first JSON endpoints with never-raise wrappers.

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
