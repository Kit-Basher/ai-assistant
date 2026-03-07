# Architecture (Current Branch)

This file describes the structural shape of the project as implemented on `brief-v0.2-clean`.
Canonical product/runtime target is defined in [`PRODUCT_RUNTIME_SPEC.md`](/home/c/personal-agent/PRODUCT_RUNTIME_SPEC.md).
v1 functional focus is local-first PC health management with read-only diagnostics/guidance first.

## Runtime Entry Points

- Core runtime service (authoritative): `agent/api_server.py`
- Native UI (primary user surface): served by core runtime (`GET /` from `agent/webui/dist`)
- Unified operator CLI: `python -m agent` (`agent/cli.py`, `agent/__main__.py`)
- Optional Telegram adapter surface: `telegram_adapter/bot.py`

All surfaces must depend on shared core modules (`agent/orchestrator.py`, skills, memory DB, llm router) and must not become separate business-logic owners.

## Source Of Truth

- Core agent + LLM runtime state is authoritative (`agent/api_server.py`, `agent/llm/*`).
- User-facing transports (Telegram/CLI) should surface runtime facts and avoid speculative state.
- Operational checks/fixes are centralized under `python -m agent doctor`.
- Runtime-mode contract is centralized in `agent/runtime_contract.py`.
- Onboarding and recovery state contracts are centralized in:
  - `agent/onboarding_contract.py`
  - `agent/recovery_contract.py`
  - `agent/setup_wizard.py`
- Telegram canonical UX bridge is centralized in:
  - `agent/telegram_bridge.py`

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
  -> API listen + native UI available
  -> optional embedded Telegram runner (transport only)
  -> user sends natural-language request (UI/CLI/Telegram)
  -> deterministic pre-routing at transport boundary when needed
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

## Onboarding + Recovery

- Canonical first-run command: `python -m agent setup`.
- `agent/setup_wizard.py` consumes already-fetched runtime state and returns:
  - onboarding state
  - recovery mode
  - one exact next action
  - safe suggestions (read-only by default)
- Telegram `setup/help/why isn’t this working` and API `/ready` consume the same contract fields to avoid conflicting advice.

## Unified Error UX

- Deterministic error blocks use `agent/error_response_ux.py::deterministic_error_message`.
- User-facing failures provide exactly one next action.
- Telegram send path retries safely on parse errors and logs a definitive `telegram.out` on success.
- Runtime mode mapping is shared across API/CLI/Telegram/Orchestrator:
  - `READY`
  - `BOOTSTRAP_REQUIRED`
  - `DEGRADED`
  - `FAILED`

## LLM Tool-Use Contract

- Canonical LLM tool request schema is centralized in `agent/tool_contract.py`.
- A single execution gate in `agent/tool_executor.py` validates, authorizes, executes, and returns deterministic results.
- Permission decisions are centralized in `agent/permission_contract.py`:
  - one allow/deny decision format
  - one reason string set
  - one next-action policy for blocked actions
- Orchestrator LLM tool requests, directive shims, and deterministic heuristic tool paths share this execution gate.

## LLM Control Plane

- Deterministic LLM control-plane modules live under `agent/llm/*` and stay inside the core runtime.
- Current control-plane responsibilities:
  - `agent/llm/control_contract.py`: canonical inventory/task/selection shapes
  - `agent/llm/model_inventory.py`: local-first inventory of registry + installed Ollama models
  - `agent/llm/model_health_check.py`: lightweight local/provider health classification
  - `agent/llm/task_classifier.py`: deterministic task classification
  - `agent/llm/model_selector.py`: policy-aware model selection
  - `agent/llm/install_planner.py`: approved install/download planning only
- Surfaces should ask the core runtime/control-plane for model truth instead of inventing provider/model decisions locally.

## Continuity Contract

- Canonical continuity data shapes are in `agent/memory_contract.py`:
  - `thread_state`
  - `pending_item`
  - `memory_summary`
- Runtime continuity access is centralized in `agent/memory_runtime.py`:
  - thread state read/write
  - pending lifecycle (including expiry)
  - resumable snapshot summary
  - follow-up resolution rules
- Thread fidelity rule:
  - short follow-ups (`yes/no/do it/that one/show me more`) bind only when exactly one valid pending item exists for the active thread.
  - ambiguous or expired follow-ups are refused deterministically (no implicit thread mixing).

## Surface Routing

- CLI (`python -m agent`): operator workflows (`doctor/status/health/brief/logs/version`).
- Telegram: transport adapter with canonical routing delegated through `agent/telegram_bridge.py`.
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
  - transport adapters should not own scheduling/business logic

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
