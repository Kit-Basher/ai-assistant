# PROJECT_STATUS.md

Branch: `brief-v0.2-clean`  
HEAD: `see git rev-parse --short HEAD`  
Date (UTC): `2026-03-06`  
Tests: `832 passed in 16.28s (pytest -q)`

## Canonical
This file is the authoritative current-state document for this branch.
If any other doc disagrees with this file, trust `PRODUCT_RUNTIME_SPEC.md` first, then this file.
Documentation hierarchy:
- root: `PRODUCT_RUNTIME_SPEC.md`, `README.md`, `ARCHITECTURE.md`, `PROJECT_STATUS.md`, `STABILITY.md`
- operator docs: `docs/operator/*`
- design docs: `docs/design/*`
- history docs: `docs/history/*` and `docs/archive/*`

## Operator CLI
Single operator entrypoint:
- `python -m agent setup`
- `python -m agent doctor`
- `python -m agent status`
- `python -m agent health`
- `python -m agent health_system`
- `python -m agent llm_inventory`
- `python -m agent llm_select --task \"...\"`
- `python -m agent llm_plan --task \"...\"`
- `python -m agent brief`
- `python -m agent memory`

## Current Product Shape
- v1 scope: local-first PC health management, read-only guidance first.
- PC health monitoring (read-only) implemented.
- System health output now includes deterministic warning analysis, severity, and actionable suggestions.
- LLM control plane implemented:
  - deterministic local-first inventory
  - deterministic task classification
  - deterministic model selection
  - approved install planning when no suitable local model exists
- Golden path:
  1. Start `personal-agent-api.service`
  2. Run `python -m agent setup --dry-run`
  3. Verify `python -m agent status`
  4. Use native UI as primary user surface
  5. Use `doctor/status/health/brief` deterministically when needed
  6. Use Telegram as optional transport surface when enabled
  7. Telegram text commands (`help/setup/status/health/doctor/memory`) are routed through the same canonical runtime/setup/doctor/memory contracts as CLI.
  8. Telegram `status` no longer uses legacy ENABLE_WRITES/audit text; it uses canonical runtime mode/next-action semantics.
  9. Meta summary actions (`memory/setup/status/doctor/resume`) do not overwrite continuity `last action`.
- Startup safety:
  - API + Telegram startup checks run via `agent/startup_checks.py`
  - FAIL exits non-zero with one next action
  - WARN logs continue
- User-facing truth:
  - identity is centralized in `agent/identity.py`
  - fallback guidance for no-chat-model is centralized in `agent/golden_path.py`
  - runtime mode contract is centralized in `agent/runtime_contract.py`
  - Telegram canonical product UX delegation is centralized in `agent/telegram_bridge.py`
  - onboarding/recovery contracts are centralized in:
    - `agent/onboarding_contract.py`
    - `agent/recovery_contract.py`
    - `agent/setup_wizard.py`

## Onboarding + Recovery Contract
- Onboarding states:
  - `NOT_STARTED`
  - `TOKEN_MISSING`
  - `LLM_MISSING`
  - `SERVICES_DOWN`
  - `READY`
  - `DEGRADED`
- Recovery modes:
  - `TELEGRAM_DOWN`
  - `API_DOWN`
  - `TOKEN_INVALID`
  - `LLM_UNAVAILABLE`
  - `LOCK_CONFLICT`
  - `DEGRADED_READ_ONLY`
  - `UNKNOWN_FAILURE`
- Canonical first-run command:
  - `python -m agent setup`

## Runtime Contract
- Shared mode names across CLI, Telegram, API, and orchestrator:
  - `READY`
  - `BOOTSTRAP_REQUIRED`
  - `DEGRADED`
  - `FAILED`
- Shared helpers:
  - `get_runtime_mode(...)`
  - `get_effective_llm_identity(...)`
  - `get_effective_next_action(...)`
  - `normalize_user_facing_status(...)`
- Truthfulness rule:
  - user-facing provider/model identity can explicitly be `unknown` when certainty is missing.

## LLM Control Plane
- Canonical control-plane modules:
  - `agent/llm/control_contract.py`
  - `agent/llm/model_inventory.py`
  - `agent/llm/model_health_check.py`
  - `agent/llm/task_classifier.py`
  - `agent/llm/model_selector.py`
  - `agent/llm/install_planner.py`
- Operator surfaces:
  - `python -m agent llm_inventory`
  - `python -m agent llm_select --task \"...\"`
  - `python -m agent llm_plan --task \"...\"`
- Selection behavior:
  - healthy
  - approved
  - local-first
  - capability/context match
  - policy/cost-cap aware
  - deterministic tie-break

## Tool Execution Contract
- Canonical LLM tool request schema: `agent/tool_contract.py`
  - fields: `tool`, `args`, `reason`, `read_only`, `confidence`
  - strict allowlist of supported tools
- Single execution gate: `agent/tool_executor.py`
  - request validation
  - permission decision
  - deterministic execution envelope
- Single permission decision helper: `agent/permission_contract.py`
  - read-only/write gating semantics are centralized
  - block reasons + next actions are deterministic and reused

## Continuity Contract
- Canonical continuity shapes and normalization: `agent/memory_contract.py`
- Single continuity store access layer: `agent/memory_runtime.py`
- Thread fidelity hardening:
  - follow-up bind only when exactly one valid pending item exists in the active thread
  - ambiguity/expiry/no-resumable paths are deterministic and never guessed
- User-facing continuity summary:
  - CLI: `python -m agent memory`
  - Telegram text aliases: `what are we doing?`, `where were we`, `resume`
  - Meta summaries never recursively store their own rendered output as `last_agent_action`.

## Remaining Rough Edges
- Root-level docs are now canonicalized, but some historical wording remains inside archived files.
- Some legacy fallback strings can still appear in older persisted notifications until they roll out.
- API surface remains broad; operator golden path should prefer the unified CLI over direct endpoint usage.

## A) API Contract
- Top-level user endpoints (`/chat`, `/ask`, `/done`) are wrapped to avoid uncaught exceptions and always return JSON.
- Contract keys for top-level envelope responses: `ok`, `intent`, `confidence`, `did_work`, `error_kind`, `message`, `next_question`, `actions`, `errors`, `trace_id`.
- Message is guaranteed non-empty for failure/clarification paths.
- Clarification flow is a soft response (`ok=true`, `did_work=false`, `error_kind="needs_clarification"`) and asks at most one question.

## B) Intent Resilience Phases Implemented
- Phase 1: deterministic low-confidence guard before orchestrator (`agent/intent/low_confidence.py`).
- Phase 2: deterministic clarification planner with stable reasons/hints (`agent/intent/clarification.py`).
- Phase 3: deterministic thread-integrity guard for likely topic/thread drift (`agent/intent/thread_integrity.py`).
- Phase 4: deterministic intent assessment layer with optional bounded LLM rerank (`agent/intent/assessment.py`, `agent/intent/llm_rerank.py`).

## C) ModelOps Advisor
- Read/advise surfaces:
  - `POST /llm/models/check`
  - `POST /llm/models/recommend`
- Consent-gated change surface:
  - `POST /llm/models/switch`
- Switching is explicit-user-action only; no automatic provider/model switching.

## D) LLM Ops Tooling (Current Surface)
- Health and remediation:
  - `/llm/health`, `/llm/health/run`
  - `/llm/support/diagnose`, `/llm/support/bundle`
  - `/llm/support/remediate/plan`, `/llm/support/remediate/execute`
- Catalog/capabilities/registry:
  - `/llm/catalog`, `/llm/catalog/run`, `/llm/catalog/status`
  - `/llm/capabilities/reconcile/plan`, `/llm/capabilities/reconcile/apply`
  - `/llm/registry/snapshots`, `/llm/registry/rollback`
- Autopilot/notifications:
  - `/llm/autopilot/*`
  - `/llm/notifications/*`
- Model watch (deterministic + consent-gated):
  - Provider catalog diff proposals can route through fix-it (`issue_code=model_watch.proposal`).
  - Optional HF watch (`AGENT_MODEL_WATCH_HF_ENABLED=1`) scans allowlisted HF repos/orgs only, persists diff state, and can produce `local_download` proposals.
  - HF local acquisition is always confirmation-gated: no download/install runs before explicit `confirm=true`.

## Runtime Note
- Default runtime model is one main service (`personal-agent-api.service`).
- Telegram is optional and treated as a transport adapter surface (disabled by default; set `TELEGRAM_ENABLED=1` to enable).
- Readiness check after restart:
  1. `systemctl --user restart personal-agent-api.service`
  2. `python tools/wait_ready.py`
  3. Optional detail: `curl -s http://127.0.0.1:8765/ready`

## E) Active Endpoints (Generated)
Endpoint inventory below is generated from `agent/api_server.py` using `tools/dump_routes.py`.

To refresh endpoint list:
1. `python3 tools/dump_routes.py > /tmp/routes.md`
2. Replace the generated endpoint section in this file with `/tmp/routes.md` content.

## Active Endpoints
### GET
- /audit
- /config
- /defaults
- /health
- /ready
- /llm/autopilot/explain_last
- /llm/autopilot/ledger
- /llm/autopilot/ledger/{part3}
- /llm/catalog
- /llm/catalog/status
- /llm/health
- /llm/notifications
- /llm/notifications/last_change
- /llm/notifications/policy
- /llm/notifications/status
- /llm/registry/snapshots
- /llm/support/bundle
- /llm/support/diagnose
- /model_scout/sources
- /model_scout/status
- /model_scout/suggestions
- /model_watch/latest
- /model_watch/hf/status
- /models
- /permissions
- /providers
- /telegram/status
- /version

### POST
- /ask
- /chat
- /done
- /llm/autoconfig/apply
- /llm/autoconfig/plan
- /llm/autopilot/bootstrap
- /llm/autopilot/undo
- /llm/autopilot/unpause
- /llm/capabilities/reconcile/apply
- /llm/capabilities/reconcile/plan
- /llm/catalog/run
- /llm/cleanup/apply
- /llm/cleanup/plan
- /llm/health/run
- /llm/hygiene/apply
- /llm/hygiene/plan
- /llm/models/check
- /llm/models/recommend
- /llm/models/switch
- /llm/notifications/mark_read
- /llm/notifications/prune
- /llm/notifications/test
- /llm/registry/rollback
- /llm/self_heal/apply
- /llm/self_heal/plan
- /llm/support/remediate/execute
- /llm/support/remediate/plan
- /model_scout/run
- /model_scout/suggestions/{part2}/dismiss
- /model_scout/suggestions/{part2}/mark_installed
- /model_watch/refresh
- /model_watch/hf/scan
- /model_watch/run
- /modelops/execute
- /modelops/plan
- /models/refresh
- /providers
- /providers/test
- /providers/{part1}/models
- /providers/{part1}/models/refresh
- /providers/{part1}/secret
- /providers/{part1}/test
- /telegram/secret
- /telegram/test

### PUT
- /config
- /defaults
- /permissions
- /providers/{part1}

### DELETE
- /providers/{part1}
