# PROJECT_STATUS.md

Branch: `brief-v0.2-clean`  
HEAD: `see git rev-parse --short HEAD`  
Date (UTC): `2026-03-06`  
Tests: `770 passed in 15.67s (pytest -q)`

## Canonical
This file is the authoritative current-state document for this branch.
If any other doc disagrees with this file, trust this file.
Documentation hierarchy:
- root: `README.md`, `ARCHITECTURE.md`, `PROJECT_STATUS.md`, `STABILITY.md`
- operator docs: `docs/operator/*`
- design docs: `docs/design/*`
- history docs: `docs/history/*` and `docs/archive/*`

## Operator CLI
Single operator entrypoint:
- `python -m agent doctor`
- `python -m agent status`
- `python -m agent health`
- `python -m agent brief`

## Current Product Shape
- Golden path:
  1. Start `personal-agent-api.service`
  2. Verify `python -m agent status`
  3. Use Telegram in plain English
  4. Use `doctor/status/health/brief` deterministically when needed
- Startup safety:
  - API + Telegram startup checks run via `agent/startup_checks.py`
  - FAIL exits non-zero with one next action
  - WARN logs continue
- User-facing truth:
  - identity is centralized in `agent/identity.py`
  - fallback guidance for no-chat-model is centralized in `agent/golden_path.py`
  - runtime mode contract is centralized in `agent/runtime_contract.py`

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
- Telegram bot polling now runs inside `personal-agent-api.service` when a Telegram token is configured; no separate `python -m telegram_adapter` process is required.
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
