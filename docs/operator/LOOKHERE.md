# Personal Agent - Operator Runbook

## Source Priority

- Product/runtime contract: `PRODUCT_RUNTIME_SPEC.md`
- Current branch reality: `PROJECT_STATUS.md`
- Mission/behavior contract: `docs/design/CANONICAL_HANDOFF_V3.md`
- Setup/API/UI details: `README.md`

## Runtime Modes

- Canonical runtime: `personal-agent-api.service` (core runtime brain)
- Native UI is primary user surface.
- Telegram is optional transport adapter; it must not become a second brain.
- Telegram is disabled by default; inspect/control it with:
  - `python -m agent telegram_status`
  - `python -m agent telegram_enable`
  - `python -m agent telegram_disable`
- Telegram canonical UX routing is delegated to `agent/telegram_bridge.py`; keep transport safety in `telegram_adapter/bot.py`.
- Manual debug runtime: `.venv/bin/python -m agent.api_server --host 127.0.0.1 --port 8765`

## Service Control

User service:
- `systemctl --user status personal-agent-api.service`
- `systemctl --user restart personal-agent-api.service`

## Quick Diagnostics

- Core health:
  - `python -m agent setup`
  - `python -m agent health`
  - `python -m agent version`
  - `python -m agent status`
  - `python -m agent brief`
  - `python -m agent memory`
- Logs:
  - `python -m agent logs`
- Doctor checks:
  - `.venv/bin/python -m agent doctor`
  - JSON: `.venv/bin/python -m agent doctor --json`
  - safe local fixes: `.venv/bin/python -m agent doctor --fix`
  - strict mode (legacy timer checks): `AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS=1 .venv/bin/python scripts/doctor.py`
- Test sweep:
  - `. .venv/bin/activate && pytest -q`

Runtime mode contract (all surfaces):
- `READY`: normal operation.
- `BOOTSTRAP_REQUIRED`: setup path.
- `DEGRADED`: partial operation with one next action.
- `FAILED`: deterministic error block with trace id.

Onboarding/recovery contract:
- first-run command: `python -m agent setup`
- onboarding states: `NOT_STARTED`, `TOKEN_MISSING`, `LLM_MISSING`, `SERVICES_DOWN`, `READY`, `DEGRADED`
- recovery modes: `TELEGRAM_DOWN`, `API_DOWN`, `TOKEN_INVALID`, `LLM_UNAVAILABLE`, `LOCK_CONFLICT`, `DEGRADED_READ_ONLY`, `UNKNOWN_FAILURE`

Tool execution contract:
- LLM tool requests are validated via `agent/tool_contract.py`.
- Execution and permission gating are centralized in:
  - `agent/tool_executor.py`
  - `agent/permission_contract.py`

Continuity contract:
- Thread/pending summary and follow-up resolution live in `agent/memory_runtime.py`.
- Use `python -m agent memory` or Telegram `what are we doing?` for current resumable state.
- Follow-up actions never cross threads silently; ambiguous follow-ups are rejected.
- Meta summary actions (`memory/setup/status/doctor/resume`) do not overwrite last meaningful action.

## Timers

- Observe: `personal-agent-observe.service` + `personal-agent-observe.timer`
- Daily brief: `personal-agent-daily-brief.service` + `personal-agent-daily-brief.timer`
- Check timer state:
  - system scope: `sudo systemctl status <timer>`
  - user scope: `systemctl --user status <timer>`

## Telegram Commands (Golden Path)

- `help`
- `status`
- `health`
- `brief`
- `doctor`
- `setup`
- `memory` (or plain text: `what are we doing?`, `where were we?`, `resume`)

All of the above now use the same canonical runtime/setup/doctor/memory contracts as the unified CLI.
Telegram plain text `status` is canonical runtime status output (legacy ENABLE_WRITES/audit block removed).
Telegram `setup/status` readiness semantics are sourced from the same runtime truth as CLI (`/ready` contract).

## Logs

- Default app log: `logs/agent.jsonl`
- Supported runtime model: entrypoints bootstrap stdout logging automatically, so journald/stdout should always show runtime logs unless an explicit external logging config replaces it.
- System journal:
  - system scope: `journalctl -u personal-agent.service -n 200 --no-pager`
  - user scope: `journalctl --user -u personal-agent-api.service -n 200 --no-pager`

## Common Failures

Bot/API not responding:
- verify active service and restart it
- check journal logs
- if Telegram is enabled, confirm token exists (`telegram:bot_token` secret or `TELEGRAM_BOT_TOKEN`)

Telegram conflict (`terminated by other getUpdates request`):
- ensure only one process uses the same bot token
- run `python -m agent telegram_status`
- if blocked by a stale lock, run `python -m agent telegram_enable`
- if blocked by a live duplicate poller, stop the duplicate process and keep only the intended service

No reports/snapshots yet:
- run `/storage_snapshot` once to seed baseline
- verify timers are enabled and active
