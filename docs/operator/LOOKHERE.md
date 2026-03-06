# Personal Agent - Operator Runbook

## Source Priority

- Current branch reality: `PROJECT_STATUS.md`
- Mission/behavior contract: `docs/design/CANONICAL_HANDOFF_V3.md`
- Setup/API/UI details: `README.md`

## Runtime Modes

- Canonical runtime: `personal-agent-api.service` (API + embedded Telegram)
- Manual debug runtime: `.venv/bin/python -m agent.api_server --host 127.0.0.1 --port 8765`

## Service Control

User service:
- `systemctl --user status personal-agent-api.service`
- `systemctl --user restart personal-agent-api.service`

## Quick Diagnostics

- Core health:
  - `python -m agent health`
  - `python -m agent version`
  - `python -m agent status`
  - `python -m agent brief`
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

Tool execution contract:
- LLM tool requests are validated via `agent/tool_contract.py`.
- Execution and permission gating are centralized in:
  - `agent/tool_executor.py`
  - `agent/permission_contract.py`

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

## Logs

- Default app log: `logs/agent.jsonl`
- System journal:
  - system scope: `journalctl -u personal-agent.service -n 200 --no-pager`
  - user scope: `journalctl --user -u personal-agent-api.service -n 200 --no-pager`

## Common Failures

Bot/API not responding:
- verify active service and restart it
- check journal logs
- confirm token exists (`telegram:bot_token` secret or `TELEGRAM_BOT_TOKEN`)

Telegram conflict (`terminated by other getUpdates request`):
- ensure only one process uses the same bot token
- stop duplicate pollers and restart the intended service

No reports/snapshots yet:
- run `/storage_snapshot` once to seed baseline
- verify timers are enabled and active
