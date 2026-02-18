# Personal Agent - Operator Runbook

## Source Priority

- Current branch reality: `PROJECT_STATUS.md`
- Mission/behavior contract: `CANONICAL_HANDOFF_V3.md`
- Setup/API/UI details: `README.md`

## Runtime Modes

- Telegram bot runtime: `.venv/bin/python -m telegram_adapter`
- Local API/UI runtime: `.venv/bin/python -m agent.api_server --host 127.0.0.1 --port 8765`

## Service Control

System service install:
- `sudo systemctl status personal-agent.service`
- `sudo systemctl restart personal-agent.service`

User service install:
- `systemctl --user status personal-agent-api.service`
- `systemctl --user restart personal-agent-api.service`

If unsure which mode is installed, check both and use the one that exists.

## Quick Diagnostics

- Core health:
  - `curl -s http://127.0.0.1:8765/health`
  - `curl -s http://127.0.0.1:8765/version`
- Doctor checks:
  - `. .venv/bin/activate && python scripts/doctor.py`
  - strict mode (fail when units missing): `AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS=1 .venv/bin/python scripts/doctor.py`
- Test sweep:
  - `. .venv/bin/activate && pytest -q`

## Timers

- Observe: `personal-agent-observe.service` + `personal-agent-observe.timer`
- Daily brief: `personal-agent-daily-brief.service` + `personal-agent-daily-brief.timer`
- Check timer state:
  - system scope: `sudo systemctl status <timer>`
  - user scope: `systemctl --user status <timer>`

## Telegram Commands (High-Use)

- `/brief`
- `/today`
- `/task_add <title>`
- `/done <id>`
- `/open_loops [all|due|important]`
- `/daily_brief_status`
- `/health`
- `/status`
- `/ask <prompt>`
- `/ask_opinion <prompt>`
- `/scout`

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
