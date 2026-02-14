# Personal Agent (Telegram-first)

Local-first personal assistant with SQLite memory, deterministic routing, and systemd-managed scheduling.

## Documentation Order (Source of Truth)
Use docs in this order when context conflicts:
1. `CANONICAL_HANDOFF_V3.md` - mission, behavioral rules, long-horizon direction.
2. `PROJECT_STATUS.md` - current branch reality, active work, test health.
3. `LOOKHERE.md` - day-to-day operator runbook.
4. `README.md` - setup and command surface overview.

Historical snapshots (not current truth):
- `docs/archive/CANONICAL_TRACKING.md`
- `docs/archive/FINAL_CHECK.md`
- `docs/archive/RELEASE_NOTES.md`

## Quick Start (Local)
1. Create venv:
   - `python3 -m venv .venv`
   - `. .venv/bin/activate`
2. Install deps:
   - `pip install -r requirements.txt`
3. Run bot:
   - `TELEGRAM_BOT_TOKEN=... .venv/bin/python -m telegram_adapter`

## Environment Variables
Required:
- `TELEGRAM_BOT_TOKEN`

Common optional:
- `AGENT_TIMEZONE` (default `America/Regina`)
- `AGENT_DB_PATH` (default `memory/agent.db`)
- `AGENT_LOG_PATH` (default `logs/agent.jsonl`)
- `AGENT_SKILLS_PATH` (default `skills/`)
- `ENABLE_SCHEDULED_SNAPSHOTS` (`1` to enable periodic snapshots)
- `ENABLE_WRITES` (default off)

LLM/provider optional:
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- `OLLAMA_HOST`, `OLLAMA_MODEL`

## Install (systemd)
Recommended user install:
- `bash ops/install.sh --user`

System install:
- `bash ops/install.sh`

Uninstall:
- `bash ops/install.sh uninstall`

## Telegram Commands (Registered in `telegram_adapter/bot.py`)
- `/remind <YYYY-MM-DD HH:MM> | <text>`
- `/status`
- `/runtime_status`
- `/disk_grow [path]`
- `/audit`
- `/storage_snapshot`
- `/storage_report`
- `/resource_report`
- `/brief`
- `/network_report`
- `/weekly_reflection`
- `/today`
- `/task_add <title>`
- `/done <id>`
- `/open_loops [all|due|important]`
- `/health`
- `/daily_brief_status`
- `/ask <question>`
- `/ask_opinion <question>`

Notes:
- `/task_add` also accepts advanced pipe syntax:
  - `/task_add <project> | <title> | <effort_mins> | <impact_1to5>`
- `/done` requires numeric task id.

## Daily Brief Scheduling
Daily brief scheduling is systemd-driven via:
- `ops/systemd/personal-agent-daily-brief.service`
- `ops/systemd/personal-agent-daily-brief.timer`

Entrypoint:
- `.venv/bin/python -m agent.scheduled_daily_brief`

User timer status:
- `systemctl --user status personal-agent-daily-brief.timer`

## Testing
- Full suite: `pytest -q`
- Current local result (2026-02-14): `168 passed`

## Architecture References
- `ARCHITECTURE.md`
- `STABILITY.md`
