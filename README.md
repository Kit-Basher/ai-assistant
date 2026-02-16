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

## Local API (Desktop Runtime)
Run local HTTP API (no Telegram token required):
- `. .venv/bin/activate`
- `.venv/bin/python -m agent.api_server --host 127.0.0.1 --port 8765`

Endpoints:
- `GET /health`
- `GET /models`
- `POST /chat`
- `GET /config`
- `PUT /config`
- `GET /providers`
- `POST /providers`
- `PUT /providers/{id}`
- `DELETE /providers/{id}`
- `POST /providers/{id}/secret`
- `POST /providers/{id}/test`
- `GET /defaults`
- `PUT /defaults`
- `POST /models/refresh`

## Desktop App (Tauri + React)
Located in `desktop/`.

Dev run:
1. Start API: `.venv/bin/python -m agent.api_server`
2. In another shell:
   - `cd desktop`
   - `npm install`
   - `npm run tauri:dev`

Build:
- `cd desktop`
- `npm run tauri:build`

Secret storage:
- Primary: OS keychain via `keyring` (if available in runtime environment)
- Fallback: encrypted local file at `~/.local/share/personal-agent/secrets.enc.json`

Provider/model registry:
- File: `llm_registry.json` (schema v2, JSON on disk)
- Backward compatibility: v1 files are loaded via migration/compat logic at runtime.

Routing (high-level):
- `auto`: balanced quality/cost ordering.
- `prefer_cheap`: prioritize lower-cost models.
- `prefer_best`: prioritize higher-quality models.
- `prefer_local_lowest_cost_capable`: local-capable models first; otherwise lowest expected token-cost among capable remote models.
  - Expected cost uses rolling usage averages per `(task_type, provider, model)` from `llm_usage_stats.json` (next to DB by default).
  - Local models with unknown pricing are treated as zero-cost for ranking.

Side-by-side staging run:
1. Use a different API port:
   - `.venv/bin/python -m agent.api_server --port 8876`
2. Use separate data/config paths:
   - `AGENT_DB_PATH=/tmp/personal-agent-staging/agent.db`
   - `LLM_REGISTRY_PATH=/tmp/personal-agent-staging/llm_registry.json`
   - `AGENT_SECRET_STORE_PATH=/tmp/personal-agent-staging/secrets.enc.json`
   - `LLM_USAGE_STATS_PATH=/tmp/personal-agent-staging/llm_usage_stats.json`

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
- `LLM_REGISTRY_PATH` (defaults to `llm_registry.json` if present)
- `LLM_ROUTING_MODE` (`auto`, `prefer_cheap`, `prefer_best`, `prefer_local_lowest_cost_capable`)
- `LLM_RETRY_ATTEMPTS`
- `LLM_RETRY_BASE_DELAY_MS`
- `LLM_CIRCUIT_BREAKER_FAILURES`
- `LLM_CIRCUIT_BREAKER_WINDOW_SECONDS`
- `LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS`
- `LLM_USAGE_STATS_PATH` (default: `llm_usage_stats.json` next to DB)

Desktop/API optional:
- `AGENT_API_HOST` (default `127.0.0.1`)
- `AGENT_API_PORT` (default `8765`)
- `AGENT_SECRET_STORE_PATH` (encrypted file fallback location)

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
- Current local result (2026-02-16): `341 passed`

## Architecture References
- `ARCHITECTURE.md`
- `STABILITY.md`
