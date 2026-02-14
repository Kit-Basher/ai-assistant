# Personal Agent (Telegram) — How To Use

## What This Is
This is a local‑first Telegram assistant designed to be trustable and auditable.  
It runs on your machine, uses explicit confirmations for sensitive actions, and keeps a clear log of what happened.

## Current Status (Authoritative)
- Use `PROJECT_STATUS.md` for current branch status, active work, and latest test health.
- If another status/tracking doc conflicts, prefer `PROJECT_STATUS.md`.

## Safety Model (Plain English)
- No surprise actions: sensitive operations require explicit confirmation.
- Facts → opinions: the assistant will not give advice without facts first.
- Audit trail: actions and results are logged to a local JSONL file.

## Quick Start (Systemd)
If installed via `ops/install.sh`, the service is already set up.

Common service commands:
- Start: `sudo systemctl start personal-agent.service`
- Stop: `sudo systemctl stop personal-agent.service`
- Restart: `sudo systemctl restart personal-agent.service`
- Status: `sudo systemctl status personal-agent.service`

## Usage Examples (Natural Language)
Try these in Telegram:
- “show me my last disk report”
- “what changed this week?”
- “any anomalies lately?”
- “largest directory growth in /home”

## Slash Commands (Telegram)
Core:
- `/remind <YYYY-MM-DD HH:MM> | <text>`
- `/status`
- `/runtime_status`
- `/weekly`

Storage/Reports:
- `/storage_snapshot`
- `/storage_report`
- `/resource_report`
- `/network_report`
- `/weekly_reflection`

Ask/Opinion:
- `/ask <question>`
- `/ask_opinion <question>`

Ops (if enabled):
- `/restart` (requires confirmation via `/confirm`)
- `/service_status`
- `/logs [lines]`

## Logs (Where To Look)
Application event log (JSONL):
- `/home/c/personal-agent/logs/agent.jsonl`

Journal logs:
- `journalctl -u personal-agent.service -n 200 --no-pager`

## Troubleshooting
**Bot not responding**
- Check status: `sudo systemctl status personal-agent.service`
- Check logs: `journalctl -u personal-agent.service -n 200 --no-pager`

**Telegram Conflict: another poller**
- Error looks like: `Conflict: terminated by other getUpdates request`
- Ensure only one instance is running (no other machines polling this bot token).
- Restart service: `sudo systemctl restart personal-agent.service`

**No snapshots found**
- Run `/storage_snapshot` once to seed data.
- Or enable scheduled snapshots (set `ENABLE_SCHEDULED_SNAPSHOTS=1` in `/etc/personal-agent/agent.env` and restart).

## Scheduled Observe Timer
- Installed unit: `personal-agent-observe.service`
- Installed timer: `personal-agent-observe.timer`
- Default cadence: hourly (`OnCalendar=hourly`)
- Override cadence with a systemd drop-in (example 3x/day: `OnCalendar=*-*-* 09,15,21:00:00`)
- Commands:
  - `sudo systemctl status personal-agent-observe.timer`
  - `sudo systemctl list-timers | grep personal-agent-observe`

## Daily Brief (Opt-In)
- Enable via explicit chat phrase:
  - `daily brief on at 09:00`
  - `daily brief off`
- Optional brief controls:
  - `daily brief quiet on|off`
  - `set disk delta threshold to 300 mb`
  - `only send if service unhealthy on|off`
  - `set open loops due window to 3 days`
- Stored preferences:
  - `daily_brief_enabled` (`on|off`)
  - `daily_brief_time` (`HH:MM`, local agent timezone)
- Daily brief sends Telegram cards only; it does not run actions.

## Open Loops (Explicit Only)
- Add: `remember that I need to <title> by <YYYY-MM-DD>`
- Priority add: `remember that ! <title> by <YYYY-MM-DD>` (priority 1)
- List:
  - `/open_loops` (recent open)
  - `/open_loops due` (due-first open)
  - `/open_loops important` (priority-first, then due)
  - `/open_loops all` (open + done)
- Complete: `mark <title> done`

## Operational Trust
- `/health` shows cards for bot status, observe scheduler state, DB path/schema, daily brief config, and redacted last error.
- `/daily_brief_status` (or NL: `daily brief status`) explains why brief did/didn't send today.

## Versions
- App version file: `VERSION`
- DB schema version: stored in `schema_meta.schema_version`

## Security Notes
- Secrets live in `/etc/personal-agent/agent.env` (keep it `600` and root‑owned).
- Do not paste tokens or secrets into chat, logs, or GitHub issues.
