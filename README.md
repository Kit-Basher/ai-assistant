# Personal Agent (Telegram-first)

A minimal Telegram-first personal agent with SQLite memory, a safe skills manifest, and a systemd service for 24/7 uptime.

## Features (v0.1)
- Telegram bot with slash commands
- Natural language intent routing for common requests (non-slash messages)
- SQLite memory initialized on first run
- Skill loader with manifest-based permissions
- Structured JSONL logging
- Reminder scheduler (checks every 30s)

## Setup
1. Create a virtual environment:
   - `python3.11 -m venv .venv`
   - `. .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`

## Environment Variables
- Required:
  - `TELEGRAM_BOT_TOKEN`
- Optional:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (default: `gpt-4o-mini`)
  - `AGENT_TIMEZONE` (default: `America/Regina`)
  - `AGENT_DB_PATH` (default: `memory/agent.db`)
  - `AGENT_LOG_PATH` (default: `logs/agent.jsonl`)
  - `AGENT_SKILLS_PATH` (default: `skills/`)

## Run Locally
- `TELEGRAM_BOT_TOKEN=... .venv/bin/python -m telegram_adapter`

## Telegram Commands
- `/remember <text>`
- `/projects`
- `/project_new <name> | <pitch>`
- `/task_add <project> | <title> | <effort_mins> | <impact_1to5>`
- `/remind <YYYY-MM-DD HH:MM> | <text>`
- `/next <minutes> <energy:low|med|high>`
- `/plan <minutes> <energy:low|med|high>`
- `/done <free text>`
- `/weekly`

## Natural Language Examples
- "remember buy milk #groceries"
- "what should I do next? 30 minutes low energy"
- "plan my evening, 2 hours high"
- "weekly review"
- "projects"
- "remind me 2026-02-05 15:00 to call dentist"

## Intent Phrases (v0.1)
- Remember note:
  - "remember ..."
  - "save this ..."
  - "make a note ..."
  - "jot this down ..."
  - "don't forget ..."
- Next-best task:
  - "what should I do next"
  - "what next"
  - "next task"
  - "what's next"
  - "what should I do now"
- Daily plan:
  - "plan my day"
  - "plan my evening"
  - "plan my morning"
  - "plan my afternoon"
  - "make me a plan"
- Weekly review:
  - "weekly review"
  - "how did I do this week"
  - "review this week"
  - "weekly check-in"
  - "weekly check in"
- List projects:
  - "projects"
  - "my projects"
  - "show my projects"
  - "list my projects"
  - "what am I working on"
- Reminders (explicit timestamp required):
  - "remind me ..."
  - "set a reminder ..."
  - "set reminder ..."
  - "reminder for ..."

## Clarifications
- If a natural language request is missing required details, the bot asks one clarifying question with a few options.
- Clarifications expire after 10 minutes.
- Reminder clarification suggestions are generated in the user's timezone with future-only defaults (next day at 09:00, 17:00, 20:00).
- If you reply with a time-only string (e.g., `2:00`), the bot responds with explicit AM/PM timestamp choices.
- Slash commands still work unchanged.

## Intent Metrics
- Logged as `intent_metric` events in JSONL:
  - `intent_matched` (includes intent/function name)
  - `clarification_triggered` (includes intent when known)
  - `no_intent` (fallback)

## Systemd Install
1. Copy the service file:
   - `sudo cp infra/personal-agent.service /etc/systemd/system/personal-agent.service`
2. Edit the service file if needed (paths, env vars).
3. Reload and enable:
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable --now personal-agent`
4. Check status:
   - `sudo systemctl status personal-agent`

## Notes
- Logs are append-only JSONL at `logs/agent.jsonl`.
- Reminders use `AGENT_TIMEZONE` for parsing, then store UTC internally.
- Skills are loaded from `skills/*/manifest.json` at startup.
