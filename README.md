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
  - `OLLAMA_HOST` (default: empty)
  - `OLLAMA_MODEL` (default: empty)
  - `OLLAMA_MODEL_SENTINEL` (optional override)
  - `OLLAMA_MODEL_WORKER` (optional override)
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (default: `gpt-4.1-mini`)
  - `OPENAI_MODEL_WORKER` (optional override)
  - `ALLOW_CLOUD` (default: `1`)
  - `PREFER_LOCAL` (default: `1`)
  - `LLM_TIMEOUT_SECONDS` (default: `20`)
  - `AGENT_TIMEZONE` (default: `America/Regina`)
  - `AGENT_DB_PATH` (default: `memory/agent.db`)
  - `AGENT_LOG_PATH` (default: `logs/agent.jsonl`)
  - `AGENT_SKILLS_PATH` (default: `skills/`)

## LLM Routing (v0.2)
- Builds candidates by capability tier (low, mid, high) and provider.
- Prefers local models first when `PREFER_LOCAL=1`.
- Falls back on timeout/error to the next cheapest sufficient candidate.
- Cloud models are only considered when `ALLOW_CLOUD=1`.
- Watchdog tasks are forced local-only and degrade to Dummy if needed.

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

## Git Skill Notes
- `git_push` is intentionally blocked in this environment; run `git push` on the host machine.

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

## Architecture & Stability
- `STABILITY.md`
- `ARCHITECTURE.md`
- `RELEASE_NOTES.md`

## Governance & Safety
- `THREAT_MODEL.md`
- `SECURITY.md`
- `RELEASES/v0.5-sandbox-runner-audited.md`

## LLM Models (Optional)
- `LLM_MODELS.md`

## Why This Agent Cannot Surprise You
This agent is deliberately conservative. It is designed to explain what it can do, refuse what it cannot do, and leave a paper trail for anything that might matter. The goal is not to be clever; the goal is to be predictable.

What the agent can do:
- Observe system state, such as disk usage and recent changes.
- Report facts and summaries based on local data.
- Simulate actions in sandbox mode without touching the system.

What the agent cannot do:
- It cannot modify the system by default.
- It cannot act without an explicit mode and explicit consent.
- It cannot bypass audit logging.
- It cannot act if audit logging fails.
- It cannot silently escalate its own autonomy or scope.

Three execution modes define its behavior:
- off: Transparent passthrough. The agent does not enforce autonomy rules and does not simulate actions. This mode preserves existing behavior and avoids accidental changes to runtime semantics.
- sandbox: Simulation only. Commands are checked against allowlists, mutating commands are blocked, and a would-run record is produced. No real effects are permitted.
- live: Gated, audited, opt-in only. This mode is disabled unless explicitly enabled and is treated as a higher-risk execution path.

Core invariant: “If it can’t be audited, it didn’t happen.”
Sandbox and live executions are required to write an audit entry. If audit logging fails, the operation is aborted and the user receives the exact message: “Audit logging failed. Operation aborted.” This is not a suggestion or a warning. It is a hard stop, enforced by tests.

Freeze discipline keeps the trust model stable. v0.5 semantics are frozen. New powers require a new guardrail document and a new tag. There are no “temporary” bypasses and no silent behavior shifts. If something changes, it must be explicit, documented, and tested.

This section is intentionally plain and restrictive. You should be able to trust the system without reading the code. If any behavior here ever becomes untrue, it is a bug that must be fixed before new features are added.
