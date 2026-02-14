# PERSONAL AGENT - AUTHORITATIVE PROJECT STATUS

Last updated: 2026-02-14

A living snapshot of where the project stands, what exists, and what the next logical steps are. Designed to be honest, concise, and grounded in reality, not aspiration.

## A. Project Overview

Name: Personal Agent  
Purpose: A local-first digital companion that helps organize life, remembers context, and interacts via Telegram.

Key characteristics:
- Runs locally (Python + SQLite)
- Uses systemd timers for scheduling
- Predictable behavior; no hidden autonomy
- Friendly, evolving personality potential
- Memory that can be reinforced and retrieved

Philosophy:  
We are building a companion that is:
- useful
- safe
- persistent
- adaptive but not emotionally manipulative
- able to grow in capability and subtle personality shift

## B. Current Implementation (Code Reality)

### Branch and Version
- Current branch: `brief-v0.2-clean`
- HEAD: `c5420da`
- Active version: `v0.2.0`
- Daily brief and core responses are implemented.

### Dependencies
- `python-telegram-bot >=22.6`
- `openai >=1.0.0`

## C. Commands Surface

Registered Telegram commands:

| Command | Purpose |
| --- | --- |
| `/ask` | Ask a question (LLM) |
| `/ask_opinion` | Ask for opinion synthesis |
| `/audit` | Show recent audit log |
| `/brief` | Manual daily brief |
| `/daily_brief_status` | Check scheduled brief |
| `/disk_grow` | Disk growth view for a target path |
| `/health` | System health summary |
| `/network_report` | Network report |
| `/open_loops` | List open loops |
| `/remind` | Set a reminder |
| `/resource_report` | Resource usage summary |
| `/runtime_status` | Runtime diagnostics |
| `/storage_snapshot` | Trigger storage snapshot collection |
| `/storage_report` | Storage usage summary |
| `/task_add` | Add a task |
| `/done` | Mark a task done by id |
| `/status` | Agent runtime status |
| `/today` | Show today's view |
| `/weekly_reflection` | Weekly reflection prompt |

Other command logic exists in orchestrator that is not Telegram-registered yet (for example: `/project_new`, `/remember`, `/plan`, `/next`).

## D. Storage and Schema

DB: SQLite (`memory/agent.db`)  
Schema: Loaded from `memory/schema.sql`

Tables include:
- `projects`
- `tasks`
- `open_loops`
- `reminders`
- `preferences`
- `activity_log`
- `audit_log`
- various metric/snapshot tables (`resource_`, `network_`)
- `schema_meta` for versioning
- `last_report_registry`, `report_history`

Important: Tasks already exist and are used by brief logic; task planning UI is still stubbed.

## E. Daily Brief - What Exists

Daily brief functionality now runs via systemd only, not in-process Telegram scheduler.

What it does:
- Reads open tasks and open loops
- Filters by due dates
- Builds structured Markdown summary
- Sends Telegram message once per day
- Prevents duplicate sends via `daily_brief_last_sent_date`

Sections generated:
- Today
- Top tasks (sorted by due/impact)
- Due soon
- Open loops
- A single nudge (oldest item)

## F. What Works Today

- Tasks can be added via `/task_add`
- Tasks can be marked done (`/done`)
- Daily brief sends with summary
- Systemd services are installed via `ops/install.sh`
- Doctor service checks daily brief timer health
- Preferences stored in DB for config
- Open loops tracked

## G. Known Gaps / Stubs

The following commands are present in orchestrator but the UX is currently stubbed or incomplete:
- `/next` -> placeholder
- `/plan` -> placeholder
- `/weekly` -> placeholder
- Task editing / rich task workflows
- More expressive daily interaction beyond a report

## H. Agent Identity, Memory and Personality (Intent)

This section does not yet exist in code but defines the vision.

Memory system goals:
- Multi-tier memory (working / episodic / semantic)
- Decay and reinforcement
- Tagging + retrieval
- Pinning for non-forget

Personality and evolution:
- Adapt tone to user preferences
- Allow personality drift based on interaction history
- Never use emotional punishment mechanics
- Preferences override personality drift

This will be a focus of the next iteration.

## I. Next Logical Steps (Concrete)

### Foundational
- Finalize Telegram handlers for important commands.
- Keep `/task_add` and `/done` behavior stable as command surface evolves.
- Fix any malformed test files.
- Ensure naming, imports, and `if __name__ == "__main__"` are correct.
- Clean up orchestrator stub handlers.
- Fill out `/next`, `/plan`, `/weekly` in simple iterations.

### Memory and Personality
- Build Memory Layer Tier 0.
- Start with a table for episodic events + reinforcement counts.
- Build simple retrieval.
- Tag + retrieve relevant memories for daily check-ins.
- Daily companion message.
- Replace or extend daily brief with friend-like check-in.

### Vision Phase
- Adaptive personality.
- Slow drift based on user interaction.
- Style preferences in DB.
- Controlled sandbox actions.

## J. Testing Reality

- 168 tests passing
- New tests added for daily brief entrypoint + scheduler
- Doctor tests include daily brief timer health
- If tests fail after changes, new behavior should be documented.

## K. Definition of Done (Current Goals)

- Tasks/states are stored and retrievable
- Daily companion message version exists
- Memory can be recalled and reinforced
- Personality evolves slowly over time
- Telegram UX is consistent and helpful

## L. What This Project Actually Is

A companion you raise: a semi-autonomous agent with memory, personality, and a sense of continuity that helps you organize your life, and grows in capability over time without ever punishing you emotionally.
