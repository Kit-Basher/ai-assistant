# Doctor

Canonical local diagnostics entrypoint:

- `python -m agent doctor`
- `python -m agent doctor --json`
- `python -m agent doctor --collect-diagnostics`
- `python -m agent doctor --fix` (safe local fixes only)

Canonical release gate:

- `python scripts/release_smoke.py`
- heavier follow-up validation: `python scripts/release_validation_extended.py`

## Guarantees

- deterministic check ordering and stable output fields
- offline by default (`--online` enables Telegram getMe check)
- no sudo required
- no secret/token leakage (redacted output)
- trace id included in stdout and per-check logs
- stdout/journald visibility is part of the supported runtime model; `logging.stdout` should pass unless logging is explicitly misconfigured
- aligns with runtime permission semantics: read-only diagnostics are always preferred

## Canonical Collect-Diagnostics Path

Use this when you need one operator-safe support bundle without changing local state:

- `python -m agent doctor --collect-diagnostics`

It writes one redacted local bundle in a temp directory and prints the bundle path.
The bundle includes:

- the doctor report and a plain-text summary
- startup/self-check snapshots for API and Telegram
- redacted local API snapshots (`/health`, `/ready`, `/runtime`, `/llm/status`, `/memory`) when available
- canonical paths, backup targets, and restore guidance

If the local API is down, diagnostics collection still succeeds and records that
the fetch failed instead of pretending the runtime is healthy.

## Safe Fix Scope (`--fix`)

- creates missing local agent/systemd user directories
- creates missing `~/.config/personal-agent` operator config directory
- writes Telegram drop-in with `AGENT_SECRET_STORE_PATH` when safe
- removes stale Telegram lock files only when PID is not running
- copies legacy repo-local `memory/agent.db` and `logs/agent.jsonl` into
  `~/.local/share/personal-agent` when the canonical state files are missing
- generates local support bundle in a temp directory

## Backup / Restore Story

Supported today is a manual export/restore story, not a full backup subsystem.

Backup/export:

- stop the user services first
- copy:
  - `~/.config/personal-agent`
  - `~/.local/share/personal-agent`
  - `~/.config/systemd/user`

Restore/import:

- restore those copied paths to the same locations
- run `python -m agent doctor --fix`
- restart the user services
- verify `python -m agent status` and `/ready`

## Failed Upgrade / Corrupt State Recovery

- failed upgrade:
  - reinstall the supported repo checkout or wheel
  - run `python -m agent doctor --fix`
  - restart the user services
- corrupt state:
  - collect diagnostics first: `python -m agent doctor --collect-diagnostics`
  - move the corrupt file aside instead of editing it in place
  - restore from backup when available
  - run `python -m agent doctor --fix`

## Telegram

- Telegram command/text `doctor` runs the same checks (without fixes) and returns a short summary with trace id.
- Telegram command/text `setup` runs deterministic onboarding/recovery guidance (`python -m agent setup` equivalent, read-only).

## Output Semantics

- `PASS`: system is healthy; no operator action needed.
- `WARN`: degraded but running; follow the single `Next action` line.
- `FAIL`: startup/runtime blocking issue; follow the single `Next action` line immediately.
- Canonical first-run/recovery command remains: `python -m agent setup`.
- Runtime mode mapping used by user-facing surfaces:
  - `READY` ~= `PASS`
  - `BOOTSTRAP_REQUIRED` ~= setup-needed `WARN`
  - `DEGRADED` ~= partial-operation `WARN`
  - `FAILED` ~= blocking `FAIL`

Example:

```text
Trace ID: doctor-1700000000-1234
Status: WARN
Checks:
- [OK] env.python: python=3.12.3 exec=/home/c/personal-agent/.venv/bin/python
- [WARN] llm.availability: /llm/status unavailable: URLError:...
Next action: Run: python -m agent doctor --fix
```
