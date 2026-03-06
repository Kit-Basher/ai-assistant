# Doctor

Canonical local diagnostics entrypoint:

- `python -m agent doctor`
- `python -m agent doctor --json`
- `python -m agent doctor --fix` (safe local fixes only)

## Guarantees

- deterministic check ordering and stable output fields
- offline by default (`--online` enables Telegram getMe check)
- no sudo required
- no secret/token leakage (redacted output)
- trace id included in stdout and per-check logs
- aligns with runtime permission semantics: read-only diagnostics are always preferred

## Safe Fix Scope (`--fix`)

- creates missing local agent/systemd user directories
- writes Telegram drop-in with `AGENT_SECRET_STORE_PATH` when safe
- removes stale Telegram lock files only when PID is not running
- generates local support bundle in a temp directory

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
