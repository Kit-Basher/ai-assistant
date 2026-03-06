# Golden Path

## Runtime Modes
- `READY`: normal operation.
- `BOOTSTRAP_REQUIRED`: setup guidance only.
- `DEGRADED`: short degraded status + one next action.
- `FAILED`: deterministic error block (trace + component + next_action).

## Setup Modes
- Onboarding states:
  - `NOT_STARTED`
  - `TOKEN_MISSING`
  - `LLM_MISSING`
  - `SERVICES_DOWN`
  - `READY`
  - `DEGRADED`
- Recovery modes:
  - `TELEGRAM_DOWN`
  - `API_DOWN`
  - `TOKEN_INVALID`
  - `LLM_UNAVAILABLE`
  - `LOCK_CONFLICT`
  - `DEGRADED_READ_ONLY`
  - `UNKNOWN_FAILURE`

## Tool Execution Path
- LLM-driven actions use one contract (`agent/tool_contract.py`) and one executor (`agent/tool_executor.py`).
- Read-only checks stay available in degraded/bootstrap when possible.
- Write actions remain policy-gated and may be blocked with one explicit next step.

## Continuity Path
- Continuity state is centralized (`agent/memory_runtime.py`).
- Follow-up phrases (`yes/no/do it/that one/show me more`) only bind when one valid pending item exists in the active thread.
- If ambiguous/expired/missing, the agent returns one deterministic next step instead of guessing.
- Meta summary commands (`memory/setup/status/doctor/resume`) do not overwrite the last meaningful action.

## When Everything Works
1. Restart service: `systemctl --user restart personal-agent-api.service`
2. Verify setup: `python -m agent setup --dry-run`
3. Verify runtime: `python -m agent status`
4. Talk to Telegram naturally.
5. Use `doctor/status/health/brief` when needed.
6. Telegram `help/setup/status/health/doctor/memory` semantics match the CLI/runtime contracts.
7. Telegram typo handling keeps continuity paths safe (`breif` -> brief, `memory/resume` -> continuity summary).
8. `READY` semantics are aligned across CLI and Telegram from the same runtime readiness source.

## When LLM Is Down
- Telegram/CLI should show deterministic setup or recovery guidance.
- Next step is explicit: run `python -m agent doctor`.

## When Telegram Token Is Wrong
- Startup checks fail with `failure_code=telegram_token_missing`.
- Fix token:
  - `python -m agent.secrets set telegram:bot_token`
  - `systemctl --user restart personal-agent-telegram.service`

## When Operator Is Confused
- Run exactly one command first: `python -m agent setup`
- If still blocked, run: `python -m agent doctor`
- Follow the single `Next action` line.
