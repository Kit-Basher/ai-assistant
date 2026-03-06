# Golden Path

## Runtime Modes
- `READY`: normal operation.
- `BOOTSTRAP_REQUIRED`: setup guidance only.
- `DEGRADED`: short degraded status + one next action.
- `FAILED`: deterministic error block (trace + component + next_action).

## Tool Execution Path
- LLM-driven actions use one contract (`agent/tool_contract.py`) and one executor (`agent/tool_executor.py`).
- Read-only checks stay available in degraded/bootstrap when possible.
- Write actions remain policy-gated and may be blocked with one explicit next step.

## When Everything Works
1. Restart service: `systemctl --user restart personal-agent-api.service`
2. Verify: `python -m agent status`
3. Talk to Telegram naturally.
4. Use `doctor/status/health/brief` when needed.

## When LLM Is Down
- Telegram/CLI should show deterministic setup or recovery guidance.
- Next step is explicit: run `python -m agent doctor`.

## When Telegram Token Is Wrong
- Startup checks fail with `failure_code=telegram_token_missing`.
- Fix token:
  - `python -m agent.secrets set telegram:bot_token`
  - `systemctl --user restart personal-agent-telegram.service`

## When Operator Is Confused
- Run exactly one command first: `python -m agent doctor`
- Follow the single `Next action` line.
