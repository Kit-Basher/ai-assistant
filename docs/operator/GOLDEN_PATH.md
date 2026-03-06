# Golden Path

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
  - `systemctl --user restart personal-agent-api.service`

## When Operator Is Confused
- Run exactly one command first: `python -m agent doctor`
- Follow the single `Next action` line.

