# Setup

Canonical product/runtime source: [`PRODUCT_RUNTIME_SPEC.md`](/home/c/personal-agent/PRODUCT_RUNTIME_SPEC.md).

Canonical first-run command:

- `python -m agent setup`
- JSON: `python -m agent setup --json`
- Dry-run: `python -m agent setup --dry-run`

## First Run

1. Run `python -m agent setup`.
2. Follow exactly one `Next action`.
3. Re-run `python -m agent status` until runtime is stable.
4. Use native UI as primary setup/recovery surface; Telegram mirrors runtime setup state when enabled.
5. Telegram is optional and off by default (`TELEGRAM_ENABLED=0`).

## Setup Complete

Setup is complete when onboarding state is `READY` and:

- `python -m agent status` reports healthy runtime mode.
- Telegram `help/setup/status/health/doctor/memory` follows the same canonical contract text as CLI outputs.
- Telegram `memory` / `what are we doing?` / `resume` routes to continuity summary (not setup fallback).

## Onboarding States

- `NOT_STARTED`
- `TOKEN_MISSING`
- `LLM_MISSING`
- `SERVICES_DOWN`
- `READY`
- `DEGRADED`

## Recovery States

- `TELEGRAM_DOWN`
- `API_DOWN`
- `TOKEN_INVALID`
- `LLM_UNAVAILABLE`
- `LOCK_CONFLICT`
- `DEGRADED_READ_ONLY`
- `UNKNOWN_FAILURE`

## Short Recovery Paths

- token missing:
  - (only when Telegram is enabled)
  - `python -m agent.secrets set telegram:bot_token`
  - `systemctl --user restart personal-agent-telegram.service`
- telegram down:
  - (only when Telegram is enabled)
  - `systemctl --user restart personal-agent-telegram.service`
- api down:
  - `systemctl --user restart personal-agent-api.service`
- llm unavailable:
  - `python -m agent setup`
  - `python -m agent doctor`

## If Setup Fails

1. Run `python -m agent doctor`.
2. Follow the single `Next action`.
3. Re-run `python -m agent setup`.
