# Telegram Setup

## Token source
- Secret key: `telegram:bot_token`
- Preferred workflow:
  - `python -m agent.secrets set telegram:bot_token` previews an authorized
    secret Plan; it never writes directly. Apply needs the exact Plan plus an
    already valid scoped confirmation artifact.
  - `python -m agent.secrets get telegram:bot_token --redacted`
  - `systemctl --user restart personal-agent-telegram.service`

## Verify
- Service logs:
  - `journalctl --user -u personal-agent-telegram.service -n 100 --no-pager`
- Setup guidance:
  - `python -m agent setup`
- Runtime split and bridge contract:
  - `python -m agent split_status`
  - `python scripts/telegram_bridge_smoke.py`
- Healthy startup log markers:
  - `telegram.started`
  - `telegram.out`
- Conflict check:
  - look for `getUpdates conflict — another poller is active for this token`
  - duplicate replies usually mean two pollers are running with the same token

## Duplicate poller guard
- Telegram polling uses a token-scoped local lock under:
  - `~/.local/share/personal-agent/telegram_poll.<token_hash>.lock`
- On lock conflict, the second poller exits and logs a warning instead of thrashing.

## Stable API alignment
- Maintainer/dev workstations may run `personal-agent-telegram.service` from
  `~/personal-agent/.venv` while `personal-agent-api.service` runs from
  `~/.local/share/personal-agent/runtime/current`.
- In that split, Telegram is a transport adapter: ordinary non-command text must
  proxy to the stable API `POST /chat`.
- Keep assistant behavior in the API path. Do not duplicate model switching,
  memory, continuity, or fallback logic in Telegram.
- `python scripts/telegram_bridge_smoke.py` exercises the local Telegram
  message-to-`/chat` bridge without hitting real Telegram servers.

## systemd override cleanup
- If `systemctl --user edit` fails because of a staging file, remove temp editor files:
  - `~/.config/systemd/user/personal-agent-telegram.service.d/.#override.conf*`
