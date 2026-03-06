# Telegram Setup

## Token source
- Secret key: `telegram:bot_token`
- Preferred workflow:
  - `python -m agent.secrets set telegram:bot_token`
  - `python -m agent.secrets get telegram:bot_token --redacted`
  - `systemctl --user restart personal-agent-telegram.service`

## Verify
- Service logs:
  - `journalctl --user -u personal-agent-telegram.service -n 100 --no-pager`
- Setup guidance:
  - `python -m agent setup`
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

## systemd override cleanup
- If `systemctl --user edit` fails because of a staging file, remove temp editor files:
  - `~/.config/systemd/user/personal-agent-telegram.service.d/.#override.conf*`
