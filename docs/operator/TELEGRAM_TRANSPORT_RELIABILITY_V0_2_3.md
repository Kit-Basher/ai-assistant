# Telegram Transport Reliability v0.2.3

## Reported Failure

Normal use found that Telegram could be configured while the bot did not answer
messages. Existing checks proved configuration and code-path survival, but they
did not prove the full live chain:

```text
Telegram update -> transport -> handler -> runtime -> response -> reply
```

## Architecture

Telegram can run as the embedded runtime poller or through the managed
`personal-agent-telegram.service`. The adapter uses polling, registers command
and text handlers in `telegram_adapter.bot`, and dispatches text messages into
the same local runtime path used by the API.

## Diagnostics

`/telegram/status`, `python -m agent telegram_status`, and
`scripts/telegram_transport_diagnostic.py` now expose a no-send status model:

- configured/token present;
- transport mode;
- polling activity;
- webhook-active placeholder;
- handler registration;
- last update received/processed timestamp;
- last reply attempt/success timestamp;
- bounded last error code/summary;
- duplicate consumer suspicion;
- runtime reachability.

The diagnostic never prints the bot token or incoming message text.

## Conflict Handling

Polling/webhook and duplicate-poller conflicts are reported as degraded status.
The diagnostic may recommend a repair, but it does not delete webhooks, restart
services, or send messages automatically.

## Logging

New redacted events include:

- `telegram_update_received`
- `telegram_update_dispatched`
- `telegram_reply_attempted`
- `telegram_reply_succeeded`
- `telegram_reply_failed`

These events include timestamps, route, redacted chat id, update type, latency,
and error class. They do not include tokens or message bodies.

## Proof

`scripts/telegram_transport_smoke.py` uses fake Telegram updates and fake reply
delivery to prove the inbound-to-reply path without contacting Telegram.

`scripts/telegram_transport_diagnostic.py` inspects the live configured
transport without sending a message. A real live canary remains optional because
it would require explicit user intent to send or reply through Telegram.

## Limitations

- Live delivery still depends on Telegram availability and a valid user token.
- Per-user authorization and allowed-chat rules remain enforced by the existing
adapter path.
- Repairs remain bounded service/config actions and must use the normal
authorization/confirmation boundary when added or invoked.
