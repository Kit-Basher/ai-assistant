# Telegram First-Reply Reliability v0.2.4

## Reported Failure

The first Telegram message after service startup, `are you ther?`, took about a
minute and returned a weak generic fallback. A later greeting worked normally.

## Changes

Simple greeting and presence-check messages now use a local fast path before
the web/tool/model route:

- `hello`
- `hello?>`
- `hi`
- `hey`
- `are you there?`
- `are you ther?`
- `you there?`
- `ping`
- `test`

These responses do not call web search, local tools, or the LLM path.

Telegram health now distinguishes:

- `CONFIGURED`
- `POLLING`
- `RECEIVING`
- `DISPATCHING`
- `REPLYING`
- `HEALTHY`
- `DEGRADED`
- `UNVERIFIED`

The diagnostic state includes bounded redacted trace fields for last update,
dispatch, reply attempt, reply success/failure, round trip, and first cold
start timing. It does not include message text or raw Telegram payloads.

## Verification

Run:

```bash
python scripts/telegram_first_reply_latency_smoke.py
python scripts/telegram_transport_smoke.py
python scripts/telegram_transport_diagnostic.py
```

The fixture smoke proves the first typo greeting returns a presence response
without calling the local API. A live Telegram canary remains optional because
it requires an operator-controlled real message.

## Remaining Limitations

Live Telegram delivery still depends on the optional user service being active,
Telegram availability, and a valid rotated token. The canary is not automatic
because it may send or receive real Telegram messages.
