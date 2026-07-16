# Telegram Token Redaction v0.2.4

## Reported Failure

Real Telegram testing after v0.2.3 showed that HTTP client logging could write
Telegram Bot API URLs to the user journal. Those URLs contain the bot token.

Treat any previously logged Telegram bot token as compromised.

If any log previously contained `https://api.telegram.org/bot...`, rotate the
bot token in BotFather, update the Personal Agent secret store, and restart
`personal-agent-telegram.service`.

This patch does not rotate secrets automatically.

## Redaction Boundary

Personal Agent now uses a shared redaction helper for runtime logs, structured
event logs, audit records, diagnostics, and smoke evidence.

It redacts:

- `https://api.telegram.org/bot<TOKEN>/<method>` while preserving the method;
- `http://api.telegram.org/bot<TOKEN>/<method>`;
- raw `bot<TOKEN>` fragments;
- known secret values supplied by trusted callers;
- `Authorization: Bearer ...`;
- `token=...`, `api_key=...`, and related query/header forms.

It also redacts nested JSON-like structures and fails closed to a redaction
error marker rather than raising during logging.

## Logging Policy

Telegram transport events keep operational metadata only:

- event name;
- trace id;
- redacted chat id;
- stable scope hash;
- route;
- message length;
- bounded redacted error class/summary;
- timing spans.

They do not include:

- bot tokens;
- Telegram API URLs with real tokens;
- raw update payloads;
- message bodies;
- full chat ids;
- authorization headers.

## Verification

Run:

```bash
python scripts/telegram_token_redaction_smoke.py
python scripts/telegram_transport_diagnostic.py
```

After promoting the installed runtime, manually inspect journald without
pasting secrets into reports:

```bash
journalctl --user -u personal-agent-telegram.service -n 100 --no-pager -l
```

Expected result: no full Telegram token and no
`https://api.telegram.org/bot<real-token>` URL. Method names may appear with
`bot<redacted>`.
