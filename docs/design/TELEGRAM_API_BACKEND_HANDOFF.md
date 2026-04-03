Telegram ordinary chat now uses the local API as its authoritative backend.

Before this change:
- `run_polling_with_backoff()` built a local `AgentRuntime` and injected it into the Telegram app.
- ordinary non-command Telegram text could hit `runtime.chat()` inside `agent/telegram_bridge.py`
- that meant Telegram could crash or diverge when its local runtime was not initialized the same way as the API runtime

After this change:
- `run_polling_with_backoff()` no longer builds or injects a local runtime by default
- `_handle_message()` now passes local API callbacks into `handle_telegram_text()`
- ordinary non-command Telegram text is proxied to `POST /chat`
- Telegram formats the API/orchestrator response and preserves route/provenance metadata from that response

What still stays local in Telegram:
- polling, send/retry behavior, truncation, and audit logging
- explicit slash-command parsing
- `/help`, `/setup`, and `/status` formatting in the bridge, backed by local API `GET` calls when no runtime is injected
- `/doctor`, `/health`, `/brief`, `/memory`, reminders, and fix-it wizard transport state

Compatibility note:
- `agent/telegram_bridge.py` still has local-runtime and local-orchestrator fallback branches for direct/unit callers that do not provide API callbacks
- the live Telegram polling path no longer relies on those branches for ordinary chat
