# Live Behavior Smokes

Run these smokes after promoting the current checkout into the stable runtime and before calling the assistant "working".

Recommended sequence:

```bash
bash scripts/promote_local_stable.sh
python scripts/live_user_barrage.py --base-url http://127.0.0.1:8765 --telegram-bridge --timeout 90
python scripts/telegram_bridge_smoke.py --base-url http://127.0.0.1:8765 --timeout 90
python scripts/live_model_switch_smoke.py --base-url http://127.0.0.1:8765 --timeout 90
```

`scripts/live_user_barrage.py` sends messy real-user prompts through the live `/chat` path using a stable `user_id` and `thread_id`. With `--telegram-bridge`, it also runs the same prompt set through `agent.telegram_bridge.handle_telegram_text()` using the local API chat proxy, without contacting Telegram servers.

The barrage refuses to run unless `/ready` reports `ok: true` and `ready: true`. If it fails at readiness, fix the degraded runtime first instead of interpreting prompt failures.

The barrage fails on empty answers, stale `OK` / `Done.` answers, diagnostic prompts answered with stale "ready to help" wording, internal schema/debug leaks, temporary model switches claiming a default update, and browser/skill-install requests being treated as model acquisition.
