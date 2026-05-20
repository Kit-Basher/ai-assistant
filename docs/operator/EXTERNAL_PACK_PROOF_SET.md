# External Pack Proof Set

Run this proof set after any change to external-pack source handling, catalog validation, archive intake, ingestion, lifecycle, lifecycle actions, managed adapters, adapter invocation, support redaction, or related assistant routing.

Required sequence:

```bash
bash scripts/promote_local_stable.sh
python scripts/external_pack_safety_smoke.py
python -u scripts/live_user_barrage.py --base-url http://127.0.0.1:8765 --telegram-bridge --timeout 90 --strict-quality
git status
```

`scripts/external_pack_safety_smoke.py` proves the hostile intake gates still hold: remote source trust, strict catalog schema, archive extraction hardening, imported instruction handling, lifecycle/managed-adapter gates, and support/tombstone redaction.

`scripts/live_user_barrage.py --strict-quality` proves normal assistant behavior still works after the external-pack change. It checks the live `/chat` path and the Telegram bridge proxy for stale context, low-value replies, contradictions, bad routing, and normal runtime/open-chat behavior.

Both smokes must pass before expanding online skill ecosystem support. Passing this proof set does not make remote content trusted; it only shows the current hostile-by-default intake chain and normal assistant behavior did not regress.
