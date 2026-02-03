from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

SECRET_KEYS = {"openai_api_key", "telegram_bot_token", "authorization", "api_key"}


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted = {}
        for key, value in obj.items():
            if key.lower() in SECRET_KEYS:
                redacted[key] = "***redacted***"
            else:
                redacted[key] = _redact(value)
        return redacted
    if isinstance(obj, list):
        return [_redact(item) for item in obj]
    return obj


def log_event(log_path: str, event_type: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "payload": _redact(payload),
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
