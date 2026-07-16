from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from agent.security.redaction import redact_value


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = redact_value(payload)
    return redacted if isinstance(redacted, dict) else {}


def log_event(log_path: str, event_type: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "payload": redact_payload(payload),
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
