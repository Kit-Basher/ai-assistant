from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from agent.security.redaction import redact_value
from agent.internal_writer_authority import perform_registered_internal_write


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = redact_value(payload)
    return redacted if isinstance(redacted, dict) else {}


def log_event(log_path: str, event_type: str, payload: dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "payload": redact_payload(payload),
    }
    def _append() -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    perform_registered_internal_write(
        writer_id="logging_utils",
        operation="append_log",
        resource_type="redacted_log",
        target_scope="state:logs",
        arguments={"event_type": str(event_type), "timestamp": record["ts"]},
        callback=_append,
        journal_path=f"{log_path}.internal-writer.sqlite3",
        operation_id=f"log:{record['ts']}",
    )
