from __future__ import annotations

import json
from typing import Any


def _load_json(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return None
    return None


def _get_last_report(db: Any, user_id: str, report_key: str) -> dict[str, Any] | None:
    fn = getattr(db, "get_last_report", None)
    if callable(fn):
        row = fn(user_id, report_key)
        if isinstance(row, dict):
            payload = _load_json(row.get("payload") or row.get("payload_json"))
            if payload is not None:
                return payload
        if isinstance(row, tuple) and len(row) >= 2:
            payload = _load_json(row[1])
            if payload is not None:
                return payload

    # Fallback: query the sqlite connection directly if present.
    conn = getattr(db, "_conn", None)
    if conn is None:
        return None
    try:
        cur = conn.execute(
            """
            SELECT payload_json
            FROM last_report_registry
            WHERE user_id = ? AND report_key = ?
            """,
            (user_id, report_key),
        )
        row = cur.fetchone()
        if not row:
            return None
        return _load_json(row["payload_json"] if isinstance(row, dict) else row[0])
    except Exception:
        return None


def resource_followup(db: Any, user_id: str, kind: str, tz: str) -> str:
    _ = tz
    payload = _get_last_report(db, user_id, "resource_report")
    if not payload:
        return "Run /resource_report first so I have a recent snapshot to explain."

    if kind == "top_memory":
        rss = payload.get("rss_samples") or []
        lines = ["Top memory processes:"]
        for p in rss:
            name = p.get("name", "unknown")
            rss_b = int(p.get("rss_bytes") or 0)
            lines.append(f"- {name}: {rss_b}B RSS")
        return "\n".join(lines)

    if kind == "compare":
        fn = getattr(db, "get_report_history", None)
        history = None
        if callable(fn):
            history = fn(user_id, "resource_report", limit=2)

        if not history:
            # Best-effort fallback: return deterministic current value.
            mem_used = int((payload.get("memory") or {}).get("used") or 0)
            return f"Memory used delta: (previous report unavailable) current_used={mem_used}"

        newest = _load_json(history[0].get("payload_json") if isinstance(history[0], dict) else history[0]) or {}
        prev = _load_json(history[1].get("payload_json") if isinstance(history[1], dict) else history[1]) or {}
        newest_used = int((newest.get("memory") or {}).get("used") or 0)
        prev_used = int((prev.get("memory") or {}).get("used") or 0)
        return f"Memory used delta: {newest_used - prev_used}"

    return "Unknown followup."

