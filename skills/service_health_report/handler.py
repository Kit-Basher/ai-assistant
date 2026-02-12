from __future__ import annotations

from typing import Any

from agent.cards import normalize_card
from agent.runtime_status import build_runtime_status_report


def service_health_report(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    db = context.get("db") if context else None
    if not db:
        text = "Database not available."
        return {"status": "blocked", "text": text, "message": text}
    report_text = build_runtime_status_report(db)
    lines = report_text.splitlines()
    service_line = next((line for line in lines if line.startswith("- status:")), "- status: unknown")
    log_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "3. Recent Logs":
            log_start = i + 1
            break
    log_lines = [line for line in lines[log_start:log_start + 6] if line.strip()]
    cards = [
        normalize_card(
            {
                "title": "Service health",
                "lines": [service_line.replace("- ", "", 1)],
                "severity": "ok" if "active" in service_line else "warn",
            },
            0,
        ),
        normalize_card(
            {
                "title": "Recent service logs",
                "lines": log_lines or ["No recent logs."],
                "severity": "ok",
            },
            1,
        ),
    ]
    return {"status": "ok", "text": report_text, "payload": {"report": report_text}, "cards_payload": {"cards": cards, "raw_available": True}}
