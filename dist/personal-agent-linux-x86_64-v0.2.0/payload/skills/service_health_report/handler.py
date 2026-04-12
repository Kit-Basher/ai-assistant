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
    service_lines = [
        line
        for line in lines
        if line.startswith("- personal-agent-api.service:") or line.startswith("- personal-agent-telegram.service:")
    ]
    if not service_lines:
        service_lines = ["personal-agent-api.service: status=unknown"]
    api_line = next(
        (line for line in service_lines if line.startswith("- personal-agent-api.service:")),
        service_lines[0],
    )
    api_active = "status=active" in api_line.lower()
    telegram_failed = any(
        line.startswith("- personal-agent-telegram.service:") and "status=failed" in line.lower()
        for line in service_lines
    )
    log_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "2. Recent Logs":
            log_start = i + 1
            break
    log_lines = [line for line in lines[log_start:log_start + 6] if line.strip()]
    cards = [
        normalize_card(
            {
                "title": "Service health",
                "lines": [line.replace("- ", "", 1) for line in service_lines],
                "severity": "warn" if (not api_active or telegram_failed) else "ok",
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
