from __future__ import annotations

from typing import Any

from agent.runtime_status import build_runtime_status_report


def runtime_status(context: dict[str, Any]) -> dict[str, Any]:
    db = context["db"]
    report_text = build_runtime_status_report(db)
    return {"text": report_text}
