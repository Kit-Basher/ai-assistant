from __future__ import annotations


def build_safe_mode_paused_message(
    *,
    reason: str | None,
    blocked_detail: str | None = None,
    unpause_endpoint: str = "/llm/autopilot/unpause",
    unpause_command: str = 'POST /llm/autopilot/unpause {"confirm": true}',
) -> str:
    reason_text = str(reason or "").strip() or "churn_detected"
    detail_text = str(blocked_detail or "").strip()
    detail_clause = f" Blocked change: {detail_text}." if detail_text else ""
    return (
        "Safe mode is a guardrail that pauses automatic apply changes after unstable configuration churn. "
        f"It is currently paused. Reason: {reason_text}.{detail_clause} "
        f"To unpause, call {unpause_command} (endpoint: {unpause_endpoint}). "
        "While paused, chat and read-only endpoints still work, but autopilot apply actions remain blocked."
    )
