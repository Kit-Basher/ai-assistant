from __future__ import annotations

from agent.public_chat import build_public_sentence_text


def build_safe_mode_paused_message(
    *,
    reason: str | None,
    blocked_detail: str | None = None,
    unpause_endpoint: str = "/llm/autopilot/unpause",
    unpause_command: str = 'POST /llm/autopilot/unpause {"confirm": true}',
) -> str:
    return build_public_sentence_text(
        "Automatic changes are paused for safety",
        "I can't apply that change right now.",
        f"To unpause, call {unpause_command} (endpoint: {unpause_endpoint})",
    )
