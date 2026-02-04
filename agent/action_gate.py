from __future__ import annotations

from typing import Any


def handle_action_text(_db: Any, _user_id: str, _text: str, _enable_writes: bool) -> dict[str, Any] | None:
    return None


def propose_action(
    _db: Any,
    _user_id: str,
    _action_type: str,
    _action_id: str,
    _details: dict[str, Any],
) -> str:
    return "Action proposals are disabled in this build."
