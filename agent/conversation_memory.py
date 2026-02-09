from __future__ import annotations

from typing import Any


def record_event(db: Any, user_id: str, topic: str, intent_type: str) -> None:
    """
    Best-effort conversation event recorder.

    This codebase has historically had multiple implementations for conversation memory.
    Orchestrator depends on a stable symbol; this wrapper must never raise.
    """
    try:
        fn = getattr(db, "insert_conversation_event", None)
        if callable(fn):
            fn(user_id=user_id, topic=topic, intent_type=intent_type)
            return
    except Exception:
        return

    # No backing store available in this revision. Intentionally a no-op.
    return

