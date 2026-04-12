from __future__ import annotations

from typing import Any


def _semantic_service(db: Any) -> Any | None:
    for attr in ("_semantic_memory_service", "semantic_memory"):
        service = getattr(db, attr, None)
        if service is not None:
            return service
    return None


def _notify_memory_issue(db: Any, *, operation: str, error: str, details: dict[str, Any] | None = None) -> None:
    notifier = getattr(db, "_memory_issue_notifier", None)
    if not callable(notifier):
        return
    try:
        notifier(
            subsystem="semantic_memory",
            operation=operation,
            error=error,
            details=details or {},
        )
    except Exception:
        return


def record_event(db: Any, user_id: str, topic: str, intent_type: str) -> None:
    """
    Best-effort conversation event recorder.

    This codebase has historically had multiple implementations for conversation memory.
    Orchestrator depends on a stable symbol; this wrapper must never raise.
    """
    try:
        service = _semantic_service(db)
        if service is None:
            return
        normalized_topic = str(topic or "").strip()
        if not normalized_topic:
            return
        scope = f"user:{str(user_id or '').strip()}" if str(user_id or "").strip() else "global"
        metadata = {
            "topic": normalized_topic,
            "intent_type": str(intent_type or "").strip() or None,
        }
        ingest = getattr(service, "ingest_conversation_text", None)
        if callable(ingest):
            ingest(
                source_ref=f"topic:{user_id}:{normalized_topic}",
                text=normalized_topic,
                scope=scope,
                metadata=metadata,
            )
    except Exception as exc:
        _notify_memory_issue(
            db,
            operation="compat_record_event",
            error=exc.__class__.__name__,
            details={"topic": str(topic or "").strip() or None},
        )
        return

    return
