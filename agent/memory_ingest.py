from __future__ import annotations

from typing import Any, Tuple


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


def ingest_event(
    db: Any,
    user_id: str,
    kind: str,
    text: str,
    tags: list[str] | None = None,
    *,
    override: bool | None = None,
) -> None:
    try:
        kind_norm = str(kind or "").strip().lower()
        if kind_norm not in {"user", "conversation", "message", "event"}:
            return
        service = _semantic_service(db)
        if service is None:
            return
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return
        tag_values = [str(item).strip() for item in (tags or []) if str(item).strip()]
        scope = f"user:{str(user_id or '').strip()}" if str(user_id or "").strip() else "global"
        metadata = {
            "kind": kind_norm or "event",
            "tags": tag_values,
            "override": bool(override) if override is not None else None,
        }
        ingest = getattr(service, "ingest_conversation_text", None)
        if callable(ingest):
            ingest(
                source_ref=f"{kind}:{user_id}",
                text=normalized_text,
                scope=scope,
                pinned=False,
                metadata=metadata,
            )
            return
    except Exception as exc:
        _notify_memory_issue(
            db,
            operation="compat_ingest_event",
            error=exc.__class__.__name__,
            details={"kind": str(kind or "").strip().lower() or None},
        )
        return


def parse_memory_override(text: str) -> Tuple[bool, str]:
    """
    Returns (override, cleaned_text).
    Minimal behavior: no overrides supported.
    """
    return False, text
