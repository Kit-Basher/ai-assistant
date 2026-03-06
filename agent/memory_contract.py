from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


PENDING_STATUS_PENDING = "PENDING"
PENDING_STATUS_WAITING_FOR_USER = "WAITING_FOR_USER"
PENDING_STATUS_READY_TO_RESUME = "READY_TO_RESUME"
PENDING_STATUS_EXPIRED = "EXPIRED"
PENDING_STATUS_DONE = "DONE"
PENDING_STATUS_ABORTED = "ABORTED"

_PENDING_STATUSES = {
    PENDING_STATUS_PENDING,
    PENDING_STATUS_WAITING_FOR_USER,
    PENDING_STATUS_READY_TO_RESUME,
    PENDING_STATUS_EXPIRED,
    PENDING_STATUS_DONE,
    PENDING_STATUS_ABORTED,
}

_PENDING_KINDS = {"clarification", "confirmation", "followup", "task"}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _as_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_runtime_mode(value: Any) -> str:
    mode = str(value or "").strip().upper()
    if mode in {"READY", "BOOTSTRAP_REQUIRED", "DEGRADED", "FAILED"}:
        return mode
    return "DEGRADED"


def _normalize_status(value: Any, default: str = "active") -> str:
    status = str(value or "").strip().lower()
    return status or default


@dataclass(frozen=True)
class ThreadState:
    thread_id: str
    user_id: str
    current_topic: str | None
    last_tool: str | None
    runtime_mode: str
    updated_at: int
    status: str


@dataclass(frozen=True)
class PendingItem:
    pending_id: str
    kind: str
    origin_tool: str | None
    question: str
    options: tuple[str, ...]
    created_at: int
    expires_at: int
    thread_id: str
    status: str


@dataclass(frozen=True)
class MemorySummary:
    current_topic: str | None
    pending_count: int
    last_tool: str | None
    last_meaningful_user_request: str | None
    last_agent_action: str | None
    resumable: bool


def normalize_thread_state(
    raw: Mapping[str, Any] | None,
    *,
    user_id: str,
    default_thread_id: str,
    now_ts: int | None = None,
) -> dict[str, Any]:
    source = dict(raw or {})
    timestamp = int(now_ts if isinstance(now_ts, int) else _now_epoch())
    thread_id = str(source.get("thread_id") or "").strip() or str(default_thread_id).strip()
    if not thread_id:
        thread_id = f"user:{str(user_id).strip() or 'unknown'}"
    normalized = ThreadState(
        thread_id=thread_id,
        user_id=str(source.get("user_id") or "").strip() or str(user_id).strip() or "unknown",
        current_topic=_as_text(source.get("current_topic")),
        last_tool=_as_text(source.get("last_tool")),
        runtime_mode=_normalize_runtime_mode(source.get("runtime_mode")),
        updated_at=_coerce_int(source.get("updated_at"), timestamp),
        status=_normalize_status(source.get("status"), default="active"),
    )
    return {
        "thread_id": normalized.thread_id,
        "user_id": normalized.user_id,
        "current_topic": normalized.current_topic,
        "last_tool": normalized.last_tool,
        "runtime_mode": normalized.runtime_mode,
        "updated_at": int(normalized.updated_at),
        "status": normalized.status,
    }


def normalize_pending_item(
    raw: Mapping[str, Any] | None,
    *,
    default_thread_id: str,
    now_ts: int | None = None,
) -> dict[str, Any]:
    source = dict(raw or {})
    timestamp = int(now_ts if isinstance(now_ts, int) else _now_epoch())
    pending_id = str(source.get("pending_id") or "").strip()
    if not pending_id:
        pending_id = f"pending-{timestamp}"
    kind = str(source.get("kind") or "").strip().lower()
    if kind not in _PENDING_KINDS:
        kind = "followup"
    options_raw = source.get("options") if isinstance(source.get("options"), list | tuple) else []
    options = tuple(
        option
        for option in [str(item or "").strip() for item in options_raw]
        if option
    )
    status = str(source.get("status") or "").strip().upper()
    if status not in _PENDING_STATUSES:
        status = PENDING_STATUS_WAITING_FOR_USER
    created_at = _coerce_int(source.get("created_at"), timestamp)
    expires_at = _coerce_int(source.get("expires_at"), created_at + 600)
    thread_id = str(source.get("thread_id") or "").strip() or str(default_thread_id).strip()
    if not thread_id:
        thread_id = "user:unknown"
    normalized = PendingItem(
        pending_id=pending_id,
        kind=kind,
        origin_tool=_as_text(source.get("origin_tool")),
        question=str(source.get("question") or "").strip() or "Pending follow-up.",
        options=options,
        created_at=created_at,
        expires_at=expires_at,
        thread_id=thread_id,
        status=status,
    )
    return {
        "pending_id": normalized.pending_id,
        "kind": normalized.kind,
        "origin_tool": normalized.origin_tool,
        "question": normalized.question,
        "options": list(normalized.options),
        "created_at": int(normalized.created_at),
        "expires_at": int(normalized.expires_at),
        "thread_id": normalized.thread_id,
        "status": normalized.status,
    }


def build_memory_summary(
    *,
    thread_state: Mapping[str, Any] | None,
    pending_items: list[Mapping[str, Any]] | None,
    last_meaningful_user_request: str | None,
    last_agent_action: str | None,
    now_ts: int | None = None,
) -> dict[str, Any]:
    timestamp = int(now_ts if isinstance(now_ts, int) else _now_epoch())
    thread = normalize_thread_state(
        thread_state or {},
        user_id=str((thread_state or {}).get("user_id") or "unknown"),
        default_thread_id=str((thread_state or {}).get("thread_id") or "user:unknown"),
        now_ts=timestamp,
    )
    normalized_pending = [
        normalize_pending_item(item, default_thread_id=thread["thread_id"], now_ts=timestamp)
        for item in (pending_items or [])
    ]
    active_pending = [
        item
        for item in normalized_pending
        if item["status"] in {
            PENDING_STATUS_PENDING,
            PENDING_STATUS_WAITING_FOR_USER,
            PENDING_STATUS_READY_TO_RESUME,
        }
        and int(item["expires_at"]) > timestamp
    ]
    summary = MemorySummary(
        current_topic=_as_text(thread.get("current_topic")),
        pending_count=len(active_pending),
        last_tool=_as_text(thread.get("last_tool")),
        last_meaningful_user_request=_as_text(last_meaningful_user_request),
        last_agent_action=_as_text(last_agent_action),
        resumable=len(active_pending) > 0,
    )
    return {
        "current_topic": summary.current_topic,
        "pending_count": int(summary.pending_count),
        "last_tool": summary.last_tool,
        "last_meaningful_user_request": summary.last_meaningful_user_request,
        "last_agent_action": summary.last_agent_action,
        "resumable": bool(summary.resumable),
    }


def deterministic_memory_snapshot(
    *,
    thread_state: Mapping[str, Any] | None,
    pending_items: list[Mapping[str, Any]] | None,
    last_meaningful_user_request: str | None,
    last_agent_action: str | None,
    now_ts: int | None = None,
) -> dict[str, Any]:
    timestamp = int(now_ts if isinstance(now_ts, int) else _now_epoch())
    thread = normalize_thread_state(
        thread_state or {},
        user_id=str((thread_state or {}).get("user_id") or "unknown"),
        default_thread_id=str((thread_state or {}).get("thread_id") or "user:unknown"),
        now_ts=timestamp,
    )
    normalized_pending = [
        normalize_pending_item(item, default_thread_id=thread["thread_id"], now_ts=timestamp)
        for item in (pending_items or [])
    ]
    normalized_pending.sort(key=lambda row: (int(row.get("created_at") or 0), str(row.get("pending_id") or "")))
    summary = build_memory_summary(
        thread_state=thread,
        pending_items=normalized_pending,
        last_meaningful_user_request=last_meaningful_user_request,
        last_agent_action=last_agent_action,
        now_ts=timestamp,
    )
    return {
        "thread_state": thread,
        "pending_items": normalized_pending,
        "memory_summary": summary,
        "snapshot_ts": timestamp,
    }


__all__ = [
    "PENDING_STATUS_ABORTED",
    "PENDING_STATUS_DONE",
    "PENDING_STATUS_EXPIRED",
    "PENDING_STATUS_PENDING",
    "PENDING_STATUS_READY_TO_RESUME",
    "PENDING_STATUS_WAITING_FOR_USER",
    "build_memory_summary",
    "deterministic_memory_snapshot",
    "normalize_pending_item",
    "normalize_thread_state",
]
