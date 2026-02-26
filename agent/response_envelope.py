from __future__ import annotations

import hashlib
import time
from typing import Any, TypedDict

from agent.error_kind import ensure_user_message, normalize_error_kind

class ResponseAction(TypedDict):
    label: str
    command: str


class ResponseEnvelope(TypedDict):
    ok: bool
    intent: str
    confidence: float
    did_work: bool
    error_kind: str | None
    message: str
    next_question: str | None
    actions: list[ResponseAction]
    errors: list[str]
    trace_id: str


def _clamp_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _normalize_trace_id(trace_id: str | None) -> str:
    text = str(trace_id or "").strip()
    if text:
        return text
    seed = f"{time.time_ns()}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:16]


def validate_envelope(env: dict[str, Any]) -> ResponseEnvelope:
    ok = bool(env.get("ok"))
    raw_error_kind = env.get("error_kind")
    error_kind: str | None
    if raw_error_kind is None or not str(raw_error_kind).strip():
        error_kind = None
    else:
        error_kind = normalize_error_kind(raw_error_kind, default="unknown")

    message = ensure_user_message(
        str(env.get("message") or "").strip(),
        error_kind=error_kind or ("unknown" if ok else "internal_error"),
    )
    if not message:
        raise ValueError("message must be non-empty")

    next_question_raw = env.get("next_question")
    next_question = str(next_question_raw).strip() if next_question_raw is not None else None
    if next_question:
        if "?" not in next_question:
            raise ValueError("next_question must contain '?'")
        if next_question.count("?") > 1:
            raise ValueError("next_question must contain at most one '?'")

    actions_raw = env.get("actions") if isinstance(env.get("actions"), list) else []
    if len(actions_raw) > 3:
        raise ValueError("actions must have at most 3 items")
    actions: list[ResponseAction] = []
    for row in actions_raw:
        if not isinstance(row, dict):
            raise ValueError("actions rows must be objects")
        label = str(row.get("label") or "").strip()
        command = str(row.get("command") or "").strip()
        if not label or not command:
            raise ValueError("actions rows require non-empty label and command")
        actions.append({"label": label, "command": command})

    errors_raw = env.get("errors") if isinstance(env.get("errors"), list) else []
    errors = [str(item).strip() for item in errors_raw if str(item).strip()]

    normalized: ResponseEnvelope = {
        "ok": ok,
        "intent": str(env.get("intent") or "").strip() or "unknown",
        "confidence": _clamp_confidence(env.get("confidence")),
        "did_work": bool(env.get("did_work")),
        "error_kind": error_kind,
        "message": message,
        "next_question": next_question if next_question else None,
        "actions": actions,
        "errors": errors,
        "trace_id": _normalize_trace_id(str(env.get("trace_id") or "").strip() or None),
    }
    return normalized


def ok_result(
    *,
    intent: str,
    message: str,
    confidence: float = 1.0,
    did_work: bool = True,
    next_question: str | None = None,
    actions: list[dict[str, str]] | None = None,
    errors: list[str] | None = None,
    trace_id: str | None = None,
) -> ResponseEnvelope:
    return validate_envelope(
        {
            "ok": True,
            "intent": intent,
            "confidence": confidence,
            "did_work": did_work,
            "error_kind": None,
            "message": message,
            "next_question": next_question,
            "actions": actions or [],
            "errors": errors or [],
            "trace_id": trace_id,
        }
    )


def needs_clarification(
    question: str,
    *,
    intent: str = "clarification",
    confidence: float = 0.35,
    message: str = "I need one detail before I continue.",
    actions: list[dict[str, str]] | None = None,
    errors: list[str] | None = None,
    trace_id: str | None = None,
) -> ResponseEnvelope:
    return validate_envelope(
        {
            "ok": True,
            "intent": intent,
            "confidence": confidence,
            "did_work": False,
            "error_kind": None,
            "message": message,
            "next_question": question,
            "actions": actions or [],
            "errors": errors or [],
            "trace_id": trace_id,
        }
    )


def failure(
    message: str,
    *,
    intent: str,
    confidence: float = 0.0,
    error_kind: str = "internal_error",
    next_question: str | None = None,
    actions: list[dict[str, str]] | None = None,
    errors: list[str] | None = None,
    trace_id: str | None = None,
) -> ResponseEnvelope:
    return validate_envelope(
        {
            "ok": False,
            "intent": intent,
            "confidence": confidence,
            "did_work": False,
            "error_kind": error_kind,
            "message": message,
            "next_question": next_question,
            "actions": actions or [],
            "errors": errors or [],
            "trace_id": trace_id,
        }
    )


__all__ = [
    "ResponseAction",
    "ResponseEnvelope",
    "failure",
    "needs_clarification",
    "ok_result",
    "validate_envelope",
]
