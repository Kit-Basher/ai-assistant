from __future__ import annotations

from typing import Any, Callable

from agent.error_kind import classify_error_kind
from agent.error_response_ux import friendly_error_message
from agent.logging_utils import log_event
from agent.response_envelope import ResponseEnvelope, failure, validate_envelope


_DEFAULT_FAILURE_MESSAGE = "I hit an internal error, but I’m still running. Try one of these:"


def _suggested_actions(context: dict[str, Any]) -> list[dict[str, str]]:
    raw = context.get("actions") if isinstance(context.get("actions"), list) else []
    actions: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        command = str(row.get("command") or "").strip()
        if not label or not command:
            continue
        actions.append({"label": label, "command": command})
    return actions[:2]


def _log_safe(log_path: str | None, event_type: str, payload: dict[str, Any]) -> None:
    if not log_path:
        return
    try:
        log_event(log_path, event_type, payload)
    except Exception:
        return


def _minimal_failure(
    context: dict[str, Any],
    *,
    error_code: str,
    error_kind: str = "internal_error",
) -> ResponseEnvelope:
    trace_id = str(context.get("trace_id") or "").strip() or None
    message = (
        friendly_error_message(
            error_kind=error_kind,
            current_message=_DEFAULT_FAILURE_MESSAGE,
            context=context if isinstance(context, dict) else {},
        )
        or _DEFAULT_FAILURE_MESSAGE
    )
    return failure(
        message,
        intent=str(context.get("intent") or "unknown"),
        actions=_suggested_actions(context),
        errors=[error_code],
        error_kind=error_kind,
        trace_id=trace_id,
    )


def run_with_fallback(*, fn: Callable[[], ResponseEnvelope], context: dict[str, Any]) -> ResponseEnvelope:
    log_path = str(context.get("log_path") or "").strip() or None
    trace_id = str(context.get("trace_id") or "").strip() or None
    intent = str(context.get("intent") or "unknown").strip() or "unknown"
    actions = _suggested_actions(context)

    try:
        env = fn()
    except Exception as exc:
        error_kind = classify_error_kind(error=exc, context=context)
        _log_safe(
            log_path,
            "fallback_ladder_exception",
            {
                "intent": intent,
                "trace_id": trace_id,
                "error": exc.__class__.__name__,
                "error_kind": error_kind,
            },
        )
        return _minimal_failure(context, error_code=exc.__class__.__name__, error_kind=error_kind)

    try:
        return validate_envelope(dict(env))
    except Exception as exc:
        _log_safe(
            log_path,
            "fallback_ladder_invalid_envelope",
            {
                "intent": intent,
                "trace_id": trace_id,
                "error": exc.__class__.__name__,
                "error_kind": "internal_error",
            },
        )
        try:
            return failure(
                _DEFAULT_FAILURE_MESSAGE,
                intent=intent,
                actions=actions,
                errors=["InvalidEnvelope"],
                error_kind="internal_error",
                trace_id=trace_id,
            )
        except Exception:
            return _minimal_failure(context, error_code="InvalidEnvelope", error_kind="internal_error")


__all__ = ["run_with_fallback"]
