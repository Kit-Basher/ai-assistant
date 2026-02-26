from __future__ import annotations

import socket
from typing import Any
import urllib.error


ERROR_KINDS = (
    "bad_request",
    "payment_required",
    "upstream_down",
    "timeout",
    "internal_error",
    "rate_limited",
    "policy_blocked",
    "unknown",
)

INTERNAL_ERROR_SUPPORT_HINT = "If this keeps happening, run: python -m agent.support_bundle"


def normalize_error_kind(value: Any, *, default: str = "unknown") -> str:
    candidate = str(value or "").strip().lower()
    if candidate in ERROR_KINDS:
        return candidate
    return default if default in ERROR_KINDS else "unknown"


def ensure_user_message(message: str, *, error_kind: str) -> str:
    text = str(message or "").strip()
    if not text:
        return ""
    if normalize_error_kind(error_kind) != "internal_error":
        return text
    if INTERNAL_ERROR_SUPPORT_HINT in text:
        return text
    return f"{text}\n{INTERNAL_ERROR_SUPPORT_HINT}"


def _flatten_marker_values(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            key_text = str(key).strip().lower()
            if key_text in {
                "error",
                "error_class",
                "error_kind",
                "reason",
                "detail",
                "message",
                "status",
                "safe_mode_blocked_reason",
            }:
                output.append(str(value.get(key) or ""))
            nested = value.get(key)
            if key_text in {"meta", "attempts", "health", "providers", "models"}:
                output.extend(_flatten_marker_values(nested))
            if key_text == "attempts" and isinstance(nested, list):
                for row in nested:
                    if isinstance(row, dict):
                        output.append(str(row.get("reason") or ""))
                        output.append(str(row.get("status_code") or ""))
    elif isinstance(value, list):
        for row in value:
            output.extend(_flatten_marker_values(row))
    else:
        output.append(str(value or ""))
    return output


def _has_any(texts: list[str], needles: set[str]) -> bool:
    merged = " | ".join(texts).lower()
    return any(token in merged for token in needles)


def _health_is_down(context: dict[str, Any]) -> bool:
    health_state = context.get("health_state") if isinstance(context.get("health_state"), dict) else {}
    providers = health_state.get("providers") if isinstance(health_state.get("providers"), dict) else {}
    models = health_state.get("models") if isinstance(health_state.get("models"), dict) else {}
    provider_id = str(context.get("provider") or "").strip().lower()
    model_id = str(context.get("model") or "").strip()

    if model_id:
        row = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
        status = str(row.get("status") or "").strip().lower()
        if status == "down":
            return True
    if provider_id:
        row = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
        status = str(row.get("status") or "").strip().lower()
        if status == "down":
            return True
    return False


def classify_error_kind(
    *,
    error: Exception | None = None,
    payload: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    payload_data = payload if isinstance(payload, dict) else {}
    context_data = context if isinstance(context, dict) else {}
    payload_meta = payload_data.get("meta") if isinstance(payload_data.get("meta"), dict) else {}

    explicit = normalize_error_kind(
        context_data.get("error_kind")
        or payload_data.get("error_kind")
        or payload_meta.get("error_kind")
    )
    if explicit != "unknown":
        return explicit

    if bool(context_data.get("policy_blocked")):
        return "policy_blocked"
    if bool(context_data.get("validation_error")):
        return "bad_request"

    if error is not None:
        if isinstance(error, (TimeoutError, socket.timeout)):
            return "timeout"
        if isinstance(error, urllib.error.HTTPError) and int(getattr(error, "code", 0) or 0) == 402:
            return "payment_required"
        if isinstance(error, urllib.error.HTTPError) and int(getattr(error, "code", 0) or 0) == 429:
            return "rate_limited"
        name = error.__class__.__name__.lower()
        text = str(error).lower()
        merged = f"{name} {text}"
        if "timeout" in merged or "timed out" in merged or "connecttimeout" in merged:
            return "timeout"
        if "rate limit" in merged or "ratelimit" in merged or "429" in merged:
            return "rate_limited"
        if any(
            token in merged
            for token in {
                "payment required",
                "insufficient credits",
                "credits_insufficient",
                "insufficient_credits",
                "402",
            }
        ):
            return "payment_required"
        if "policy" in merged or "permission" in merged or "blocked" in merged:
            return "policy_blocked"
        if any(token in merged for token in {"provider_unavailable", "circuit_open", "provider_down", "model_down", "cooldown"}):
            return "upstream_down"
        if isinstance(error, (ValueError, KeyError, TypeError)):
            return "bad_request"
        return "internal_error"

    markers = [item.strip().lower() for item in _flatten_marker_values(payload_data) if str(item).strip()]
    markers.extend([str(context_data.get("route") or "").strip().lower()])
    markers = [item for item in markers if item]

    validation_markers = {
        "is required",
        "must be",
        "invalid",
        "bad_request",
        "missing field",
        "messages must be a non-empty list",
        "provider_not_supported",
        "provider not found",
    }
    if _has_any(markers, validation_markers):
        return "bad_request"

    timeout_markers = {"timeout", "timed out", "deadline exceeded", "connecttimeout"}
    if _has_any(markers, timeout_markers):
        return "timeout"

    rate_markers = {"rate_limit", "rate limited", "429", "too many requests"}
    if _has_any(markers, rate_markers):
        return "rate_limited"

    payment_markers = {
        "payment_required",
        "credits_insufficient",
        "insufficient_credits",
        "insufficient credits",
        "402",
    }
    if _has_any(markers, payment_markers):
        return "payment_required"

    policy_markers = {
        "policy_blocked",
        "permission_required",
        "action_not_permitted",
        "manual_confirm_required",
        "safe_mode_blocked",
        "blocked",
    }
    if _has_any(markers, policy_markers):
        return "policy_blocked"

    upstream_markers = {
        "provider_down",
        "model_down",
        "provider_unavailable",
        "circuit_open",
        "cooldown",
        "server_error",
        "auth_error",
        "model_not_installed",
    }
    if _has_any(markers, upstream_markers):
        return "upstream_down"
    if _health_is_down(context_data):
        return "upstream_down"

    if bool(payload_data) and payload_data.get("ok") is False:
        return "unknown"
    return "unknown"


__all__ = [
    "ERROR_KINDS",
    "INTERNAL_ERROR_SUPPORT_HINT",
    "classify_error_kind",
    "ensure_user_message",
    "normalize_error_kind",
]
