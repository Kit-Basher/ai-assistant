from __future__ import annotations

from typing import Any, Mapping


_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "brief": {"read_only": True},
    "status": {"read_only": True},
    "health": {"read_only": True},
    "doctor": {"read_only": True},
    "observe_now": {"read_only": False},
}

_ORDERED_KEYS = ("tool", "args", "reason", "read_only", "confidence")


def supported_tools() -> tuple[str, ...]:
    return tuple(sorted(_TOOL_SPECS.keys()))


def normalize_tool_request(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    tool = str(payload.get("tool") or "").strip().lower()
    args = payload.get("args") if isinstance(payload.get("args"), Mapping) else {}
    reason = str(payload.get("reason") or "").strip()
    read_only_default = bool((_TOOL_SPECS.get(tool) or {}).get("read_only", False))
    read_only = bool(payload.get("read_only", read_only_default))
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0
    normalized: dict[str, Any] = {}
    normalized["tool"] = tool
    normalized["args"] = {str(k): v for k, v in sorted(dict(args).items(), key=lambda item: str(item[0]))}
    normalized["reason"] = reason
    normalized["read_only"] = read_only
    normalized["confidence"] = round(confidence, 4)
    return normalized


def validate_tool_request(raw: Mapping[str, Any] | None) -> tuple[bool, dict[str, Any], str | None]:
    normalized = normalize_tool_request(raw)
    tool = str(normalized.get("tool") or "").strip().lower()
    if not tool:
        return False, normalized, "tool_missing"
    if tool not in _TOOL_SPECS:
        return False, normalized, "tool_unsupported"
    if not isinstance(normalized.get("args"), dict):
        return False, normalized, "args_invalid"
    expected_read_only = bool((_TOOL_SPECS.get(tool) or {}).get("read_only", False))
    if bool(normalized.get("read_only")) != expected_read_only:
        normalized["read_only"] = expected_read_only
    return True, normalized, None


def tool_request_to_public_summary(raw: Mapping[str, Any] | None) -> str:
    ok, normalized, error_code = validate_tool_request(raw)
    if not ok:
        return f"tool_request invalid ({error_code or 'unknown_error'})"
    tool = str(normalized.get("tool") or "").strip().lower()
    reason = str(normalized.get("reason") or "").strip() or "none"
    confidence = float(normalized.get("confidence") or 0.0)
    args = normalized.get("args") if isinstance(normalized.get("args"), dict) else {}
    arg_keys = ",".join(sorted(args.keys())) if args else "none"
    return (
        f"tool={tool} read_only={str(bool(normalized.get('read_only'))).lower()} "
        f"confidence={confidence:.2f} args={arg_keys} reason={reason}"
    )


__all__ = [
    "normalize_tool_request",
    "supported_tools",
    "tool_request_to_public_summary",
    "validate_tool_request",
]

