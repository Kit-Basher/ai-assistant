from __future__ import annotations

from typing import Any, Mapping

from agent.runtime_contract import RUNTIME_MODE_BOOTSTRAP_REQUIRED, RUNTIME_MODE_DEGRADED, RUNTIME_MODE_FAILED


def is_tool_allowed(
    *,
    tool_request: Mapping[str, Any],
    runtime_mode: str,
    enable_writes: bool,
    safe_mode: bool,
) -> bool:
    decision = permission_decision(
        tool_request=tool_request,
        runtime_mode=runtime_mode,
        enable_writes=enable_writes,
        safe_mode=safe_mode,
    )
    return bool(decision.get("allowed", False))


def why_blocked(
    *,
    tool_request: Mapping[str, Any],
    runtime_mode: str,
    enable_writes: bool,
    safe_mode: bool,
) -> str | None:
    decision = permission_decision(
        tool_request=tool_request,
        runtime_mode=runtime_mode,
        enable_writes=enable_writes,
        safe_mode=safe_mode,
    )
    return str(decision.get("reason") or "").strip() or None


def next_action(*, reason: str | None) -> str | None:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return None
    if normalized == "runtime_failed":
        return "Run: python -m agent doctor"
    if normalized in {"writes_disabled", "safe_mode_blocked"}:
        return "Use a read-only request or enable writes explicitly."
    return "Run: python -m agent doctor"


def permission_decision(
    *,
    tool_request: Mapping[str, Any],
    runtime_mode: str,
    enable_writes: bool,
    safe_mode: bool,
) -> dict[str, Any]:
    read_only = bool(tool_request.get("read_only", False))
    mode = str(runtime_mode or "").strip().upper() or RUNTIME_MODE_DEGRADED

    if mode == RUNTIME_MODE_FAILED:
        return {
            "allowed": False,
            "reason": "runtime_failed",
            "next_action": next_action(reason="runtime_failed"),
        }
    if read_only:
        # Read-only tools remain usable in degraded/bootstrap modes.
        return {"allowed": True, "reason": "allow_read_only", "next_action": None}
    if not enable_writes:
        return {
            "allowed": False,
            "reason": "writes_disabled",
            "next_action": next_action(reason="writes_disabled"),
        }
    if safe_mode:
        return {
            "allowed": False,
            "reason": "safe_mode_blocked",
            "next_action": next_action(reason="safe_mode_blocked"),
        }
    if mode in {RUNTIME_MODE_DEGRADED, RUNTIME_MODE_BOOTSTRAP_REQUIRED}:
        return {
            "allowed": False,
            "reason": "runtime_degraded_write_blocked",
            "next_action": "Run: python -m agent doctor",
        }
    return {"allowed": True, "reason": "allow_write", "next_action": None}


__all__ = [
    "is_tool_allowed",
    "next_action",
    "permission_decision",
    "why_blocked",
]

