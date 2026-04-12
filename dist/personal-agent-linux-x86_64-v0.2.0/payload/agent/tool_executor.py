from __future__ import annotations

import uuid
from typing import Any, Callable, Mapping

from agent.permission_contract import permission_decision
from agent.tool_contract import tool_request_to_public_summary, validate_tool_request


ToolHandler = Callable[[dict[str, Any], str], dict[str, Any]]
LogEmitter = Callable[[str, dict[str, Any]], None]


def _default_trace_id() -> str:
    return f"tool-{uuid.uuid4().hex[:10]}"


class ToolExecutor:
    def __init__(
        self,
        *,
        handlers: Mapping[str, ToolHandler],
        emit_log: LogEmitter | None = None,
        component: str = "tool_executor",
    ) -> None:
        self._handlers = dict(handlers)
        self._emit_log = emit_log
        self._component = str(component or "tool_executor")

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if not callable(self._emit_log):
            return
        try:
            self._emit_log(event, payload)
        except Exception:
            return

    def execute(
        self,
        *,
        request: Mapping[str, Any] | None,
        user_id: str,
        surface: str,
        runtime_mode: str,
        enable_writes: bool,
        safe_mode: bool,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        request_trace_id = str(trace_id or "").strip() or _default_trace_id()
        ok, normalized, error_code = validate_tool_request(request)
        tool_name = str(normalized.get("tool") or "").strip().lower()

        self._log(
            "tool.request",
            {
                "trace_id": request_trace_id,
                "surface": str(surface or "").strip().lower() or "unknown",
                "tool": tool_name or None,
                "read_only": bool(normalized.get("read_only", False)),
                "runtime_mode": str(runtime_mode or "").strip().upper() or "DEGRADED",
                "request_summary": tool_request_to_public_summary(request),
            },
        )

        if not ok:
            result = {
                "ok": False,
                "tool": tool_name or None,
                "trace_id": request_trace_id,
                "component": self._component,
                "data": {},
                "user_text": f"Tool request rejected ({error_code}).",
                "next_action": "Run: python -m agent doctor",
                "error_code": str(error_code or "tool_request_invalid"),
            }
            self._log(
                "tool.result",
                {
                    "trace_id": request_trace_id,
                    "surface": str(surface or "").strip().lower() or "unknown",
                    "tool": tool_name or None,
                    "read_only": bool(normalized.get("read_only", False)),
                    "runtime_mode": str(runtime_mode or "").strip().upper() or "DEGRADED",
                    "allowed": False,
                    "error_code": result["error_code"],
                    "next_action": result["next_action"],
                },
            )
            return result

        decision = permission_decision(
            tool_request=normalized,
            runtime_mode=runtime_mode,
            enable_writes=bool(enable_writes),
            safe_mode=bool(safe_mode),
        )
        allowed = bool(decision.get("allowed", False))
        self._log(
            "tool.decision",
            {
                "trace_id": request_trace_id,
                "surface": str(surface or "").strip().lower() or "unknown",
                "tool": tool_name,
                "read_only": bool(normalized.get("read_only", False)),
                "runtime_mode": str(runtime_mode or "").strip().upper() or "DEGRADED",
                "allowed": allowed,
                "reason": str(decision.get("reason") or ""),
                "next_action": decision.get("next_action"),
            },
        )
        if not allowed:
            result = {
                "ok": False,
                "tool": tool_name,
                "trace_id": request_trace_id,
                "component": self._component,
                "data": {},
                "user_text": "This action is blocked by policy.",
                "next_action": str(decision.get("next_action") or "Run: python -m agent doctor"),
                "error_code": str(decision.get("reason") or "permission_blocked"),
            }
            self._log(
                "tool.result",
                {
                    "trace_id": request_trace_id,
                    "surface": str(surface or "").strip().lower() or "unknown",
                    "tool": tool_name,
                    "read_only": bool(normalized.get("read_only", False)),
                    "runtime_mode": str(runtime_mode or "").strip().upper() or "DEGRADED",
                    "allowed": False,
                    "error_code": result.get("error_code"),
                    "next_action": result.get("next_action"),
                },
            )
            return result

        handler = self._handlers.get(tool_name)
        if not callable(handler):
            result = {
                "ok": False,
                "tool": tool_name,
                "trace_id": request_trace_id,
                "component": self._component,
                "data": {},
                "user_text": "Tool is not implemented.",
                "next_action": "Run: python -m agent doctor",
                "error_code": "tool_unimplemented",
            }
            self._log(
                "tool.result",
                {
                    "trace_id": request_trace_id,
                    "surface": str(surface or "").strip().lower() or "unknown",
                    "tool": tool_name,
                    "read_only": bool(normalized.get("read_only", False)),
                    "runtime_mode": str(runtime_mode or "").strip().upper() or "DEGRADED",
                    "allowed": True,
                    "error_code": result.get("error_code"),
                    "next_action": result.get("next_action"),
                },
            )
            return result

        self._log(
            "tool.execute",
            {
                "trace_id": request_trace_id,
                "surface": str(surface or "").strip().lower() or "unknown",
                "tool": tool_name,
                "read_only": bool(normalized.get("read_only", False)),
                "runtime_mode": str(runtime_mode or "").strip().upper() or "DEGRADED",
                "allowed": True,
            },
        )
        try:
            payload = handler(normalized, user_id)
        except Exception as exc:
            result = {
                "ok": False,
                "tool": tool_name,
                "trace_id": request_trace_id,
                "component": self._component,
                "data": {},
                "user_text": "Tool execution failed.",
                "next_action": "Run: python -m agent doctor",
                "error_code": f"tool_execution_failed:{exc.__class__.__name__}",
            }
            self._log(
                "tool.result",
                {
                    "trace_id": request_trace_id,
                    "surface": str(surface or "").strip().lower() or "unknown",
                    "tool": tool_name,
                    "read_only": bool(normalized.get("read_only", False)),
                    "runtime_mode": str(runtime_mode or "").strip().upper() or "DEGRADED",
                    "allowed": True,
                    "error_code": result["error_code"],
                    "next_action": result["next_action"],
                },
            )
            return result

        result = {
            "ok": bool(payload.get("ok", True)),
            "tool": tool_name,
            "trace_id": request_trace_id,
            "component": self._component,
            "data": payload.get("data") if isinstance(payload.get("data"), dict) else {},
            "user_text": str(payload.get("user_text") or "").strip(),
            "next_action": str(payload.get("next_action") or "").strip() or None,
            "error_code": str(payload.get("error_code") or "").strip() or None,
        }
        self._log(
            "tool.result",
            {
                "trace_id": request_trace_id,
                "surface": str(surface or "").strip().lower() or "unknown",
                "tool": tool_name,
                "read_only": bool(normalized.get("read_only", False)),
                "runtime_mode": str(runtime_mode or "").strip().upper() or "DEGRADED",
                "allowed": True,
                "error_code": result.get("error_code"),
                "next_action": result.get("next_action"),
            },
        )
        return result


__all__ = ["ToolExecutor", "ToolHandler"]
