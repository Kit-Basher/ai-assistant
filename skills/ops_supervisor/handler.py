from __future__ import annotations

from typing import Any

from agent.ops import supervisor_client


def _format_error(op: str, response: dict[str, Any]) -> dict[str, Any]:
    error = response.get("error") or "request_failed"
    text = f"Supervisor {op} failed: {error}"
    return {"text": text, "data": {"ok": False, "error": error}}


def restart_agent(context: dict[str, Any]) -> dict[str, Any]:
    response = supervisor_client.send_request("restart", {})
    if not response.get("ok"):
        return _format_error("restart", response)
    status = (response.get("result") or {}).get("status") or "unknown"
    return {
        "text": f"Restart requested. Status: {status}",
        "data": response,
    }


def service_status(context: dict[str, Any]) -> dict[str, Any]:
    response = supervisor_client.send_request("status", {})
    if not response.get("ok"):
        return _format_error("status", response)
    result = response.get("result") or {}
    active = result.get("active") or "unknown"
    show = result.get("show") or ""
    text = f"Service status: {active}"
    if show:
        text = f"{text}\n{show}"
    return {"text": text, "data": response}


def service_logs(context: dict[str, Any], lines: int | None = None) -> dict[str, Any]:
    bounded = supervisor_client.clamp_log_lines(lines)
    response = supervisor_client.send_request("logs", {"lines": bounded})
    if not response.get("ok"):
        return _format_error("logs", response)
    result = response.get("result") or {}
    content = result.get("lines") or ""
    text = f"Last {bounded} lines:\n{content}".rstrip()
    return {"text": text, "data": response}
