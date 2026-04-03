from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.orchestrator import OrchestratorResponse


@dataclass(frozen=True)
class SerializedChatResponse:
    ok: bool
    body: dict[str, Any]
    route: str
    route_reason: str
    generic_fallback_allowed: bool
    generic_fallback_reason: str | None


def _response_data(response: OrchestratorResponse) -> dict[str, Any]:
    return dict(response.data) if isinstance(response.data, dict) else {}


def _runtime_payload(response_data: dict[str, Any]) -> dict[str, Any] | None:
    payload = response_data.get("runtime_payload")
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def _selection_metadata_allowed(*, route: str, response_data: dict[str, Any]) -> bool:
    if str(route or "").strip().lower() == "generic_chat":
        return True
    return bool(response_data.get("used_llm", False))


def _resolve_provider(response_data: dict[str, Any], runtime_payload: dict[str, Any] | None) -> str | None:
    provider = str(response_data.get("provider") or "").strip().lower() or None
    if provider is not None or not isinstance(runtime_payload, dict):
        return provider
    return str(runtime_payload.get("provider") or "").strip().lower() or None


def _resolve_model(response_data: dict[str, Any], runtime_payload: dict[str, Any] | None) -> str | None:
    model = str(response_data.get("model") or "").strip() or None
    if model is not None or not isinstance(runtime_payload, dict):
        return model
    return (
        str(runtime_payload.get("model_id") or "").strip()
        or str(runtime_payload.get("current_model_id") or "").strip()
        or None
    )


def serialize_orchestrator_chat_response(
    response: OrchestratorResponse,
    *,
    source_surface: str,
    user_id: str,
    thread_id: str,
    autopilot_meta: dict[str, Any],
) -> SerializedChatResponse:
    response_data = _response_data(response)
    route = str(response_data.get("route") or "generic_chat").strip().lower() or "generic_chat"
    route_reason = str(response_data.get("route_reason") or route).strip().lower() or route
    runtime_payload = _runtime_payload(response_data)
    provider = _resolve_provider(response_data, runtime_payload) if _selection_metadata_allowed(route=route, response_data=response_data) else None
    model = _resolve_model(response_data, runtime_payload) if _selection_metadata_allowed(route=route, response_data=response_data) else None
    generic_fallback_used = route == "generic_chat"
    generic_fallback_allowed = route == "generic_chat"
    generic_fallback_reason = str(response_data.get("generic_fallback_reason") or "").strip() or None
    assistant_text = str(response.text or "").strip() or "Done."
    ok = bool(response_data.get("ok", True))

    meta: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "route": route,
        "route_reason": route_reason,
        "source_surface": source_surface,
        "fallback_used": bool(response_data.get("fallback_used", False)),
        "generic_fallback_used": generic_fallback_used,
        "generic_fallback_allowed": generic_fallback_allowed,
        "generic_fallback_reason": generic_fallback_reason,
        "attempts": response_data.get("attempts") or [],
        "duration_ms": int(response_data.get("duration_ms") or 0),
        "error": str(response_data.get("error_kind") or "").strip() or None,
        "autopilot": dict(autopilot_meta) if isinstance(autopilot_meta, dict) else {},
        "used_runtime_state": bool(response_data.get("used_runtime_state", False)),
        "used_llm": bool(response_data.get("used_llm", False)),
        "used_memory": bool(response_data.get("used_memory", False)),
        "used_tools": [
            str(item).strip()
            for item in (
                response_data.get("used_tools")
                if isinstance(response_data.get("used_tools"), list)
                else []
            )
            if str(item).strip()
        ],
        "thread_id": thread_id,
        "user_id": user_id,
    }
    if isinstance(runtime_payload, dict):
        meta["setup_type"] = str(runtime_payload.get("type") or "").strip() or None
        meta["runtime_state_failure_reason"] = str(runtime_payload.get("reason") or "").strip() or None
    if isinstance(response_data.get("selection_policy"), dict):
        meta["selection_policy"] = dict(response_data.get("selection_policy"))

    body: dict[str, Any] = {
        "ok": ok,
        "assistant": {
            "role": "assistant",
            "content": assistant_text,
        },
        "message": assistant_text,
        "meta": meta,
    }
    if isinstance(runtime_payload, dict):
        body["setup"] = runtime_payload
    next_question = str(response_data.get("next_question") or "").strip()
    if next_question:
        body["next_question"] = next_question
    error_kind = str(response_data.get("error_kind") or "").strip()
    if error_kind:
        body["error_kind"] = error_kind

    return SerializedChatResponse(
        ok=ok,
        body=body,
        route=route,
        route_reason=route_reason,
        generic_fallback_allowed=generic_fallback_allowed,
        generic_fallback_reason=generic_fallback_reason,
    )
