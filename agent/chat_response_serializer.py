from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.orchestrator import OrchestratorResponse
from agent.public_chat import build_public_chat_meta, normalize_public_assistant_text


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
    include_debug: bool = False,
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
    assistant_text = normalize_public_assistant_text(response.text, fallback="Done.")
    ok = bool(response_data.get("ok", True))

    meta = build_public_chat_meta(
        response_data,
        include_debug=include_debug,
        source_surface=source_surface,
        user_id=user_id,
        thread_id=thread_id,
        autopilot_meta=autopilot_meta,
    )

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
