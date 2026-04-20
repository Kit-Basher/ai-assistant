from __future__ import annotations

import json
import re
from typing import Any

from agent.persona import normalize_persona_text


_INTERNAL_TEXT_MARKERS = (
    "trace_id:",
    "component:",
    "failure_code:",
    "next_action:",
    "local_observations",
    "route_reason:",
    "selection_policy",
    "runtime_payload",
    "runtime_state_failure_reason",
    "setup_type:",
    "generic_fallback_reason:",
    "autopilot:",
    "operator_only:",
    "thread_id:",
    "user_id:",
    "source_surface:",
)
_INTERNAL_JSON_KEYS = {
    "trace_id",
    "component",
    "failure_code",
    "next_action",
    "local_observations",
    "route_reason",
    "selection_policy",
    "runtime_payload",
    "runtime_state_failure_reason",
    "setup_type",
    "generic_fallback_reason",
    "autopilot",
    "operator_only",
    "thread_id",
    "user_id",
    "source_surface",
}
_JSON_OBJECT_RE = re.compile(r"^\s*[\[{].*[\]}]\s*$", re.DOTALL)
_NO_LLM_PUBLIC_MESSAGE = "I’m not ready to chat yet. Open Setup to finish getting me ready."
_READY_LLM_PUBLIC_MESSAGE = "The runtime is ready, but I can’t reach chat right now. Try again in a moment or ask for status or setup help."
_NO_LLM_ERROR_KINDS = {
    "llm_unavailable",
    "no_chat_model",
    "provider_unhealthy",
    "model_unhealthy",
    "router_unavailable",
}
_TRIVIAL_SOCIAL_TURNS = {
    "hello": "greeting",
    "hi": "greeting",
    "hey": "greeting",
    "thanks": "thanks",
    "ok": "ack",
    "okay": "ack",
    "good morning": "morning",
    "good evening": "evening",
}
_TRIVIAL_SOCIAL_TURN_REPLIES = {
    "greeting": "Hi — I’m here and ready to help. What can I do for you?",
    "thanks": "You’re welcome. What would you like to do next?",
    "ack": "Got it. What should I do next?",
    "morning": "Good morning. What can I help with?",
    "evening": "Good evening. What can I help with?",
}


def _normalize_social_turn_text(text: str | None) -> str:
    cleaned = str(text or "").strip().lower()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def _looks_like_internal_json(text: str) -> bool:
    trimmed = str(text or "").strip()
    if not trimmed or not _JSON_OBJECT_RE.match(trimmed):
        return False
    try:
        parsed = json.loads(trimmed)
    except Exception:
        return False
    if isinstance(parsed, dict):
        return any(str(key).strip().lower() in _INTERNAL_JSON_KEYS for key in parsed.keys())
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and any(str(key).strip().lower() in _INTERNAL_JSON_KEYS for key in item.keys()):
                return True
    return False


def _looks_like_internal_text(text: str) -> bool:
    lowered = str(text or "").lower()
    if _looks_like_internal_json(text):
        return True
    if any(marker in lowered for marker in _INTERNAL_TEXT_MARKERS):
        return True
    if "```json" in lowered and any(marker in lowered for marker in _INTERNAL_TEXT_MARKERS):
        return True
    return False


def normalize_public_assistant_text(text: str | None, *, fallback: str = "Done.") -> str:
    cleaned = normalize_persona_text(text)
    if not cleaned:
        return normalize_persona_text(fallback)
    if _looks_like_internal_text(cleaned):
        return normalize_persona_text(fallback)
    return cleaned


def build_public_sentence_text(*parts: str | None) -> str:
    sentences: list[str] = []
    for part in parts:
        cleaned = " ".join(str(part or "").strip().split())
        if not cleaned:
            continue
        sentences.append(cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}.")
    return normalize_persona_text(" ".join(sentences))


def build_no_llm_public_message(*, runtime_ready: bool = False) -> str:
    return _READY_LLM_PUBLIC_MESSAGE if runtime_ready else _NO_LLM_PUBLIC_MESSAGE


def classify_trivial_social_turn(text: str | None) -> str | None:
    normalized = _normalize_social_turn_text(text)
    if not normalized:
        return None
    return _TRIVIAL_SOCIAL_TURNS.get(normalized)


def build_trivial_social_turn_message(text: str | None) -> str | None:
    turn_kind = classify_trivial_social_turn(text)
    if not turn_kind:
        return None
    return normalize_persona_text(_TRIVIAL_SOCIAL_TURN_REPLIES[turn_kind])


def is_no_llm_error_kind(error_kind: str | None) -> bool:
    return str(error_kind or "").strip().lower() in _NO_LLM_ERROR_KINDS


def build_public_chat_meta(
    response_data: dict[str, Any],
    *,
    include_debug: bool = False,
    source_surface: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    autopilot_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = str(response_data.get("provider") or "").strip().lower() or None
    model = str(response_data.get("model") or "").strip() or None
    used_tools = [
        str(item).strip()
        for item in (
            response_data.get("used_tools")
            if isinstance(response_data.get("used_tools"), list)
            else []
        )
        if str(item).strip()
    ]
    route = str(response_data.get("route") or "generic_chat").strip().lower() or "generic_chat"
    meta: dict[str, Any] = {
        "route": route,
        "provider": provider,
        "model": model,
        "fallback_used": bool(response_data.get("fallback_used", False)),
        "generic_fallback_used": bool(response_data.get("generic_fallback_used", False)) or route == "generic_chat",
        "generic_fallback_allowed": bool(response_data.get("generic_fallback_allowed", False)) or route == "generic_chat",
        "used_runtime_state": bool(response_data.get("used_runtime_state", False)),
        "used_llm": bool(response_data.get("used_llm", False)),
        "used_memory": bool(response_data.get("used_memory", False)),
        "used_tools": used_tools,
        "assistant_turn_type": str(response_data.get("assistant_turn_type") or "").strip().lower() or None,
        "assistant_turn_kind": str(response_data.get("assistant_turn_kind") or "").strip().lower() or None,
    }
    if "skip_post_response_hooks" in response_data:
        meta["skip_post_response_hooks"] = bool(response_data.get("skip_post_response_hooks", False))
    if include_debug:
        meta["route_reason"] = str(response_data.get("route_reason") or "").strip().lower() or None
        meta["source_surface"] = str(source_surface or response_data.get("source_surface") or "").strip().lower() or None
        meta["attempts"] = response_data.get("attempts") or []
        meta["duration_ms"] = int(response_data.get("duration_ms") or 0)
        meta["error"] = str(response_data.get("error_kind") or "").strip() or None
        meta["autopilot"] = dict(autopilot_meta) if isinstance(autopilot_meta, dict) else {}
        meta["thread_id"] = str(thread_id or "").strip() or None
        meta["user_id"] = str(user_id or "").strip() or None
        if isinstance(response_data.get("selection_policy"), dict):
            meta["selection_policy"] = dict(response_data.get("selection_policy"))
        if isinstance(response_data.get("runtime_payload"), dict):
            runtime_payload = dict(response_data.get("runtime_payload"))
            meta["setup_type"] = str(runtime_payload.get("type") or "").strip() or None
            meta["runtime_state_failure_reason"] = str(runtime_payload.get("reason") or "").strip() or None
    chat_timing_ms = response_data.get("chat_timing_ms")
    if isinstance(chat_timing_ms, dict) and chat_timing_ms:
        meta["chat_timing_ms"] = dict(chat_timing_ms)
    orchestrator_timing_ms = response_data.get("orchestrator_timing_ms")
    if isinstance(orchestrator_timing_ms, dict) and orchestrator_timing_ms:
        meta["orchestrator_timing_ms"] = dict(orchestrator_timing_ms)
    truth_timing_ms = response_data.get("truth_timing_ms")
    if not isinstance(truth_timing_ms, dict) or not truth_timing_ms:
        runtime_payload = response_data.get("runtime_payload")
        if isinstance(runtime_payload, dict):
            truth_timing_ms = runtime_payload.get("truth_timing_ms")
    if isinstance(truth_timing_ms, dict) and truth_timing_ms:
        meta["truth_timing_ms"] = dict(truth_timing_ms)
    return meta


__all__ = [
    "build_no_llm_public_message",
    "build_public_chat_meta",
    "build_public_sentence_text",
    "build_trivial_social_turn_message",
    "classify_trivial_social_turn",
    "is_no_llm_error_kind",
    "normalize_public_assistant_text",
]
