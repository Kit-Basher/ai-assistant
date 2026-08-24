from __future__ import annotations

from typing import Any, Mapping

from agent.public_chat import build_no_llm_public_message


ONBOARDING_NOT_STARTED = "NOT_STARTED"
ONBOARDING_TOKEN_MISSING = "TOKEN_MISSING"
ONBOARDING_LLM_MISSING = "LLM_MISSING"
ONBOARDING_SERVICES_DOWN = "SERVICES_DOWN"
ONBOARDING_READY = "READY"
ONBOARDING_DEGRADED = "DEGRADED"

_ONBOARDING_STATES = {
    ONBOARDING_NOT_STARTED,
    ONBOARDING_TOKEN_MISSING,
    ONBOARDING_LLM_MISSING,
    ONBOARDING_SERVICES_DOWN,
    ONBOARDING_READY,
    ONBOARDING_DEGRADED,
}

_TOKEN_FAILURE_CODES = {
    "telegram_token_missing",
    "missing_token",
    "token_invalid",
    "telegram_token_invalid",
}

_LLM_FAILURE_CODES = {
    "llm_unavailable",
    "no_chat_model",
    "provider_unhealthy",
    "model_unhealthy",
    "router_unavailable",
}

_SERVICE_FAILURE_CODES = {
    "api_down",
    "service_down",
    "startup_check_failed",
    "config_load_failed",
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _telegram_enabled(ready_payload: Mapping[str, Any]) -> bool:
    telegram = _as_map(ready_payload.get("telegram"))
    raw = telegram.get("enabled")
    if isinstance(raw, bool):
        return raw
    normalized = _norm(raw)
    if normalized in {"0", "false", "off", "no"}:
        return False
    if normalized in {"1", "true", "on", "yes"}:
        return True
    return True


def _telegram_required(ready_payload: Mapping[str, Any]) -> bool:
    telegram = _as_map(ready_payload.get("telegram"))
    raw = telegram.get("required")
    if isinstance(raw, bool):
        return raw
    normalized = _norm(raw)
    if normalized in {"1", "true", "on", "yes"}:
        return True
    return False


def _health_status(status_payload: Mapping[str, Any], key: str) -> str:
    row = _as_map(status_payload.get(key))
    return _norm(row.get("status"))


def _default_model(status_payload: Mapping[str, Any]) -> str:
    return str(
        status_payload.get("resolved_default_model")
        or status_payload.get("chat_model")
        or status_payload.get("default_model")
        or ""
    ).strip()


def detect_onboarding_state(
    *,
    ready_payload: Mapping[str, Any] | None = None,
    llm_status: Mapping[str, Any] | None = None,
    startup_report: Mapping[str, Any] | None = None,
) -> str:
    ready = _as_map(ready_payload)
    status = _as_map(llm_status)
    startup = _as_map(startup_report)
    if not ready and not status and not startup:
        return ONBOARDING_NOT_STARTED

    runtime_status = _as_map(ready.get("runtime_status"))
    failure_code = _norm(
        startup.get("failure_code")
        or ready.get("failure_code")
        or ready.get("llm_reason")
        or runtime_status.get("failure_code")
    )

    telegram = _as_map(ready.get("telegram"))
    telegram_state = _norm(telegram.get("state"))
    telegram_configured = telegram.get("configured")
    telegram_enabled = _telegram_enabled(ready)
    telegram_required = _telegram_required(ready)

    if telegram_enabled and telegram_required and failure_code in _TOKEN_FAILURE_CODES:
        return ONBOARDING_TOKEN_MISSING
    if telegram_enabled and telegram_required and telegram_state == "disabled_missing_token":
        return ONBOARDING_TOKEN_MISSING
    if telegram_enabled and telegram_required and telegram_configured is False:
        return ONBOARDING_TOKEN_MISSING

    ready_flag = bool(ready.get("ready", False))
    phase = _norm(ready.get("startup_phase") or ready.get("phase"))
    runtime_mode = _norm(runtime_status.get("runtime_mode") or ready.get("runtime_mode"))
    startup_status = _norm(startup.get("status"))

    status_source = status if status else ready
    model_id = _default_model(status_source)
    provider_state = _health_status(status_source, "active_provider_health")
    model_state = _health_status(status_source, "active_model_health")

    if ready_flag and runtime_mode == "ready":
        return ONBOARDING_READY

    if failure_code in _SERVICE_FAILURE_CODES or startup_status == "fail":
        return ONBOARDING_SERVICES_DOWN
    if phase in {"starting", "listening", "warming"}:
        return ONBOARDING_SERVICES_DOWN

    if failure_code in _LLM_FAILURE_CODES:
        return ONBOARDING_LLM_MISSING
    if not model_id:
        return ONBOARDING_LLM_MISSING
    if provider_state and provider_state != "ok":
        return ONBOARDING_LLM_MISSING
    if model_state and model_state != "ok":
        return ONBOARDING_LLM_MISSING
    if (
        not ready
        and model_id
        and provider_state in {"", "ok", "unknown"}
        and model_state in {"", "ok", "unknown"}
    ):
        return ONBOARDING_READY

    if runtime_mode in {"degraded", "failed"} or not ready_flag:
        return ONBOARDING_DEGRADED
    return ONBOARDING_NOT_STARTED


def onboarding_next_action(
    state: str,
    *,
    ready_payload: Mapping[str, Any] | None = None,
) -> str:
    normalized = str(state or "").strip().upper()
    ready = _as_map(ready_payload)
    telegram = _as_map(ready.get("telegram"))
    telegram_state = _norm(telegram.get("state"))
    telegram_enabled = _telegram_enabled(ready)
    telegram_required = _telegram_required(ready)
    if normalized == ONBOARDING_READY:
        return "No action needed."
    if normalized == ONBOARDING_TOKEN_MISSING:
        return "Open Setup > Basics, add your Telegram bot token, then choose Save and test."
    if normalized == ONBOARDING_SERVICES_DOWN:
        if telegram_enabled and telegram_required and telegram_state in {"stopped", "crash_loop"}:
            return "Open Setup > Optional capabilities and choose Restart Telegram."
        return "Open Diagnostics and choose Restart Personal Agent."
    if normalized == ONBOARDING_LLM_MISSING:
        return "Open Setup > Basics, choose an installed healthy chat model, and save it."
    if normalized == ONBOARDING_DEGRADED:
        return "Open Diagnostics, refresh status, and follow the suggested recovery action."
    return "Open Setup > Basics and follow the short setup checklist."


def onboarding_summary(
    state: str,
    *,
    ready_payload: Mapping[str, Any] | None = None,
    llm_status: Mapping[str, Any] | None = None,
) -> str:
    normalized = str(state or "").strip().upper()
    if normalized == ONBOARDING_READY:
        return "Setup complete. The agent is ready."
    if normalized == ONBOARDING_TOKEN_MISSING:
        return "Telegram bot token is missing."
    if normalized == ONBOARDING_SERVICES_DOWN:
        return "Core services are down or still starting."
    if normalized == ONBOARDING_LLM_MISSING:
        status = _as_map(llm_status if llm_status else ready_payload)
        model = _default_model(status)
        if model:
            return f"Chat model {model} is not healthy."
        return build_no_llm_public_message()
    if normalized == ONBOARDING_DEGRADED:
        return "Setup is partially complete but degraded."
    return "Setup has not started."


def onboarding_steps(state: str) -> list[str]:
    normalized = str(state or "").strip().upper()
    if normalized == ONBOARDING_READY:
        return [
            "Describe what you want in chat.",
            "Open Setup > Basics to review local access, model, and permission choices.",
            "Open Diagnostics if anything looks wrong.",
        ]
    if normalized == ONBOARDING_TOKEN_MISSING:
        return [
            "Open Setup > Basics.",
            "Paste the Telegram bot token into the hidden token field.",
            "Choose Save and test; Telegram is optional unless your policy requires it.",
        ]
    if normalized == ONBOARDING_SERVICES_DOWN:
        return [
            "Open Diagnostics and refresh status.",
            "Choose the offered restart action for the affected Personal Agent service.",
            "Wait for Ready, then return to chat.",
        ]
    if normalized == ONBOARDING_LLM_MISSING:
        return [
            "Open Setup > Basics and review installed models.",
            "Choose a healthy chat model and save it.",
            "Use Refresh status to confirm it is responding.",
        ]
    if normalized == ONBOARDING_DEGRADED:
        return [
            "Open Diagnostics and refresh status.",
            "Follow the one suggested recovery action.",
            "Export a redacted diagnostics bundle if you need help.",
        ]
    return [
        "Open Setup > Basics and choose what you want the assistant to help with.",
        "Review local folders, model, and Safe Mode before saving.",
        "Optional network and skill features stay off until you enable and approve them.",
    ]


def normalize_onboarding_state(state: str) -> str:
    normalized = str(state or "").strip().upper()
    if normalized in _ONBOARDING_STATES:
        return normalized
    return ONBOARDING_NOT_STARTED


__all__ = [
    "ONBOARDING_DEGRADED",
    "ONBOARDING_LLM_MISSING",
    "ONBOARDING_NOT_STARTED",
    "ONBOARDING_READY",
    "ONBOARDING_SERVICES_DOWN",
    "ONBOARDING_TOKEN_MISSING",
    "detect_onboarding_state",
    "normalize_onboarding_state",
    "onboarding_next_action",
    "onboarding_steps",
    "onboarding_summary",
]
