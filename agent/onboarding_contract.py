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

    if telegram_enabled and failure_code in _TOKEN_FAILURE_CODES:
        return ONBOARDING_TOKEN_MISSING
    if telegram_enabled and telegram_state == "disabled_missing_token":
        return ONBOARDING_TOKEN_MISSING
    if telegram_enabled and telegram_configured is False:
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
    if normalized == ONBOARDING_READY:
        return "No action needed."
    if normalized == ONBOARDING_TOKEN_MISSING:
        return "Run: python -m agent.secrets set telegram:bot_token"
    if normalized == ONBOARDING_SERVICES_DOWN:
        if telegram_enabled and telegram_state in {"stopped", "crash_loop"}:
            return "Run: systemctl --user restart personal-agent-telegram.service"
        return "Run: systemctl --user restart personal-agent-api.service"
    if normalized == ONBOARDING_LLM_MISSING:
        return "Run: python -m agent setup"
    if normalized == ONBOARDING_DEGRADED:
        return "Run: python -m agent doctor"
    return "Run: python -m agent setup"


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
            "Use Telegram naturally.",
            "Run: python -m agent status",
            "Run: python -m agent doctor if anything looks wrong.",
        ]
    if normalized == ONBOARDING_TOKEN_MISSING:
        return [
            "Run: python -m agent.secrets set telegram:bot_token",
            "Run: systemctl --user restart personal-agent-telegram.service",
            "Run: python -m agent status",
        ]
    if normalized == ONBOARDING_SERVICES_DOWN:
        return [
            "Run: systemctl --user restart personal-agent-api.service",
            "If Telegram is enabled: run systemctl --user restart personal-agent-telegram.service",
            "Run: python -m agent status",
        ]
    if normalized == ONBOARDING_LLM_MISSING:
        return [
            "Run: python -m agent doctor",
            "Run: python -m agent setup --dry-run",
            "Run: python -m agent status",
        ]
    if normalized == ONBOARDING_DEGRADED:
        return [
            "Run: python -m agent doctor",
            "Run: python -m agent setup --dry-run",
            "Run: python -m agent status",
        ]
    return [
        "Run: python -m agent setup",
        "Run: python -m agent doctor",
        "Run: python -m agent status",
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
