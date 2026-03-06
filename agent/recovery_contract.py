from __future__ import annotations

from typing import Any, Mapping


RECOVERY_TELEGRAM_DOWN = "TELEGRAM_DOWN"
RECOVERY_API_DOWN = "API_DOWN"
RECOVERY_TOKEN_INVALID = "TOKEN_INVALID"
RECOVERY_LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
RECOVERY_LOCK_CONFLICT = "LOCK_CONFLICT"
RECOVERY_DEGRADED_READ_ONLY = "DEGRADED_READ_ONLY"
RECOVERY_UNKNOWN_FAILURE = "UNKNOWN_FAILURE"

_RECOVERY_MODES = {
    RECOVERY_TELEGRAM_DOWN,
    RECOVERY_API_DOWN,
    RECOVERY_TOKEN_INVALID,
    RECOVERY_LLM_UNAVAILABLE,
    RECOVERY_LOCK_CONFLICT,
    RECOVERY_DEGRADED_READ_ONLY,
    RECOVERY_UNKNOWN_FAILURE,
}

_TOKEN_CODES = {"token_invalid", "telegram_token_invalid", "telegram_token_missing", "missing_token"}
_LOCK_CODES = {"lock_unavailable", "lock_path_unavailable", "telegram_conflict", "lock_conflict"}
_LLM_CODES = {"llm_unavailable", "provider_unhealthy", "model_unhealthy", "no_chat_model", "router_unavailable"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _health_status(payload: Mapping[str, Any], key: str) -> str:
    row = _as_map(payload.get(key))
    return _norm(row.get("status"))


def detect_recovery_mode(
    *,
    ready_payload: Mapping[str, Any] | None = None,
    llm_status: Mapping[str, Any] | None = None,
    failure_code: str | None = None,
    api_reachable: bool | None = None,
) -> str:
    ready = _as_map(ready_payload)
    status = _as_map(llm_status)
    code = _norm(
        failure_code
        or ready.get("failure_code")
        or ready.get("llm_reason")
        or _as_map(ready.get("runtime_status")).get("failure_code")
    )

    if api_reachable is False:
        return RECOVERY_API_DOWN
    if code in _LOCK_CODES:
        return RECOVERY_LOCK_CONFLICT
    if code in _TOKEN_CODES:
        return RECOVERY_TOKEN_INVALID

    telegram = _as_map(ready.get("telegram"))
    telegram_state = _norm(telegram.get("state"))
    telegram_configured = telegram.get("configured")
    if telegram_state == "crash_loop":
        return RECOVERY_TELEGRAM_DOWN
    if telegram_configured is True and telegram_state == "stopped":
        return RECOVERY_TELEGRAM_DOWN
    if telegram_state == "disabled_missing_token":
        return RECOVERY_TOKEN_INVALID

    status_source = status if status else ready
    provider_state = _health_status(status_source, "active_provider_health")
    model_state = _health_status(status_source, "active_model_health")
    model_id = str(
        status_source.get("resolved_default_model")
        or status_source.get("chat_model")
        or status_source.get("default_model")
        or ""
    ).strip()
    runtime_mode = _norm(_as_map(ready.get("runtime_status")).get("runtime_mode") or ready.get("runtime_mode"))
    safe_mode = _as_map(status_source.get("safe_mode"))
    read_only_safe = bool(safe_mode.get("safe_mode")) if "safe_mode" in safe_mode else False

    if code in _LLM_CODES:
        return RECOVERY_LLM_UNAVAILABLE
    if not model_id:
        return RECOVERY_LLM_UNAVAILABLE
    if provider_state and provider_state != "ok":
        return RECOVERY_LLM_UNAVAILABLE
    if model_state and model_state != "ok":
        return RECOVERY_LLM_UNAVAILABLE
    if runtime_mode == "degraded" or read_only_safe:
        return RECOVERY_DEGRADED_READ_ONLY
    if ready and not bool(ready.get("ready", True)):
        return RECOVERY_UNKNOWN_FAILURE
    return RECOVERY_UNKNOWN_FAILURE


def recovery_next_action(mode: str) -> str:
    normalized = str(mode or "").strip().upper()
    if normalized == RECOVERY_TELEGRAM_DOWN:
        return "Run: systemctl --user restart personal-agent-telegram.service"
    if normalized == RECOVERY_API_DOWN:
        return "Run: systemctl --user restart personal-agent-api.service"
    if normalized == RECOVERY_TOKEN_INVALID:
        return "Run: python -m agent.secrets set telegram:bot_token"
    if normalized == RECOVERY_LLM_UNAVAILABLE:
        return "Run: python -m agent setup"
    if normalized == RECOVERY_LOCK_CONFLICT:
        return "Stop duplicate Telegram pollers, then restart personal-agent-telegram.service"
    if normalized == RECOVERY_DEGRADED_READ_ONLY:
        return "Run: python -m agent doctor"
    return "Run: python -m agent doctor"


def recovery_summary(mode: str) -> str:
    normalized = str(mode or "").strip().upper()
    if normalized == RECOVERY_TELEGRAM_DOWN:
        return "Telegram delivery is down."
    if normalized == RECOVERY_API_DOWN:
        return "API service is down."
    if normalized == RECOVERY_TOKEN_INVALID:
        return "Telegram token is missing or invalid."
    if normalized == RECOVERY_LLM_UNAVAILABLE:
        return "Chat LLM is unavailable."
    if normalized == RECOVERY_LOCK_CONFLICT:
        return "Telegram poll lock conflict detected."
    if normalized == RECOVERY_DEGRADED_READ_ONLY:
        return "Runtime is degraded in read-only mode."
    return "Runtime failure reason is unknown."


def normalize_recovery_mode(mode: str) -> str:
    normalized = str(mode or "").strip().upper()
    if normalized in _RECOVERY_MODES:
        return normalized
    return RECOVERY_UNKNOWN_FAILURE


__all__ = [
    "RECOVERY_API_DOWN",
    "RECOVERY_DEGRADED_READ_ONLY",
    "RECOVERY_LLM_UNAVAILABLE",
    "RECOVERY_LOCK_CONFLICT",
    "RECOVERY_TELEGRAM_DOWN",
    "RECOVERY_TOKEN_INVALID",
    "RECOVERY_UNKNOWN_FAILURE",
    "detect_recovery_mode",
    "normalize_recovery_mode",
    "recovery_next_action",
    "recovery_summary",
]
