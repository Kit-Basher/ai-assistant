from __future__ import annotations

from typing import Any, Iterable, Mapping


RUNTIME_MODE_READY = "READY"
RUNTIME_MODE_BOOTSTRAP_REQUIRED = "BOOTSTRAP_REQUIRED"
RUNTIME_MODE_DEGRADED = "DEGRADED"
RUNTIME_MODE_FAILED = "FAILED"

_FAILED_CODES = {
    "config_load_failed",
    "router_unavailable",
    "startup_check_failed",
    "secret_store_unreadable",
    "secret_store_decrypt_failed",
    "registry_invalid_json",
    "registry_unreadable",
    "internal_error",
}

_BOOTSTRAP_CODES = {
    "llm_unavailable",
    "no_chat_model",
    "provider_unhealthy",
    "model_unhealthy",
    "safe_mode_paused",
    "telegram_token_missing",
    "missing_token",
}


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def get_runtime_mode(
    *,
    ready: bool | None = None,
    bootstrap_required: bool | None = None,
    failure_code: str | None = None,
    phase: str | None = None,
) -> str:
    normalized_code = _norm_text(failure_code)
    normalized_phase = _norm_text(phase)
    ready_value = _bool_or_none(ready)
    bootstrap_value = bool(bootstrap_required)

    if normalized_phase == "failed":
        return RUNTIME_MODE_FAILED
    if normalized_code in _FAILED_CODES:
        return RUNTIME_MODE_FAILED
    if bootstrap_value or normalized_code in _BOOTSTRAP_CODES:
        return RUNTIME_MODE_BOOTSTRAP_REQUIRED
    if ready_value is True and normalized_phase not in {"starting", "listening", "warming", "degraded"}:
        return RUNTIME_MODE_READY
    return RUNTIME_MODE_DEGRADED


def get_effective_llm_identity(
    *,
    provider: str | None,
    model: str | None,
    local_providers: Iterable[str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    provider_id = str(provider or "").strip().lower() or None
    model_id = str(model or "").strip() or None
    local_set = {
        str(item).strip().lower()
        for item in (local_providers or [])
        if str(item).strip()
    }
    known = bool(provider_id and model_id)
    if not known:
        return {
            "provider": provider_id,
            "model": model_id,
            "local_remote": "unknown",
            "known": False,
            "reason": str(reason or "unknown_provider_model").strip().lower() or "unknown_provider_model",
        }
    local_remote = "local" if provider_id in local_set else "remote"
    return {
        "provider": provider_id,
        "model": model_id,
        "local_remote": local_remote,
        "known": True,
        "reason": str(reason or "ok").strip().lower() or "ok",
    }


def _next_action_from_failure_code(failure_code: str | None) -> str:
    code = _norm_text(failure_code)
    if code in {"telegram_token_missing", "missing_token"}:
        return "Run: python -m agent.secrets set telegram:bot_token"
    if code in {"registry_unreadable", "registry_invalid_json"}:
        return "Check LLM_REGISTRY_PATH and run: python -m agent doctor"
    if code in {"lock_unavailable", "lock_path_unavailable"}:
        return "Stop duplicate Telegram processes and restart the service"
    if code in {"config_load_failed", "config_invalid"}:
        return "Review config and run: python -m agent doctor"
    return "Run: python -m agent doctor"


def get_effective_next_action(
    *,
    runtime_mode: str,
    failure_code: str | None = None,
) -> str | None:
    mode = str(runtime_mode or "").strip().upper()
    if mode == RUNTIME_MODE_READY:
        return None
    return _next_action_from_failure_code(failure_code)


def normalize_user_facing_status(
    *,
    ready: bool | None = None,
    bootstrap_required: bool | None = None,
    failure_code: str | None = None,
    phase: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    local_providers: Iterable[str] | None = None,
) -> dict[str, Any]:
    runtime_mode = get_runtime_mode(
        ready=ready,
        bootstrap_required=bootstrap_required,
        failure_code=failure_code,
        phase=phase,
    )
    identity = get_effective_llm_identity(
        provider=provider,
        model=model,
        local_providers=local_providers,
        reason=failure_code,
    )
    next_action = get_effective_next_action(
        runtime_mode=runtime_mode,
        failure_code=failure_code,
    )
    if runtime_mode == RUNTIME_MODE_READY:
        if bool(identity.get("known")):
            summary = f"Agent is ready. Using {identity.get('provider')} / {identity.get('model')}."
        else:
            summary = "Agent is ready."
    elif runtime_mode == RUNTIME_MODE_BOOTSTRAP_REQUIRED:
        summary = "Setup needed. No chat model is ready yet."
    elif runtime_mode == RUNTIME_MODE_FAILED:
        summary = "Agent failed."
    else:
        summary = "Agent is starting or degraded."
    if runtime_mode != RUNTIME_MODE_READY and next_action:
        summary = f"{summary}\nNext: {next_action}"
    return {
        "runtime_mode": runtime_mode,
        "ready": runtime_mode == RUNTIME_MODE_READY,
        "bootstrap_required": runtime_mode == RUNTIME_MODE_BOOTSTRAP_REQUIRED,
        "degraded": runtime_mode == RUNTIME_MODE_DEGRADED,
        "failed": runtime_mode == RUNTIME_MODE_FAILED,
        "failure_code": _norm_text(failure_code) or None,
        "provider": identity.get("provider"),
        "model": identity.get("model"),
        "local_remote": identity.get("local_remote"),
        "identity_known": bool(identity.get("known", False)),
        "identity_reason": str(identity.get("reason") or "").strip().lower() or "unknown",
        "next_action": next_action,
        "summary": summary,
    }


__all__ = [
    "RUNTIME_MODE_BOOTSTRAP_REQUIRED",
    "RUNTIME_MODE_DEGRADED",
    "RUNTIME_MODE_FAILED",
    "RUNTIME_MODE_READY",
    "get_effective_llm_identity",
    "get_effective_next_action",
    "get_runtime_mode",
    "normalize_user_facing_status",
]

