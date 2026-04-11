from __future__ import annotations

from typing import Any, Mapping

from agent.public_chat import build_no_llm_public_message
from agent.runtime_contract import (
    RUNTIME_MODE_DEGRADED,
    get_effective_next_action,
    normalize_user_facing_status,
)


def is_runtime_ready(
    *,
    ready_payload: Mapping[str, Any] | None = None,
    llm_status: Mapping[str, Any] | None = None,
) -> bool:
    ready_row = ready_payload if isinstance(ready_payload, Mapping) else {}
    if "ready" in ready_row:
        return bool(ready_row.get("ready", False))
    status = llm_status if isinstance(llm_status, Mapping) else {}
    provider_health = (
        status.get("active_provider_health")
        if isinstance(status.get("active_provider_health"), Mapping)
        else {}
    )
    model_health = (
        status.get("active_model_health")
        if isinstance(status.get("active_model_health"), Mapping)
        else {}
    )
    provider_ok = str(provider_health.get("status") or "").strip().lower() == "ok"
    model_ok = str(model_health.get("status") or "").strip().lower() == "ok"
    model_name = (
        str(status.get("resolved_default_model") or "").strip()
        or str(status.get("default_model") or "").strip()
    )
    return bool(provider_ok and model_ok and model_name)


def bootstrap_needed(
    *,
    llm_available: bool | None = None,
    availability_reason: str | None = None,
    defaults: Mapping[str, Any] | None = None,
    llm_status: Mapping[str, Any] | None = None,
) -> bool:
    if llm_available is False:
        return True
    reason = str(availability_reason or "").strip().lower()
    if reason in {"llm_unavailable", "provider_unhealthy", "no_routable_model", "no_chat_model"}:
        return True
    default_row = defaults if isinstance(defaults, Mapping) else {}
    if default_row:
        chat_model = str(default_row.get("chat_model") or default_row.get("default_model") or "").strip()
        if not chat_model:
            return True
    status = llm_status if isinstance(llm_status, Mapping) else {}
    model_name = (
        str(status.get("resolved_default_model") or "").strip()
        or str(status.get("default_model") or "").strip()
    )
    if status and not model_name:
        return True
    model_health = (
        status.get("active_model_health")
        if isinstance(status.get("active_model_health"), Mapping)
        else {}
    )
    model_status = str(model_health.get("status") or "").strip().lower()
    if model_status in {"down", "unknown"} and not bool(status.get("allow_remote_fallback", True)):
        return True
    return False


def next_step_for_failure(failure_code: str | None) -> str:
    return str(
        get_effective_next_action(
            runtime_mode=RUNTIME_MODE_DEGRADED,
            failure_code=failure_code,
        )
        or "Run: python -m agent doctor"
    )


def bootstrap_guidance() -> str:
    return build_no_llm_public_message()


def user_safe_summary(
    *,
    ready: bool,
    provider: str | None = None,
    model: str | None = None,
    bootstrap: bool = False,
    failure_code: str | None = None,
) -> str:
    normalized = normalize_user_facing_status(
        ready=bool(ready),
        bootstrap_required=bool(bootstrap),
        failure_code=failure_code,
        provider=provider,
        model=model,
    )
    return str(normalized.get("summary") or "").strip()


__all__ = [
    "bootstrap_guidance",
    "bootstrap_needed",
    "is_runtime_ready",
    "next_step_for_failure",
    "user_safe_summary",
]
