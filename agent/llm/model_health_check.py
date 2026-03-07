from __future__ import annotations

import os
from typing import Any, Callable

from agent.config import Config
from agent.llm.approved_local_models import approved_local_profile_for_ref
from agent.llm.probes import probe_model, probe_provider
from agent.llm.registry import ModelConfig, ProviderConfig, Registry


ProviderProbeFn = Callable[..., dict[str, Any]]
ModelProbeFn = Callable[..., dict[str, Any]]


def _provider_probe_config(provider_cfg: ProviderConfig, config: Config, *, allow_remote_fallback: bool) -> dict[str, Any]:
    auth_present = False
    if provider_cfg.auth_env_var:
        auth_present = bool(os.getenv(provider_cfg.auth_env_var, "").strip())
    return {
        "id": provider_cfg.id,
        "provider_type": provider_cfg.provider_type,
        "base_url": provider_cfg.base_url,
        "chat_path": provider_cfg.chat_path,
        "enabled": provider_cfg.enabled,
        "local": provider_cfg.local,
        "allow_remote_fallback": allow_remote_fallback,
        "api_key_source": (
            {
                "type": provider_cfg.api_key_source.source_type,
                "name": provider_cfg.api_key_source.name,
            }
            if provider_cfg.api_key_source is not None
            else None
        ),
        "headers": {},
        "_resolved_api_key_present": auth_present,
        "available": True,
        "config_ollama_base_url": config.ollama_base_url,
    }


def _map_probe_failure_kind(error_kind: str | None) -> str:
    normalized = str(error_kind or "").strip().lower()
    if normalized in {"timeout"}:
        return "timed_out"
    if normalized in {"bad_request", "bad_status_code", "invalid_json"}:
        return "bad_response"
    if normalized in {"provider_disabled", "provider_unavailable", "connection_refused", "connection_error", "unreachable"}:
        return "provider_down"
    if normalized:
        return normalized
    return "unavailable"


def check_provider_health(
    *,
    config: Config,
    registry: Registry,
    provider_id: str,
    provider_probe_fn: ProviderProbeFn | None = None,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    provider_cfg = registry.providers.get(str(provider_id or "").strip().lower())
    if provider_cfg is None:
        return {
            "healthy": False,
            "failure_kind": "unavailable",
            "provider_health": {"status": "down", "error_kind": "provider_missing"},
        }
    probe = (provider_probe_fn or probe_provider)(
        _provider_probe_config(
            provider_cfg,
            config,
            allow_remote_fallback=bool(registry.defaults.allow_remote_fallback),
        ),
        timeout_seconds=float(timeout_seconds),
    )
    status = str(probe.get("status") or "unknown").strip().lower()
    error_kind = str(probe.get("error_kind") or probe.get("last_error_kind") or "").strip().lower() or None
    if status == "ok":
        return {
            "healthy": True,
            "failure_kind": None,
            "provider_health": probe,
        }
    return {
        "healthy": False,
        "failure_kind": _map_probe_failure_kind(error_kind),
        "provider_health": probe,
    }


def check_model_health(
    *,
    config: Config,
    registry: Registry,
    model: ModelConfig,
    installed: bool,
    provider_health: dict[str, Any] | None = None,
    model_probe_fn: ModelProbeFn | None = None,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    provider_cfg = registry.providers.get(model.provider)
    if provider_cfg is None:
        return {
            "healthy": False,
            "failure_kind": "provider_down",
            "provider_health": provider_health or {"status": "down", "error_kind": "provider_missing"},
            "model_health": None,
        }
    if not provider_cfg.enabled or str((provider_health or {}).get("status") or "").strip().lower() != "ok":
        return {
            "healthy": False,
            "failure_kind": "provider_down",
            "provider_health": provider_health or {"status": "down", "error_kind": "provider_disabled"},
            "model_health": None,
        }
    if provider_cfg.local and not installed:
        return {
            "healthy": False,
            "failure_kind": "not_installed",
            "provider_health": provider_health,
            "model_health": {"status": "down", "error_kind": "not_installed"},
        }
    if not model.enabled or not model.available:
        return {
            "healthy": False,
            "failure_kind": "unavailable",
            "provider_health": provider_health,
            "model_health": {"status": "down", "error_kind": "unavailable"},
        }
    approved_profile = approved_local_profile_for_ref(model.id) or approved_local_profile_for_ref(model.model)
    approved_caps = {
        str(item).strip().lower()
        for item in ((approved_profile or {}).get("capabilities") or [])
        if str(item).strip()
    }
    if provider_cfg.local and installed and "vision" in approved_caps:
        return {
            "healthy": True,
            "failure_kind": None,
            "provider_health": provider_health,
            "model_health": {
                "status": "ok",
                "error_kind": None,
                "detail": "metadata verified local vision profile",
                "health_reason": "approved_profile_local_vision",
                "probe_mode": "metadata",
            },
        }
    # Only actively probe local models. Remote models inherit provider health to stay cheap and deterministic.
    if not provider_cfg.local:
        return {
            "healthy": True,
            "failure_kind": None,
            "provider_health": provider_health,
            "model_health": {"status": "ok", "error_kind": None, "detail": "provider_health_ok"},
        }

    probe = (model_probe_fn or probe_model)(
        _provider_probe_config(
            provider_cfg,
            config,
            allow_remote_fallback=bool(registry.defaults.allow_remote_fallback),
        ),
        model.id,
        timeout_seconds=float(timeout_seconds),
        model_capabilities=sorted(model.capabilities),
    )
    status = str(probe.get("status") or "unknown").strip().lower()
    error_kind = str(probe.get("error_kind") or "").strip().lower() or None
    if status == "ok":
        return {
            "healthy": True,
            "failure_kind": None,
            "provider_health": provider_health,
            "model_health": probe,
        }
    return {
        "healthy": False,
        "failure_kind": _map_probe_failure_kind(error_kind),
        "provider_health": provider_health,
        "model_health": probe,
    }


__all__ = [
    "check_model_health",
    "check_provider_health",
]
