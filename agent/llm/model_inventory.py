from __future__ import annotations

import re
from typing import Any, Iterable

from agent.config import Config
from agent.llm.capabilities import capability_list_from_inference, infer_capabilities_from_catalog, is_embedding_model_name
from agent.llm.control_contract import normalize_model_inventory
from agent.llm.model_health_check import check_model_health, check_provider_health
from agent.llm.registry import ModelConfig, Registry, load_registry
from agent.modelops.discovery import list_models_ollama


def _size_label(model_name: str) -> str | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", str(model_name or "").strip().lower())
    if not match:
        return None
    return f"{match.group(1)}B"


def _configured_model_ids(registry: Registry) -> set[str]:
    defaults = registry.defaults
    return {
        str(item).strip()
        for item in (
            defaults.chat_model,
            defaults.embed_model,
            defaults.default_model,
        )
        if str(item or "").strip()
    }


def _approved_model_ids(config: Config) -> set[str]:
    default_allowlist = config.default_policy.get("allowlist") if isinstance(config.default_policy, dict) else []
    premium_allowlist = config.premium_policy.get("allowlist") if isinstance(config.premium_policy, dict) else []
    values = {
        str(item).strip()
        for item in [*(default_allowlist or []), *(premium_allowlist or [])]
        if str(item).strip()
    }
    return values


def _discover_local_model_names(discovered_local_models: Iterable[str] | None) -> set[str]:
    if discovered_local_models is not None:
        return {
            str(item).strip()
            for item in discovered_local_models
            if str(item).strip()
        }
    discovered: set[str] = set()
    for row in list_models_ollama():
        model_id = str(getattr(row, "model_id", "") or "").strip()
        if model_id:
            discovered.add(model_id)
    return discovered


def _inferred_capabilities_for_discovered(model_name: str) -> list[str]:
    inferred = infer_capabilities_from_catalog(
        "ollama",
        {
            "id": f"ollama:{model_name}",
            "provider_id": "ollama",
            "model": model_name,
            "capabilities": ["embedding"] if is_embedding_model_name(model_name) else ["chat"],
        },
    )
    return capability_list_from_inference(inferred)


def _inventory_row_from_model(
    *,
    model: ModelConfig,
    installed_local_names: set[str],
    config: Config,
    approved_ids: set[str],
    registry: Registry,
    provider_health_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    provider_cfg = registry.providers.get(model.provider)
    local = bool(provider_cfg.local) if provider_cfg is not None else False
    installed = model.model in installed_local_names if local else False
    available = bool(model.enabled and (installed if local else model.available))
    configured = model.id in _configured_model_ids(registry)
    provider_health = provider_health_cache.get(model.provider) or {}
    health = check_model_health(
        config=config,
        registry=registry,
        model=model,
        installed=installed,
        provider_health=provider_health,
    )
    approved = bool(local)
    if not approved:
        if approved_ids:
            approved = model.id in approved_ids
        else:
            approved = bool(registry.defaults.allow_remote_fallback)
    reason = "healthy" if bool(health.get("healthy")) else str(health.get("failure_kind") or "unavailable")
    return {
        "id": model.id,
        "provider": model.provider,
        "installed": installed,
        "available": available,
        "healthy": bool(health.get("healthy", False)),
        "capabilities": sorted(model.capabilities),
        "size": _size_label(model.model),
        "context_window": model.max_context_tokens,
        "local": local,
        "approved": approved,
        "reason": reason,
        "quality_rank": model.quality_rank,
        "cost_rank": model.cost_rank,
        "price_in": model.input_cost_per_million_tokens,
        "price_out": model.output_cost_per_million_tokens,
        "health_status": "ok" if bool(health.get("healthy", False)) else "down",
        "health_failure_kind": health.get("failure_kind"),
        "model_name": model.model,
        "source": "registry",
        "configured": configured,
    }


def _discovered_rows(
    *,
    config: Config,
    registry: Registry,
    installed_local_names: set[str],
    approved_ids: set[str],
    provider_health_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = {model.model for model in registry.models.values() if model.provider == "ollama"}
    rows: list[dict[str, Any]] = []
    provider_health = provider_health_cache.get("ollama") or {}
    for model_name in sorted(installed_local_names):
        if model_name in existing:
            continue
        model_id = f"ollama:{model_name}"
        capabilities = _inferred_capabilities_for_discovered(model_name)
        temp_model = ModelConfig(
            id=model_id,
            provider="ollama",
            model=model_name,
            capabilities=frozenset(capabilities),
            quality_rank=0,
            cost_rank=0,
            default_for=tuple(),
            enabled=True,
            available=True,
            input_cost_per_million_tokens=None,
            output_cost_per_million_tokens=None,
            max_context_tokens=None,
        )
        health = check_model_health(
            config=config,
            registry=registry,
            model=temp_model,
            installed=True,
            provider_health=provider_health,
        )
        rows.append(
            {
                "id": model_id,
                "provider": "ollama",
                "installed": True,
                "available": True,
                "healthy": bool(health.get("healthy", False)),
                "capabilities": capabilities,
                "size": _size_label(model_name),
                "context_window": None,
                "local": True,
                "approved": True if not approved_ids else model_id in approved_ids,
                "reason": "discovered_local_model" if bool(health.get("healthy", False)) else str(health.get("failure_kind") or "unavailable"),
                "quality_rank": 0,
                "cost_rank": 0,
                "price_in": None,
                "price_out": None,
                "health_status": "ok" if bool(health.get("healthy", False)) else "down",
                "health_failure_kind": health.get("failure_kind"),
                "model_name": model_name,
                "source": "ollama_list",
                "configured": False,
            }
        )
    return rows


def build_model_inventory(
    *,
    config: Config,
    registry: Registry | None = None,
    discovered_local_models: Iterable[str] | None = None,
    timeout_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    active_registry = registry or load_registry(config)
    installed_local_names = _discover_local_model_names(discovered_local_models)
    approved_ids = _approved_model_ids(config)
    provider_health_cache: dict[str, dict[str, Any]] = {}
    for provider_id in sorted(active_registry.providers.keys()):
        provider_cfg = active_registry.providers.get(provider_id)
        if provider_cfg is not None and not provider_cfg.local:
            provider_health_cache[provider_id] = {
                "status": "ok" if provider_cfg.enabled else "down",
                "error_kind": None if provider_cfg.enabled else "provider_disabled",
                "detail": "remote_provider_not_probed_by_default",
            }
            continue
        provider_health_cache[provider_id] = check_provider_health(
            config=config,
            registry=active_registry,
            provider_id=provider_id,
            timeout_seconds=float(timeout_seconds),
        ).get("provider_health", {})

    rows = [
        _inventory_row_from_model(
            model=model,
            installed_local_names=installed_local_names,
            config=config,
            approved_ids=approved_ids,
            registry=active_registry,
            provider_health_cache=provider_health_cache,
        )
        for model in active_registry.sorted_models()
    ]
    rows.extend(
        _discovered_rows(
            config=config,
            registry=active_registry,
            installed_local_names=installed_local_names,
            approved_ids=approved_ids,
            provider_health_cache=provider_health_cache,
        )
    )
    return normalize_model_inventory(rows)


__all__ = ["build_model_inventory"]
