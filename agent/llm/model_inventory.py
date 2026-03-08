from __future__ import annotations

import re
from typing import Any, Iterable

from agent.config import Config
from agent.llm.approved_local_models import approved_local_profile_for_ref
from agent.llm.capabilities import capability_list_from_inference, infer_capabilities_from_catalog, is_embedding_model_name
from agent.llm.control_contract import normalize_model_inventory
from agent.llm.registry import Registry, load_registry
from agent.llm.runtime_model_snapshot import build_runtime_model_snapshot


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
    return {
        str(item).strip()
        for item in [*(default_allowlist or []), *(premium_allowlist or [])]
        if str(item).strip()
    }


def _normalize_caps(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(item).strip().lower()
            for item in values
            if str(item).strip()
        }
    )


def _approved_profile_caps(model_id: str, model_name: str) -> list[str]:
    profile = approved_local_profile_for_ref(model_id) or approved_local_profile_for_ref(model_name)
    if not isinstance(profile, dict):
        return []
    return _normalize_caps(profile.get("capabilities") or [])


def _inferred_caps(model_id: str, model_name: str) -> list[str]:
    inferred = infer_capabilities_from_catalog(
        "ollama",
        {
            "id": model_id,
            "provider_id": "ollama",
            "model": model_name,
            "capabilities": ["embedding"] if is_embedding_model_name(model_name) else ["chat"],
        },
    )
    return capability_list_from_inference(inferred)


def _inventory_reason(*, health_status: str, failure_kind: str | None) -> str:
    if health_status == "ok":
        return "healthy"
    if str(failure_kind or "").strip():
        return str(failure_kind or "").strip()
    return health_status or "unknown"


def _runtime_inventory_rows(
    *,
    config: Config,
    registry: Registry,
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    configured_ids = _configured_model_ids(registry)
    approved_ids = _approved_model_ids(config)
    rows: list[dict[str, Any]] = []
    for model in snapshot.get("models") if isinstance(snapshot.get("models"), list) else []:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "").strip()
        provider_id = str(model.get("provider") or "").strip().lower()
        model_name = str(model.get("model") or "").strip()
        if not model_id or not provider_id or not model_name:
            continue
        local = bool(model.get("local", False))
        installed = bool(model.get("installed", False))
        health = model.get("health") if isinstance(model.get("health"), dict) else {}
        health_status = str(health.get("status") or "unknown").strip().lower() or "unknown"
        health_failure_kind = str(health.get("last_error_kind") or "").strip().lower() or None
        approved = bool(local)
        if not approved:
            approved = model_id in approved_ids if approved_ids else bool(registry.defaults.allow_remote_fallback)
        rows.append(
            {
                "id": model_id,
                "provider": provider_id,
                "installed": installed,
                "available": bool(model.get("available", False)),
                "healthy": health_status == "ok",
                "capabilities": list(model.get("capabilities") or []) if isinstance(model.get("capabilities"), list) else [],
                "size": _size_label(model_name),
                "context_window": model.get("max_context_tokens"),
                "local": local,
                "approved": approved,
                "reason": _inventory_reason(health_status=health_status, failure_kind=health_failure_kind),
                "quality_rank": int(model.get("quality_rank") or 0),
                "cost_rank": int(model.get("cost_rank") or 0),
                "price_in": model.get("input_cost_per_million_tokens"),
                "price_out": model.get("output_cost_per_million_tokens"),
                "health_status": health_status,
                "health_failure_kind": health_failure_kind,
                "health_reason": str(health.get("last_error_kind") or health_status or "unknown"),
                "model_name": model_name,
                "source": "registry",
                "configured": model_id in configured_ids,
                "capability_source": str(model.get("capability_source") or "runtime_snapshot"),
                "capability_provenance": list(model.get("capability_provenance") or []) if isinstance(model.get("capability_provenance"), list) else [],
                "runtime_known": True,
                "routable": bool(model.get("routable", False)),
            }
        )
    return rows


def _discovered_rows(
    *,
    config: Config,
    registry: Registry,
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    configured_ids = _configured_model_ids(registry)
    approved_ids = _approved_model_ids(config)
    installed_local_names = snapshot.get("installed_local_names")
    installed_names = (
        {
            str(item).strip()
            for item in installed_local_names
            if str(item).strip()
        }
        if isinstance(installed_local_names, set)
        else set()
    )
    existing_names = {
        str((row or {}).get("model") or "").strip()
        for row in (snapshot.get("models") if isinstance(snapshot.get("models"), list) else [])
        if isinstance(row, dict) and str((row or {}).get("provider") or "").strip().lower() == "ollama"
    }
    rows: list[dict[str, Any]] = []
    for model_name in sorted(installed_names):
        if model_name in existing_names:
            continue
        model_id = f"ollama:{model_name}"
        approved_caps = _approved_profile_caps(model_id, model_name)
        inferred_caps = _inferred_caps(model_id, model_name)
        capabilities = approved_caps or inferred_caps
        capability_source = "approved_profile" if approved_caps else "catalog_inference"
        provenance: list[dict[str, Any]] = []
        if approved_caps:
            provenance.append({"source": "approved_profile", "capabilities": approved_caps})
        if inferred_caps:
            provenance.append({"source": "catalog_inference", "capabilities": inferred_caps})
        rows.append(
            {
                "id": model_id,
                "provider": "ollama",
                "installed": True,
                "available": False,
                "healthy": False,
                "capabilities": capabilities,
                "size": _size_label(model_name),
                "context_window": None,
                "local": True,
                "approved": True if not approved_ids else model_id in approved_ids,
                "reason": "not_registered",
                "quality_rank": 0,
                "cost_rank": 0,
                "price_in": None,
                "price_out": None,
                "health_status": "unknown",
                "health_failure_kind": "not_registered",
                "health_reason": "discovered_local_model_not_registered_in_runtime",
                "model_name": model_name,
                "source": "ollama_list",
                "configured": model_id in configured_ids,
                "capability_source": capability_source,
                "capability_provenance": provenance,
                "runtime_known": False,
                "routable": False,
            }
        )
    return rows


def build_model_inventory(
    *,
    config: Config,
    registry: Registry | None = None,
    discovered_local_models: Iterable[str] | None = None,
    router_snapshot: dict[str, Any] | None = None,
    timeout_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    _ = timeout_seconds
    active_registry = registry or load_registry(config)
    snapshot = build_runtime_model_snapshot(
        config=config,
        registry=active_registry,
        router_snapshot=router_snapshot,
        discovered_local_models=discovered_local_models,
    )
    rows = _runtime_inventory_rows(
        config=config,
        registry=active_registry,
        snapshot=snapshot,
    )
    rows.extend(
        _discovered_rows(
            config=config,
            registry=active_registry,
            snapshot=snapshot,
        )
    )
    return normalize_model_inventory(rows)


__all__ = ["build_model_inventory"]
