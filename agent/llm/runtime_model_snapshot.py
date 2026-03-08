from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

from agent.config import Config
from agent.llm.approved_local_models import approved_local_profile_for_ref
from agent.llm.capabilities import capability_list_from_inference, infer_capabilities_from_catalog, is_embedding_model_name
from agent.llm.registry import Registry, load_registry
from agent.llm.router import LLMRouter
from agent.modelops.discovery import list_models_ollama


_HEALTH_STATUSES = {"ok", "degraded", "down", "unknown", "not_applicable"}


def _health_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _HEALTH_STATUSES else "unknown"


def _health_epoch(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _health_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _health_iso(epoch: int | None) -> str | None:
    if epoch is None or int(epoch) <= 0:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(int(epoch)))
    except (OSError, OverflowError, ValueError):
        return None


def normalize_health_record(
    raw: Mapping[str, Any] | None,
    *,
    status_hint: str | None = None,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    now_value = int(now_epoch) if isinstance(now_epoch, int) and now_epoch > 0 else int(time.time())
    status = _health_status(payload.get("status") if payload else status_hint)
    last_checked_at = _health_epoch(payload.get("last_checked_at"))
    last_ts = _health_float(payload.get("last_ts"))
    if last_checked_at is None:
        inferred_checked = _health_epoch(last_ts)
        if inferred_checked is not None:
            last_checked_at = inferred_checked
    if status in {"ok", "degraded", "down"} and last_checked_at is None:
        last_checked_at = now_value
    if last_ts is None and last_checked_at is not None:
        last_ts = float(last_checked_at)
    status_code = _health_epoch(payload.get("status_code"))
    last_status_code = _health_epoch(payload.get("last_status_code"))
    if last_status_code is None:
        last_status_code = status_code
    cooldown_until = _health_epoch(payload.get("cooldown_until"))
    down_since = _health_epoch(payload.get("down_since"))
    if status == "down" and down_since is None and last_checked_at is not None:
        down_since = last_checked_at
    last_error_kind = str(payload.get("last_error_kind") or "").strip().lower() or None
    try:
        successes = max(0, int(payload.get("successes") or 0))
    except (TypeError, ValueError):
        successes = 0
    try:
        failures = max(0, int(payload.get("failures") or 0))
    except (TypeError, ValueError):
        failures = 0
    try:
        failure_streak = max(0, int(payload.get("failure_streak") or 0))
    except (TypeError, ValueError):
        failure_streak = 0
    if status == "ok":
        last_error_kind = None
        status_code = None
        last_status_code = None
        cooldown_until = None
        down_since = None
        failure_streak = 0
    return {
        "status": status,
        "last_checked_at": last_checked_at,
        "last_checked_at_iso": _health_iso(last_checked_at),
        "last_error_kind": last_error_kind,
        "status_code": status_code,
        "last_status_code": last_status_code,
        "cooldown_until": cooldown_until,
        "cooldown_until_iso": _health_iso(cooldown_until),
        "down_since": down_since,
        "down_since_iso": _health_iso(down_since),
        "successes": successes,
        "failures": failures,
        "failure_streak": failure_streak,
        "last_ts": last_ts,
        "last_ts_iso": _health_iso(_health_epoch(last_ts)),
    }


def _health_has_runtime_evidence(raw: Mapping[str, Any] | None) -> bool:
    payload = raw if isinstance(raw, Mapping) else {}
    if str(payload.get("last_error_kind") or "").strip():
        return True
    for key in ("status_code", "last_status_code", "last_checked_at", "last_ts", "cooldown_until", "down_since"):
        if payload.get(key) is not None:
            return True
    for key in ("successes", "failures", "failure_streak"):
        try:
            if int(payload.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _normalize_capabilities(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(item).strip().lower()
            for item in values
            if str(item).strip()
        }
    )


def _approved_profile_capabilities(model_id: str, model_name: str) -> list[str]:
    profile = approved_local_profile_for_ref(model_id) or approved_local_profile_for_ref(model_name)
    if not isinstance(profile, Mapping):
        return []
    return _normalize_capabilities(profile.get("capabilities") or [])


def _inferred_capabilities_for_model(model_id: str, provider_id: str, model_name: str) -> list[str]:
    inferred = infer_capabilities_from_catalog(
        provider_id,
        {
            "id": model_id,
            "provider_id": provider_id,
            "model": model_name,
            "capabilities": ["embedding"] if is_embedding_model_name(model_name) else ["chat"],
        },
    )
    return capability_list_from_inference(inferred)


def _capability_payload(
    *,
    model_id: str,
    provider_id: str,
    model_name: str,
    runtime_capabilities: Iterable[Any],
    runtime_known: bool,
) -> tuple[list[str], str, list[dict[str, Any]]]:
    runtime_caps = _normalize_capabilities(runtime_capabilities)
    approved_caps = _approved_profile_capabilities(model_id, model_name)
    inferred_caps = [] if runtime_known else _inferred_capabilities_for_model(model_id, provider_id, model_name)
    provenance: list[dict[str, Any]] = []
    if runtime_caps:
        provenance.append({"source": "runtime_snapshot", "capabilities": runtime_caps})
    if approved_caps:
        provenance.append({"source": "approved_profile", "capabilities": approved_caps})
    if inferred_caps:
        provenance.append({"source": "catalog_inference", "capabilities": inferred_caps})
    if runtime_known:
        return runtime_caps, "runtime_snapshot", provenance
    if approved_caps:
        return approved_caps, "approved_profile", provenance
    return inferred_caps, "catalog_inference", provenance


def discover_local_model_names(discovered_local_models: Iterable[str] | None = None) -> set[str]:
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


def _build_router_snapshot(*, config: Config, registry: Registry) -> dict[str, Any]:
    router = LLMRouter(config, registry=registry)
    return router.doctor_snapshot()


def build_runtime_model_snapshot(
    *,
    config: Config,
    registry: Registry | None = None,
    router_snapshot: Mapping[str, Any] | None = None,
    discovered_local_models: Iterable[str] | None = None,
) -> dict[str, Any]:
    active_registry = registry or load_registry(config)
    snapshot = (
        dict(router_snapshot)
        if isinstance(router_snapshot, Mapping)
        else _build_router_snapshot(config=config, registry=active_registry)
    )
    now_epoch = int(time.time())
    installed_local_names = discover_local_model_names(discovered_local_models)

    providers_raw = snapshot.get("providers") if isinstance(snapshot.get("providers"), list) else []
    providers: list[dict[str, Any]] = []
    provider_lookup: dict[str, dict[str, Any]] = {}
    disabled_provider_ids: set[str] = set()
    disabled_provider_health: dict[str, dict[str, Any]] = {}
    for row in providers_raw:
        if not isinstance(row, Mapping):
            continue
        provider_id = str(row.get("id") or "").strip().lower()
        if not provider_id:
            continue
        local = bool(row.get("local", False))
        health_raw = dict(row.get("health") or {}) if isinstance(row.get("health"), Mapping) else {}
        provider_is_disabled = (not bool(row.get("enabled", True))) or (
            str(health_raw.get("last_error_kind") or "").strip().lower() == "provider_disabled"
        )
        if provider_is_disabled:
            health_raw = {
                **health_raw,
                "status": "down",
                "last_error_kind": "provider_disabled",
            }
        if not local and _health_status(health_raw.get("status")) == "ok" and not _health_has_runtime_evidence(health_raw):
            health_raw["status"] = "unknown"
        normalized_provider_health = normalize_health_record(health_raw, now_epoch=now_epoch)
        provider_row = {
            **dict(row),
            "id": provider_id,
            "local": local,
            "health": normalized_provider_health,
        }
        providers.append(provider_row)
        provider_lookup[provider_id] = provider_row
        if provider_is_disabled:
            disabled_provider_ids.add(provider_id)
            disabled_provider_health[provider_id] = normalized_provider_health

    models_raw = snapshot.get("models") if isinstance(snapshot.get("models"), list) else []
    models: list[dict[str, Any]] = []
    model_lookup: dict[str, dict[str, Any]] = {}
    for row in models_raw:
        if not isinstance(row, Mapping):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        provider_id = str(row.get("provider") or "").strip().lower()
        registry_model = active_registry.models.get(model_id)
        if registry_model is not None and not provider_id:
            provider_id = str(registry_model.provider or "").strip().lower()
        provider_row = provider_lookup.get(provider_id)
        local = bool(provider_row.get("local", False)) if isinstance(provider_row, dict) else provider_id == "ollama"
        model_name = (
            str(row.get("model") or "").strip()
            or (str(registry_model.model).strip() if registry_model is not None else "")
            or (model_id.split(":", 1)[1] if ":" in model_id else model_id)
        )
        health_raw = dict(row.get("health") or {}) if isinstance(row.get("health"), Mapping) else {}
        raw_status = _health_status(health_raw.get("status"))
        explicit_local_install_evidence = local and (
            bool(row.get("routable", False))
            or bool(row.get("available", False))
            or raw_status == "ok"
            or (
                _health_has_runtime_evidence(health_raw)
                and str(health_raw.get("last_error_kind") or "").strip().lower()
                not in {"not_installed", "model_not_installed"}
            )
        )
        installed = bool(not local or model_name in installed_local_names or explicit_local_install_evidence)
        provider_health = (
            provider_row.get("health")
            if isinstance(provider_row, dict) and isinstance(provider_row.get("health"), Mapping)
            else {}
        )
        provider_status = _health_status(provider_health.get("status"))
        if provider_id in disabled_provider_ids:
            disabled_health = disabled_provider_health.get(provider_id) or {}
            health_raw = {
                **health_raw,
                "status": "down",
                "last_error_kind": "provider_disabled",
                "status_code": disabled_health.get("status_code"),
                "last_status_code": disabled_health.get("last_status_code"),
            }
        elif local and not installed:
            health_raw = {"status": "down", "last_error_kind": "not_installed"}
        elif not local and raw_status == "ok" and not _health_has_runtime_evidence(health_raw):
            derived_status = provider_status if provider_status in {"degraded", "down", "unknown", "not_applicable"} else "unknown"
            health_raw = {
                **health_raw,
                "status": derived_status,
                "last_error_kind": (
                    str(provider_health.get("last_error_kind") or "").strip().lower() or None
                    if derived_status in {"degraded", "down"}
                    else None
                ),
            }
        capabilities, capability_source, capability_provenance = _capability_payload(
            model_id=model_id,
            provider_id=provider_id,
            model_name=model_name,
            runtime_capabilities=row.get("capabilities") or [],
            runtime_known=True,
        )
        available = bool(row.get("available", False))
        if local:
            available = bool(available and installed)
        normalized_health = normalize_health_record(health_raw, now_epoch=now_epoch)
        routable = bool(row.get("routable", False)) and bool(available)
        if provider_status != "ok" or str(normalized_health.get("status") or "").strip().lower() != "ok":
            routable = False
        model_row = {
            **dict(row),
            "id": model_id,
            "provider": provider_id,
            "model": model_name,
            "local": local,
            "installed": installed,
            "available": available,
            "routable": routable,
            "capabilities": capabilities,
            "capability_source": capability_source,
            "capability_provenance": capability_provenance,
            "runtime_known": True,
            "quality_rank": int(registry_model.quality_rank) if registry_model is not None else 0,
            "cost_rank": int(registry_model.cost_rank) if registry_model is not None else 0,
            "default_for": list(registry_model.default_for) if registry_model is not None else [],
            "health": normalized_health,
        }
        models.append(model_row)
        model_lookup[model_id] = model_row

    defaults = snapshot.get("defaults") if isinstance(snapshot.get("defaults"), Mapping) else {}
    default_provider = (
        str(defaults.get("default_provider") or "").strip().lower()
        or str(active_registry.defaults.default_provider or "").strip().lower()
        or None
    )
    default_model = (
        str(defaults.get("default_model") or "").strip()
        or str(active_registry.defaults.chat_model or "").strip()
        or str(active_registry.defaults.default_model or "").strip()
        or None
    )
    return {
        "default_provider": default_provider,
        "default_model": default_model,
        "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", active_registry.defaults.allow_remote_fallback)),
        "providers": providers,
        "models": models,
        "provider_lookup": provider_lookup,
        "model_lookup": model_lookup,
        "installed_local_names": installed_local_names,
    }


__all__ = [
    "build_runtime_model_snapshot",
    "discover_local_model_names",
    "normalize_health_record",
]
