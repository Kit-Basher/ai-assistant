from __future__ import annotations

import copy
import os
from typing import Any, Callable


def build_autoconfig_plan(
    registry_document: dict[str, Any],
    health_summary: dict[str, Any] | None = None,
    *,
    secret_lookup: Callable[[str], str | None] | None = None,
    disable_auth_failed_providers: bool = True,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    document = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
    models = document.get("models") if isinstance(document.get("models"), dict) else {}
    defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
    env_map = env if env is not None else dict(os.environ)
    secret_fn = secret_lookup or (lambda _name: None)
    health = _health_maps(health_summary)

    local_candidate = _best_model_candidate(
        providers,
        models,
        health,
        local_only=True,
        env_map=env_map,
        secret_lookup=secret_fn,
    )
    remote_candidate = _best_model_candidate(
        providers,
        models,
        health,
        local_only=False,
        env_map=env_map,
        secret_lookup=secret_fn,
    )

    changes: list[dict[str, Any]] = []
    reasons: list[str] = []
    proposed_defaults = {
        "routing_mode": str(defaults.get("routing_mode") or "auto").strip().lower() or "auto",
        "default_provider": str(defaults.get("default_provider") or "").strip().lower() or None,
        "default_model": str(defaults.get("default_model") or "").strip() or None,
        "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
    }
    if not bool(proposed_defaults["allow_remote_fallback"]):
        remote_candidate = None

    canonical_default_model = _canonical_default_model_id(
        proposed_defaults["default_provider"],
        proposed_defaults["default_model"],
        models,
    )
    if canonical_default_model and proposed_defaults["default_model"] != canonical_default_model:
        changes.append(
            {
                "kind": "defaults",
                "field": "default_model",
                "before": proposed_defaults["default_model"],
                "after": canonical_default_model,
                "reason": "fully_qualified_default_model",
            }
        )
        proposed_defaults["default_model"] = canonical_default_model

    current_candidate = _candidate_from_model_id(
        proposed_defaults["default_model"],
        providers,
        models,
        health,
        env_map=env_map,
        secret_lookup=secret_fn,
        allow_remote_fallback=bool(proposed_defaults["allow_remote_fallback"]),
    )
    current_is_healthy = bool(
        current_candidate and str(current_candidate.get("health_status") or "unknown").strip().lower() in {"ok", "unknown"}
    )

    selected = current_candidate if current_is_healthy else (local_candidate or remote_candidate)
    if selected is not None:
        selected_provider = str(selected["provider_id"])
        selected_model = str(selected["model_id"])
        selected_is_local = bool(selected["is_local"])
        desired_routing_mode = "prefer_local_lowest_cost_capable" if selected_is_local else "prefer_best"
        desired_allow_remote = bool(proposed_defaults["allow_remote_fallback"])
        if not selected_is_local:
            desired_allow_remote = True

        if current_is_healthy and current_candidate is not None:
            reasons.append("keep_current_default_model")
        else:
            if proposed_defaults["routing_mode"] != desired_routing_mode:
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "routing_mode",
                        "before": proposed_defaults["routing_mode"],
                        "after": desired_routing_mode,
                        "reason": "selected_best_available_candidate",
                    }
                )
                proposed_defaults["routing_mode"] = desired_routing_mode
            if proposed_defaults["default_provider"] != selected_provider:
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "default_provider",
                        "before": proposed_defaults["default_provider"],
                        "after": selected_provider,
                        "reason": "selected_best_available_candidate",
                    }
                )
                proposed_defaults["default_provider"] = selected_provider
            if proposed_defaults["default_model"] != selected_model:
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "default_model",
                        "before": proposed_defaults["default_model"],
                        "after": selected_model,
                        "reason": "selected_best_available_candidate",
                    }
                )
                proposed_defaults["default_model"] = selected_model
            if bool(proposed_defaults["allow_remote_fallback"]) != bool(desired_allow_remote):
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "allow_remote_fallback",
                        "before": bool(proposed_defaults["allow_remote_fallback"]),
                        "after": bool(desired_allow_remote),
                        "reason": "selected_best_available_candidate",
                    }
                )
                proposed_defaults["allow_remote_fallback"] = bool(desired_allow_remote)

        reasons.append(
            "selected {scope} model {model_id} (provider={provider_id}, health={health_status})".format(
                scope="local" if selected_is_local else "remote",
                model_id=selected_model,
                provider_id=selected_provider,
                health_status=str(selected.get("health_status") or "unknown"),
            )
        )
    else:
        reasons.append("no_working_candidate_found")

    providers_to_disable: list[str] = []
    if disable_auth_failed_providers:
        for provider_id, provider_payload in sorted(providers.items()):
            if not isinstance(provider_payload, dict):
                continue
            if not bool(provider_payload.get("enabled", True)):
                continue
            provider_health = health["providers"].get(str(provider_id)) if isinstance(health["providers"], dict) else None
            if not isinstance(provider_health, dict):
                continue
            status = str(provider_health.get("status") or "unknown").strip().lower()
            last_error = str(provider_health.get("last_error_kind") or "").strip().lower()
            status_code = provider_health.get("status_code")
            auth_failed = last_error == "auth_error" or int(status_code or 0) in {401, 403}
            if status == "down" and auth_failed:
                providers_to_disable.append(str(provider_id))

    for provider_id in sorted(set(providers_to_disable)):
        before_enabled = bool((providers.get(provider_id) or {}).get("enabled", True))
        if not before_enabled:
            continue
        changes.append(
            {
                "kind": "provider",
                "id": provider_id,
                "field": "enabled",
                "before": True,
                "after": False,
                "reason": "auth_hard_failure",
            }
        )
        reasons.append(f"disable provider {provider_id} due to auth_hard_failure")

    changes.sort(key=_change_sort_key)
    reasons = sorted(set(reasons))
    return {
        "ok": True,
        "changes": changes,
        "reasons": reasons,
        "selected_candidate": selected,
        "proposed_defaults": proposed_defaults,
        "impact": {
            "changes_count": len(changes),
            "providers_to_disable": sorted(set(providers_to_disable)),
            "selected_model": (selected or {}).get("model_id"),
        },
    }


def apply_autoconfig_plan(registry_document: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    providers = updated.get("providers") if isinstance(updated.get("providers"), dict) else {}
    defaults = updated.get("defaults") if isinstance(updated.get("defaults"), dict) else {}
    changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []

    for row in sorted((item for item in changes if isinstance(item, dict)), key=_change_sort_key):
        kind = str(row.get("kind") or "").strip().lower()
        field = str(row.get("field") or "").strip()
        if kind == "defaults":
            defaults[field] = row.get("after")
            continue
        if kind == "provider":
            provider_id = str(row.get("id") or "").strip().lower()
            if not provider_id or provider_id not in providers or not isinstance(providers.get(provider_id), dict):
                continue
            providers[provider_id] = {
                **providers[provider_id],
                field: row.get("after"),
            }

    updated["providers"] = providers
    updated["defaults"] = defaults
    return updated


def _health_maps(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    payload = summary if isinstance(summary, dict) else {}
    provider_rows = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    model_rows = payload.get("models") if isinstance(payload.get("models"), list) else []
    providers = {
        str(row.get("id") or "").strip().lower(): row
        for row in provider_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    models = {
        str(row.get("id") or "").strip(): row
        for row in model_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    return {"providers": providers, "models": models}


def _provider_has_auth(
    provider_payload: dict[str, Any],
    *,
    env_map: dict[str, str],
    secret_lookup: Callable[[str], str | None],
) -> bool:
    source = provider_payload.get("api_key_source") if isinstance(provider_payload.get("api_key_source"), dict) else None
    if not source:
        # local providers generally do not require keys.
        return bool(provider_payload.get("local", False))
    source_type = str(source.get("type") or "").strip().lower()
    source_name = str(source.get("name") or "").strip()
    if source_type == "env":
        return bool(env_map.get(source_name, "").strip())
    if source_type == "secret":
        return bool(str(secret_lookup(source_name) or "").strip())
    return False


def _best_model_candidate(
    providers: dict[str, Any],
    models: dict[str, Any],
    health: dict[str, dict[str, Any]],
    *,
    local_only: bool,
    env_map: dict[str, str],
    secret_lookup: Callable[[str], str | None],
) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, int, str], dict[str, Any]]] = []
    for model_id, model_payload in sorted(models.items()):
        if not isinstance(model_payload, dict):
            continue
        provider_id = str(model_payload.get("provider") or "").strip().lower()
        provider_payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else None
        if provider_payload is None:
            continue
        is_local = bool(provider_payload.get("local", False))
        if local_only and not is_local:
            continue
        if not local_only and is_local:
            continue
        if not bool(provider_payload.get("enabled", True)):
            continue
        if not bool(model_payload.get("enabled", True)) or not bool(model_payload.get("available", True)):
            continue
        capabilities = {
            str(item).strip().lower()
            for item in (model_payload.get("capabilities") or [])
            if str(item).strip()
        }
        if "chat" not in capabilities:
            continue
        if not is_local and not _provider_has_auth(provider_payload, env_map=env_map, secret_lookup=secret_lookup):
            continue

        model_health = health["models"].get(str(model_id)) if isinstance(health.get("models"), dict) else None
        provider_health = health["providers"].get(provider_id) if isinstance(health.get("providers"), dict) else None
        health_status = _effective_health_status(model_health, provider_health)
        if health_status == "down":
            continue
        health_rank = {"ok": 0, "degraded": 1, "unknown": 2}.get(health_status, 2)
        quality_rank = -int(model_payload.get("quality_rank") or 0)
        cost_rank = int(model_payload.get("cost_rank") or 0)

        candidates.append(
            (
                (health_rank, quality_rank, cost_rank, str(model_id)),
                {
                    "provider_id": provider_id,
                    "model_id": str(model_id),
                    "is_local": is_local,
                    "health_status": health_status,
                    "quality_rank": int(model_payload.get("quality_rank") or 0),
                    "cost_rank": int(model_payload.get("cost_rank") or 0),
                },
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _canonical_default_model_id(
    default_provider: str | None,
    default_model: str | None,
    models: dict[str, Any],
) -> str | None:
    provider_id = str(default_provider or "").strip().lower()
    model_value = str(default_model or "").strip()
    if not model_value:
        return None
    if model_value in models:
        return model_value
    if ":" in model_value:
        return None
    if not provider_id:
        return None

    scoped = f"{provider_id}:{model_value}"
    if scoped in models:
        return scoped
    matches = [
        model_id
        for model_id, payload in sorted(models.items())
        if isinstance(payload, dict)
        and str(payload.get("provider") or "").strip().lower() == provider_id
        and str(payload.get("model") or "").strip() == model_value
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _candidate_from_model_id(
    model_id: str | None,
    providers: dict[str, Any],
    models: dict[str, Any],
    health: dict[str, dict[str, Any]],
    *,
    env_map: dict[str, str],
    secret_lookup: Callable[[str], str | None],
    allow_remote_fallback: bool,
) -> dict[str, Any] | None:
    target_model_id = str(model_id or "").strip()
    if not target_model_id:
        return None
    model_payload = models.get(target_model_id) if isinstance(models.get(target_model_id), dict) else None
    if model_payload is None:
        return None

    provider_id = str(model_payload.get("provider") or "").strip().lower()
    provider_payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else None
    if provider_payload is None:
        return None
    if not bool(provider_payload.get("enabled", True)):
        return None
    if not bool(model_payload.get("enabled", True)) or not bool(model_payload.get("available", True)):
        return None

    capabilities = {
        str(item).strip().lower()
        for item in (model_payload.get("capabilities") or [])
        if str(item).strip()
    }
    if "chat" not in capabilities:
        return None

    is_local = bool(provider_payload.get("local", False))
    if not is_local and not allow_remote_fallback:
        return None
    if not is_local and not _provider_has_auth(provider_payload, env_map=env_map, secret_lookup=secret_lookup):
        return None

    model_health = health["models"].get(target_model_id) if isinstance(health.get("models"), dict) else None
    provider_health = health["providers"].get(provider_id) if isinstance(health.get("providers"), dict) else None
    health_status = _effective_health_status(model_health, provider_health)
    if health_status == "down":
        return None
    return {
        "provider_id": provider_id,
        "model_id": target_model_id,
        "is_local": is_local,
        "health_status": health_status,
        "quality_rank": int(model_payload.get("quality_rank") or 0),
        "cost_rank": int(model_payload.get("cost_rank") or 0),
    }


def _effective_health_status(model_health: dict[str, Any] | None, provider_health: dict[str, Any] | None) -> str:
    model_status = str((model_health or {}).get("status") or "unknown").strip().lower()
    provider_status = str((provider_health or {}).get("status") or "unknown").strip().lower()
    if model_status == "down" or provider_status == "down":
        return "down"
    if model_status == "degraded" or provider_status == "degraded":
        return "degraded"
    if model_status == "ok" or provider_status == "ok":
        return "ok"
    return "unknown"


def _change_sort_key(change: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(change.get("kind") or ""),
        str(change.get("id") or ""),
        str(change.get("field") or ""),
    )
