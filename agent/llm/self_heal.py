from __future__ import annotations

import copy
from typing import Any


_CHAT_REQUIRED_ROUTING_MODES = {
    "auto",
    "prefer_cheap",
    "prefer_best",
    "prefer_local_lowest_cost_capable",
}


def build_drift_report(
    registry_document: dict[str, Any],
    health_summary: dict[str, Any] | None = None,
    *,
    router_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document = registry_document if isinstance(registry_document, dict) else {}
    providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
    models = document.get("models") if isinstance(document.get("models"), dict) else {}
    defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
    health_payload = health_summary if isinstance(health_summary, dict) else {}
    snapshot = router_snapshot if isinstance(router_snapshot, dict) else {}

    provider_rows = snapshot.get("providers") if isinstance(snapshot.get("providers"), list) else []
    model_rows = snapshot.get("models") if isinstance(snapshot.get("models"), list) else []
    provider_lookup = {
        str(row.get("id") or "").strip().lower(): row
        for row in provider_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    model_lookup = {
        str(row.get("id") or "").strip(): row
        for row in model_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }

    provider_health_rows = health_payload.get("providers") if isinstance(health_payload.get("providers"), list) else []
    model_health_rows = health_payload.get("models") if isinstance(health_payload.get("models"), list) else []
    provider_health_lookup = {
        str(row.get("id") or "").strip().lower(): row
        for row in provider_health_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    model_health_lookup = {
        str(row.get("id") or "").strip(): row
        for row in model_health_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }

    routing_mode = str(defaults.get("routing_mode") or "auto").strip().lower() or "auto"
    default_provider = str(defaults.get("default_provider") or "").strip().lower()
    raw_default_model = str(defaults.get("default_model") or "").strip()
    resolved_default_model = _resolve_default_model_id(raw_default_model, default_provider, models)

    provider_payload = (
        providers.get(default_provider)
        if default_provider and isinstance(providers.get(default_provider), dict)
        else None
    )
    provider_row = provider_lookup.get(default_provider) if default_provider else None
    provider_health_row = provider_health_lookup.get(default_provider) if default_provider else None

    provider_exists = provider_payload is not None
    provider_enabled = bool((provider_payload or {}).get("enabled", False))
    provider_available = _provider_available(provider_row, provider_health_row, provider_payload)
    provider_health_status = _provider_status(provider_row, provider_health_row)

    model_payload = (
        models.get(resolved_default_model)
        if resolved_default_model and isinstance(models.get(resolved_default_model), dict)
        else None
    )
    model_row = model_lookup.get(resolved_default_model) if resolved_default_model else None
    model_health_row = model_health_lookup.get(resolved_default_model) if resolved_default_model else None

    model_exists = model_payload is not None
    model_provider = str((model_payload or {}).get("provider") or "").strip().lower()
    model_enabled = bool((model_payload or {}).get("enabled", False))
    model_available = bool((model_payload or {}).get("available", False))
    model_has_chat = _model_has_chat_capability(model_payload)
    model_routable = _model_routable(model_row, model_payload, provider_available)
    model_health_status = _model_status(model_row, model_health_row)
    requires_chat = routing_mode in _CHAT_REQUIRED_ROUTING_MODES

    reasons: set[str] = set()
    if not default_provider:
        reasons.add("default_provider_missing")
    elif not provider_exists:
        reasons.add("default_provider_missing")
    elif not provider_enabled:
        reasons.add("default_provider_disabled")
    elif provider_available is False:
        reasons.add("default_provider_unavailable")

    if not raw_default_model:
        reasons.add("default_model_missing")
    elif not model_exists:
        reasons.add("default_model_missing")
    else:
        if default_provider and model_provider and model_provider != default_provider:
            reasons.add("default_model_provider_mismatch")
        if not model_enabled:
            reasons.add("default_model_disabled")
        if not model_available:
            reasons.add("default_model_not_in_provider_inventory")
        if requires_chat and not model_has_chat:
            reasons.add("default_model_not_chat_capable")
        if model_health_status in {"degraded", "down"}:
            reasons.add("default_model_health_not_ok")
        if not model_routable:
            reasons.add("default_model_unroutable")

    details = {
        "routing_mode": routing_mode,
        "default_provider": default_provider or None,
        "default_model": raw_default_model or None,
        "resolved_default_model": resolved_default_model or None,
        "requires_chat": bool(requires_chat),
        "provider_exists": bool(provider_exists),
        "provider_enabled": bool(provider_enabled),
        "provider_available": provider_available,
        "provider_health_status": provider_health_status,
        "model_exists": bool(model_exists),
        "model_provider": model_provider or None,
        "model_enabled": bool(model_enabled),
        "model_available": bool(model_available),
        "model_has_chat_capability": bool(model_has_chat),
        "model_routable": bool(model_routable),
        "model_health_status": model_health_status,
    }
    reason_rows = sorted(reasons)
    return {
        "has_drift": bool(reason_rows),
        "reasons": reason_rows,
        "details": details,
    }


def build_self_heal_plan(
    registry_document: dict[str, Any],
    health_summary: dict[str, Any] | None = None,
    *,
    router_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document = registry_document if isinstance(registry_document, dict) else {}
    defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
    proposed_defaults = {
        "routing_mode": str(defaults.get("routing_mode") or "auto").strip().lower() or "auto",
        "default_provider": str(defaults.get("default_provider") or "").strip().lower() or None,
        "default_model": str(defaults.get("default_model") or "").strip() or None,
        "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
    }
    drift = build_drift_report(document, health_summary, router_snapshot=router_snapshot)
    reasons = drift.get("reasons") if isinstance(drift.get("reasons"), list) else []
    drift_reasons = [str(item).strip() for item in reasons if str(item).strip()]

    if not bool(drift.get("has_drift")):
        return {
            "ok": True,
            "drift": drift,
            "changes": [],
            "reasons": ["no_drift"],
            "rationale": "defaults_healthy_and_routable",
            "selected_candidate": None,
            "proposed_defaults": proposed_defaults,
            "impact": {
                "changes_count": 0,
                "selected_model": None,
                "selected_provider": None,
            },
        }

    selected = _select_replacement_candidate(
        document,
        health_summary or {},
        router_snapshot=router_snapshot or {},
        default_provider=str(proposed_defaults.get("default_provider") or "").strip().lower(),
        allow_remote_fallback=bool(proposed_defaults.get("allow_remote_fallback", True)),
    )
    if selected is None:
        joined_drift = ",".join(drift_reasons) or "unknown_drift"
        return {
            "ok": True,
            "drift": drift,
            "changes": [],
            "reasons": [f"drift_detected({joined_drift}); no_healthy_replacement"],
            "rationale": "no_healthy_replacement",
            "selected_candidate": None,
            "proposed_defaults": proposed_defaults,
            "impact": {
                "changes_count": 0,
                "selected_model": None,
                "selected_provider": None,
            },
        }

    changes: list[dict[str, Any]] = []
    selected_provider = str(selected.get("provider_id") or "").strip().lower() or None
    selected_model = str(selected.get("model_id") or "").strip() or None
    if proposed_defaults.get("default_provider") != selected_provider:
        changes.append(
            {
                "kind": "defaults",
                "field": "default_provider",
                "before": proposed_defaults.get("default_provider"),
                "after": selected_provider,
                "reason": "self_heal_drift_repair",
            }
        )
        proposed_defaults["default_provider"] = selected_provider
    if proposed_defaults.get("default_model") != selected_model:
        changes.append(
            {
                "kind": "defaults",
                "field": "default_model",
                "before": proposed_defaults.get("default_model"),
                "after": selected_model,
                "reason": "self_heal_drift_repair",
            }
        )
        proposed_defaults["default_model"] = selected_model

    changes.sort(key=_change_sort_key)
    rationale = str(selected.get("rationale") or "").strip() or "selected_healthy_replacement"
    joined_drift = ",".join(drift_reasons) or "unknown_drift"
    combined_reason = f"drift_repair({joined_drift}); {rationale}"
    return {
        "ok": True,
        "drift": drift,
        "changes": changes,
        "reasons": [combined_reason],
        "rationale": rationale,
        "selected_candidate": selected,
        "proposed_defaults": proposed_defaults,
        "impact": {
            "changes_count": len(changes),
            "selected_model": selected_model,
            "selected_provider": selected_provider,
        },
    }


def apply_self_heal_plan(registry_document: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    defaults = updated.get("defaults") if isinstance(updated.get("defaults"), dict) else {}
    changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []

    for row in sorted((item for item in changes if isinstance(item, dict)), key=_change_sort_key):
        if str(row.get("kind") or "").strip().lower() != "defaults":
            continue
        field = str(row.get("field") or "").strip()
        if not field:
            continue
        defaults[field] = row.get("after")
    updated["defaults"] = defaults
    return updated


def _resolve_default_model_id(
    default_model: str,
    default_provider: str,
    models: dict[str, Any],
) -> str:
    candidate = str(default_model or "").strip()
    if not candidate:
        return ""
    if candidate in models:
        return candidate
    if default_provider:
        scoped = f"{default_provider}:{candidate}"
        if scoped in models:
            return scoped
    return candidate


def _provider_status(provider_row: dict[str, Any] | None, provider_health_row: dict[str, Any] | None) -> str:
    if isinstance(provider_row, dict):
        health = provider_row.get("health") if isinstance(provider_row.get("health"), dict) else {}
        status = str(health.get("status") or "").strip().lower()
        if status in {"ok", "degraded", "down", "unknown"}:
            return status
    if isinstance(provider_health_row, dict):
        status = str(provider_health_row.get("status") or "").strip().lower()
        if status in {"ok", "degraded", "down", "unknown"}:
            return status
    return "unknown"


def _provider_available(
    provider_row: dict[str, Any] | None,
    provider_health_row: dict[str, Any] | None,
    provider_payload: dict[str, Any] | None,
) -> bool | None:
    if isinstance(provider_row, dict) and "available" in provider_row:
        return bool(provider_row.get("available"))
    status = _provider_status(provider_row, provider_health_row)
    if status == "down":
        return False
    if status in {"ok", "degraded"}:
        return True
    if isinstance(provider_payload, dict):
        return bool(provider_payload.get("enabled", False))
    return None


def _model_status(model_row: dict[str, Any] | None, model_health_row: dict[str, Any] | None) -> str:
    if isinstance(model_row, dict):
        health = model_row.get("health") if isinstance(model_row.get("health"), dict) else {}
        status = str(health.get("status") or "").strip().lower()
        if status in {"ok", "degraded", "down", "unknown"}:
            return status
    if isinstance(model_health_row, dict):
        status = str(model_health_row.get("status") or "").strip().lower()
        if status in {"ok", "degraded", "down", "unknown"}:
            return status
    return "unknown"


def _model_routable(
    model_row: dict[str, Any] | None,
    model_payload: dict[str, Any] | None,
    provider_available: bool | None,
) -> bool:
    if isinstance(model_row, dict):
        return bool(model_row.get("routable", False))
    payload = model_payload if isinstance(model_payload, dict) else {}
    provider_ok = True if provider_available is None else bool(provider_available)
    return bool(payload.get("enabled", False)) and bool(payload.get("available", False)) and provider_ok


def _model_has_chat_capability(model_payload: dict[str, Any] | None) -> bool:
    payload = model_payload if isinstance(model_payload, dict) else {}
    capabilities = {
        str(item).strip().lower()
        for item in (payload.get("capabilities") or [])
        if str(item).strip()
    }
    return "chat" in capabilities


def _observed_cost(model_payload: dict[str, Any], model_row: dict[str, Any] | None) -> float | None:
    pricing = model_payload.get("pricing") if isinstance(model_payload.get("pricing"), dict) else {}
    if pricing:
        input_cost = pricing.get("input_per_million_tokens")
        output_cost = pricing.get("output_per_million_tokens")
        if _is_number(input_cost) and _is_number(output_cost):
            return float(input_cost) + float(output_cost)
    if isinstance(model_row, dict):
        input_cost = model_row.get("input_cost_per_million_tokens")
        output_cost = model_row.get("output_cost_per_million_tokens")
        if _is_number(input_cost) and _is_number(output_cost):
            return float(input_cost) + float(output_cost)
    return None


def _select_replacement_candidate(
    registry_document: dict[str, Any],
    health_summary: dict[str, Any],
    *,
    router_snapshot: dict[str, Any],
    default_provider: str,
    allow_remote_fallback: bool,
) -> dict[str, Any] | None:
    document = registry_document if isinstance(registry_document, dict) else {}
    providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
    models = document.get("models") if isinstance(document.get("models"), dict) else {}
    health_payload = health_summary if isinstance(health_summary, dict) else {}
    snapshot = router_snapshot if isinstance(router_snapshot, dict) else {}

    provider_rows = snapshot.get("providers") if isinstance(snapshot.get("providers"), list) else []
    model_rows = snapshot.get("models") if isinstance(snapshot.get("models"), list) else []
    provider_lookup = {
        str(row.get("id") or "").strip().lower(): row
        for row in provider_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    model_lookup = {
        str(row.get("id") or "").strip(): row
        for row in model_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    provider_health_rows = health_payload.get("providers") if isinstance(health_payload.get("providers"), list) else []
    model_health_rows = health_payload.get("models") if isinstance(health_payload.get("models"), list) else []
    provider_health_lookup = {
        str(row.get("id") or "").strip().lower(): row
        for row in provider_health_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    model_health_lookup = {
        str(row.get("id") or "").strip(): row
        for row in model_health_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }

    candidates: list[dict[str, Any]] = []
    for model_id in sorted(models.keys()):
        model_payload = models.get(model_id)
        if not isinstance(model_payload, dict):
            continue
        provider_id = str(model_payload.get("provider") or "").strip().lower()
        provider_payload = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else None
        if provider_payload is None:
            continue
        provider_row = provider_lookup.get(provider_id)
        provider_health_row = provider_health_lookup.get(provider_id)
        model_row = model_lookup.get(model_id)
        model_health_row = model_health_lookup.get(model_id)

        is_local = bool(provider_payload.get("local", False))
        provider_enabled = bool(provider_payload.get("enabled", True))
        provider_available = _provider_available(provider_row, provider_health_row, provider_payload)
        model_enabled = bool(model_payload.get("enabled", True))
        model_available = bool(model_payload.get("available", True))
        has_chat = _model_has_chat_capability(model_payload)
        routable = _model_routable(model_row, model_payload, provider_available)
        health_status = _model_status(model_row, model_health_row)
        health_ok = health_status == "ok"
        if not provider_enabled or provider_available is False:
            continue
        if not model_enabled or not model_available:
            continue
        if not has_chat or not routable or not health_ok:
            continue
        observed_cost = _observed_cost(model_payload, model_row)
        max_context_tokens = _safe_context(model_payload.get("max_context_tokens"), model_row)
        candidates.append(
            {
                "provider_id": provider_id,
                "model_id": model_id,
                "is_local": is_local,
                "health_status": health_status,
                "observed_cost": observed_cost,
                "max_context_tokens": max_context_tokens,
            }
        )

    same_provider_local = [
        row
        for row in candidates
        if row.get("provider_id") == default_provider and bool(row.get("is_local"))
    ]
    any_local = [row for row in candidates if bool(row.get("is_local"))]
    remote = [row for row in candidates if not bool(row.get("is_local"))]

    if same_provider_local:
        selected = _rank_replacement_candidates(same_provider_local)[0]
        group = "same_provider_local"
    elif any_local:
        selected = _rank_replacement_candidates(any_local)[0]
        group = "any_local"
    elif allow_remote_fallback and remote:
        selected = _rank_replacement_candidates(remote)[0]
        group = "remote_fallback"
    else:
        return None

    cost = selected.get("observed_cost")
    cost_label = f"cost={cost:.6f}" if isinstance(cost, float) else "cost=unknown"
    selected["rationale"] = (
        f"picked {selected['model_id']} because {group}+chat+routable+health_ok+{cost_label}"
    )
    return selected


def _rank_replacement_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(row: dict[str, Any]) -> tuple[int, float, int, str]:
        cost = row.get("observed_cost")
        model_id = str(row.get("model_id") or "")
        if isinstance(cost, float):
            return (0, float(cost), -int(row.get("max_context_tokens") or 0), model_id)
        return (1, 0.0, 0, model_id)

    return sorted(rows, key=_key)


def _safe_context(raw_context: Any, model_row: dict[str, Any] | None) -> int:
    if _is_number(raw_context):
        return max(0, int(raw_context))
    if isinstance(model_row, dict) and _is_number(model_row.get("max_context_tokens")):
        return max(0, int(model_row.get("max_context_tokens")))
    return 0


def _is_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _change_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("kind") or ""),
        str(row.get("id") or ""),
        str(row.get("field") or ""),
    )
