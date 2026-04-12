from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iso_day_from_epoch(epoch: int | None) -> str | None:
    if epoch is None or int(epoch) <= 0:
        return None
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).date().isoformat()


def _change_sort_key(change: dict[str, Any]) -> tuple[int, str, str, str]:
    kind_order = {"defaults": 0, "provider": 1, "model": 2}
    kind = str(change.get("kind") or "").strip().lower()
    item_id = str(change.get("id") or "")
    field = str(change.get("field") or "")
    reason = str(change.get("reason") or "")
    return (kind_order.get(kind, 99), item_id, field, reason)


def _usage_maps(usage_stats_snapshot: dict[str, Any] | None) -> tuple[dict[str, int], dict[str, int]]:
    model_samples: dict[str, int] = {}
    provider_samples: dict[str, int] = {}
    usage = usage_stats_snapshot if isinstance(usage_stats_snapshot, dict) else {}
    for key, payload in sorted(usage.items()):
        if not isinstance(payload, dict):
            continue
        parts = str(key).split("::", 2)
        if len(parts) != 3:
            continue
        provider_id = str(parts[1] or "").strip().lower()
        model_id = str(parts[2] or "").strip()
        if not provider_id or not model_id:
            continue
        samples = max(0, _safe_int(payload.get("samples"), 0))
        if samples <= 0:
            continue
        model_samples[model_id] = model_samples.get(model_id, 0) + samples
        provider_samples[provider_id] = provider_samples.get(provider_id, 0) + samples
    return model_samples, provider_samples


def _health_maps(health_summary: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = health_summary if isinstance(health_summary, dict) else {}
    providers_rows = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    models_rows = payload.get("models") if isinstance(payload.get("models"), list) else []
    providers = {
        str(row.get("id") or "").strip().lower(): row
        for row in providers_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    models = {
        str(row.get("id") or "").strip(): row
        for row in models_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    return providers, models


def _catalog_models_map(catalog_snapshot: dict[str, Any] | None) -> tuple[dict[str, set[str]], set[str]]:
    payload = catalog_snapshot if isinstance(catalog_snapshot, dict) else {}
    providers = payload.get("providers") if isinstance(payload.get("providers"), dict) else {}
    provider_models: dict[str, set[str]] = {}
    authoritative: set[str] = set()
    for provider_id_raw, row in sorted(providers.items()):
        provider_id = str(provider_id_raw).strip().lower()
        if not provider_id or not isinstance(row, dict):
            continue
        models = row.get("models") if isinstance(row.get("models"), list) else []
        ids: set[str] = set()
        for item in models:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            model_name = str(item.get("model") or "").strip()
            if model_id:
                ids.add(model_id)
            elif model_name:
                ids.add(f"{provider_id}:{model_name}")
        provider_models[provider_id] = ids
        if row.get("last_refresh_at") is not None and not str(row.get("last_error_kind") or "").strip():
            authoritative.add(provider_id)
    return provider_models, authoritative


def build_registry_cleanup_plan(
    registry_document: dict[str, Any],
    usage_stats_snapshot: dict[str, Any] | None,
    health_summary: dict[str, Any] | None,
    catalog_snapshot: dict[str, Any] | None,
    *,
    unused_days: int = 30,
    disable_failing_provider: bool = False,
    provider_failure_streak: int = 8,
    apply_prune: bool = False,
) -> dict[str, Any]:
    original = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    updated = copy.deepcopy(original)
    providers = updated.get("providers") if isinstance(updated.get("providers"), dict) else {}
    models = updated.get("models") if isinstance(updated.get("models"), dict) else {}
    defaults = updated.get("defaults") if isinstance(updated.get("defaults"), dict) else {}
    changes: list[dict[str, Any]] = []
    prune_candidates: list[dict[str, Any]] = []
    reasons: set[str] = set()

    model_samples, provider_samples = _usage_maps(usage_stats_snapshot)
    provider_health, model_health = _health_maps(health_summary)
    catalog_models, catalog_authoritative = _catalog_models_map(catalog_snapshot)
    prune_unused_days = max(1, int(unused_days))
    failure_threshold = max(1, int(provider_failure_streak))

    for model_id, payload in sorted(models.items()):
        if not isinstance(payload, dict):
            continue
        provider_id = str(payload.get("provider") or "").strip().lower()
        if not provider_id or provider_id not in catalog_authoritative:
            continue
        if not bool(payload.get("enabled", True)):
            continue
        if not bool(payload.get("available", True)):
            continue
        catalog_ids = catalog_models.get(provider_id, set())
        if model_id in catalog_ids:
            continue
        model_name = str(payload.get("model") or "").strip()
        if model_name and f"{provider_id}:{model_name}" in catalog_ids:
            continue
        down_since = _safe_int((model_health.get(model_id) or {}).get("down_since"), 0) or None
        reason = "missing_from_catalog"
        changes.append(
            {
                "kind": "model",
                "id": model_id,
                "field": "available",
                "before": True,
                "after": False,
                "reason": reason,
                "details": {
                    "provider_id": provider_id,
                    "missing_from_catalog": True,
                    "usage_samples": int(model_samples.get(model_id, 0)),
                    "health_down_since_day": _iso_day_from_epoch(down_since),
                },
            }
        )
        payload["available"] = False
        models[model_id] = payload
        reasons.add(reason)

    if disable_failing_provider:
        for provider_id, payload in sorted(providers.items()):
            if not isinstance(payload, dict):
                continue
            if not bool(payload.get("enabled", True)):
                continue
            if bool(payload.get("local", False)):
                continue
            health_row = provider_health.get(str(provider_id).strip().lower())
            if not isinstance(health_row, dict):
                continue
            status = str(health_row.get("status") or "unknown").strip().lower()
            failure_streak = _safe_int(health_row.get("failure_streak"), 0)
            if status != "down" or failure_streak < failure_threshold:
                continue
            if int(provider_samples.get(provider_id, 0)) > 0:
                continue
            down_since = _safe_int(health_row.get("down_since"), 0) or None
            reason = f"unused_repeated_failures_{failure_threshold}"
            changes.append(
                {
                    "kind": "provider",
                    "id": provider_id,
                    "field": "enabled",
                    "before": True,
                    "after": False,
                    "reason": reason,
                    "details": {
                        "unused_days": prune_unused_days,
                        "last_used_day": "unknown",
                        "failure_streak": failure_streak,
                        "health_down_since_day": _iso_day_from_epoch(down_since),
                    },
                }
            )
            payload["enabled"] = False
            providers[provider_id] = payload
            reasons.add(reason)

    for model_id, payload in sorted(models.items()):
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("enabled", True)):
            continue
        if int(model_samples.get(model_id, 0)) > 0:
            continue
        prune_candidates.append(
            {
                "kind": "model",
                "id": model_id,
                "field": "deleted",
                "before": False,
                "after": True,
                "reason": "prune_disabled_unused_model",
                "details": {
                    "unused_days": prune_unused_days,
                    "last_used_day": "unknown",
                    "usage_samples": int(model_samples.get(model_id, 0)),
                },
            }
        )

    models_remaining_by_provider: dict[str, set[str]] = {}
    for model_id, payload in sorted(models.items()):
        if not isinstance(payload, dict):
            continue
        provider_id = str(payload.get("provider") or "").strip().lower()
        if not provider_id:
            continue
        models_remaining_by_provider.setdefault(provider_id, set()).add(model_id)
    pruned_model_ids = {str(item.get("id") or "").strip() for item in prune_candidates if item.get("kind") == "model"}
    for provider_id in sorted(models_remaining_by_provider.keys()):
        models_remaining_by_provider[provider_id] = {
            item_id for item_id in models_remaining_by_provider[provider_id] if item_id not in pruned_model_ids
        }

    for provider_id, payload in sorted(providers.items()):
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("enabled", True)):
            continue
        if int(provider_samples.get(provider_id, 0)) > 0:
            continue
        if models_remaining_by_provider.get(provider_id):
            continue
        source = payload.get("api_key_source")
        has_key_source = isinstance(source, dict) and bool(str(source.get("name") or "").strip())
        if has_key_source:
            continue
        prune_candidates.append(
            {
                "kind": "provider",
                "id": provider_id,
                "field": "deleted",
                "before": False,
                "after": True,
                "reason": "prune_disabled_empty_provider",
                "details": {
                    "unused_days": prune_unused_days,
                    "last_used_day": "unknown",
                    "usage_samples": int(provider_samples.get(provider_id, 0)),
                },
            }
        )

    if apply_prune:
        for row in sorted(prune_candidates, key=_change_sort_key):
            kind = str(row.get("kind") or "").strip().lower()
            item_id = str(row.get("id") or "").strip()
            if kind == "model":
                models.pop(item_id, None)
            elif kind == "provider":
                providers.pop(item_id, None)
                for model_id in list(models.keys()):
                    model_payload = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
                    if str(model_payload.get("provider") or "").strip().lower() == item_id:
                        models.pop(model_id, None)
        changes.extend(prune_candidates)
        if defaults:
            default_provider = str(defaults.get("default_provider") or "").strip().lower() or None
            default_model = str(defaults.get("default_model") or "").strip() or None
            if default_provider and default_provider not in providers:
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "default_provider",
                        "before": defaults.get("default_provider"),
                        "after": None,
                        "reason": "default_provider_pruned",
                    }
                )
                defaults["default_provider"] = None
            if default_model and default_model not in models:
                changes.append(
                    {
                        "kind": "defaults",
                        "field": "default_model",
                        "before": defaults.get("default_model"),
                        "after": None,
                        "reason": "default_model_pruned",
                    }
                )
                defaults["default_model"] = None
        reasons.update(
            {
                str(row.get("reason") or "").strip()
                for row in prune_candidates
                if str(row.get("reason") or "").strip()
            }
        )

    updated["providers"] = providers
    updated["models"] = models
    updated["defaults"] = defaults
    changes.sort(key=_change_sort_key)
    prune_candidates.sort(key=_change_sort_key)
    reasons.update(
        {
            str(row.get("reason") or "").strip()
            for row in changes
            if str(row.get("reason") or "").strip()
        }
    )

    return {
        "ok": True,
        "changes": changes,
        "prune_candidates": prune_candidates,
        "reasons": sorted(item for item in reasons if item),
        "impact": {
            "changes_count": len(changes),
            "prune_candidates_count": len(prune_candidates),
            "models_marked_unavailable": sorted(
                {
                    str(row.get("id") or "")
                    for row in changes
                    if row.get("kind") == "model" and row.get("field") == "available" and row.get("after") is False
                }
            ),
            "providers_disabled": sorted(
                {
                    str(row.get("id") or "")
                    for row in changes
                    if row.get("kind") == "provider" and row.get("field") == "enabled" and row.get("after") is False
                }
            ),
            "models_pruned": sorted(
                {
                    str(row.get("id") or "")
                    for row in changes
                    if row.get("kind") == "model" and row.get("field") == "deleted" and row.get("after") is True
                }
            ),
            "providers_pruned": sorted(
                {
                    str(row.get("id") or "")
                    for row in changes
                    if row.get("kind") == "provider" and row.get("field") == "deleted" and row.get("after") is True
                }
            ),
        },
        "policy": {
            "unused_days": prune_unused_days,
            "disable_failing_provider": bool(disable_failing_provider),
            "provider_failure_streak": failure_threshold,
            "apply_prune": bool(apply_prune),
        },
        "before_document": original,
        "updated_document": updated,
    }


def apply_registry_cleanup_plan(registry_document: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    updated_document = plan.get("updated_document") if isinstance(plan.get("updated_document"), dict) else None
    if updated_document is not None:
        return copy.deepcopy(updated_document)
    return copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})


__all__ = [
    "apply_registry_cleanup_plan",
    "build_registry_cleanup_plan",
]
