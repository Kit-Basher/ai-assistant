from __future__ import annotations

import copy
import time
from typing import Any


def build_hygiene_plan(
    registry_document: dict[str, Any],
    health_summary: dict[str, Any] | None = None,
    *,
    unavailable_days: int = 7,
    remove_empty_disabled_providers: bool = True,
    provider_inventory: dict[str, Any] | None = None,
    disable_repeatedly_failing_providers: bool = False,
    provider_failure_streak: int = 8,
) -> dict[str, Any]:
    original = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    updated = copy.deepcopy(original)
    providers = updated.get("providers") if isinstance(updated.get("providers"), dict) else {}
    models = updated.get("models") if isinstance(updated.get("models"), dict) else {}
    defaults = updated.get("defaults") if isinstance(updated.get("defaults"), dict) else {}
    changes: list[dict[str, Any]] = []

    _normalize_provider_urls(providers, changes)
    _normalize_default_model(defaults, models, changes)
    _mark_missing_provider_models_unavailable(
        models,
        provider_inventory or {},
        changes,
    )
    _mark_stale_down_models(
        models,
        health_summary or {},
        changes,
        unavailable_days=max(1, int(unavailable_days)),
    )
    if disable_repeatedly_failing_providers:
        _disable_repeatedly_failing_providers(
            providers,
            health_summary or {},
            changes,
            provider_failure_streak=max(1, int(provider_failure_streak)),
        )
    if remove_empty_disabled_providers:
        _prune_empty_disabled_providers(providers, models, changes)

    changes.sort(key=_change_sort_key)
    return {
        "ok": True,
        "changes": changes,
        "impact": {
            "changes_count": len(changes),
            "providers_removed": sorted(
                {
                    str(item.get("id") or "")
                    for item in changes
                    if item.get("kind") == "provider" and item.get("field") == "deleted" and item.get("after") is True
                }
            ),
            "providers_disabled": sorted(
                {
                    str(item.get("id") or "")
                    for item in changes
                    if item.get("kind") == "provider" and item.get("field") == "enabled" and item.get("after") is False
                }
            ),
            "models_marked_unavailable": sorted(
                {
                    str(item.get("id") or "")
                    for item in changes
                    if item.get("kind") == "model" and item.get("field") == "available" and item.get("after") is False
                }
            ),
        },
        "updated_document": updated,
        "before_document": original,
    }


def apply_hygiene_plan(registry_document: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    updated_document = plan.get("updated_document") if isinstance(plan.get("updated_document"), dict) else None
    if updated_document is not None:
        return copy.deepcopy(updated_document)
    # Fallback defensive behavior.
    fallback = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    return fallback


def _normalize_provider_urls(providers: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    for provider_id in sorted(providers.keys()):
        payload = providers.get(provider_id)
        if not isinstance(payload, dict):
            continue
        base_before = str(payload.get("base_url") or "")
        chat_before = str(payload.get("chat_path") or "")
        base_after = base_before.strip().rstrip("/")
        chat_after = chat_before.strip() or "/v1/chat/completions"
        if not chat_after.startswith("/"):
            chat_after = "/" + chat_after
        if base_after.endswith("/v1") and chat_after.startswith("/v1/"):
            base_after = base_after[:-3]
            if base_after.endswith("/"):
                base_after = base_after.rstrip("/")

        if base_before != base_after:
            changes.append(
                {
                    "kind": "provider",
                    "id": provider_id,
                    "field": "base_url",
                    "before": base_before,
                    "after": base_after,
                    "reason": "normalize_base_url",
                }
            )
            payload["base_url"] = base_after
        if chat_before != chat_after:
            changes.append(
                {
                    "kind": "provider",
                    "id": provider_id,
                    "field": "chat_path",
                    "before": chat_before,
                    "after": chat_after,
                    "reason": "normalize_chat_path",
                }
            )
            payload["chat_path"] = chat_after
        providers[provider_id] = payload


def _normalize_default_model(defaults: dict[str, Any], models: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    default_provider = str(defaults.get("default_provider") or "").strip().lower()
    default_model = str(defaults.get("default_model") or "").strip()
    if not default_provider or not default_model:
        return
    if ":" in default_model:
        return

    fq = f"{default_provider}:{default_model}"
    candidate = fq
    if candidate not in models:
        matching_ids = [
            model_id
            for model_id, payload in sorted(models.items())
            if isinstance(payload, dict)
            and str(payload.get("provider") or "").strip().lower() == default_provider
            and str(payload.get("model") or "").strip() == default_model
        ]
        if len(matching_ids) == 1:
            candidate = matching_ids[0]
        else:
            return

    changes.append(
        {
            "kind": "defaults",
            "field": "default_model",
            "before": default_model,
            "after": candidate,
            "reason": "fully_qualified_default_model",
        }
    )
    defaults["default_model"] = candidate


def _mark_stale_down_models(
    models: dict[str, Any],
    health_summary: dict[str, Any],
    changes: list[dict[str, Any]],
    *,
    unavailable_days: int,
) -> None:
    model_health_rows = health_summary.get("models") if isinstance(health_summary.get("models"), list) else []
    model_health = {
        str(row.get("id") or "").strip(): row
        for row in model_health_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    now = int(time.time())
    threshold_seconds = max(1, int(unavailable_days)) * 86400

    for model_id, payload in sorted(models.items()):
        if not isinstance(payload, dict):
            continue
        if not bool(payload.get("enabled", True)):
            continue
        if not bool(payload.get("available", True)):
            continue
        row = model_health.get(model_id)
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "unknown").strip().lower()
        down_since = int(row.get("down_since") or 0)
        if status != "down" or down_since <= 0:
            continue
        if now - down_since < threshold_seconds:
            continue
        changes.append(
            {
                "kind": "model",
                "id": model_id,
                "field": "available",
                "before": True,
                "after": False,
                "reason": f"down_for_{unavailable_days}_days",
            }
        )
        payload["available"] = False
        models[model_id] = payload


def _mark_missing_provider_models_unavailable(
    models: dict[str, Any],
    provider_inventory: dict[str, Any],
    changes: list[dict[str, Any]],
) -> None:
    normalized_inventory: dict[str, dict[str, Any]] = {}
    for provider_id, payload in sorted(provider_inventory.items()):
        pid = str(provider_id or "").strip().lower()
        if not pid:
            continue

        authoritative = False
        model_values: list[Any] = []
        if isinstance(payload, dict):
            authoritative = bool(payload.get("authoritative", False))
            model_values = payload.get("models") if isinstance(payload.get("models"), list) else []
        elif isinstance(payload, list):
            authoritative = True
            model_values = payload

        model_names: set[str] = set()
        for row in model_values:
            text = str(row or "").strip()
            if not text:
                continue
            if ":" in text:
                prefix, remainder = text.split(":", 1)
                if prefix.strip().lower() == pid:
                    text = remainder.strip()
            if text:
                model_names.add(text)

        normalized_inventory[pid] = {
            "authoritative": authoritative,
            "models": model_names,
        }

    for model_id, payload in sorted(models.items()):
        if not isinstance(payload, dict):
            continue
        if not bool(payload.get("enabled", True)):
            continue
        if not bool(payload.get("available", True)):
            continue
        provider_id = str(payload.get("provider") or "").strip().lower()
        inventory = normalized_inventory.get(provider_id)
        if not isinstance(inventory, dict):
            continue
        if not bool(inventory.get("authoritative", False)):
            continue
        present_models = inventory.get("models")
        if not isinstance(present_models, set):
            continue
        model_name = str(payload.get("model") or "").strip()
        if not model_name or model_name in present_models:
            continue
        changes.append(
            {
                "kind": "model",
                "id": model_id,
                "field": "available",
                "before": True,
                "after": False,
                "reason": "missing_from_provider_inventory",
            }
        )
        payload["available"] = False
        models[model_id] = payload


def _disable_repeatedly_failing_providers(
    providers: dict[str, Any],
    health_summary: dict[str, Any],
    changes: list[dict[str, Any]],
    *,
    provider_failure_streak: int,
) -> None:
    provider_health_rows = health_summary.get("providers") if isinstance(health_summary.get("providers"), list) else []
    provider_health = {
        str(row.get("id") or "").strip().lower(): row
        for row in provider_health_rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    threshold = max(1, int(provider_failure_streak))

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
        failure_streak = int(health_row.get("failure_streak") or 0)
        if status != "down" or failure_streak < threshold:
            continue
        changes.append(
            {
                "kind": "provider",
                "id": provider_id,
                "field": "enabled",
                "before": True,
                "after": False,
                "reason": f"down_failure_streak_{threshold}",
            }
        )
        payload["enabled"] = False
        providers[provider_id] = payload


def _prune_empty_disabled_providers(
    providers: dict[str, Any],
    models: dict[str, Any],
    changes: list[dict[str, Any]],
) -> None:
    used_provider_ids = {
        str(payload.get("provider") or "").strip().lower()
        for payload in models.values()
        if isinstance(payload, dict) and str(payload.get("provider") or "").strip()
    }
    for provider_id in sorted(list(providers.keys())):
        payload = providers.get(provider_id)
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("enabled", True)):
            continue
        if provider_id in used_provider_ids:
            continue
        source = payload.get("api_key_source") if isinstance(payload.get("api_key_source"), dict) else None
        if source:
            continue
        changes.append(
            {
                "kind": "provider",
                "id": provider_id,
                "field": "deleted",
                "before": False,
                "after": True,
                "reason": "disabled_empty_provider_without_key_source",
            }
        )
        providers.pop(provider_id, None)


def _change_sort_key(change: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(change.get("kind") or ""),
        str(change.get("id") or ""),
        str(change.get("field") or ""),
    )
