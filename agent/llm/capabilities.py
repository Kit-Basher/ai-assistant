from __future__ import annotations

import copy
from typing import Any


_EMBEDDING_ONLY_NAME_MARKERS = (
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
    "bge-m3",
    "jina-embeddings",
    "granite-embedding",
    "text-embedding",
)

_OLLAMA_CHAT_NAME_MARKERS = (
    "llama",
    "qwen",
    "gemma",
    "mistral",
    "mixtral",
    "phi",
    "deepseek",
    "yi",
    "command-r",
)

_ORDERED_CAPABILITIES = ("chat", "embedding", "image", "json", "tools", "vision")


def _normalized_model_name(entry: dict[str, Any]) -> str:
    name = str(entry.get("model") or "").strip()
    if name:
        return name
    item_id = str(entry.get("id") or "").strip()
    if ":" in item_id:
        return item_id.split(":", 1)[1].strip()
    return item_id


def _normalize_capability_list(raw: Any) -> list[str]:
    values = raw if isinstance(raw, list) else [raw] if isinstance(raw, str) else []
    normalized = sorted(
        {
            str(item).strip().lower()
            for item in values
            if str(item).strip()
        }
    )
    return normalized


def is_embedding_model_name(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return False
    if "embed" in normalized:
        return True
    return any(marker in normalized for marker in _EMBEDDING_ONLY_NAME_MARKERS)


def infer_capabilities_from_catalog(provider_id: str, catalog_entry: dict[str, Any]) -> dict[str, bool]:
    provider = str(provider_id or "").strip().lower()
    entry = catalog_entry if isinstance(catalog_entry, dict) else {}
    model_name = _normalized_model_name(entry)
    normalized_name = model_name.strip().lower()
    source_caps = set(_normalize_capability_list(entry.get("capabilities")))

    inferred = {key: False for key in _ORDERED_CAPABILITIES}
    if "image" in source_caps:
        inferred["image"] = True
    if "vision" in source_caps:
        inferred["vision"] = True
    if "json" in source_caps:
        inferred["json"] = True
    if "tools" in source_caps:
        inferred["tools"] = True

    if provider == "ollama":
        if is_embedding_model_name(normalized_name):
            inferred["embedding"] = True
            inferred["chat"] = False
            inferred["json"] = False
            inferred["tools"] = False
            inferred["image"] = False
            inferred["vision"] = False
            return inferred

        chat_hint = (
            normalized_name.endswith("-instruct")
            or ":instruct" in normalized_name
            or any(marker in normalized_name for marker in _OLLAMA_CHAT_NAME_MARKERS)
        )
        if "embedding" in source_caps and "chat" not in source_caps:
            if chat_hint:
                inferred["chat"] = True
                inferred["embedding"] = False
            else:
                inferred["embedding"] = True
                inferred["chat"] = False
        else:
            inferred["chat"] = True
            inferred["embedding"] = "embedding" in source_caps and "chat" in source_caps
        return inferred

    if is_embedding_model_name(normalized_name):
        inferred["embedding"] = True
        inferred["chat"] = False
        inferred["json"] = False
        inferred["tools"] = False
        return inferred

    if "embedding" in source_caps and "chat" not in source_caps:
        inferred["embedding"] = True
        inferred["chat"] = False
        inferred["json"] = False
        inferred["tools"] = False
        return inferred

    inferred["chat"] = True
    if "embedding" in source_caps and "chat" in source_caps:
        inferred["embedding"] = True
    return inferred


def capability_list_from_inference(inferred: dict[str, bool]) -> list[str]:
    payload = inferred if isinstance(inferred, dict) else {}
    values = [key for key in _ORDERED_CAPABILITIES if bool(payload.get(key))]
    if not values:
        return ["chat"]
    return values


def _change_sort_key(change: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(change.get("kind") or ""),
        str(change.get("id") or ""),
        str(change.get("field") or ""),
    )


def _default_for_from_capabilities(current: Any, capabilities: list[str]) -> list[str]:
    normalized_current = sorted(
        {
            str(item).strip().lower()
            for item in (current if isinstance(current, list) else [])
            if str(item).strip()
        }
    )
    caps = {str(item).strip().lower() for item in capabilities if str(item).strip()}
    if "chat" in caps:
        if not normalized_current:
            return ["chat"]
        if "chat" not in normalized_current:
            return sorted({"chat", *normalized_current})
        return normalized_current
    filtered = [item for item in normalized_current if item != "chat"]
    if filtered:
        return filtered
    if "embedding" in caps:
        return ["embedding"]
    return []


def _catalog_model_map(catalog_snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshot = catalog_snapshot if isinstance(catalog_snapshot, dict) else {}
    providers = snapshot.get("providers") if isinstance(snapshot.get("providers"), dict) else {}
    output: dict[str, dict[str, Any]] = {}
    for provider_id in sorted(providers.keys()):
        provider_row = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
        models = provider_row.get("models") if isinstance(provider_row.get("models"), list) else []
        for model_row in models:
            if not isinstance(model_row, dict):
                continue
            model_id = str(model_row.get("id") or "").strip()
            if not model_id:
                continue
            output[model_id] = model_row
    return output


def build_capabilities_reconcile_plan(
    registry_document: dict[str, Any],
    catalog_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document = registry_document if isinstance(registry_document, dict) else {}
    models = document.get("models") if isinstance(document.get("models"), dict) else {}
    catalog_models = _catalog_model_map(catalog_snapshot or {})
    changes: list[dict[str, Any]] = []

    for model_id in sorted(models.keys()):
        model_row = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
        provider_id = str(model_row.get("provider") or "").strip().lower()
        if not provider_id:
            continue

        catalog_row = catalog_models.get(model_id)
        if not isinstance(catalog_row, dict):
            catalog_row = {
                "id": model_id,
                "provider_id": provider_id,
                "model": str(model_row.get("model") or "").strip(),
                "capabilities": list(model_row.get("capabilities") or []),
            }

        inferred = infer_capabilities_from_catalog(provider_id, catalog_row)
        inferred_caps = capability_list_from_inference(inferred)
        current_caps = _normalize_capability_list(model_row.get("capabilities"))
        if current_caps != inferred_caps:
            changes.append(
                {
                    "kind": "model",
                    "id": model_id,
                    "field": "capabilities",
                    "before": current_caps,
                    "after": inferred_caps,
                    "reason": "inferred_capability_mismatch",
                }
            )

        desired_default_for = _default_for_from_capabilities(model_row.get("default_for"), inferred_caps)
        current_default_for = _default_for_from_capabilities(model_row.get("default_for"), current_caps)
        if current_default_for != desired_default_for:
            changes.append(
                {
                    "kind": "model",
                    "id": model_id,
                    "field": "default_for",
                    "before": current_default_for,
                    "after": desired_default_for,
                    "reason": "routability_adjustment_from_capabilities",
                }
            )

    changes.sort(key=_change_sort_key)
    reasons = sorted(
        {
            str(item.get("reason") or "").strip()
            for item in changes
            if isinstance(item, dict) and str(item.get("reason") or "").strip()
        }
    )
    if not reasons:
        reasons = ["no_changes"]
    return {
        "ok": True,
        "changes": changes,
        "reasons": reasons,
        "impact": {
            "changes_count": len(changes),
            "models_with_mismatch": len(
                {
                    str(item.get("id") or "")
                    for item in changes
                    if isinstance(item, dict) and str(item.get("id") or "")
                }
            ),
        },
    }


def apply_capabilities_reconcile_plan(
    registry_document: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(registry_document if isinstance(registry_document, dict) else {})
    models = updated.get("models") if isinstance(updated.get("models"), dict) else {}
    changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []
    for change in sorted((item for item in changes if isinstance(item, dict)), key=_change_sort_key):
        if str(change.get("kind") or "").strip().lower() != "model":
            continue
        model_id = str(change.get("id") or "").strip()
        field = str(change.get("field") or "").strip()
        if not model_id or not field:
            continue
        model_row = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
        if not isinstance(model_row, dict):
            continue
        model_row = copy.deepcopy(model_row)
        model_row[field] = copy.deepcopy(change.get("after"))
        models[model_id] = model_row
    updated["models"] = models
    return updated


__all__ = [
    "apply_capabilities_reconcile_plan",
    "build_capabilities_reconcile_plan",
    "capability_list_from_inference",
    "infer_capabilities_from_catalog",
    "is_embedding_model_name",
]
