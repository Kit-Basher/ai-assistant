from __future__ import annotations

from typing import Any, Iterable, Mapping


_ALLOWED_TASK_TYPES = {
    "chat",
    "coding",
    "vision",
    "reasoning",
    "tool_use",
    "health",
}


def _sorted_unique_strings(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        }
    )


def _normalize_inventory_item(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    context_window_value = payload.get("context_window")
    try:
        context_window = int(context_window_value) if context_window_value is not None else None
    except (TypeError, ValueError):
        context_window = None
    normalized: dict[str, Any] = {
        "id": str(payload.get("id") or "").strip(),
        "provider": str(payload.get("provider") or "").strip().lower(),
        "installed": bool(payload.get("installed", False)),
        "available": bool(payload.get("available", False)),
        "healthy": bool(payload.get("healthy", False)),
        "capabilities": _sorted_unique_strings(payload.get("capabilities") or []),
        "task_types": _sorted_unique_strings(payload.get("task_types") or []),
        "size": str(payload.get("size") or "").strip() or None,
        "context_window": context_window,
        "local": bool(payload.get("local", False)),
        "approved": bool(payload.get("approved", False)),
        "reason": str(payload.get("reason") or "").strip() or "unknown",
    }
    for key in (
        "quality_rank",
        "cost_rank",
        "price_in",
        "price_out",
        "architecture_modality",
        "input_modalities",
        "output_modalities",
        "params_b",
        "health_status",
        "health_failure_kind",
        "health_reason",
        "model_name",
        "source",
        "configured",
        "capability_source",
        "capability_provenance",
        "runtime_known",
        "routable",
        "availability_state",
        "availability_reason",
        "eligibility_state",
        "eligibility_reason",
        "acquirable",
        "acquisition_state",
        "acquisition_reason",
        "lifecycle_state",
        "provider_connection_state",
        "provider_selection_state",
        "auth_required",
        "comfortable_local",
        "local_fit_state",
        "local_fit_reason",
        "local_fit_margin_gb",
        "min_memory_gb",
    ):
        if key in payload:
            normalized[key] = payload.get(key)
    return normalized


def normalize_model_inventory(items: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    normalized = [_normalize_inventory_item(item) for item in (items or []) if isinstance(item, Mapping)]
    normalized.sort(key=lambda item: (0 if bool(item.get("local", False)) else 1, str(item.get("provider") or ""), str(item.get("id") or "")))
    return normalized


def normalize_task_request(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    task_type = str(payload.get("task_type") or "chat").strip().lower() or "chat"
    if task_type not in _ALLOWED_TASK_TYPES:
        task_type = "chat"
    return {
        "task_type": task_type,
        "requirements": _sorted_unique_strings(payload.get("requirements") or []),
        "preferred_local": bool(payload.get("preferred_local", True)),
    }


def normalize_selection_result(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    fallbacks_raw = payload.get("fallbacks") if isinstance(payload.get("fallbacks"), list) else []
    seen: set[str] = set()
    fallbacks: list[str] = []
    for item in fallbacks_raw:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        fallbacks.append(normalized)
    return {
        "selected_model": str(payload.get("selected_model") or "").strip() or None,
        "provider": str(payload.get("provider") or "").strip().lower() or None,
        "reason": str(payload.get("reason") or "").strip() or "no_selection",
        "fallbacks": fallbacks,
        "trace_id": str(payload.get("trace_id") or "").strip() or None,
    }


__all__ = [
    "normalize_model_inventory",
    "normalize_selection_result",
    "normalize_task_request",
]
