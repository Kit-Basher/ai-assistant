from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent.model_watch_catalog import load_latest_snapshot


def _normalized_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted(
        {
            str(item).strip().lower()
            for item in values
            if str(item).strip()
        }
    )


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _capabilities_from_snapshot_row(row: dict[str, Any]) -> list[str]:
    modalities = set(_normalized_string_list(row.get("modalities")))
    supports_tools = row.get("supports_tools")
    if "embedding" in modalities or "embed" in modalities:
        return ["embedding"]
    values: set[str] = set()
    if "text" in modalities:
        values.add("chat")
    if "image" in modalities:
        values.add("image")
        values.add("vision")
    if supports_tools is True:
        values.add("tools")
    return sorted(values)


def external_model_discovery_rows_from_openrouter_snapshot(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    payload = snapshot if isinstance(snapshot, dict) else {}
    rows = payload.get("models") if isinstance(payload.get("models"), list) else []
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        provider_id = str(row.get("provider_id") or "").strip().lower()
        model_name = str(row.get("model") or "").strip()
        if not model_id or provider_id != "openrouter" or not model_name or model_id in seen_ids:
            continue
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
        modalities = _normalized_string_list(row.get("modalities"))
        task_types = _normalized_string_list(row.get("task_types"))
        normalized = {
            "model_id": model_id,
            "provider_id": provider_id,
            "model_name": model_name,
            "capabilities": _capabilities_from_snapshot_row(row),
            "task_types": task_types,
            "modalities": modalities,
            "architecture_modality": None,
            "input_modalities": [],
            "output_modalities": [],
            "context_window": _safe_int(row.get("context_length")),
            "price_in": _safe_float(pricing.get("prompt_per_million")),
            "price_out": _safe_float(pricing.get("completion_per_million")),
            "available": True,
            "local": False,
            "source": "external_openrouter_snapshot",
            "external": True,
            "review_required": True,
            "non_canonical": True,
            "canonical_status": "not_adopted",
        }
        output.append(normalized)
        seen_ids.add(model_id)
    output.sort(key=lambda item: str(item.get("model_id") or ""))
    return output


def load_external_model_discovery_rows(
    *,
    provider_id: str | None = None,
    openrouter_snapshot_path: str | Path | None = None,
    snapshot_loader: Callable[[Path], tuple[dict[str, Any] | None, str | None]] = load_latest_snapshot,
) -> dict[str, Any]:
    normalized_provider = str(provider_id or "").strip().lower() or None
    if normalized_provider and normalized_provider != "openrouter":
        return {
            "sources": [],
            "models": [],
        }
    if openrouter_snapshot_path is None:
        return {
            "sources": [
                {
                    "provider_id": "openrouter",
                    "source": "external_openrouter_snapshot",
                    "ok": False,
                    "error_kind": "snapshot_path_missing",
                    "model_count": 0,
                }
            ],
            "models": [],
        }
    snapshot, error_kind = snapshot_loader(Path(openrouter_snapshot_path))
    if error_kind:
        return {
            "sources": [
                {
                    "provider_id": "openrouter",
                    "source": "external_openrouter_snapshot",
                    "ok": False,
                    "error_kind": str(error_kind),
                    "model_count": 0,
                }
            ],
            "models": [],
        }
    models = external_model_discovery_rows_from_openrouter_snapshot(snapshot)
    return {
        "sources": [
            {
                "provider_id": "openrouter",
                "source": "external_openrouter_snapshot",
                "ok": True,
                "error_kind": None,
                "model_count": len(models),
            }
        ],
        "models": models,
    }


__all__ = [
    "external_model_discovery_rows_from_openrouter_snapshot",
    "load_external_model_discovery_rows",
]
