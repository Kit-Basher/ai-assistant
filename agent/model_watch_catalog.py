from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


CATALOG_SCHEMA_VERSION = 1
_PARAMS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*[bB]")


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN guard
        return None
    return parsed


def _params_b_from_name(model_name: str) -> float | None:
    text = str(model_name or "").strip()
    if not text:
        return None
    match = _PARAMS_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _normalize_modalities(row: dict[str, Any]) -> list[str]:
    architecture = row.get("architecture") if isinstance(row.get("architecture"), dict) else {}
    values: set[str] = set()
    modality = str(architecture.get("modality") or "").strip().lower()
    if modality:
        values.add(modality)
    for key in ("input_modalities", "output_modalities"):
        entries = architecture.get(key) if isinstance(architecture.get(key), list) else []
        for item in entries:
            text = str(item or "").strip().lower()
            if text:
                values.add(text)
    if not values:
        values.add("text")
    return sorted(values)


def _supports_tools(row: dict[str, Any]) -> bool | None:
    params = row.get("supported_parameters") if isinstance(row.get("supported_parameters"), list) else None
    if not isinstance(params, list):
        return None
    normalized = {str(item or "").strip().lower() for item in params if str(item or "").strip()}
    if not normalized:
        return None
    return any("tool" in item for item in normalized)


def _normalize_capabilities(*, model_name: str, modalities: list[str], supports_tools: bool | None) -> list[str]:
    lowered = str(model_name or "").strip().lower()
    modality_set = {str(item).strip().lower() for item in modalities if str(item).strip()}
    if "embedding" in modality_set or "embed" in lowered:
        return ["embed"]
    values = {"chat"}
    if supports_tools is True:
        values.add("tools")
        values.add("json")
    if "image" in modality_set:
        values.add("vision")
    return sorted(values)


def _normalize_openrouter_rows(raw_payload: Any) -> tuple[list[dict[str, Any]], str]:
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    rows = payload.get("data") if isinstance(payload.get("data"), list) else []
    raw_sha256 = hashlib.sha256(
        json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_name = str(row.get("id") or "").strip()
        if not model_name:
            continue
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
        prompt_per_million = _safe_float(
            pricing.get("prompt_per_million")
            if pricing.get("prompt_per_million") is not None
            else pricing.get("input_per_million_tokens")
        )
        completion_per_million = _safe_float(
            pricing.get("completion_per_million")
            if pricing.get("completion_per_million") is not None
            else pricing.get("output_per_million_tokens")
        )
        if prompt_per_million is None:
            prompt_token = _safe_float(pricing.get("prompt") if pricing.get("prompt") is not None else pricing.get("input"))
            if prompt_token is not None:
                prompt_per_million = prompt_token * 1_000_000.0
        if completion_per_million is None:
            completion_token = _safe_float(
                pricing.get("completion") if pricing.get("completion") is not None else pricing.get("output")
            )
            if completion_token is not None:
                completion_per_million = completion_token * 1_000_000.0

        context_length = _safe_int(row.get("context_length"))
        if context_length is None:
            context_length = _safe_int(row.get("max_context_length"))

        modalities = _normalize_modalities(row)
        supports_tools = _supports_tools(row)
        normalized.append(
            {
                "id": f"openrouter:{model_name}",
                "provider_id": "openrouter",
                "model": model_name,
                "context_length": context_length,
                "modalities": modalities,
                "supports_tools": supports_tools,
                "pricing": {
                    "prompt_per_million": prompt_per_million,
                    "completion_per_million": completion_per_million,
                },
                "capabilities": _normalize_capabilities(
                    model_name=model_name,
                    modalities=modalities,
                    supports_tools=supports_tools,
                ),
                "params_b": _safe_float(row.get("params_b")) or _params_b_from_name(model_name),
                "quality_percentile": _safe_float(row.get("quality_percentile")),
            }
        )
    normalized.sort(key=lambda item: str(item.get("id") or ""))
    return normalized, raw_sha256


def normalize_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    base = payload if isinstance(payload, dict) else {}
    models_raw = base.get("models") if isinstance(base.get("models"), list) else []
    models: list[dict[str, Any]] = []
    for row in models_raw:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        provider_id = str(row.get("provider_id") or "").strip().lower()
        model_name = str(row.get("model") or "").strip()
        if not model_id or not provider_id or not model_name:
            continue
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
        modalities_raw = row.get("modalities") if isinstance(row.get("modalities"), list) else []
        capabilities_raw = row.get("capabilities") if isinstance(row.get("capabilities"), list) else []
        models.append(
            {
                "id": model_id,
                "provider_id": provider_id,
                "model": model_name,
                "context_length": _safe_int(row.get("context_length")),
                "modalities": sorted(
                    {
                        str(item).strip().lower()
                        for item in modalities_raw
                        if str(item).strip()
                    }
                ),
                "supports_tools": (
                    bool(row.get("supports_tools"))
                    if row.get("supports_tools") is not None
                    else None
                ),
                "pricing": {
                    "prompt_per_million": _safe_float(pricing.get("prompt_per_million")),
                    "completion_per_million": _safe_float(pricing.get("completion_per_million")),
                },
                "capabilities": sorted(
                    {
                        str(item).strip().lower()
                        for item in capabilities_raw
                        if str(item).strip()
                    }
                )
                or ["chat"],
                "params_b": _safe_float(row.get("params_b")),
                "quality_percentile": _safe_float(row.get("quality_percentile")),
            }
        )
    models.sort(key=lambda item: str(item.get("id") or ""))
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source": str(base.get("source") or "openrouter_models").strip() or "openrouter_models",
        "provider": str(base.get("provider") or "openrouter").strip().lower() or "openrouter",
        "fetched_at": _safe_int(base.get("fetched_at")),
        "raw_sha256": str(base.get("raw_sha256") or "").strip() or None,
        "model_count": len(models),
        "models": models,
    }


def build_openrouter_snapshot(*, raw_payload: Any, fetched_at: int, source: str = "openrouter_models") -> dict[str, Any]:
    models, raw_sha256 = _normalize_openrouter_rows(raw_payload)
    return normalize_snapshot(
        {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "source": str(source or "openrouter_models").strip() or "openrouter_models",
            "provider": "openrouter",
            "fetched_at": int(fetched_at),
            "raw_sha256": raw_sha256,
            "models": models,
        }
    )


def snapshot_path_for_config(config: Any) -> Path:
    env_path = os.getenv("AGENT_MODEL_WATCH_CATALOG_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    configured = str(getattr(config, "model_watch_catalog_path", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    state_path = str(getattr(config, "model_watch_state_path", "") or "").strip()
    if not state_path:
        state_path = os.getenv("AGENT_MODEL_WATCH_STATE_PATH", "").strip()
    if state_path:
        return (Path(state_path).expanduser().resolve().parent / "model_watch_catalog_snapshot.json").resolve()
    return (Path.home() / ".local" / "share" / "personal-agent" / "model_watch_catalog_snapshot.json").resolve()


def write_snapshot_atomic(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_snapshot(payload if isinstance(payload, dict) else {})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
    return normalized


def load_latest_snapshot(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "unreadable"
    if not isinstance(parsed, dict):
        return None, "invalid_payload"
    normalized = normalize_snapshot(parsed)
    if not isinstance(normalized.get("models"), list):
        return None, "invalid_models"
    return normalized, None


def build_feature_index(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = snapshot.get("models") if isinstance(snapshot.get("models"), list) else []
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
        context_tokens = _safe_int(row.get("context_length"))
        price_in = _safe_float(pricing.get("prompt_per_million"))
        price_out = _safe_float(pricing.get("completion_per_million"))
        missing_features: list[str] = []
        if context_tokens is None:
            missing_features.append("missing:context_length")
        if price_in is None or price_out is None:
            missing_features.append("missing:pricing")
        if _safe_float(row.get("params_b")) is None:
            missing_features.append("missing:params_b")
        output[model_id] = {
            "context_tokens": context_tokens,
            "price_in": price_in,
            "price_out": price_out,
            "capabilities": [
                str(item).strip().lower()
                for item in (row.get("capabilities") if isinstance(row.get("capabilities"), list) else [])
                if str(item).strip()
            ]
            or ["chat"],
            "params_b": _safe_float(row.get("params_b")),
            "quality_percentile": _safe_float(row.get("quality_percentile")),
            "missing_features": sorted(set(missing_features)),
        }
    return {model_id: output[model_id] for model_id in sorted(output.keys())}


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "build_feature_index",
    "build_openrouter_snapshot",
    "load_latest_snapshot",
    "normalize_snapshot",
    "snapshot_path_for_config",
    "write_snapshot_atomic",
]
