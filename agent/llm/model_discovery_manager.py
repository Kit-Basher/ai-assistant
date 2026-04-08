from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import urllib.parse
import urllib.request

from agent.llm.capabilities import capability_list_from_inference, infer_capabilities_from_catalog
from agent.llm.model_discovery_external import load_external_model_discovery_rows
from agent.model_watch_catalog import build_openrouter_snapshot, snapshot_path_for_config
from agent.modelops.discovery import list_models_ollama
from agent.llm.registry import load_registry_document


_SOURCE_IDS = ("huggingface", "openrouter", "ollama", "external_snapshots")
_SOURCE_ALIASES = {
    "hf": "huggingface",
    "huggingface_trending": "huggingface",
    "ollama_catalog": "ollama",
    "openrouter_models": "openrouter",
    "external": "external_snapshots",
    "external_snapshot": "external_snapshots",
    "external_snapshot_rows": "external_snapshots",
}
_CONFIDENCE_LABELS = {
    "high": 0.9,
    "medium": 0.65,
    "low": 0.35,
}
_SOURCE_CONFIDENCE = {
    "huggingface": 0.55,
    "openrouter": 0.75,
    "ollama": 0.9,
    "external_snapshots": 0.7,
}
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "available",
    "best",
    "better",
    "check",
    "discover",
    "for",
    "find",
    "from",
    "give",
    "list",
    "me",
    "model",
    "models",
    "of",
    "or",
    "please",
    "recommend",
    "search",
    "show",
    "the",
    "to",
    "use",
    "what",
    "with",
}


def _as_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if not isinstance(values, (list, tuple, set, frozenset)):
        return []
    return [
        text
        for text in (
            str(item).strip()
            for item in values
        )
        if text
    ]


def _normalized_strings(values: Any) -> list[str]:
    return sorted({item.lower() for item in _as_string_list(values)})


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _query_tokens(query: str | None) -> list[str]:
    tokens = [token for token in (str(query or "").strip().lower().split()) if token]
    return [token for token in tokens if token not in _QUERY_STOPWORDS]


def _row_text(row: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "id",
        "provider",
        "provider_id",
        "source",
        "source_origin",
        "model",
        "model_name",
        "display_name",
        "repo_id",
        "install_name",
        "kind",
        "reason",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip().lower())
    for key in ("capabilities", "task_types", "modalities", "tags"):
        values = row.get(key)
        if isinstance(values, (list, tuple, set, frozenset)):
            pieces.extend(
                text
                for text in (
                    str(item).strip().lower()
                    for item in values
                )
                if text
            )
    return " ".join(pieces)


def _row_matches_query(row: Mapping[str, Any], query: str | None) -> bool:
    tokens = _query_tokens(query)
    if not tokens:
        return True
    haystack = _row_text(row)
    return all(token in haystack for token in tokens)


def _row_matches_filters(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    provider_ids = {
        item.lower()
        for item in _as_string_list(filters.get("provider_ids"))
        if item
    }
    if provider_ids and str(row.get("provider") or row.get("provider_id") or "").strip().lower() not in provider_ids:
        return False

    if filters.get("local_only") is True and not bool(row.get("local", False)):
        return False
    if filters.get("remote_only") is True and bool(row.get("local", False)):
        return False
    if filters.get("installable_only") is True and not bool(row.get("installable", False)):
        return False

    return True


def _canonical_model_id(provider_id: str, row: Mapping[str, Any]) -> str:
    raw_id = str(
        row.get("id")
        or row.get("model_id")
        or row.get("model")
        or row.get("display_name")
        or row.get("repo_id")
        or row.get("install_name")
        or ""
    ).strip()
    if not raw_id:
        return f"{provider_id}:unknown" if provider_id else "unknown"
    if ":" in raw_id:
        return raw_id
    if provider_id and not raw_id.lower().startswith(f"{provider_id.lower()}:"):
        return f"{provider_id}:{raw_id}"
    return raw_id


def _capabilities_for_row(provider_id: str, row: Mapping[str, Any]) -> list[str]:
    capabilities = _as_string_list(row.get("capabilities"))
    if not capabilities:
        derived = infer_capabilities_from_catalog(
            provider_id or "",
            {
                "id": row.get("id") or row.get("model_id") or row.get("model") or row.get("display_name"),
                "provider_id": provider_id or "",
                "model": row.get("model") or row.get("model_name") or row.get("display_name") or row.get("id") or "",
                "capabilities": row.get("capabilities") or [],
            },
        )
        capabilities = capability_list_from_inference(derived)
    return sorted({item.lower() for item in capabilities if item})


def _installable_for_row(source_id: str, row: Mapping[str, Any]) -> bool:
    if "installable" in row:
        return bool(row.get("installable"))
    installability = str(row.get("installability") or "").strip().lower()
    if installability == "installable_ollama":
        return True
    if row.get("selected_gguf"):
        return True
    if source_id == "ollama":
        return True
    if source_id == "external_snapshots" and bool(row.get("available", False)):
        return True
    return bool(row.get("available", False))


def _confidence_for_row(source_id: str, row: Mapping[str, Any]) -> float:
    raw = row.get("confidence")
    if isinstance(raw, (int, float)):
        parsed = _safe_float(raw)
        if parsed is not None:
            return max(0.0, min(1.0, parsed))
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in _CONFIDENCE_LABELS:
            return _CONFIDENCE_LABELS[normalized]
    score = _safe_float(row.get("score"))
    if score is not None:
        if 0.0 <= score <= 1.0:
            return score
        if 1.0 < score <= 100.0:
            return max(0.0, min(1.0, score / 100.0))
    return _SOURCE_CONFIDENCE.get(source_id, 0.5)


def _normalize_result_row(
    *,
    source_id: str,
    row: Mapping[str, Any],
    query: str | None,
    source_origin: str | None = None,
) -> dict[str, Any]:
    payload = dict(row)
    provider_id = str(
        payload.get("provider")
        or payload.get("provider_id")
        or (
            "openrouter" if source_id in {"openrouter", "external_snapshots"} else source_id
        )
        or ""
    ).strip().lower()
    canonical_id = _canonical_model_id(provider_id or source_id, payload)
    original_source = str(payload.get("source") or "").strip() or None
    model_name = str(
        payload.get("model_name")
        or payload.get("model")
        or payload.get("display_name")
        or canonical_id
    ).strip()
    payload["id"] = canonical_id
    payload["provider"] = provider_id or source_id
    payload.setdefault("provider_id", provider_id or source_id)
    payload["source"] = source_id
    if original_source and original_source != source_id:
        payload["source_origin"] = original_source
    elif source_origin:
        payload["source_origin"] = source_origin
    payload["model_name"] = model_name
    payload["capabilities"] = _capabilities_for_row(provider_id or source_id, payload)
    payload["local"] = bool(payload.get("local", False))
    payload["installable"] = _installable_for_row(source_id, payload)
    payload["confidence"] = _confidence_for_row(source_id, payload)
    payload["query_match"] = _row_matches_query(payload, query)
    return payload


def _query_huggingface(
    query: str | None,
    *,
    fetch_json: Callable[[str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fetch = fetch_json or _fetch_json
    search_query = str(query or "").strip()
    if search_query:
        url = f"https://huggingface.co/api/models?search={urllib.parse.quote(search_query, safe='')}"
    else:
        url = "https://huggingface.co/api/trending?type=model"
    try:
        payload = fetch(url)
    except Exception as exc:
        return [], {
            "source": "huggingface",
            "enabled": True,
            "queried": True,
            "ok": False,
            "count": 0,
            "error_kind": "fetch_failed",
            "error": str(exc) or "hf_fetch_failed",
        }

    rows = payload if isinstance(payload, list) else (payload.get("models") if isinstance(payload, dict) else [])
    normalized: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        repo_id = str(row.get("id") or row.get("repo_id") or row.get("modelId") or "").strip()
        if not repo_id:
            continue
        tags = _normalized_strings(row.get("tags"))
        pipeline_tag = str(row.get("pipeline_tag") or "").strip().lower()
        capabilities: list[str] = []
        if "embed" in repo_id.lower() or pipeline_tag in {"feature-extraction", "sentence-similarity"}:
            capabilities.append("embedding")
        elif pipeline_tag in {"text-generation", "text2text-generation", "chat-completion"}:
            capabilities.append("chat")
        elif pipeline_tag in {"image-to-text", "image-generation", "zero-shot-image-classification"}:
            capabilities.extend(["image", "vision"])
        elif pipeline_tag:
            capabilities.append("chat")
        if "tool" in pipeline_tag or any("tool" in tag for tag in tags):
            capabilities.append("tools")
        if "vision" in tags or "image" in tags:
            capabilities.extend(["image", "vision"])
        if not capabilities:
            capabilities.append("chat")
        normalized.append(
            _normalize_result_row(
                source_id="huggingface",
                row={
                    **row,
                    "id": repo_id,
                    "provider_id": "huggingface",
                    "model": repo_id,
                    "model_name": repo_id,
                    "capabilities": capabilities,
                    "local": False,
                    "available": True,
                    "installable": False,
                    "source": "huggingface_models_search" if search_query else "huggingface_trending",
                },
                query=query,
            )
        )

    normalized.sort(key=lambda item: str(item.get("id") or ""))
    return normalized, {
        "source": "huggingface",
        "enabled": True,
        "queried": True,
        "ok": True,
        "count": len(normalized),
        "error_kind": None,
        "error": None,
    }


def _fetch_json(url: str, *, headers: Mapping[str, str] | None = None, timeout_seconds: float = 20.0) -> Any:
    request = urllib.request.Request(url, method="GET", headers=dict(headers or {"Accept": "application/json"}))
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw or "null")


def _openrouter_api_key(runtime: Any, registry_document: Mapping[str, Any], secret_lookup: Callable[[str], str | None] | None) -> str | None:
    config = getattr(runtime, "config", None)
    config_key = str(getattr(config, "openrouter_api_key", "") or "").strip()
    if config_key:
        return config_key
    provider_payload = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
    openrouter = provider_payload.get("openrouter") if isinstance(provider_payload.get("openrouter"), dict) else None
    source = openrouter.get("api_key_source") if isinstance(openrouter, dict) and isinstance(openrouter.get("api_key_source"), dict) else None
    if isinstance(source, dict):
        source_type = str(source.get("type") or "").strip().lower()
        source_name = str(source.get("name") or "").strip()
        if source_type == "env" and source_name:
            env_value = str(os.getenv(source_name, "")).strip()
            if env_value:
                return env_value
        if source_type == "secret" and source_name and callable(secret_lookup):
            secret_value = str(secret_lookup(source_name) or "").strip()
            if secret_value:
                return secret_value
    env_value = str(os.getenv("OPENROUTER_API_KEY", "")).strip()
    return env_value or None


def _query_openrouter(
    *,
    runtime: Any,
    registry_document: Mapping[str, Any],
    query: str | None,
    secret_lookup: Callable[[str], str | None] | None,
    fetch_json: Callable[[str], Any] | None = None,
    source_name: str = "openrouter",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fetch = fetch_json or _fetch_json
    providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
    provider_payload = providers.get("openrouter") if isinstance(providers.get("openrouter"), dict) else None
    if not isinstance(provider_payload, dict):
        return [], {
            "source": source_name,
            "enabled": False,
            "queried": False,
            "ok": False,
            "count": 0,
            "error_kind": "provider_missing",
            "error": "openrouter provider missing from registry",
        }
    if not bool(provider_payload.get("enabled", True)):
        return [], {
            "source": source_name,
            "enabled": False,
            "queried": False,
            "ok": False,
            "count": 0,
            "error_kind": "provider_disabled",
            "error": "openrouter provider disabled",
        }
    api_key = _openrouter_api_key(runtime, registry_document, secret_lookup)
    if not api_key:
        return [], {
            "source": source_name,
            "enabled": False,
            "queried": False,
            "ok": False,
            "count": 0,
            "error_kind": "missing_api_key",
            "error": "openrouter api key missing",
        }

    try:
        raw_payload = fetch(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout_seconds=6.0,
        )
    except Exception as exc:
        return [], {
            "source": source_name,
            "enabled": True,
            "queried": True,
            "ok": False,
            "count": 0,
            "error_kind": "fetch_failed",
            "error": str(exc) or "openrouter_fetch_failed",
        }

    snapshot = build_openrouter_snapshot(
        raw_payload=raw_payload,
        fetched_at=int(time.time()),
        source="openrouter_models",
    )
    normalized = []
    for row in snapshot.get("models") if isinstance(snapshot.get("models"), list) else []:
        if not isinstance(row, dict):
            continue
        normalized.append(
            _normalize_result_row(
                source_id=source_name,
                row={
                    **row,
                    "available": True,
                    "local": False,
                    "installable": False,
                },
                query=query,
                source_origin="openrouter_models",
            )
        )
    normalized.sort(key=lambda item: str(item.get("id") or ""))
    return normalized, {
        "source": source_name,
        "enabled": True,
        "queried": True,
        "ok": True,
        "count": len(normalized),
        "error_kind": None,
        "error": None,
    }


def _query_ollama(query: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        discovered = list_models_ollama()
    except Exception as exc:
        return [], {
            "source": "ollama",
            "enabled": True,
            "queried": True,
            "ok": False,
            "count": 0,
            "error_kind": "fetch_failed",
            "error": str(exc) or "ollama_fetch_failed",
        }

    normalized = []
    for item in discovered:
        row = dict(item.__dict__) if hasattr(item, "__dict__") else {}
        model_id = str(row.get("model_id") or row.get("id") or "").strip()
        if not model_id:
            continue
        normalized.append(
            _normalize_result_row(
                source_id="ollama",
                row={
                    **row,
                    "id": model_id,
                    "provider_id": "ollama",
                    "provider": "ollama",
                    "model": model_id,
                    "model_name": model_id,
                    "local": True,
                    "available": True,
                    "installable": True,
                    "source": "ollama_list",
                    "confidence": row.get("confidence") or 0.9,
                },
                query=query,
                source_origin="ollama_list",
            )
        )
    normalized.sort(key=lambda item: str(item.get("id") or ""))
    return normalized, {
        "source": "ollama",
        "enabled": True,
        "queried": True,
        "ok": True,
        "count": len(normalized),
        "error_kind": None,
        "error": None,
    }


def _query_external_snapshots(
    *,
    runtime: Any,
    query: str | None,
    filters: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = getattr(runtime, "config", None)
    explicit_path = str(filters.get("external_snapshot_path") or "").strip()
    snapshot_path = Path(explicit_path).expanduser().resolve() if explicit_path else snapshot_path_for_config(config)
    if not snapshot_path.is_file():
        return [], {
            "source": "external_snapshots",
            "enabled": False,
            "queried": False,
            "ok": False,
            "count": 0,
            "error_kind": "snapshot_path_missing",
            "error": f"snapshot not found at {snapshot_path}",
        }

    provider_filter = _as_string_list(filters.get("provider_ids"))
    provider_id = provider_filter[0].strip().lower() if len(provider_filter) == 1 else None
    if provider_id and provider_id != "openrouter":
        return [], {
            "source": "external_snapshots",
            "enabled": False,
            "queried": False,
            "ok": False,
            "count": 0,
            "error_kind": "unsupported_provider_filter",
            "error": f"external snapshots only support openrouter rows, not {provider_id}",
        }

    rows_payload = load_external_model_discovery_rows(
        provider_id=provider_id,
        openrouter_snapshot_path=snapshot_path,
    )
    rows = rows_payload.get("models") if isinstance(rows_payload.get("models"), list) else []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            _normalize_result_row(
                source_id="external_snapshots",
                row={
                    **row,
                    "source_origin": row.get("source"),
                    "available": True,
                    "local": False,
                    "installable": True,
                    "confidence": row.get("confidence") or (0.65 if bool(row.get("review_required", False)) else 0.75),
                },
                query=query,
                source_origin="external_openrouter_snapshot",
            )
        )
    normalized.sort(key=lambda item: str(item.get("id") or ""))
    return normalized, {
        "source": "external_snapshots",
        "enabled": True,
        "queried": True,
        "ok": True,
        "count": len(normalized),
        "error_kind": None,
        "error": None,
        "snapshot_path": str(snapshot_path),
    }


def _source_summary_text(statuses: Sequence[Mapping[str, Any]]) -> str:
    failed = [row for row in statuses if not bool(row.get("ok", False))]
    if failed:
        parts = []
        for row in failed[:3]:
            source = str(row.get("source") or "unknown")
            error = str(row.get("error") or row.get("error_kind") or "unknown_error")
            parts.append(f"{source}: {error}")
        return "; ".join(parts)
    return ""


def _actionable_message(
    *,
    query: str | None,
    statuses: Sequence[Mapping[str, Any]],
    matched_count: int,
    enabled_sources: Sequence[str],
    filtered_out_count: int,
) -> str:
    queried = [row for row in statuses if bool(row.get("queried", False))]
    failed = [row for row in statuses if not bool(row.get("ok", False))]
    if not queried:
        return (
            "No discovery sources are enabled. "
            "Enable Hugging Face/OpenRouter/Ollama in the registry or point AGENT_MODEL_WATCH_CATALOG_PATH at a snapshot."
        )
    if queried and all(not bool(row.get("ok", False)) for row in queried):
        return (
            "No discovery source returned usable data. "
            f"Check the source configuration and network/auth setup. "
            f"Source errors: {_source_summary_text(statuses)}."
        )
    if matched_count > 0:
        source_count = len({str(row.get("source") or "") for row in statuses if bool(row.get("ok", False))})
        if failed:
            return (
                f"Found {matched_count} model(s) across {source_count} source(s), but some sources failed. "
                f"Source errors: {_source_summary_text(statuses)}."
            )
        return f"Found {matched_count} model(s) across {source_count} source(s)."
    if filtered_out_count > 0:
        return (
            f"No models matched '{query or ''}'. "
            "Try broader terms or remove filters."
        )
    if failed:
        return (
            f"No models were returned for '{query or ''}'. "
            f"Source errors: {_source_summary_text(statuses)}."
        )
    return (
        f"No models matched '{query or ''}'. "
        "Try broader terms or check whether the queried sources are enabled."
    )


@dataclass(frozen=True)
class ModelDiscoveryQueryResult:
    ok: bool
    query: str | None
    message: str
    models: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    debug: dict[str, Any]


class ModelDiscoveryManager:
    def __init__(
        self,
        *,
        runtime: Any,
        secret_lookup: Callable[[str], str | None] | None = None,
        fetch_json: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.secret_lookup = secret_lookup
        self._fetch_json = fetch_json or _fetch_json

    @classmethod
    def source_registry(cls) -> tuple[str, ...]:
        return _SOURCE_IDS

    def _config(self) -> Any:
        return getattr(self.runtime, "config", None)

    def _registry_document(self) -> dict[str, Any]:
        registry_document = getattr(self.runtime, "registry_document", None)
        if isinstance(registry_document, dict):
            return dict(registry_document)
        config = self._config()
        registry_path = str(getattr(config, "llm_registry_path", "") or "").strip() or None
        if registry_path:
            return load_registry_document(registry_path)
        return {}

    def _query_huggingface(self, query: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _query_huggingface(query, fetch_json=self._fetch_json)

    def _query_openrouter(
        self,
        query: str | None,
        *,
        registry_document: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _query_openrouter(
            runtime=self.runtime,
            registry_document=registry_document,
            query=query,
            secret_lookup=self.secret_lookup,
            fetch_json=self._fetch_json,
        )

    def _query_ollama(self, query: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _query_ollama(query)

    def _query_external_snapshots(
        self,
        query: str | None,
        *,
        filters: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _query_external_snapshots(
            runtime=self.runtime,
            query=query,
            filters=filters,
        )

    def query(self, query: str | None, filters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        normalized_filters = dict(filters) if isinstance(filters, Mapping) else {}
        selected_sources = _as_string_list(normalized_filters.get("sources") or normalized_filters.get("source"))
        selected_sources = [
            _SOURCE_ALIASES.get(source.lower(), source.lower())
            for source in selected_sources
            if source
        ]
        if not selected_sources:
            selected_sources = list(_SOURCE_IDS)

        registry_document = self._registry_document()
        source_rows: list[dict[str, Any]] = []
        source_statuses: list[dict[str, Any]] = []

        for source_id in _SOURCE_IDS:
            if source_id not in selected_sources:
                source_statuses.append(
                    {
                        "source": source_id,
                        "enabled": False,
                        "queried": False,
                        "ok": True,
                        "count": 0,
                        "error_kind": "not_requested",
                        "error": None,
                    }
                )
                continue

            if source_id == "huggingface":
                rows, status = self._query_huggingface(query)
            elif source_id == "openrouter":
                rows, status = self._query_openrouter(query, registry_document=registry_document)
            elif source_id == "ollama":
                rows, status = self._query_ollama(query)
            else:
                rows, status = self._query_external_snapshots(query, filters=normalized_filters)

            source_rows.extend(rows)
            source_statuses.append(dict(status))

        seen_ids: set[str] = set()
        merged_rows: list[dict[str, Any]] = []
        for row in source_rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "").strip()
            if not model_id or model_id in seen_ids:
                continue
            if not _row_matches_query(row, query):
                continue
            if not _row_matches_filters(row, normalized_filters):
                continue
            seen_ids.add(model_id)
            merged_rows.append(dict(row))

        merged_rows.sort(key=lambda row: (str(row.get("source") or ""), str(row.get("provider") or ""), str(row.get("id") or "")))

        limit = _safe_int(normalized_filters.get("limit"))
        if limit is not None:
            merged_rows = merged_rows[:limit]

        enabled_sources = [row.get("source") for row in source_statuses if bool(row.get("queried", False)) and bool(row.get("ok", False))]
        filtered_out_count = max(0, len(source_rows) - len(merged_rows))
        matched_count = len(merged_rows)
        message = _actionable_message(
            query=query,
            statuses=source_statuses,
            matched_count=matched_count,
            enabled_sources=[str(item) for item in enabled_sources if str(item).strip()],
            filtered_out_count=filtered_out_count,
        )
        any_source_ok = any(bool(row.get("ok", False)) for row in source_statuses if bool(row.get("queried", False)))
        if not any_source_ok and not merged_rows:
            ok = False
        else:
            ok = True

        debug = {
            "query": query,
            "filters": normalized_filters,
            "source_statuses": source_statuses,
            "source_errors": {
                str(row.get("source") or f"source_{idx}"): {
                    "error_kind": row.get("error_kind"),
                    "error": row.get("error"),
                }
                for idx, row in enumerate(source_statuses)
                if not bool(row.get("ok", False))
            },
            "source_counts": {
                str(row.get("source") or f"source_{idx}"): int(row.get("count") or 0)
                for idx, row in enumerate(source_statuses)
                if bool(row.get("queried", False))
            },
            "source_registry": list(_SOURCE_IDS),
            "matched_count": len(merged_rows),
        }

        return {
            "ok": ok,
            "query": query,
            "message": message,
            "models": merged_rows,
            "sources": source_statuses,
            "debug": debug,
        }


__all__ = [
    "ModelDiscoveryManager",
    "ModelDiscoveryQueryResult",
]
