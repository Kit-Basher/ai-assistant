from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from agent.llm.capabilities import capability_list_from_inference, infer_capabilities_from_catalog
from agent.llm.known_model_metadata import known_model_metadata


_CATALOG_SCHEMA_VERSION = 1


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _rounded_cost(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return round(parsed, 6)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _openrouter_cost_per_million(*, per_token: Any, per_million: Any) -> float | None:
    explicit_per_million = _rounded_cost(per_million)
    if explicit_per_million is not None:
        return explicit_per_million
    parsed_per_token = _safe_float(per_token)
    if parsed_per_token is None:
        return None
    return round(parsed_per_token * 1_000_000.0, 6)


def _iso_from_epoch(epoch: int | None) -> str | None:
    if epoch is None or int(epoch) <= 0:
        return None
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()


def _normalize_capabilities(raw: Any) -> list[str]:
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    normalized = sorted(
        {
            str(item).strip().lower()
            for item in values
            if str(item).strip()
        }
    )
    if not normalized:
        return ["chat"]
    return normalized


def _normalize_labels(raw: Any) -> list[str]:
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    return sorted(
        {
            str(item).strip().lower()
            for item in values
            if str(item).strip()
        }
    )


def _normalize_task_types(raw: Any) -> list[str]:
    return _normalize_labels(raw)


def _normalize_modalities(raw: Any) -> list[str]:
    return _normalize_labels(raw)


def _normalize_modality(raw: Any) -> str | None:
    value = str(raw or "").strip().lower()
    return value or None


def normalize_catalog_entry(
    provider_id: str,
    model_name: str,
    *,
    capabilities: Any,
    task_types: Any = None,
    architecture_modality: Any = None,
    input_modalities: Any = None,
    output_modalities: Any = None,
    max_context_tokens: Any,
    input_cost_per_million_tokens: Any,
    output_cost_per_million_tokens: Any,
    source: str,
) -> dict[str, Any]:
    provider = str(provider_id or "").strip().lower()
    model = str(model_name or "").strip()
    max_context = _safe_int(max_context_tokens)
    if max_context is not None and max_context <= 0:
        max_context = None
    return {
        "id": f"{provider}:{model}",
        "provider_id": provider,
        "model": model,
        "capabilities": _normalize_capabilities(capabilities),
        "task_types": _normalize_task_types(task_types) if task_types is not None else [],
        "architecture_modality": _normalize_modality(architecture_modality),
        "input_modalities": _normalize_modalities(input_modalities),
        "output_modalities": _normalize_modalities(output_modalities),
        "max_context_tokens": max_context,
        "input_cost_per_million_tokens": _rounded_cost(input_cost_per_million_tokens),
        "output_cost_per_million_tokens": _rounded_cost(output_cost_per_million_tokens),
        "source": str(source or "manual").strip() or "manual",
    }


def _http_get_json_with_policy(
    url: str,
    *,
    headers: dict[str, str],
    timeout_seconds: float,
    allowed_hosts: set[str],
) -> dict[str, Any]:
    parsed_url = urllib.parse.urlparse(url)
    host = (parsed_url.hostname or "").strip().lower()
    normalized_hosts = {str(item).strip().lower() for item in allowed_hosts if str(item).strip()}
    if host and host not in normalized_hosts:
        raise RuntimeError("domain_not_allowed")

    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"http_{int(exc.code)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("unreachable") from exc

    try:
        parsed = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid_json") from exc

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"data": parsed}
    return {}


def _capabilities_from_ollama(model_name: str, row: dict[str, Any]) -> list[str]:
    name = str(model_name or "").strip().lower()
    if "embed" in name:
        return ["embedding"]
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    family = str(details.get("family") or "").strip().lower()
    families = details.get("families") if isinstance(details.get("families"), list) else []
    family_values = {family} | {str(item).strip().lower() for item in families if str(item).strip()}
    if any("embed" in item for item in family_values):
        return ["embedding"]
    return ["chat"]


def _capabilities_from_remote_model_id(model_id: str) -> list[str]:
    value = str(model_id or "").strip().lower()
    if "embed" in value:
        return ["embedding"]
    caps = {"chat"}
    if "vision" in value or "image" in value:
        caps.add("image")
    return sorted(caps)


def _openrouter_capabilities(row: dict[str, Any]) -> list[str]:
    model_id = str(row.get("id") or "").strip()
    caps = set(_capabilities_from_remote_model_id(model_id))
    architecture = row.get("architecture") if isinstance(row.get("architecture"), dict) else {}
    modality = str(architecture.get("modality") or "").strip().lower()
    if "embed" in modality:
        return ["embedding"]
    if "image" in modality:
        caps.add("image")
    input_modalities = architecture.get("input_modalities") if isinstance(architecture.get("input_modalities"), list) else []
    output_modalities = architecture.get("output_modalities") if isinstance(architecture.get("output_modalities"), list) else []
    modality_values = {
        str(item).strip().lower()
        for item in list(input_modalities) + list(output_modalities)
        if str(item).strip()
    }
    if "image" in modality_values:
        caps.add("image")
        caps.add("vision")
    if "image" in modality:
        caps.add("image")
        caps.add("vision")
    if "embedding" in modality_values:
        return ["embedding"]
    return sorted(caps)


def _openrouter_modality_payload(row: dict[str, Any]) -> dict[str, Any]:
    architecture = row.get("architecture") if isinstance(row.get("architecture"), dict) else {}
    return {
        "architecture_modality": architecture.get("modality"),
        "input_modalities": architecture.get("input_modalities"),
        "output_modalities": architecture.get("output_modalities"),
    }


def _inferred_capabilities(
    provider_id: str,
    model_name: str,
    *,
    source_caps: list[str],
) -> list[str]:
    inferred = infer_capabilities_from_catalog(
        provider_id,
        {
            "id": f"{provider_id}:{model_name}",
            "provider_id": provider_id,
            "model": model_name,
            "capabilities": source_caps,
        },
    )
    return capability_list_from_inference(inferred)


def _provider_auth_header(cfg: dict[str, Any]) -> dict[str, str]:
    source = cfg.get("api_key_source") if isinstance(cfg.get("api_key_source"), dict) else None
    if not isinstance(source, dict):
        return {}
    source_type = str(source.get("type") or "").strip().lower()
    source_name = str(source.get("name") or "").strip()
    if not source_name:
        return {}
    if source_type == "env":
        key = (os.environ.get(source_name, "") or "").strip()
    else:
        # Secret-store lookup is runtime-owned; catalog fetch receives pre-resolved
        # headers through cfg["resolved_headers"] when available.
        key = ""
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _catalog_result_error(provider_id: str, source: str, error_kind: str) -> dict[str, Any]:
    return {
        "ok": False,
        "provider_id": str(provider_id or "").strip().lower(),
        "source": str(source or "manual").strip() or "manual",
        "models": [],
        "error_kind": str(error_kind or "unknown_error"),
    }


def fetch_provider_catalog(
    provider_id: str,
    cfg: dict[str, Any],
    http: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    provider = str(provider_id or "").strip().lower()
    if not provider or not isinstance(cfg, dict):
        return _catalog_result_error(provider, "manual", "invalid_provider_config")

    base_url = str(cfg.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        return _catalog_result_error(provider, "manual", "base_url_missing")

    parsed = urllib.parse.urlparse(base_url)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return _catalog_result_error(provider, "manual", "bad_base_url")

    timeout_seconds = float(cfg.get("timeout_seconds") or 6.0)
    request_headers = {"Accept": "application/json"}
    resolved_headers = cfg.get("resolved_headers") if isinstance(cfg.get("resolved_headers"), dict) else None
    if isinstance(resolved_headers, dict):
        request_headers.update(
            {
                str(key): str(value)
                for key, value in resolved_headers.items()
                if str(key).strip() and str(value).strip()
            }
        )
    else:
        request_headers.update(_provider_auth_header(cfg))

    allowed_hosts = {host}
    if provider == "openrouter":
        allowed_hosts.add("openrouter.ai")
    if provider == "openai":
        allowed_hosts.add("api.openai.com")

    source = "manual"
    try:
        if provider == "ollama" or bool(cfg.get("local", False)):
            source = "ollama_tags"
            payload = http(
                base_url + "/api/tags",
                headers=request_headers,
                timeout_seconds=timeout_seconds,
                allowed_hosts=allowed_hosts,
            )
            rows = payload.get("models") if isinstance(payload.get("models"), list) else []
            output: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                model_name = str(row.get("name") or "").strip()
                if not model_name:
                    continue
                details = row.get("details") if isinstance(row.get("details"), dict) else {}
                output.append(
                    normalize_catalog_entry(
                        provider,
                        model_name,
                        capabilities=_inferred_capabilities(
                            provider,
                            model_name,
                            source_caps=_capabilities_from_ollama(model_name, row),
                        ),
                        task_types=known_model_metadata(provider, model_name).get("task_types"),
                        architecture_modality=None,
                        input_modalities=None,
                        output_modalities=None,
                        max_context_tokens=details.get("context_length")
                        or details.get("num_ctx")
                        or row.get("context_length"),
                        input_cost_per_million_tokens=None,
                        output_cost_per_million_tokens=None,
                        source=source,
                    )
                )
            output.sort(key=lambda item: str(item.get("id") or ""))
            return {
                "ok": True,
                "provider_id": provider,
                "source": source,
                "models": output,
                "error_kind": None,
            }

        if provider == "openrouter":
            source = "openrouter_models"
            payload = http(
                base_url + "/models",
                headers=request_headers,
                timeout_seconds=timeout_seconds,
                allowed_hosts=allowed_hosts,
            )
            rows = payload.get("data") if isinstance(payload.get("data"), list) else []
            output = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                model_name = str(row.get("id") or "").strip()
                if not model_name:
                    continue
                pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
                modality_payload = _openrouter_modality_payload(row)
                output.append(
                    normalize_catalog_entry(
                        provider,
                        model_name,
                        capabilities=_inferred_capabilities(
                            provider,
                            model_name,
                            source_caps=_openrouter_capabilities(row),
                        ),
                        task_types=known_model_metadata(provider, model_name).get("task_types"),
                        architecture_modality=modality_payload.get("architecture_modality"),
                        input_modalities=modality_payload.get("input_modalities"),
                        output_modalities=modality_payload.get("output_modalities"),
                        max_context_tokens=row.get("context_length") or row.get("max_context_length"),
                        input_cost_per_million_tokens=_openrouter_cost_per_million(
                            per_token=_first_present(
                                pricing.get("prompt"),
                                pricing.get("input"),
                            ),
                            per_million=pricing.get("input_per_million_tokens"),
                        ),
                        output_cost_per_million_tokens=_openrouter_cost_per_million(
                            per_token=_first_present(
                                pricing.get("completion"),
                                pricing.get("output"),
                            ),
                            per_million=pricing.get("output_per_million_tokens"),
                        ),
                        source=source,
                    )
                )
            output.sort(key=lambda item: str(item.get("id") or ""))
            return {
                "ok": True,
                "provider_id": provider,
                "source": source,
                "models": output,
                "error_kind": None,
            }

        source = "openai_models"
        payload = http(
            base_url + "/v1/models",
            headers=request_headers,
            timeout_seconds=timeout_seconds,
            allowed_hosts=allowed_hosts,
        )
        rows = payload.get("data") if isinstance(payload.get("data"), list) else []
        output = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_name = str(row.get("id") or "").strip()
            if not model_name:
                continue
            output.append(
                normalize_catalog_entry(
                    provider,
                    model_name,
                    capabilities=_inferred_capabilities(
                        provider,
                        model_name,
                        source_caps=_capabilities_from_remote_model_id(model_name),
                    ),
                    task_types=known_model_metadata(provider, model_name).get("task_types"),
                    architecture_modality=None,
                    input_modalities=None,
                    output_modalities=None,
                    max_context_tokens=None,
                    input_cost_per_million_tokens=None,
                    output_cost_per_million_tokens=None,
                    source=source,
                )
            )
        output.sort(key=lambda item: str(item.get("id") or ""))
        return {
            "ok": True,
            "provider_id": provider,
            "source": source,
            "models": output,
            "error_kind": None,
        }
    except RuntimeError as exc:
        return _catalog_result_error(provider, source, str(exc) or "catalog_fetch_failed")
    except (OSError, ValueError, TypeError, KeyError):
        return _catalog_result_error(provider, source, "catalog_fetch_failed")


class CatalogStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.state = self.load()

    @staticmethod
    def default_path() -> str:
        env_path = os.getenv("LLM_CATALOG_PATH", "").strip()
        if env_path:
            return env_path
        return str(Path.home() / ".local" / "share" / "personal-agent" / "llm_catalog.json")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "schema_version": _CATALOG_SCHEMA_VERSION,
            "last_run_at": None,
            "providers": {},
        }

    def _normalize_provider_row(self, provider_id: str, row: dict[str, Any]) -> dict[str, Any]:
        models = row.get("models") if isinstance(row.get("models"), list) else []
        normalized_models: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in models:
            if not isinstance(item, dict):
                continue
            model_name = str(item.get("model") or "").strip()
            if not model_name:
                continue
            normalized = normalize_catalog_entry(
                provider_id,
                model_name,
                capabilities=item.get("capabilities"),
                task_types=item.get("task_types"),
                architecture_modality=item.get("architecture_modality"),
                input_modalities=item.get("input_modalities"),
                output_modalities=item.get("output_modalities"),
                max_context_tokens=item.get("max_context_tokens"),
                input_cost_per_million_tokens=item.get("input_cost_per_million_tokens"),
                output_cost_per_million_tokens=item.get("output_cost_per_million_tokens"),
                source=str(item.get("source") or row.get("source") or "manual"),
            )
            model_id = str(normalized.get("id") or "")
            if not model_id or model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            normalized_models.append(normalized)
        normalized_models.sort(key=lambda item: str(item.get("id") or ""))
        return {
            "provider_id": provider_id,
            "source": str(row.get("source") or "manual").strip() or "manual",
            "last_refresh_at": _safe_int(row.get("last_refresh_at")) or None,
            "last_error_kind": str(row.get("last_error_kind") or "").strip() or None,
            "last_error_at": _safe_int(row.get("last_error_at")) or None,
            "models": normalized_models,
        }

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.empty_state()
        state["schema_version"] = _CATALOG_SCHEMA_VERSION
        state["last_run_at"] = _safe_int(payload.get("last_run_at")) or None
        providers_raw = payload.get("providers") if isinstance(payload.get("providers"), dict) else {}
        providers: dict[str, Any] = {}
        for provider_id_raw, row in sorted(providers_raw.items()):
            provider_id = str(provider_id_raw).strip().lower()
            if not provider_id or not isinstance(row, dict):
                continue
            providers[provider_id] = self._normalize_provider_row(provider_id, row)
        state["providers"] = providers
        return state

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self.empty_state()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self.empty_state()
        if not isinstance(raw, dict):
            return self.empty_state()
        normalized = self._normalize(raw)
        raw_canonical = json.dumps(raw, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        normalized_canonical = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if raw_canonical != normalized_canonical:
            try:
                self._write(normalized)
            except OSError:
                pass
        return normalized

    def _write(self, normalized: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(state if isinstance(state, dict) else {})
        self._write(normalized)
        self.state = normalized
        return normalized

    def update_provider_result(
        self,
        provider_id: str,
        result: dict[str, Any],
        *,
        now_epoch: int,
    ) -> dict[str, Any]:
        provider = str(provider_id or "").strip().lower()
        if not provider:
            return self.state
        current = json.loads(json.dumps(self.state, ensure_ascii=True))
        providers = current.get("providers") if isinstance(current.get("providers"), dict) else {}
        existing = providers.get(provider) if isinstance(providers.get(provider), dict) else {}
        row = {
            "provider_id": provider,
            "source": str(result.get("source") or existing.get("source") or "manual").strip() or "manual",
            "last_refresh_at": int(now_epoch) if bool(result.get("ok")) else _safe_int(existing.get("last_refresh_at")),
            "last_error_kind": None,
            "last_error_at": None,
            "models": existing.get("models") if isinstance(existing.get("models"), list) else [],
        }
        if bool(result.get("ok")):
            row["models"] = result.get("models") if isinstance(result.get("models"), list) else []
        else:
            row["last_error_kind"] = str(result.get("error_kind") or "catalog_fetch_failed").strip()
            row["last_error_at"] = int(now_epoch)
        providers[provider] = row
        current["providers"] = providers
        current["last_run_at"] = int(now_epoch)
        return self.save(current)

    def all_models(self, *, provider_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        providers = self.state.get("providers") if isinstance(self.state.get("providers"), dict) else {}
        target = str(provider_id or "").strip().lower() or None
        rows: list[dict[str, Any]] = []
        for pid in sorted(providers.keys()):
            if target and pid != target:
                continue
            provider_row = providers.get(pid) if isinstance(providers.get(pid), dict) else {}
            models = provider_row.get("models") if isinstance(provider_row.get("models"), list) else []
            for model in models:
                if isinstance(model, dict):
                    rows.append(dict(model))
        rows.sort(key=lambda item: str(item.get("id") or ""))
        if limit is not None:
            rows = rows[: max(1, int(limit))]
        return rows

    def provider_models(self, provider_id: str) -> list[dict[str, Any]]:
        provider = str(provider_id or "").strip().lower()
        if not provider:
            return []
        providers = self.state.get("providers") if isinstance(self.state.get("providers"), dict) else {}
        row = providers.get(provider) if isinstance(providers.get(provider), dict) else {}
        models = row.get("models") if isinstance(row.get("models"), list) else []
        output = [dict(item) for item in models if isinstance(item, dict)]
        output.sort(key=lambda item: str(item.get("id") or ""))
        return output

    def status(self) -> dict[str, Any]:
        providers = self.state.get("providers") if isinstance(self.state.get("providers"), dict) else {}
        rows = []
        for provider_id in sorted(providers.keys()):
            row = providers.get(provider_id) if isinstance(providers.get(provider_id), dict) else {}
            last_refresh_at = _safe_int(row.get("last_refresh_at")) or None
            last_error_at = _safe_int(row.get("last_error_at")) or None
            models = row.get("models") if isinstance(row.get("models"), list) else []
            rows.append(
                {
                    "provider_id": provider_id,
                    "source": str(row.get("source") or "manual"),
                    "models_count": len(models),
                    "last_refresh_at": last_refresh_at,
                    "last_refresh_at_iso": _iso_from_epoch(last_refresh_at),
                    "last_error_kind": str(row.get("last_error_kind") or "").strip() or None,
                    "last_error_at": last_error_at,
                    "last_error_at_iso": _iso_from_epoch(last_error_at),
                }
            )
        last_run_at = _safe_int(self.state.get("last_run_at")) or None
        return {
            "last_run_at": last_run_at,
            "last_run_at_iso": _iso_from_epoch(last_run_at),
            "providers": rows,
        }

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.state, ensure_ascii=True))


__all__ = [
    "CatalogStore",
    "fetch_provider_catalog",
    "normalize_catalog_entry",
    "_http_get_json_with_policy",
]
