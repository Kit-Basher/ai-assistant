from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import urllib.parse
import urllib.request

from agent.llm.registry import load_registry_document
from agent.model_recommendation import RecommendationContext, pick_recommendation, rank_candidates
from agent.model_watch import (
    ModelWatchStore,
    evaluate_model_watch_candidates,
    latest_model_watch_batch,
    load_model_watch_config,
    model_watch_last_run_at,
    normalize_model_watch_state,
    set_model_watch_last_run_at,
    summarize_model_watch_batch,
    upsert_model_watch_batch,
)
from agent.model_watch_catalog import build_feature_index, load_latest_snapshot, snapshot_path_for_config


def _http_get_json(url: str, *, timeout_seconds: float = 20.0, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, method="GET", headers=headers or {})
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw or "null")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _watch_config_path_for_config(config: Any) -> str:
    configured = str(getattr(config, "model_watch_config_path", "") or "").strip()
    if configured:
        return configured
    env_path = os.getenv("AGENT_MODEL_WATCH_CONFIG_PATH", "").strip()
    if env_path:
        return env_path
    return str((_repo_root() / "model_watch_config.json").resolve())


def _extract_installed_model_names(tags_payload: Any) -> set[str]:
    payload = tags_payload if isinstance(tags_payload, dict) else {}
    models = payload.get("models") if isinstance(payload.get("models"), list) else []
    output: set[str] = set()
    for row in models:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name:
            output.add(name)
    return output


def _discover_installed_ollama_models(
    registry_document: dict[str, Any],
    *,
    fetch_json: Callable[[str], Any],
) -> tuple[set[str], str | None]:
    providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
    ollama = providers.get("ollama") if isinstance(providers.get("ollama"), dict) else None
    if not isinstance(ollama, dict):
        return set(), "ollama_provider_missing"
    if not bool(ollama.get("enabled", True)):
        return set(), "ollama_provider_disabled"
    base_url = str(ollama.get("base_url") or "http://127.0.0.1:11434").strip().rstrip("/")
    if not base_url:
        base_url = "http://127.0.0.1:11434"
    try:
        parsed = fetch_json(base_url + "/api/tags")
    except Exception:
        return set(), "ollama_unreachable"
    return _extract_installed_model_names(parsed), None


def _model_watch_candidates(
    authors: tuple[str, ...],
    *,
    fetch_json: Callable[[str], Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for author in authors:
        query = urllib.parse.quote(str(author), safe="")
        url = f"https://huggingface.co/api/models?author={query}"
        try:
            parsed = fetch_json(url)
        except Exception:
            errors.append(f"{author}:fetch_failed")
            continue
        source_rows = parsed if isinstance(parsed, list) else []
        for item in source_rows:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row.setdefault("author", author)
            row.setdefault("install_name", str(row.get("id") or "").strip())
            rows.append(row)
    rows.sort(key=lambda item: str(item.get("id") or ""))
    return rows, errors


def _provider_enabled_set(registry_document: dict[str, Any]) -> frozenset[str]:
    providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
    return frozenset(
        str(provider_id).strip().lower()
        for provider_id, payload in sorted(providers.items())
        if isinstance(payload, dict) and bool(payload.get("enabled", True))
    )


def _openrouter_auth_available(registry_document: dict[str, Any], config: Any) -> bool:
    providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
    openrouter = providers.get("openrouter") if isinstance(providers.get("openrouter"), dict) else None
    if not isinstance(openrouter, dict) or not bool(openrouter.get("enabled", True)):
        return False
    if str(getattr(config, "openrouter_api_key", "") or "").strip():
        return True
    source = openrouter.get("api_key_source") if isinstance(openrouter.get("api_key_source"), dict) else {}
    if isinstance(source, dict) and str(source.get("type") or "").strip().lower() == "env":
        env_name = str(source.get("name") or "").strip()
        if env_name and str(os.getenv(env_name, "")).strip():
            return True
    if str(os.getenv("OPENROUTER_API_KEY", "")).strip():
        return True
    return False


def _batch_id_for_candidates(
    candidate_ids: list[str],
    *,
    snapshot_seed: str | None = None,
) -> str:
    canonical = sorted({str(item).strip() for item in candidate_ids if str(item).strip()})
    if not canonical:
        return ""
    seed = str(snapshot_seed or "").strip()
    payload = f"{seed}|{'|'.join(canonical)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _score_reason_line(candidate: dict[str, Any]) -> str:
    reasons = candidate.get("reasons") if isinstance(candidate.get("reasons"), tuple) else tuple()
    if reasons:
        return str(reasons[0]).strip()
    tradeoffs = candidate.get("tradeoffs") if isinstance(candidate.get("tradeoffs"), tuple) else tuple()
    if tradeoffs:
        return str(tradeoffs[0]).strip()
    return "Deterministic score favored this option."


def _serialize_ranked_row(row: Any) -> dict[str, Any]:
    tradeoffs = [str(item).strip() for item in getattr(row, "tradeoffs", ()) if str(item).strip()]
    missing = [item for item in tradeoffs if item.lower().startswith("missing:")]
    non_missing = [item for item in tradeoffs if not item.lower().startswith("missing:")]
    return {
        "id": str(getattr(row, "canonical_model_id", "") or "").strip(),
        "provider": str(getattr(row, "provider_id", "") or "").strip().lower() or None,
        "model": str(getattr(row, "model_id", "") or "").strip(),
        "score": round(float(getattr(row, "total_score", 0.0) or 0.0), 2),
        "reason": _score_reason_line(row.__dict__ if hasattr(row, "__dict__") else {}),
        "tradeoffs": missing + non_missing[:2],
        "subscores": {
            "task_fit": round(float(getattr(row, "task_fit", 0.0) or 0.0), 2),
            "local_feasibility": round(float(getattr(row, "local_feasibility", 0.0) or 0.0), 2),
            "cost_efficiency": round(float(getattr(row, "cost_efficiency", 0.0) or 0.0), 2),
            "quality_proxy": round(float(getattr(row, "quality_proxy", 0.0) or 0.0), 2),
            "switch_gain": round(float(getattr(row, "switch_gain", 0.0) or 0.0), 2),
        },
    }


def _snapshot_candidate_pool(
    snapshot: dict[str, Any],
    *,
    feature_index: dict[str, dict[str, Any]],
    openrouter_available: bool,
    provider_enabled: frozenset[str],
) -> tuple[list[dict[str, Any]], int]:
    rows = snapshot.get("models") if isinstance(snapshot.get("models"), list) else []
    pool_count = len([row for row in rows if isinstance(row, dict)])
    candidates: list[dict[str, Any]] = []
    for row in sorted([item for item in rows if isinstance(item, dict)], key=lambda item: str(item.get("id") or "")):
        canonical_id = str(row.get("id") or "").strip()
        provider_id = str(row.get("provider_id") or "").strip().lower()
        model_name = str(row.get("model") or "").strip()
        if not canonical_id or not provider_id or not model_name:
            continue
        features = feature_index.get(canonical_id) if isinstance(feature_index.get(canonical_id), dict) else {}
        caps = row.get("capabilities") if isinstance(row.get("capabilities"), list) else []
        capabilities = sorted(
            {
                str(item).strip().lower()
                for item in caps
                if str(item).strip()
            }
        ) or list(features.get("capabilities") or ["chat"])
        availability = True
        if provider_id == "openrouter":
            if not openrouter_available or provider_id not in provider_enabled:
                availability = False
        candidates.append(
            {
                "provider_id": provider_id,
                "model_id": model_name,
                "params_b": features.get("params_b", row.get("params_b")),
                "context_tokens": features.get("context_tokens", row.get("context_length")),
                "price_in": features.get("price_in"),
                "price_out": features.get("price_out"),
                "capabilities": capabilities,
                "local": bool(provider_id == "ollama"),
                "availability": availability,
                "quality_percentile": features.get("quality_percentile", row.get("quality_percentile")),
                "missing_features": list(features.get("missing_features") or []),
            }
        )
    return candidates, int(pool_count)


def top_pick_plan_payload(candidate: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    if not isinstance(candidate, dict):
        return None
    candidate_id = str(candidate.get("id") or "").strip()
    provider = str(candidate.get("provider") or "").strip().lower()
    model = str(candidate.get("model") or "").strip()
    if not provider and ":" in candidate_id:
        provider = candidate_id.split(":", 1)[0].strip().lower()
    if not model and ":" in candidate_id:
        model = candidate_id.split(":", 1)[1].strip()
    if not provider or not model:
        return None
    if provider == "ollama":
        return ({"action": "install", "provider": "ollama", "model": model}, "ollama")
    if provider == "openrouter":
        return (
            {
                "action": "modelops.set_default_model",
                "params": {
                    "default_provider": "openrouter",
                    "default_model": f"openrouter:{model}" if not model.startswith("openrouter:") else model,
                },
            },
            "openrouter",
        )
    return None


def run_model_watch_check(
    config: Any,
    *,
    fetch_json: Callable[[str], Any] | None = None,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    fetch = fetch_json or (lambda url: _http_get_json(url, timeout_seconds=20.0))
    now = int(now_epoch if now_epoch is not None else time.time())

    state_path = str(getattr(config, "model_watch_state_path", "") or "").strip() or None
    state_store = ModelWatchStore(path=state_path)
    state = state_store.load()

    watch_config_path = _watch_config_path_for_config(config)
    watch_config = load_model_watch_config(watch_config_path)

    registry_document = load_registry_document(getattr(config, "llm_registry_path", None))
    installed_models, installed_error = _discover_installed_ollama_models(registry_document, fetch_json=fetch)
    hf_candidates, source_errors = _model_watch_candidates(watch_config.huggingface_watch_authors, fetch_json=fetch)
    next_state, new_recommendations = evaluate_model_watch_candidates(
        state=state,
        candidates=hf_candidates,
        installed_models=installed_models,
        config=watch_config,
        now_epoch=now,
    )

    snapshot_path = snapshot_path_for_config(config)
    snapshot, snapshot_error = load_latest_snapshot(snapshot_path)
    if not isinstance(snapshot, dict):
        saved = state_store.save(set_model_watch_last_run_at(state=next_state, last_run_at=now))
        return {
            "ok": True,
            "found": False,
            "reason": "No model catalog snapshot available; run refresh",
            "batch": None,
            "latest_batch": summarize_model_watch_batch(latest_model_watch_batch(saved)),
            "new_batch_created": False,
            "batch_id": None,
            "fetched_candidates": 0,
            "catalog_models_considered": 0,
            "catalog_snapshot_model_count": 0,
            "catalog_snapshot_error": snapshot_error,
            "catalog_snapshot_path": str(snapshot_path),
            "installed_error": installed_error,
            "source_errors": sorted(source_errors),
            "new_recommendations": len(new_recommendations),
            "last_run_at": model_watch_last_run_at(saved),
        }

    feature_index = build_feature_index(snapshot)
    openrouter_available = _openrouter_auth_available(registry_document, config)
    provider_enabled = _provider_enabled_set(registry_document)
    candidate_rows, catalog_pool_count = _snapshot_candidate_pool(
        snapshot,
        feature_index=feature_index,
        openrouter_available=openrouter_available,
        provider_enabled=provider_enabled,
    )
    ranking = rank_candidates(
        candidate_rows,
        RecommendationContext(
            purpose="chat",
            default_model=str(
                (registry_document.get("defaults") if isinstance(registry_document.get("defaults"), dict) else {}).get(
                    "default_model"
                )
                or ""
            ).strip()
            or None,
            allow_remote_fallback=bool(
                (registry_document.get("defaults") if isinstance(registry_document.get("defaults"), dict) else {}).get(
                    "allow_remote_fallback",
                    True,
                )
            ),
            enabled_providers=provider_enabled,
            vram_gb=None,
        ),
    )
    pick = pick_recommendation(ranking)

    top_rows = list(ranking.ranked[:5])
    presented_rows = [pick.pick] if (pick.pick is not None and pick.show_top_only) else list(ranking.ranked[:3])
    presented_rows = [row for row in presented_rows if row is not None]
    candidate_ids = [str(row.canonical_model_id) for row in presented_rows if str(row.canonical_model_id)]
    snapshot_seed = str(snapshot.get("raw_sha256") or snapshot.get("fetched_at") or "").strip()
    batch_id = _batch_id_for_candidates(candidate_ids, snapshot_seed=snapshot_seed)

    existing_batch = None
    if batch_id:
        for row in next_state.get("batches") if isinstance(next_state.get("batches"), list) else []:
            if isinstance(row, dict) and str(row.get("batch_id") or "") == batch_id:
                existing_batch = row
                break
    new_batch_created = bool(batch_id and existing_batch is None)

    if batch_id and new_batch_created:
        fetched_at = int(snapshot.get("fetched_at") or 0)
        created_at = max(int(now), int(fetched_at))
        next_state = upsert_model_watch_batch(
            state=next_state,
            batch={
                "batch_id": batch_id,
                "created_at": created_at,
                "status": "new",
                "deferred_until": None,
                "candidate_ids": candidate_ids,
                "top_pick_id": str(pick.pick.canonical_model_id) if pick.pick is not None else None,
                "confidence": float(pick.confidence),
                "show_top_only": bool(pick.show_top_only),
                "candidates": [_serialize_ranked_row(row) for row in top_rows],
            },
        )

    saved_state = state_store.save(set_model_watch_last_run_at(state=next_state, last_run_at=now))
    latest = summarize_model_watch_batch(latest_model_watch_batch(saved_state))

    return {
        "ok": True,
        "found": bool(latest),
        "reason": None,
        "batch": summarize_model_watch_batch(existing_batch) if (existing_batch and not new_batch_created) else latest,
        "latest_batch": latest,
        "new_batch_created": bool(new_batch_created),
        "batch_id": batch_id or None,
        "fetched_candidates": int(catalog_pool_count),
        "total_candidates_considered": len(candidate_rows),
        "catalog_models_considered": int(catalog_pool_count),
        "catalog_snapshot_model_count": int(snapshot.get("model_count") or len(feature_index)),
        "catalog_snapshot_error": snapshot_error,
        "catalog_snapshot_path": str(snapshot_path),
        "installed_error": installed_error,
        "source_errors": sorted(source_errors),
        "new_recommendations": len(new_recommendations),
        "last_run_at": model_watch_last_run_at(saved_state),
    }


def run_watch_once_for_config(
    config: Any,
    *,
    trigger: str = "manual",
    fetch_json: Callable[[str], Any] | None = None,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    _ = trigger
    return run_model_watch_check(config, fetch_json=fetch_json, now_epoch=now_epoch)


__all__ = [
    "_batch_id_for_candidates",
    "run_model_watch_check",
    "run_watch_once_for_config",
    "top_pick_plan_payload",
]
