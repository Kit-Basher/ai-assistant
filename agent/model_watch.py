from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


_SIZE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*[bB]")
_BATCH_STATUSES = frozenset({"new", "notified", "acked", "dismissed", "deferred"})


@dataclass(frozen=True)
class ModelWatchConfig:
    huggingface_watch_authors: tuple[str, ...]
    max_size_gb: float
    require_license: frozenset[str]
    score_threshold: int = 2
    recent_days: int = 30


def default_model_watch_config_document() -> dict[str, Any]:
    return {
        "huggingface_watch_authors": ["Nanbeige"],
        "max_size_gb": 8,
        "require_license": ["apache-2.0", "mit"],
    }


def load_model_watch_config(path: str) -> ModelWatchConfig:
    raw: dict[str, Any] = {}
    try:
        parsed = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            raw = parsed
    except (OSError, UnicodeError, json.JSONDecodeError):
        raw = {}

    merged = default_model_watch_config_document()
    if isinstance(raw.get("huggingface_watch_authors"), list):
        merged["huggingface_watch_authors"] = [
            str(item).strip()
            for item in raw.get("huggingface_watch_authors", [])
            if str(item).strip()
        ]
    if raw.get("max_size_gb") is not None:
        try:
            merged["max_size_gb"] = float(raw.get("max_size_gb"))
        except (TypeError, ValueError):
            pass
    if isinstance(raw.get("require_license"), list):
        merged["require_license"] = [
            str(item).strip().lower()
            for item in raw.get("require_license", [])
            if str(item).strip()
        ]
    return ModelWatchConfig(
        huggingface_watch_authors=tuple(
            sorted({str(item).strip() for item in merged.get("huggingface_watch_authors", []) if str(item).strip()})
        ),
        max_size_gb=max(0.0, float(merged.get("max_size_gb") or 0.0)),
        require_license=frozenset(
            {
                str(item).strip().lower()
                for item in merged.get("require_license", [])
                if str(item).strip()
            }
        ),
    )


class ModelWatchStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()

    @staticmethod
    def default_path() -> str:
        env_value = os.getenv("AGENT_MODEL_WATCH_STATE_PATH", "").strip()
        if env_value:
            return env_value
        return str(Path.home() / ".local" / "share" / "personal-agent" / "model_watch_state.json")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "seen": {},
            "recommendations": [],
            "batches": [],
            "active_batch_id": None,
            "scheduler": {"last_run_at": None},
        }

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self.empty_state()
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self.empty_state()
        if not isinstance(parsed, dict):
            return self.empty_state()
        return normalize_model_watch_state(parsed)

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_model_watch_state(state if isinstance(state, dict) else {})
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
        return normalized


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
    if parsed != parsed:
        return None
    return parsed


def _parse_model_size(model_id: str) -> float | None:
    text = str(model_id or "").strip()
    if not text:
        return None
    match = _SIZE_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _normalize_batch_candidates(raw_candidates: Any) -> list[dict[str, Any]]:
    rows = raw_candidates if isinstance(raw_candidates, list) else []
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("id") or "").strip()
        if not candidate_id:
            continue
        provider = str(row.get("provider") or "").strip().lower()
        model = str(row.get("model") or "").strip()
        score = _safe_float(row.get("score")) or 0.0
        reason = str(row.get("reason") or "").strip() or f"Deterministic model-watch score: {score:.2f}."
        tradeoffs_raw = row.get("tradeoffs") if isinstance(row.get("tradeoffs"), list) else []
        tradeoffs = [str(item).strip() for item in tradeoffs_raw if str(item).strip()]
        subscores_raw = row.get("subscores") if isinstance(row.get("subscores"), dict) else {}
        subscores = {
            "task_fit": round(_safe_float(subscores_raw.get("task_fit")) or 0.0, 2),
            "local_feasibility": round(_safe_float(subscores_raw.get("local_feasibility")) or 0.0, 2),
            "cost_efficiency": round(_safe_float(subscores_raw.get("cost_efficiency")) or 0.0, 2),
            "quality_proxy": round(_safe_float(subscores_raw.get("quality_proxy")) or 0.0, 2),
            "switch_gain": round(_safe_float(subscores_raw.get("switch_gain")) or 0.0, 2),
        }
        output.append(
            {
                "id": candidate_id,
                "provider": provider or None,
                "model": model,
                "score": round(score, 2),
                "reason": reason,
                "tradeoffs": tradeoffs,
                "subscores": subscores,
            }
        )
    output.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("id") or "")))
    return output


def normalize_model_watch_state(state: dict[str, Any]) -> dict[str, Any]:
    output = ModelWatchStore.empty_state()
    seen_raw = state.get("seen") if isinstance(state.get("seen"), dict) else {}
    seen: dict[str, dict[str, int]] = {}
    for model_id, payload in sorted(seen_raw.items()):
        normalized_id = str(model_id or "").strip()
        if not normalized_id:
            continue
        row = payload if isinstance(payload, dict) else {}
        first_seen = _safe_int(row.get("first_seen"))
        if first_seen is None:
            continue
        seen[normalized_id] = {"first_seen": int(first_seen)}
    output["seen"] = seen

    recommendations_raw = state.get("recommendations") if isinstance(state.get("recommendations"), list) else []
    recommendations: dict[str, dict[str, Any]] = {}
    for row in recommendations_raw:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id") or "").strip()
        if not model_id:
            continue
        recommendations[model_id] = {
            "model_id": model_id,
            "install_name": str(row.get("install_name") or "").strip(),
            "author": str(row.get("author") or "").strip() or None,
            "license": str(row.get("license") or "").strip().lower() or None,
            "size_gb": _safe_float(row.get("size_gb")),
            "score": int(_safe_int(row.get("score")) or 0),
            "chat_capable": bool(row.get("chat_capable")),
            "recent_update": bool(row.get("recent_update")),
            "link": str(row.get("link") or "").strip() or None,
            "summary": str(row.get("summary") or "").strip(),
            "first_seen": int(_safe_int(row.get("first_seen")) or 0),
        }
    output["recommendations"] = sorted(
        recommendations.values(),
        key=lambda item: (-int(item.get("score") or 0), str(item.get("model_id") or "")),
    )

    batches_raw = state.get("batches") if isinstance(state.get("batches"), list) else []
    batches_by_id: dict[str, dict[str, Any]] = {}
    for batch in batches_raw:
        if not isinstance(batch, dict):
            continue
        batch_id = str(batch.get("batch_id") or "").strip()
        if not batch_id:
            continue
        status_raw = str(batch.get("status") or "new").strip().lower()
        status = status_raw if status_raw in _BATCH_STATUSES else "new"
        created_at = int(_safe_int(batch.get("created_at")) or 0)
        deferred_until = _safe_int(batch.get("deferred_until"))
        candidate_ids_raw = batch.get("candidate_ids") if isinstance(batch.get("candidate_ids"), list) else []
        candidate_ids = sorted({str(item).strip() for item in candidate_ids_raw if str(item).strip()})
        confidence = _safe_float(batch.get("confidence"))
        if confidence is not None:
            confidence = _clamp(confidence, 0.0, 1.0)
        batches_by_id[batch_id] = {
            "batch_id": batch_id,
            "created_at": created_at,
            "status": status,
            "deferred_until": deferred_until,
            "candidate_ids": candidate_ids,
            "top_pick_id": str(batch.get("top_pick_id") or "").strip() or None,
            "confidence": confidence,
            "show_top_only": bool(batch.get("show_top_only", False)),
            "candidates": _normalize_batch_candidates(batch.get("candidates")),
        }
    output["batches"] = sorted(
        batches_by_id.values(),
        key=lambda item: (-int(item.get("created_at") or 0), str(item.get("batch_id") or "")),
    )
    active = str(state.get("active_batch_id") or "").strip() or None
    if active and any(str(batch.get("batch_id") or "") == active for batch in output["batches"]):
        output["active_batch_id"] = active
    elif output["batches"]:
        output["active_batch_id"] = str(output["batches"][0].get("batch_id") or "")
    else:
        output["active_batch_id"] = None

    scheduler_raw = state.get("scheduler") if isinstance(state.get("scheduler"), dict) else {}
    last_run_at = _safe_int(scheduler_raw.get("last_run_at"))
    output["scheduler"] = {"last_run_at": last_run_at}
    return output


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def model_watch_last_run_at(state: dict[str, Any]) -> int | None:
    normalized = normalize_model_watch_state(state if isinstance(state, dict) else {})
    scheduler = normalized.get("scheduler") if isinstance(normalized.get("scheduler"), dict) else {}
    return _safe_int(scheduler.get("last_run_at"))


def set_model_watch_last_run_at(*, state: dict[str, Any], last_run_at: int) -> dict[str, Any]:
    normalized = normalize_model_watch_state(state if isinstance(state, dict) else {})
    scheduler = normalized.get("scheduler") if isinstance(normalized.get("scheduler"), dict) else {}
    scheduler["last_run_at"] = int(last_run_at)
    normalized["scheduler"] = scheduler
    return normalize_model_watch_state(normalized)


def upsert_model_watch_batch(*, state: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_model_watch_state(state if isinstance(state, dict) else {})
    batch_id = str(batch.get("batch_id") or "").strip()
    if not batch_id:
        return normalized
    batches = normalized.get("batches") if isinstance(normalized.get("batches"), list) else []
    next_batches = [row for row in batches if isinstance(row, dict) and str(row.get("batch_id") or "") != batch_id]
    next_batches.append(dict(batch))
    normalized["batches"] = next_batches
    normalized["active_batch_id"] = batch_id
    return normalize_model_watch_state(normalized)


def latest_model_watch_batch(state: dict[str, Any]) -> dict[str, Any] | None:
    normalized = normalize_model_watch_state(state if isinstance(state, dict) else {})
    active = str(normalized.get("active_batch_id") or "").strip()
    rows = normalized.get("batches") if isinstance(normalized.get("batches"), list) else []
    if active:
        for row in rows:
            if isinstance(row, dict) and str(row.get("batch_id") or "") == active:
                return row
    for row in rows:
        if isinstance(row, dict):
            return row
    return None


def summarize_model_watch_batch(batch: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(batch, dict):
        return None
    candidates = _normalize_batch_candidates(batch.get("candidates"))
    top_pick_id = str(batch.get("top_pick_id") or "").strip() or None
    top_pick: dict[str, Any] | None = None
    if top_pick_id:
        for row in candidates:
            if str(row.get("id") or "") == top_pick_id:
                top_pick = dict(row)
                break
    if top_pick is None and candidates:
        top_pick = dict(candidates[0])
    other_candidates: list[dict[str, Any]] = []
    for row in candidates:
        if top_pick is not None and str(row.get("id") or "") == str(top_pick.get("id") or ""):
            continue
        other_candidates.append(dict(row))
    return {
        "batch_id": str(batch.get("batch_id") or "").strip(),
        "created_at": int(_safe_int(batch.get("created_at")) or 0),
        "status": str(batch.get("status") or "new").strip().lower() or "new",
        "deferred_until": _safe_int(batch.get("deferred_until")),
        "candidate_ids": sorted(
            {
                str(item).strip()
                for item in (batch.get("candidate_ids") if isinstance(batch.get("candidate_ids"), list) else [])
                if str(item).strip()
            }
        ),
        "top_pick_id": top_pick_id,
        "confidence": _safe_float(batch.get("confidence")),
        "show_top_only": bool(batch.get("show_top_only", False)),
        "top_pick": top_pick,
        "candidates": candidates,
        "other_candidates": other_candidates,
    }


def set_model_watch_batch_status(
    *,
    state: dict[str, Any],
    batch_id: str,
    status: str,
    deferred_until: int | None = None,
) -> dict[str, Any]:
    normalized = normalize_model_watch_state(state if isinstance(state, dict) else {})
    target = str(batch_id or "").strip()
    status_value = str(status or "").strip().lower()
    if not target or status_value not in _BATCH_STATUSES:
        return normalized
    rows = normalized.get("batches") if isinstance(normalized.get("batches"), list) else []
    next_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        current_id = str(row.get("batch_id") or "").strip()
        if current_id != target:
            next_rows.append(row)
            continue
        next_row = dict(row)
        next_row["status"] = status_value
        next_row["deferred_until"] = int(deferred_until) if deferred_until is not None else None
        next_rows.append(next_row)
    normalized["batches"] = next_rows
    normalized["active_batch_id"] = target
    return normalize_model_watch_state(normalized)


def evaluate_model_watch_candidates(
    *,
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    installed_models: set[str],
    config: ModelWatchConfig,
    now_epoch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = normalize_model_watch_state(state if isinstance(state, dict) else {})
    seen = normalized.get("seen") if isinstance(normalized.get("seen"), dict) else {}
    existing_recommendations = (
        normalized.get("recommendations") if isinstance(normalized.get("recommendations"), list) else []
    )
    recommendations: dict[str, dict[str, Any]] = {
        str(row.get("model_id") or "").strip(): dict(row)
        for row in existing_recommendations
        if isinstance(row, dict) and str(row.get("model_id") or "").strip()
    }
    new_rows: list[dict[str, Any]] = []

    for row in sorted([item for item in candidates if isinstance(item, dict)], key=lambda item: str(item.get("id") or "")):
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        if model_id in seen:
            continue
        install_name = str(row.get("install_name") or model_id).strip()
        if install_name in installed_models:
            continue
        license_name = str(row.get("license") or "").strip().lower()
        if config.require_license and license_name not in config.require_license:
            continue
        size_gb = _safe_float(row.get("size_gb"))
        if size_gb is None:
            size_gb = _parse_model_size(model_id)
        if size_gb is not None and float(size_gb) > float(config.max_size_gb):
            continue
        chat_capable = bool(row.get("chat_capable", True))
        recent_update = bool(row.get("recent_update", True))
        score = int(chat_capable) + int(recent_update) + int(size_gb is None or float(size_gb) <= float(config.max_size_gb))
        if score < int(config.score_threshold):
            continue
        summary = str(row.get("summary") or "").strip() or "Candidate discovered by model watch."
        payload = {
            "model_id": model_id,
            "install_name": install_name,
            "author": str(row.get("author") or "").strip() or None,
            "license": license_name or None,
            "size_gb": size_gb,
            "score": int(score),
            "chat_capable": chat_capable,
            "recent_update": recent_update,
            "link": str(row.get("link") or "").strip() or None,
            "summary": summary,
            "first_seen": int(now_epoch),
        }
        seen[model_id] = {"first_seen": int(now_epoch)}
        recommendations[model_id] = payload
        new_rows.append(payload)

    normalized["seen"] = {
        str(key): {"first_seen": int(_safe_int(value.get("first_seen")) or 0)}
        for key, value in sorted(seen.items())
        if str(key).strip() and isinstance(value, dict) and _safe_int(value.get("first_seen")) is not None
    }
    normalized["recommendations"] = sorted(
        recommendations.values(),
        key=lambda item: (-int(item.get("score") or 0), str(item.get("model_id") or "")),
    )
    return normalize_model_watch_state(normalized), sorted(
        new_rows,
        key=lambda item: (-int(item.get("score") or 0), str(item.get("model_id") or "")),
    )


__all__ = [
    "ModelWatchConfig",
    "ModelWatchStore",
    "default_model_watch_config_document",
    "evaluate_model_watch_candidates",
    "latest_model_watch_batch",
    "load_model_watch_config",
    "model_watch_last_run_at",
    "normalize_model_watch_state",
    "set_model_watch_batch_status",
    "set_model_watch_last_run_at",
    "summarize_model_watch_batch",
    "upsert_model_watch_batch",
]
