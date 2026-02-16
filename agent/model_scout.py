from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Callable
import urllib.error
import urllib.request


_HF_TRENDING_MODELS_URL = "https://huggingface.co/api/trending?type=model"
_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_])")
_STATUS_VALUES = {"new", "shown", "dismissed", "installed"}


@dataclass(frozen=True)
class ModelScoutSettings:
    enabled: bool
    notify_delta: float
    absolute_threshold: float
    max_suggestions_per_notify: int
    license_allowlist: frozenset[str]
    size_max_b: float
    cooldown_days: int = 7


@dataclass(frozen=True)
class TrendingModel:
    repo_id: str
    rank: int
    likes: int
    downloads: int
    license_name: str | None
    siblings: tuple[str, ...]


@dataclass(frozen=True)
class Suggestion:
    id: str
    kind: str
    repo_id: str | None
    provider_id: str | None
    model_id: str | None
    score: float
    rationale: str
    install_cmd: str | None

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "repo_id": self.repo_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "score": float(self.score),
            "rationale": self.rationale,
            "install_cmd": self.install_cmd,
        }


class ModelScoutStore:
    def __init__(self, db_path: str | None, json_path: str | None = None) -> None:
        self._conn: sqlite3.Connection | None = None
        self.backend = "json"
        self._json_path = Path(json_path or self._default_json_path()).expanduser().resolve()
        self._json_state: dict[str, Any] = {"runs": [], "suggestions": {}, "baselines": []}

        if db_path:
            try:
                self._init_sqlite(db_path)
            except Exception:
                self._conn = None

        if self._conn is None:
            self.backend = "json"
            self._load_json()

    @staticmethod
    def _default_json_path() -> str:
        return str(Path.home() / ".local" / "share" / "personal-agent" / "model_scout_state.json")

    def _init_sqlite(self, db_path: str) -> None:
        expanded = os.path.expanduser(db_path)
        parent = os.path.dirname(expanded)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(expanded, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_scout_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                ok INTEGER NOT NULL,
                error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_scout_suggestions (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                repo_id TEXT,
                provider_id TEXT,
                model_id TEXT,
                score REAL NOT NULL,
                rationale TEXT NOT NULL,
                install_cmd TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                last_notified_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_scout_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                snapshot_json TEXT NOT NULL
            )
            """
        )
        conn.commit()
        self._conn = conn
        self.backend = "sqlite"

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def _load_json(self) -> None:
        if not self._json_path.is_file():
            return
        try:
            parsed = json.loads(self._json_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(parsed, dict):
            return
        runs = parsed.get("runs") if isinstance(parsed.get("runs"), list) else []
        suggestions = parsed.get("suggestions") if isinstance(parsed.get("suggestions"), dict) else {}
        baselines = parsed.get("baselines") if isinstance(parsed.get("baselines"), list) else []
        self._json_state = {
            "runs": runs,
            "suggestions": suggestions,
            "baselines": baselines,
        }

    def _save_json(self) -> None:
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self._json_path.name}.", suffix=".tmp", dir=str(self._json_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self._json_state, ensure_ascii=True, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._json_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def record_run(self, started_at: str, finished_at: str, ok: bool, error: str | None) -> None:
        if self._conn is not None:
            self._conn.execute(
                "INSERT INTO model_scout_runs (started_at, finished_at, ok, error) VALUES (?, ?, ?, ?)",
                (started_at, finished_at, 1 if ok else 0, error),
            )
            self._conn.commit()
            return

        runs = self._json_state.setdefault("runs", [])
        if not isinstance(runs, list):
            runs = []
            self._json_state["runs"] = runs
        runs.append(
            {
                "id": len(runs) + 1,
                "started_at": started_at,
                "finished_at": finished_at,
                "ok": bool(ok),
                "error": error,
            }
        )
        self._json_state["runs"] = runs[-200:]
        self._save_json()

    def save_baseline(self, created_at: str, snapshot: dict[str, Any]) -> None:
        if self._conn is not None:
            self._conn.execute(
                "INSERT INTO model_scout_baselines (created_at, snapshot_json) VALUES (?, ?)",
                (created_at, json.dumps(snapshot, ensure_ascii=True)),
            )
            self._conn.commit()
            return

        baselines = self._json_state.setdefault("baselines", [])
        if not isinstance(baselines, list):
            baselines = []
        baselines.append({"created_at": created_at, "snapshot": snapshot})
        self._json_state["baselines"] = baselines[-100:]
        self._save_json()

    def _parse_suggestion_row(self, row: dict[str, Any]) -> dict[str, Any]:
        parsed = {
            "id": str(row.get("id") or "").strip(),
            "kind": str(row.get("kind") or "").strip(),
            "repo_id": row.get("repo_id"),
            "provider_id": row.get("provider_id"),
            "model_id": row.get("model_id"),
            "score": float(row.get("score") or 0.0),
            "rationale": str(row.get("rationale") or ""),
            "install_cmd": row.get("install_cmd"),
            "created_at": str(row.get("created_at") or ""),
            "updated_at": str(row.get("updated_at") or ""),
            "status": str(row.get("status") or "new").strip().lower() or "new",
            "last_notified_at": row.get("last_notified_at"),
        }
        if parsed["status"] not in _STATUS_VALUES:
            parsed["status"] = "new"
        return parsed

    def list_suggestions(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        status_filter = (status or "").strip().lower() or None

        if self._conn is not None:
            if status_filter:
                cur = self._conn.execute(
                    """
                    SELECT id, kind, repo_id, provider_id, model_id, score, rationale, install_cmd,
                           created_at, updated_at, status, last_notified_at
                    FROM model_scout_suggestions
                    WHERE status = ?
                    ORDER BY score DESC, id ASC
                    LIMIT ?
                    """,
                    (status_filter, int(limit)),
                )
            else:
                cur = self._conn.execute(
                    """
                    SELECT id, kind, repo_id, provider_id, model_id, score, rationale, install_cmd,
                           created_at, updated_at, status, last_notified_at
                    FROM model_scout_suggestions
                    ORDER BY score DESC, id ASC
                    LIMIT ?
                    """,
                    (int(limit),),
                )
            return [self._parse_suggestion_row(dict(row)) for row in cur.fetchall()]

        suggestions_map = self._json_state.get("suggestions") if isinstance(self._json_state.get("suggestions"), dict) else {}
        rows = [self._parse_suggestion_row(payload) for payload in suggestions_map.values() if isinstance(payload, dict)]
        if status_filter:
            rows = [row for row in rows if row["status"] == status_filter]
        rows.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return rows[: max(1, int(limit))]

    def get_suggestion(self, suggestion_id: str) -> dict[str, Any] | None:
        target = str(suggestion_id or "").strip()
        if not target:
            return None

        if self._conn is not None:
            cur = self._conn.execute(
                """
                SELECT id, kind, repo_id, provider_id, model_id, score, rationale, install_cmd,
                       created_at, updated_at, status, last_notified_at
                FROM model_scout_suggestions
                WHERE id = ?
                LIMIT 1
                """,
                (target,),
            )
            row = cur.fetchone()
            return self._parse_suggestion_row(dict(row)) if row is not None else None

        suggestions_map = self._json_state.get("suggestions") if isinstance(self._json_state.get("suggestions"), dict) else {}
        payload = suggestions_map.get(target)
        if not isinstance(payload, dict):
            return None
        return self._parse_suggestion_row(payload)

    def upsert_suggestions(
        self,
        suggestions: list[Suggestion],
        *,
        now_iso: str,
        cooldown_days: int,
    ) -> list[dict[str, Any]]:
        cooldown = max(0, int(cooldown_days))
        upserted: list[dict[str, Any]] = []
        for suggestion in sorted(suggestions, key=lambda item: item.id):
            existing = self.get_suggestion(suggestion.id)
            created_at = existing["created_at"] if existing else now_iso
            status = existing["status"] if existing else "new"
            last_notified_at = existing["last_notified_at"] if existing else None

            if status == "shown" and last_notified_at:
                last_notified = _parse_iso(last_notified_at)
                if last_notified is not None and _parse_iso(now_iso) is not None:
                    elapsed = _parse_iso(now_iso) - last_notified
                    if elapsed >= timedelta(days=cooldown):
                        status = "new"
            if status not in _STATUS_VALUES:
                status = "new"

            row = {
                **suggestion.as_row(),
                "created_at": created_at,
                "updated_at": now_iso,
                "status": status,
                "last_notified_at": last_notified_at,
            }
            self._write_suggestion_row(row)
            upserted.append(row)
        return sorted(upserted, key=lambda item: (-float(item["score"]), str(item["id"])))

    def _write_suggestion_row(self, row: dict[str, Any]) -> None:
        if self._conn is not None:
            self._conn.execute(
                """
                INSERT INTO model_scout_suggestions (
                    id, kind, repo_id, provider_id, model_id, score, rationale, install_cmd,
                    created_at, updated_at, status, last_notified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    repo_id = excluded.repo_id,
                    provider_id = excluded.provider_id,
                    model_id = excluded.model_id,
                    score = excluded.score,
                    rationale = excluded.rationale,
                    install_cmd = excluded.install_cmd,
                    updated_at = excluded.updated_at,
                    status = excluded.status,
                    last_notified_at = excluded.last_notified_at
                """,
                (
                    row["id"],
                    row["kind"],
                    row.get("repo_id"),
                    row.get("provider_id"),
                    row.get("model_id"),
                    float(row.get("score") or 0.0),
                    row.get("rationale") or "",
                    row.get("install_cmd"),
                    row.get("created_at") or "",
                    row.get("updated_at") or "",
                    row.get("status") or "new",
                    row.get("last_notified_at"),
                ),
            )
            self._conn.commit()
            return

        suggestions_map = self._json_state.setdefault("suggestions", {})
        if not isinstance(suggestions_map, dict):
            suggestions_map = {}
        suggestions_map[str(row["id"])] = row
        self._json_state["suggestions"] = suggestions_map
        self._save_json()

    def set_suggestion_status(self, suggestion_id: str, status: str) -> bool:
        normalized_id = str(suggestion_id or "").strip()
        normalized_status = str(status or "").strip().lower()
        if not normalized_id or normalized_status not in _STATUS_VALUES:
            return False

        row = self.get_suggestion(normalized_id)
        if row is None:
            return False

        row["status"] = normalized_status
        row["updated_at"] = _now_iso()
        self._write_suggestion_row(row)
        return True

    def mark_notified(self, suggestion_ids: list[str], at_iso: str) -> None:
        for suggestion_id in sorted({str(item or "").strip() for item in suggestion_ids if str(item or "").strip()}):
            row = self.get_suggestion(suggestion_id)
            if row is None:
                continue
            row["status"] = "shown"
            row["updated_at"] = at_iso
            row["last_notified_at"] = at_iso
            self._write_suggestion_row(row)

    def latest_run(self) -> dict[str, Any] | None:
        if self._conn is not None:
            cur = self._conn.execute(
                """
                SELECT id, started_at, finished_at, ok, error
                FROM model_scout_runs
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["ok"] = bool(payload.get("ok"))
            return payload

        runs = self._json_state.get("runs") if isinstance(self._json_state.get("runs"), list) else []
        if not runs:
            return None
        latest = runs[-1]
        if not isinstance(latest, dict):
            return None
        payload = dict(latest)
        payload["ok"] = bool(payload.get("ok"))
        return payload

    def latest_baseline(self) -> dict[str, Any] | None:
        if self._conn is not None:
            cur = self._conn.execute(
                """
                SELECT created_at, snapshot_json
                FROM model_scout_baselines
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row is None:
                return None
            try:
                snapshot = json.loads(str(row["snapshot_json"] or "{}"))
            except Exception:
                snapshot = {}
            return {
                "created_at": str(row["created_at"] or ""),
                "snapshot": snapshot,
            }

        baselines = self._json_state.get("baselines") if isinstance(self._json_state.get("baselines"), list) else []
        if not baselines:
            return None
        latest = baselines[-1]
        if not isinstance(latest, dict):
            return None
        return {
            "created_at": str(latest.get("created_at") or ""),
            "snapshot": latest.get("snapshot") if isinstance(latest.get("snapshot"), dict) else {},
        }

    def status(self) -> dict[str, Any]:
        rows = self.list_suggestions(limit=2000)
        counts = {key: 0 for key in sorted(_STATUS_VALUES)}
        for row in rows:
            status = str(row.get("status") or "new").strip().lower()
            if status not in counts:
                continue
            counts[status] += 1
        return {
            "backend": self.backend,
            "last_run": self.latest_run(),
            "counts": counts,
            "total": len(rows),
        }


class ModelScout:
    def __init__(
        self,
        settings: ModelScoutSettings,
        *,
        store: ModelScoutStore,
        fetch_json: Callable[[str], dict[str, Any]] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self._fetch_json = fetch_json or _http_get_json
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def close(self) -> None:
        self.store.close()

    def status(self) -> dict[str, Any]:
        return self.store.status()

    def list_suggestions(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return self.store.list_suggestions(status=status, limit=limit)

    def dismiss(self, suggestion_id: str) -> bool:
        return self.store.set_suggestion_status(suggestion_id, "dismissed")

    def mark_installed(self, suggestion_id: str) -> bool:
        return self.store.set_suggestion_status(suggestion_id, "installed")

    def run(
        self,
        *,
        registry_document: dict[str, Any],
        router_snapshot: dict[str, Any],
        usage_stats_snapshot: dict[str, dict[str, Any]] | None,
        notify_sender: Callable[[str, list[dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        started = self._now_fn()
        started_iso = started.isoformat()

        if not self.settings.enabled:
            finished_iso = self._now_fn().isoformat()
            self.store.record_run(started_iso, finished_iso, True, "disabled")
            return {
                "ok": True,
                "disabled": True,
                "error": "disabled",
                "suggestions": [],
                "notified": 0,
            }

        trending_error: str | None = None
        trending_models: list[TrendingModel] = []
        try:
            payload = self._fetch_json(_HF_TRENDING_MODELS_URL)
            trending_models = _extract_trending_models(payload)
        except Exception as exc:
            trending_error = str(exc) or "trending_fetch_failed"
            trending_models = []

        baselines = self._build_baselines(registry_document)
        baseline_local_score = float(baselines.get("local", {}).get("score") or 0.0)
        baseline_remote = {
            provider_id: payload
            for provider_id, payload in (baselines.get("remote") or {}).items()
            if isinstance(payload, dict)
        }

        local_suggestions = self._build_local_suggestions(
            trending_models,
            baseline_local_score=baseline_local_score,
            local_provider_id=str((baselines.get("local") or {}).get("provider_id") or "ollama"),
        )
        remote_suggestions = self._build_remote_suggestions(
            registry_document,
            router_snapshot,
            usage_stats_snapshot or {},
            trending_models,
            baseline_remote=baseline_remote,
        )

        suggestions = sorted(
            local_suggestions + remote_suggestions,
            key=lambda item: (-float(item.score), item.kind, str(item.provider_id or ""), str(item.id)),
        )

        now_iso = self._now_fn().isoformat()
        self.store.upsert_suggestions(suggestions, now_iso=now_iso, cooldown_days=self.settings.cooldown_days)
        self.store.save_baseline(now_iso, baselines)

        newly_actionable = [
            row
            for row in self.store.list_suggestions(status="new", limit=500)
            if row.get("id") in {item.id for item in suggestions}
        ]
        newly_actionable.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("id") or "")))

        notified = 0
        notification_error: str | None = None
        if notify_sender is not None and newly_actionable:
            batch = newly_actionable[: max(1, int(self.settings.max_suggestions_per_notify))]
            message = self.format_notification_message(batch)
            try:
                notify_sender(message, batch)
                notified = len(batch)
                self.store.mark_notified([str(item.get("id") or "") for item in batch], at_iso=now_iso)
            except Exception as exc:
                notification_error = str(exc) or "notification_failed"

        finished_iso = self._now_fn().isoformat()
        run_ok = trending_error is None and notification_error is None
        combined_error = "; ".join([part for part in [trending_error, notification_error] if part]) or None
        self.store.record_run(started_iso, finished_iso, run_ok, combined_error)

        return {
            "ok": run_ok,
            "error": combined_error,
            "trending_error": trending_error,
            "notification_error": notification_error,
            "suggestions": [item.as_row() for item in suggestions],
            "new_suggestions": newly_actionable,
            "notified": notified,
            "fetched_trending": len(trending_models),
        }

    def _build_baselines(self, registry_document: dict[str, Any]) -> dict[str, Any]:
        providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
        models = registry_document.get("models") if isinstance(registry_document.get("models"), dict) else {}
        defaults = registry_document.get("defaults") if isinstance(registry_document.get("defaults"), dict) else {}

        local_provider_ids = [
            str(provider_id)
            for provider_id, payload in sorted(providers.items())
            if isinstance(payload, dict) and bool(payload.get("enabled", True)) and bool(payload.get("local", False))
        ]

        local_baseline_model = None
        default_model = str(defaults.get("default_model") or "").strip()
        if default_model and default_model in models:
            model_payload = models.get(default_model)
            if isinstance(model_payload, dict):
                provider = str(model_payload.get("provider") or "").strip().lower()
                if provider in local_provider_ids and self._is_chat_model(model_payload):
                    local_baseline_model = default_model

        if local_baseline_model is None:
            candidate_rows = [
                (model_id, payload)
                for model_id, payload in sorted(models.items())
                if isinstance(payload, dict)
                and str(payload.get("provider") or "").strip().lower() in local_provider_ids
                and bool(payload.get("enabled", True))
                and bool(payload.get("available", True))
                and self._is_chat_model(payload)
            ]
            if candidate_rows:
                candidate_rows.sort(
                    key=lambda item: (
                        -int(item[1].get("quality_rank", 0) or 0),
                        int(item[1].get("cost_rank", 0) or 0),
                        str(item[0]),
                    )
                )
                local_baseline_model = candidate_rows[0][0]

        local_baseline_score = 0.0
        local_provider_id = local_provider_ids[0] if local_provider_ids else "ollama"
        if local_baseline_model and local_baseline_model in models and isinstance(models[local_baseline_model], dict):
            payload = models[local_baseline_model]
            local_provider_id = str(payload.get("provider") or local_provider_id)
            local_baseline_score = self._score_baseline_name(str(payload.get("model") or local_baseline_model))

        remote_baselines: dict[str, dict[str, Any]] = {}
        for provider_id, provider_payload in sorted(providers.items()):
            if not isinstance(provider_payload, dict):
                continue
            if not bool(provider_payload.get("enabled", True)) or bool(provider_payload.get("local", False)):
                continue
            provider_models = [
                (model_id, payload)
                for model_id, payload in sorted(models.items())
                if isinstance(payload, dict)
                and str(payload.get("provider") or "").strip().lower() == str(provider_id).strip().lower()
                and bool(payload.get("enabled", True))
                and bool(payload.get("available", True))
                and self._is_chat_model(payload)
            ]
            if not provider_models:
                continue

            baseline_model_id = None
            if default_model and default_model in models:
                payload = models.get(default_model)
                if isinstance(payload, dict) and str(payload.get("provider") or "").strip().lower() == str(provider_id).strip().lower():
                    baseline_model_id = default_model
            if baseline_model_id is None:
                provider_models.sort(key=lambda item: str(item[0]))
                baseline_model_id = provider_models[0][0]

            baseline_payload = models.get(baseline_model_id) if isinstance(models.get(baseline_model_id), dict) else {}
            baseline_score = self._score_baseline_name(str((baseline_payload or {}).get("model") or baseline_model_id))
            remote_baselines[str(provider_id)] = {
                "model_id": baseline_model_id,
                "score": baseline_score,
            }

        return {
            "local": {
                "provider_id": local_provider_id,
                "model_id": local_baseline_model,
                "score": local_baseline_score,
            },
            "remote": remote_baselines,
            "defaults": {
                "default_provider": defaults.get("default_provider"),
                "default_model": defaults.get("default_model"),
            },
        }

    @staticmethod
    def _is_chat_model(model_payload: dict[str, Any]) -> bool:
        capabilities = model_payload.get("capabilities") if isinstance(model_payload.get("capabilities"), list) else []
        normalized = {str(item).strip().lower() for item in capabilities if str(item).strip()}
        return "chat" in normalized

    def _build_local_suggestions(
        self,
        trending_models: list[TrendingModel],
        *,
        baseline_local_score: float,
        local_provider_id: str,
    ) -> list[Suggestion]:
        if not trending_models:
            return []

        by_stem: dict[str, list[TrendingModel]] = {}
        for item in trending_models:
            by_stem.setdefault(_repo_stem(item.repo_id), []).append(item)

        gguf_repo_for_stem: dict[str, str] = {}
        for stem, rows in by_stem.items():
            gguf_ids = sorted(item.repo_id for item in rows if _is_gguf_repo(item.repo_id, item.siblings))
            if gguf_ids:
                gguf_repo_for_stem[stem] = gguf_ids[0]

        results: list[Suggestion] = []
        for item in trending_models:
            direct_gguf = _is_gguf_repo(item.repo_id, item.siblings)
            stem = _repo_stem(item.repo_id)
            sibling_gguf_repo = gguf_repo_for_stem.get(stem)
            if not direct_gguf and not sibling_gguf_repo:
                continue

            install_repo = item.repo_id if direct_gguf else sibling_gguf_repo
            if not install_repo:
                continue

            size_b = _extract_model_size_b(install_repo)
            score = self._score_local_candidate(
                rank=item.rank,
                likes=item.likes,
                downloads=item.downloads,
                size_b=size_b,
                gguf_ready=True,
                license_name=item.license_name,
            )
            if not self._passes_threshold(score, baseline_local_score):
                continue

            recommended_quant = _recommended_quant(item, install_repo)
            rationale_bits = [f"trending #{item.rank}", "GGUF-ready"]
            if size_b is not None:
                rationale_bits.append(f"~{size_b:g}B fits local target")
            if item.license_name:
                license_norm = item.license_name.strip().lower()
                if license_norm in self.settings.license_allowlist:
                    rationale_bits.append(f"license {license_norm}")
                else:
                    rationale_bits.append(f"license {license_norm} (check usage)")
            else:
                rationale_bits.append("license unknown")
            rationale_bits.append(f"recommended quant {recommended_quant}")

            suggestion_id = f"local:{str(install_repo).strip().lower()}"
            results.append(
                Suggestion(
                    id=suggestion_id,
                    kind="local",
                    repo_id=install_repo,
                    provider_id=local_provider_id,
                    model_id=None,
                    score=score,
                    rationale=", ".join(rationale_bits),
                    install_cmd=f"ollama run hf.co/{install_repo}",
                )
            )

        return sorted(results, key=lambda item: (-item.score, item.id))

    def _build_remote_suggestions(
        self,
        registry_document: dict[str, Any],
        router_snapshot: dict[str, Any],
        usage_stats_snapshot: dict[str, dict[str, Any]],
        trending_models: list[TrendingModel],
        *,
        baseline_remote: dict[str, dict[str, Any]],
    ) -> list[Suggestion]:
        providers = registry_document.get("providers") if isinstance(registry_document.get("providers"), dict) else {}
        models = registry_document.get("models") if isinstance(registry_document.get("models"), dict) else {}

        provider_health = {
            str(item.get("id") or ""): item.get("health") or {}
            for item in (router_snapshot.get("providers") or [])
            if isinstance(item, dict)
        }
        model_health = {
            str(item.get("id") or ""): item.get("health") or {}
            for item in (router_snapshot.get("models") or [])
            if isinstance(item, dict)
        }

        flattened_terms: set[str] = set()
        for item in trending_models:
            if not item.repo_id:
                continue
            flattened_terms.update(_repo_name_terms(item.repo_id))

        suggestions: list[Suggestion] = []
        for provider_id, provider_payload in sorted(providers.items()):
            if not isinstance(provider_payload, dict):
                continue
            if bool(provider_payload.get("local", False)):
                continue
            if not bool(provider_payload.get("enabled", True)):
                continue

            health = provider_health.get(str(provider_id), {})
            provider_status = str(health.get("status") or "ok").strip().lower()
            provider_successes = int(health.get("successes") or 0)
            if provider_status == "down" or provider_successes <= 0:
                # Consider provider untested/unhealthy for remote model recommendations.
                continue

            provider_models = [
                (model_id, payload)
                for model_id, payload in sorted(models.items())
                if isinstance(payload, dict)
                and str(payload.get("provider") or "").strip().lower() == str(provider_id).strip().lower()
                and bool(payload.get("enabled", True))
                and bool(payload.get("available", True))
                and self._is_chat_model(payload)
            ]
            if len(provider_models) < 2:
                continue

            baseline_model_id = str((registry_document.get("defaults") or {}).get("default_model") or "").strip()
            if not baseline_model_id or baseline_model_id not in {model_id for model_id, _ in provider_models}:
                baseline_payload_from_baseline = baseline_remote.get(str(provider_id))
                if isinstance(baseline_payload_from_baseline, dict):
                    baseline_model_id = str(baseline_payload_from_baseline.get("model_id") or "")
                if not baseline_model_id or baseline_model_id not in {model_id for model_id, _ in provider_models}:
                    baseline_model_id = provider_models[0][0]

            baseline_payload = dict(models.get(baseline_model_id) or {}) if isinstance(models.get(baseline_model_id), dict) else {}
            baseline_metrics = self._remote_metrics(
                provider_id=str(provider_id),
                model_id=baseline_model_id,
                model_payload=baseline_payload,
                health=model_health.get(baseline_model_id, {}),
                usage_stats=usage_stats_snapshot,
                trending_terms=flattened_terms,
            )

            candidate_rows: list[tuple[tuple[float, float, float, str], str, dict[str, Any], dict[str, float]]] = []
            for model_id, payload in provider_models:
                metrics = self._remote_metrics(
                    provider_id=str(provider_id),
                    model_id=model_id,
                    model_payload=payload,
                    health=model_health.get(model_id, {}),
                    usage_stats=usage_stats_snapshot,
                    trending_terms=flattened_terms,
                )
                order_key = (
                    metrics["health_penalty"],
                    metrics["expected_cost"],
                    -metrics["quality"],
                    model_id,
                )
                candidate_rows.append((order_key, model_id, payload, metrics))

            candidate_rows.sort(key=lambda item: item[0])
            best_key, best_model_id, best_payload, best_metrics = candidate_rows[0]
            baseline_key = (
                baseline_metrics["health_penalty"],
                baseline_metrics["expected_cost"],
                -baseline_metrics["quality"],
                baseline_model_id,
            )
            if best_key >= baseline_key:
                continue

            baseline_score = float((baseline_remote.get(str(provider_id)) or {}).get("score") or 0.0)
            score = best_metrics["score"]
            if not self._passes_threshold(score, baseline_score):
                continue

            reason = "cheaper"
            if best_metrics["health_penalty"] + 5 < baseline_metrics["health_penalty"]:
                reason = "more reliable"
            elif best_metrics["expected_cost"] + 1e-6 < baseline_metrics["expected_cost"]:
                reason = "cheaper"
            elif best_metrics["quality"] > baseline_metrics["quality"]:
                reason = "better quality"

            suggestion_id = f"remote:{str(provider_id).strip().lower()}:{best_model_id}"
            rationale = (
                f"{reason} than {baseline_model_id} "
                f"(expected_cost={best_metrics['expected_cost']:.4f}, health_penalty={best_metrics['health_penalty']:.1f})"
            )
            suggestions.append(
                Suggestion(
                    id=suggestion_id,
                    kind="remote",
                    repo_id=None,
                    provider_id=str(provider_id),
                    model_id=best_model_id,
                    score=score,
                    rationale=rationale,
                    install_cmd=None,
                )
            )

        return sorted(suggestions, key=lambda item: (-item.score, item.id))

    def _remote_metrics(
        self,
        *,
        provider_id: str,
        model_id: str,
        model_payload: dict[str, Any],
        health: dict[str, Any],
        usage_stats: dict[str, dict[str, Any]],
        trending_terms: set[str],
    ) -> dict[str, float]:
        quality = float(int(model_payload.get("quality_rank", 0) or 0))

        pricing = model_payload.get("pricing") if isinstance(model_payload.get("pricing"), dict) else {}
        in_price = _safe_float(pricing.get("input_per_million_tokens"))
        out_price = _safe_float(pricing.get("output_per_million_tokens"))

        prompt_tokens, completion_tokens = _usage_estimate(usage_stats, provider_id, model_id)
        if in_price is None or out_price is None:
            expected_cost = 1_000_000.0
        else:
            expected_cost = (prompt_tokens * in_price / 1_000_000.0) + (completion_tokens * out_price / 1_000_000.0)

        status = str(health.get("status") or "ok").strip().lower()
        failures = float(int(health.get("failures") or 0))
        successes = float(int(health.get("successes") or 0))
        health_penalty = 0.0
        if status == "down":
            health_penalty = 1000.0
        elif status == "degraded":
            health_penalty = 80.0
        if failures > 0:
            health_penalty += (failures / max(1.0, failures + successes)) * 30.0

        trend_bonus = 0.0
        model_terms = _repo_name_terms(str(model_payload.get("model") or model_id))
        if model_terms.intersection(trending_terms):
            trend_bonus = 8.0

        score = max(
            0.0,
            100.0
            - min(expected_cost * 1000.0, 120.0)
            - health_penalty
            + (quality * 4.0)
            + trend_bonus,
        )

        return {
            "quality": quality,
            "expected_cost": expected_cost,
            "health_penalty": health_penalty,
            "score": score,
        }

    def _score_local_candidate(
        self,
        *,
        rank: int,
        likes: int,
        downloads: int,
        size_b: float | None,
        gguf_ready: bool,
        license_name: str | None,
    ) -> float:
        rank_score = max(0.0, 58.0 - (max(1, rank) - 1) * 2.0)
        likes_score = min(18.0, math.log10(max(1, likes) + 1) * 7.0)
        downloads_score = min(16.0, math.log10(max(1, downloads) + 1) * 4.0)

        fit_score = 4.0
        if size_b is not None:
            if 1.0 <= size_b <= 8.0:
                fit_score = 22.0
            elif size_b <= float(self.settings.size_max_b):
                fit_score = 12.0
            else:
                fit_score = -8.0

        readiness_score = 35.0 if gguf_ready else -8.0

        license_score = 0.0
        if license_name:
            normalized_license = license_name.strip().lower()
            if normalized_license in self.settings.license_allowlist:
                license_score = 10.0
            else:
                license_score = -10.0

        return rank_score + likes_score + downloads_score + fit_score + readiness_score + license_score

    def _score_baseline_name(self, model_name: str) -> float:
        size_b = _extract_model_size_b(model_name)
        return self._score_local_candidate(
            rank=25,
            likes=50,
            downloads=200,
            size_b=size_b,
            gguf_ready=("gguf" in model_name.lower()),
            license_name=None,
        )

    def _passes_threshold(self, candidate_score: float, baseline_score: float) -> bool:
        if candidate_score >= float(self.settings.absolute_threshold):
            return True
        return (candidate_score - baseline_score) >= float(self.settings.notify_delta)

    @staticmethod
    def format_notification_message(suggestions: list[dict[str, Any]]) -> str:
        if not suggestions:
            return "I checked trending models and nothing crossed your try-now threshold today."

        lines = ["I found a few model candidates worth trying:"]
        for item in suggestions[:3]:
            kind = str(item.get("kind") or "")
            if kind == "local":
                name = str(item.get("repo_id") or "model")
            else:
                name = str(item.get("model_id") or item.get("repo_id") or "model")
            rationale = str(item.get("rationale") or "looks promising")
            try_it = str(item.get("install_cmd") or "set it as the default in Providers/Defaults")
            lines.append(f"- {name}: {rationale}. Try it: {try_it}")

        lines.append("Reply /scout for details")
        return "\n".join(lines)


def load_model_scout_settings(config) -> ModelScoutSettings:
    allowlist = frozenset(str(item).strip().lower() for item in (config.model_scout_license_allowlist or ()) if str(item).strip())
    if not allowlist:
        allowlist = frozenset({"apache-2.0", "mit", "bsd-3-clause"})
    return ModelScoutSettings(
        enabled=bool(config.model_scout_enabled),
        notify_delta=float(config.model_scout_notify_delta),
        absolute_threshold=float(config.model_scout_absolute_threshold),
        max_suggestions_per_notify=max(1, int(config.model_scout_max_suggestions_per_notify)),
        license_allowlist=allowlist,
        size_max_b=float(config.model_scout_size_max_b),
    )


def build_model_scout(config) -> ModelScout:
    settings = load_model_scout_settings(config)
    store = ModelScoutStore(config.db_path, json_path=config.model_scout_state_path)
    return ModelScout(settings, store=store)


def _http_get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=12.0) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"hf_http_{int(exc.code)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("hf_unreachable") from exc

    try:
        parsed = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("hf_invalid_json") from exc
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"data": parsed}
    return {}


def _extract_trending_models(payload: dict[str, Any]) -> list[TrendingModel]:
    candidates = _first_list(payload, ["models", "data", "recentlyTrending", "trending", "items", "results"])
    if candidates is None and isinstance(payload.get("data"), dict):
        candidates = _first_list(payload.get("data") or {}, ["models", "items", "results"])  # type: ignore[arg-type]
    if candidates is None:
        candidates = []

    rows: list[TrendingModel] = []
    seen: set[str] = set()
    for idx, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        repo_id = _extract_repo_id(item)
        if not repo_id or repo_id in seen:
            continue
        seen.add(repo_id)

        likes = _safe_int(_first_value(item, ["likes", "likes_count", "likesCount"]))
        downloads = _safe_int(_first_value(item, ["downloads", "downloads_all_time", "downloadsAllTime"]))
        license_name = _extract_license(item)
        siblings = _extract_siblings(item)
        rows.append(
            TrendingModel(
                repo_id=repo_id,
                rank=idx,
                likes=max(0, likes),
                downloads=max(0, downloads),
                license_name=license_name,
                siblings=tuple(sorted(siblings)),
            )
        )

    return rows


def _first_list(payload: dict[str, Any], keys: list[str]) -> list[Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _first_value(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _extract_repo_id(item: dict[str, Any]) -> str:
    direct = _first_value(item, ["modelId", "model_id", "repo_id", "repoId", "id"])
    if isinstance(direct, str) and "/" in direct:
        return direct.strip()

    model_payload = item.get("model") if isinstance(item.get("model"), dict) else {}
    nested = _first_value(model_payload, ["id", "modelId", "repo_id"])
    if isinstance(nested, str) and "/" in nested:
        return nested.strip()

    card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
    card_id = _first_value(card_data, ["model_id", "id", "repo_id"])
    if isinstance(card_id, str) and "/" in card_id:
        return card_id.strip()

    return ""


def _extract_license(item: dict[str, Any]) -> str | None:
    direct = _first_value(item, ["license", "license_name", "licenseName"])
    if isinstance(direct, str) and direct.strip():
        return direct.strip().lower()

    card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
    nested = _first_value(card_data, ["license", "license_name", "licenseName"])
    if isinstance(nested, str) and nested.strip():
        return nested.strip().lower()
    return None


def _extract_siblings(item: dict[str, Any]) -> list[str]:
    siblings_raw = item.get("siblings") if isinstance(item.get("siblings"), list) else []
    siblings: list[str] = []
    for row in siblings_raw:
        if isinstance(row, str) and row.strip():
            siblings.append(row.strip())
            continue
        if not isinstance(row, dict):
            continue
        name = _first_value(row, ["rfilename", "filename", "path", "name"])
        if isinstance(name, str) and name.strip():
            siblings.append(name.strip())
    return siblings


def _is_gguf_repo(repo_id: str, siblings: tuple[str, ...]) -> bool:
    lowered = str(repo_id or "").strip().lower()
    if "gguf" in lowered:
        return True
    for sibling in siblings:
        name = str(sibling or "").strip().lower()
        if not name:
            continue
        if "gguf" in name or name.endswith(".gguf"):
            return True
    return False


def _recommended_quant(item: TrendingModel, install_repo: str) -> str:
    repo_name = str(install_repo or "").strip().upper()
    if "Q4_K_M" in repo_name:
        return "Q4_K_M"
    for sibling in item.siblings:
        candidate = str(sibling or "").strip().upper()
        if "Q4_K_M" in candidate:
            return "Q4_K_M"
    return "Q4_K_M"


def _extract_model_size_b(text: str) -> float | None:
    match = _MODEL_SIZE_RE.search(str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _repo_stem(repo_id: str) -> str:
    if "/" in repo_id:
        _, repo_name = repo_id.split("/", 1)
    else:
        repo_name = repo_id
    lowered = repo_name.strip().lower()
    lowered = lowered.replace("_gguf", "")
    lowered = lowered.replace("-gguf", "")
    return lowered


def _repo_name_terms(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    if not normalized:
        return set()
    return {part for part in normalized.split() if part and len(part) >= 2}


def _usage_estimate(
    usage_stats: dict[str, dict[str, Any]],
    provider_id: str,
    model_id: str,
) -> tuple[float, float]:
    direct_key = f"chat::{provider_id}::{model_id}"
    if direct_key in usage_stats and isinstance(usage_stats[direct_key], dict):
        row = usage_stats[direct_key]
        return (
            max(1.0, float(row.get("prompt_tokens") or 0.0)),
            max(1.0, float(row.get("completion_tokens") or 0.0)),
        )

    for key, row in usage_stats.items():
        if not isinstance(row, dict):
            continue
        parts = str(key).split("::")
        if len(parts) != 3:
            continue
        _, provider, model = parts
        if provider != provider_id or model != model_id:
            continue
        return (
            max(1.0, float(row.get("prompt_tokens") or 0.0)),
            max(1.0, float(row.get("completion_tokens") or 0.0)),
        )

    return 800.0, 220.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
