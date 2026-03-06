from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from agent.llm.capabilities import is_embedding_model_name
from agent.llm.registry import load_registry_document


_SCHEMA_VERSION = 1
_DEFAULT_MAX_SNAPSHOTS = 40


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _registry_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(document).encode("utf-8")).hexdigest()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def verify_registry_invariants(document: dict[str, Any]) -> tuple[bool, str | None]:
    payload = document if isinstance(document, dict) else {}
    providers = payload.get("providers") if isinstance(payload.get("providers"), dict) else None
    models = payload.get("models") if isinstance(payload.get("models"), dict) else None
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else None
    if providers is None or models is None or defaults is None:
        return False, "missing_sections"

    for provider_id, provider_payload in providers.items():
        if not str(provider_id or "").strip():
            return False, "invalid_provider_id"
        if not isinstance(provider_payload, dict):
            return False, "invalid_provider_payload"

    for model_id, model_payload in models.items():
        if not str(model_id or "").strip():
            return False, "invalid_model_id"
        if not isinstance(model_payload, dict):
            return False, "invalid_model_payload"
        provider_id = str(model_payload.get("provider") or "").strip().lower()
        if not provider_id:
            return False, "model_provider_missing"
        if provider_id not in providers:
            return False, "model_provider_unknown"

    default_provider = str(defaults.get("default_provider") or "").strip().lower() or None
    chat_model = str(defaults.get("chat_model") or "").strip() or None
    legacy_default_model = str(defaults.get("default_model") or "").strip() or None
    default_model = chat_model or legacy_default_model
    embed_model = str(defaults.get("embed_model") or "").strip() or None
    if default_provider and default_provider not in providers:
        return False, "default_provider_unknown"
    if chat_model and chat_model not in models:
        return False, "chat_model_unknown"
    if default_model and default_model not in models:
        return False, "default_model_unknown"
    if embed_model and embed_model not in models:
        return False, "embed_model_unknown"
    if default_model and default_provider:
        model_provider = str((models.get(default_model) or {}).get("provider") or "").strip().lower()
        if model_provider and model_provider != default_provider:
            return False, "default_model_provider_mismatch"
    if default_model:
        model_caps = {
            str(item).strip().lower()
            for item in ((models.get(default_model) or {}).get("capabilities") or [])
            if str(item).strip()
        }
        if "chat" not in model_caps:
            return False, "chat_model_not_chat_capable"
    if embed_model:
        embed_caps = {
            str(item).strip().lower()
            for item in ((models.get(embed_model) or {}).get("capabilities") or [])
            if str(item).strip()
        }
        embed_name = str((models.get(embed_model) or {}).get("model") or embed_model)
        if "embedding" not in embed_caps and "embeddings" not in embed_caps and not is_embedding_model_name(embed_name):
            return False, "embed_model_not_embedding_capable"
    if "allow_remote_fallback" in defaults and not isinstance(defaults.get("allow_remote_fallback"), bool):
        return False, "allow_remote_fallback_not_bool"
    return True, None


class RegistrySnapshotStore:
    def __init__(
        self,
        path: str | None = None,
        *,
        max_items: int = _DEFAULT_MAX_SNAPSHOTS,
    ) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.max_items = max(1, int(max_items))
        self._index_path = self.path / "index.json"
        self.state = self.load()

    @staticmethod
    def default_path() -> str:
        env_value = os.getenv("LLM_REGISTRY_SNAPSHOTS_DIR", "").strip()
        if env_value:
            return env_value
        return str(Path.home() / ".local" / "share" / "personal-agent" / "registry_snapshots")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "next_seq": 1,
            "snapshots": [],
            "pruned_count_last_run": 0,
            "pruned_count_total": 0,
        }

    @staticmethod
    def _entry_sort_key(row: dict[str, Any]) -> tuple[int, str]:
        return (_safe_int(row.get("seq"), 0), str(row.get("snapshot_id") or ""))

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.empty_state()
        state["next_seq"] = max(1, _safe_int(payload.get("next_seq"), 1))
        state["pruned_count_last_run"] = max(0, _safe_int(payload.get("pruned_count_last_run"), 0))
        state["pruned_count_total"] = max(0, _safe_int(payload.get("pruned_count_total"), 0))
        raw_rows = payload.get("snapshots") if isinstance(payload.get("snapshots"), list) else []
        rows: list[dict[str, Any]] = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            snapshot_id = str(item.get("snapshot_id") or "").strip()
            filename = str(item.get("filename") or "").strip()
            registry_hash = str(item.get("registry_hash") or "").strip()
            seq = _safe_int(item.get("seq"), 0)
            if not snapshot_id or not filename or not registry_hash or seq <= 0:
                continue
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "filename": filename,
                    "registry_hash": registry_hash,
                    "size_bytes": max(0, _safe_int(item.get("size_bytes"), 0)),
                    "seq": seq,
                }
            )
        rows.sort(key=self._entry_sort_key)
        state["snapshots"] = rows
        if rows:
            state["next_seq"] = max(int(state["next_seq"]), int(rows[-1]["seq"]) + 1)
        return state

    def load(self) -> dict[str, Any]:
        if not self._index_path.is_file():
            return self.empty_state()
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self.empty_state()
        if not isinstance(raw, dict):
            return self.empty_state()
        normalized = self._normalize(raw)
        if _stable_json(raw) != _stable_json(normalized):
            try:
                self._write(normalized)
            except OSError:
                pass
        return normalized

    def _write(self, state: dict[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self._index_path, state)

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(state if isinstance(state, dict) else {})
        self._write(normalized)
        self.state = normalized
        return normalized

    def _prune(self, state: dict[str, Any]) -> dict[str, Any]:
        snapshots = list(state.get("snapshots") if isinstance(state.get("snapshots"), list) else [])
        snapshots.sort(key=self._entry_sort_key)
        removed = 0
        if len(snapshots) > self.max_items:
            remove_count = len(snapshots) - self.max_items
            to_remove = snapshots[:remove_count]
            snapshots = snapshots[remove_count:]
            removed = len(to_remove)
            for row in to_remove:
                filename = str(row.get("filename") or "").strip()
                if not filename:
                    continue
                target = self.path / filename
                try:
                    if target.exists():
                        target.unlink()
                except OSError:
                    pass
        state["snapshots"] = snapshots
        state["pruned_count_last_run"] = removed
        state["pruned_count_total"] = max(0, _safe_int(state.get("pruned_count_total"), 0) + removed)
        return state

    def _fallback_to_registry_dir(self, registry_path: str) -> None:
        fallback_root = Path(registry_path).expanduser().resolve().parent / ".registry_snapshots"
        self.path = fallback_root
        self._index_path = self.path / "index.json"
        self.state = self.load()

    def _ensure_snapshot_dir(self, registry_path: str) -> None:
        try:
            self.path.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._fallback_to_registry_dir(registry_path)
            self.path.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, registry_path: str, document: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_snapshot_dir(registry_path)
        payload = copy.deepcopy(document if isinstance(document, dict) else load_registry_document(registry_path))
        canonical = _stable_json(payload)
        registry_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        state = copy.deepcopy(self.state)
        next_seq = max(1, _safe_int(state.get("next_seq"), 1))
        snapshot_id = f"s{next_seq:08d}-{registry_hash[:12]}"
        filename = f"{snapshot_id}.json"
        snapshot_path = self.path / filename
        _write_json_atomic(snapshot_path, payload)
        size_bytes = int(snapshot_path.stat().st_size)
        rows = state.get("snapshots") if isinstance(state.get("snapshots"), list) else []
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "filename": filename,
                "registry_hash": registry_hash,
                "size_bytes": size_bytes,
                "seq": next_seq,
            }
        )
        state["snapshots"] = rows
        state["next_seq"] = next_seq + 1
        state = self._prune(state)
        self.save(state)
        return {
            "snapshot_id": snapshot_id,
            "registry_hash": registry_hash,
            "size_bytes": size_bytes,
        }

    def list_snapshots(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.state.get("snapshots") if isinstance(self.state.get("snapshots"), list) else []
        normalized = sorted([dict(item) for item in rows if isinstance(item, dict)], key=self._entry_sort_key)
        output = []
        for row in reversed(normalized):
            output.append(
                {
                    "snapshot_id": str(row.get("snapshot_id") or ""),
                    "registry_hash": str(row.get("registry_hash") or ""),
                    "size_bytes": max(0, _safe_int(row.get("size_bytes"), 0)),
                }
            )
        return output[: max(1, int(limit))]

    def restore_snapshot(self, *, snapshot_id: str, registry_path: str) -> dict[str, Any]:
        target_id = str(snapshot_id or "").strip()
        if not target_id:
            return {"ok": False, "error_kind": "snapshot_not_found"}
        rows = self.state.get("snapshots") if isinstance(self.state.get("snapshots"), list) else []
        selected = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("snapshot_id") or "").strip() == target_id:
                selected = row
                break
        if not isinstance(selected, dict):
            return {"ok": False, "error_kind": "snapshot_not_found"}
        filename = str(selected.get("filename") or "").strip()
        if not filename:
            return {"ok": False, "error_kind": "snapshot_not_found"}
        source = self.path / filename
        if not source.is_file():
            return {"ok": False, "error_kind": "snapshot_file_missing"}
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {"ok": False, "error_kind": "snapshot_read_failed"}
        if not isinstance(payload, dict):
            return {"ok": False, "error_kind": "snapshot_read_failed"}
        ok_verify, verify_error = verify_registry_invariants(payload)
        if not ok_verify:
            return {"ok": False, "error_kind": verify_error or "verify_failed"}
        try:
            _write_json_atomic(Path(registry_path).expanduser().resolve(), payload)
        except OSError:
            return {"ok": False, "error_kind": "snapshot_restore_failed"}
        return {
            "ok": True,
            "snapshot_id": target_id,
            "resulting_registry_hash": _registry_hash(payload),
        }


def snapshot_registry(
    registry_path: str,
    *,
    snapshots_dir: str | None = None,
    max_snapshots: int = _DEFAULT_MAX_SNAPSHOTS,
) -> dict[str, Any]:
    store = RegistrySnapshotStore(path=snapshots_dir, max_items=max(1, int(max_snapshots)))
    return store.create_snapshot(registry_path)


def apply_with_rollback(
    *,
    registry_path: str,
    snapshot_store: RegistrySnapshotStore,
    plan_apply_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    current = load_registry_document(registry_path)
    snapshot_meta = snapshot_store.create_snapshot(registry_path, current)
    snapshot_id = str(snapshot_meta.get("snapshot_id") or "").strip()

    try:
        candidate = copy.deepcopy(plan_apply_fn(copy.deepcopy(current)))
    except Exception:
        return {
            "ok": False,
            "error_kind": "plan_apply_failed",
            "snapshot_id": snapshot_id,
        }
    if not isinstance(candidate, dict):
        return {
            "ok": False,
            "error_kind": "plan_apply_failed",
            "snapshot_id": snapshot_id,
        }

    try:
        _write_json_atomic(Path(registry_path).expanduser().resolve(), candidate)
    except OSError:
        _ = snapshot_store.restore_snapshot(snapshot_id=snapshot_id, registry_path=registry_path)
        return {
            "ok": False,
            "error_kind": "write_failed",
            "snapshot_id": snapshot_id,
        }

    reloaded = load_registry_document(registry_path)
    ok_verify, verify_error = verify_registry_invariants(reloaded)
    if not ok_verify:
        _ = snapshot_store.restore_snapshot(snapshot_id=snapshot_id, registry_path=registry_path)
        return {
            "ok": False,
            "error_kind": "verify_failed",
            "verify_error": verify_error,
            "snapshot_id": snapshot_id,
        }

    return {
        "ok": True,
        "snapshot_id": snapshot_id,
        "resulting_registry_hash": _registry_hash(reloaded),
    }


__all__ = [
    "RegistrySnapshotStore",
    "apply_with_rollback",
    "snapshot_registry",
    "verify_registry_invariants",
]
