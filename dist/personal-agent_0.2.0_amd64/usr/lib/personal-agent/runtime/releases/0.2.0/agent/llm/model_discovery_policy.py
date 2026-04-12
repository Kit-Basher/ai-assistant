from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


_DISCOVERY_POLICY_SCHEMA_VERSION = 1
_POLICY_STATUSES = {"known_good", "known_stale", "avoid"}
_ROLE_HINTS = {"coding", "research", "cheap_cloud", "local_best"}


def _normalize_string(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = {
        str(item).strip().lower()
        for item in values
        if str(item).strip()
    }
    return sorted(normalized)


def allowed_model_discovery_statuses() -> list[str]:
    return sorted(_POLICY_STATUSES)


def allowed_model_discovery_role_hints() -> list[str]:
    return sorted(_ROLE_HINTS)


def normalize_model_discovery_policy(document: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(document if isinstance(document, dict) else {})
    entries = raw.get("entries") if isinstance(raw.get("entries"), dict) else {}
    normalized_entries: dict[str, dict[str, Any]] = {}
    for model_id, payload in entries.items():
        normalized_model_id = _normalize_string(model_id)
        if not normalized_model_id or not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status not in _POLICY_STATUSES:
            continue
        role_hints = [
            item
            for item in _normalize_string_list(payload.get("role_hints"))
            if item in _ROLE_HINTS
        ]
        normalized_entries[normalized_model_id] = {
            "model_id": normalized_model_id,
            "status": status,
            "role_hints": role_hints,
            "notes": _normalize_string(payload.get("notes")),
            "source": _normalize_string(payload.get("source")) or "operator_review",
            "justification": _normalize_string(payload.get("justification")),
            "reviewed_at": _normalize_string(payload.get("reviewed_at")),
        }
    return {
        "schema_version": _DISCOVERY_POLICY_SCHEMA_VERSION,
        "entries": normalized_entries,
    }


class ModelDiscoveryPolicyStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.state = self.load()

    @staticmethod
    def default_path() -> str:
        configured = str(os.getenv("AGENT_MODEL_DISCOVERY_POLICY_PATH", "") or "").strip()
        if configured:
            return configured
        return str(Path.home() / ".local" / "share" / "personal-agent" / "model_discovery_policy.json")

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return normalize_model_discovery_policy({})
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return normalize_model_discovery_policy({})
        return normalize_model_discovery_policy(parsed if isinstance(parsed, dict) else {})

    def save(self, document: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_model_discovery_policy(document)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized, ensure_ascii=True, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
        self.state = normalized
        return dict(self.state)

    def list_entries(self) -> list[dict[str, Any]]:
        entries = self.state.get("entries") if isinstance(self.state.get("entries"), dict) else {}
        rows = [dict(payload) for payload in entries.values() if isinstance(payload, dict)]
        rows.sort(key=lambda row: str(row.get("model_id") or ""))
        return rows

    def get_entry(self, model_id: str) -> dict[str, Any] | None:
        normalized_model_id = _normalize_string(model_id)
        if not normalized_model_id:
            return None
        entries = self.state.get("entries") if isinstance(self.state.get("entries"), dict) else {}
        payload = entries.get(normalized_model_id)
        return dict(payload) if isinstance(payload, dict) else None

    def upsert_entry(
        self,
        model_id: str,
        *,
        status: str,
        role_hints: list[str] | None = None,
        notes: str | None = None,
        source: str | None = None,
        justification: str | None = None,
        reviewed_at: str | None = None,
    ) -> dict[str, Any]:
        normalized_model_id = _normalize_string(model_id)
        normalized_status = str(status or "").strip().lower()
        normalized_role_hints = _normalize_string_list(role_hints or [])
        if not normalized_model_id or normalized_status not in _POLICY_STATUSES:
            raise ValueError("invalid_model_discovery_policy_entry")
        invalid_role_hints = [item for item in normalized_role_hints if item not in _ROLE_HINTS]
        if invalid_role_hints:
            raise ValueError("invalid_model_discovery_role_hints")
        document = normalize_model_discovery_policy(self.state)
        entries = document.get("entries") if isinstance(document.get("entries"), dict) else {}
        entries[normalized_model_id] = {
            "model_id": normalized_model_id,
            "status": normalized_status,
            "role_hints": normalized_role_hints,
            "notes": _normalize_string(notes),
            "source": _normalize_string(source) or "operator_review",
            "justification": _normalize_string(justification),
            "reviewed_at": _normalize_string(reviewed_at),
        }
        document["entries"] = entries
        self.save(document)
        return dict(entries[normalized_model_id])

    def remove_entry(self, model_id: str) -> bool:
        normalized_model_id = _normalize_string(model_id)
        if not normalized_model_id:
            raise ValueError("invalid_model_discovery_policy_entry")
        document = normalize_model_discovery_policy(self.state)
        entries = document.get("entries") if isinstance(document.get("entries"), dict) else {}
        existed = normalized_model_id in entries
        if existed:
            entries.pop(normalized_model_id, None)
            document["entries"] = entries
            self.save(document)
        return existed


__all__ = [
    "ModelDiscoveryPolicyStore",
    "allowed_model_discovery_role_hints",
    "allowed_model_discovery_statuses",
    "normalize_model_discovery_policy",
]
