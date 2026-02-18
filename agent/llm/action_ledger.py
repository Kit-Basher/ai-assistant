from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


_SCHEMA_VERSION = 2
_DEFAULT_MAX_ITEMS = 400


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iso_from_epoch(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    if int(epoch) <= 0:
        return None
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()


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


class ActionLedgerStore:
    def __init__(self, path: str | None = None, *, max_items: int = _DEFAULT_MAX_ITEMS) -> None:
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.max_items = max(1, int(max_items))
        self.state = self.load()

    @staticmethod
    def default_path() -> str:
        env_value = os.getenv("LLM_AUTOPILOT_LEDGER_PATH", "").strip()
        if env_value:
            return env_value
        return str(Path.home() / ".local" / "share" / "personal-agent" / "autopilot_action_ledger.json")

    @staticmethod
    def empty_state() -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "next_seq": 1,
            "entries": [],
        }

    @staticmethod
    def _entry_sort_key(row: dict[str, Any]) -> tuple[int, str]:
        return (_safe_int(row.get("seq"), 0), str(row.get("id") or ""))

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.empty_state()
        state["next_seq"] = max(1, _safe_int(payload.get("next_seq"), 1))
        raw_rows = payload.get("entries") if isinstance(payload.get("entries"), list) else []
        rows: list[dict[str, Any]] = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            row_id = str(item.get("id") or "").strip()
            seq = _safe_int(item.get("seq"), 0)
            if not row_id or seq <= 0:
                continue
            changed_ids = sorted(
                {
                    str(raw).strip()
                    for raw in (item.get("changed_ids") or [])
                    if str(raw).strip()
                }
            )
            row = {
                "id": row_id,
                "seq": seq,
                "ts": _safe_int(item.get("ts"), 0),
                "ts_iso": _iso_from_epoch(_safe_int(item.get("ts"), 0)),
                "action": str(item.get("action") or "").strip(),
                "actor": str(item.get("actor") or "").strip(),
                "decision": str(item.get("decision") or "").strip(),
                "outcome": str(item.get("outcome") or "").strip(),
                "reason": str(item.get("reason") or "").strip(),
                "trigger": str(item.get("trigger") or "").strip() or None,
                "snapshot_id": (
                    str(item.get("snapshot_id_before") or "").strip()
                    or str(item.get("snapshot_id") or "").strip()
                    or None
                ),
                "snapshot_id_before": (
                    str(item.get("snapshot_id_before") or "").strip()
                    or str(item.get("snapshot_id") or "").strip()
                    or None
                ),
                "snapshot_id_after": str(item.get("snapshot_id_after") or "").strip() or None,
                "resulting_registry_hash": str(item.get("resulting_registry_hash") or "").strip() or None,
                "changed_ids": changed_ids,
            }
            rows.append(row)
        rows.sort(key=self._entry_sort_key)
        if len(rows) > self.max_items:
            rows = rows[-self.max_items :]
        if rows:
            state["next_seq"] = max(int(state["next_seq"]), int(rows[-1]["seq"]) + 1)
        state["entries"] = rows
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
        raw_s = json.dumps(raw, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        norm_s = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if raw_s != norm_s:
            try:
                self._write(normalized)
            except OSError:
                pass
        return normalized

    def _write(self, state: dict[str, Any]) -> None:
        _write_json_atomic(self.path, state)

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(state if isinstance(state, dict) else {})
        self._write(normalized)
        self.state = normalized
        return normalized

    def append(
        self,
        *,
        ts: int,
        action: str,
        actor: str,
        decision: str,
        outcome: str,
        reason: str,
        trigger: str | None,
        snapshot_id: str | None,
        snapshot_id_after: str | None = None,
        resulting_registry_hash: str | None = None,
        changed_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        state = json.loads(json.dumps(self.state, ensure_ascii=True))
        seq = max(1, _safe_int(state.get("next_seq"), 1))
        normalized_changed = sorted(
            {
                str(item).strip()
                for item in (changed_ids or [])
                if str(item).strip()
            }
        )
        payload_for_hash = {
            "seq": seq,
            "action": str(action or "").strip(),
            "actor": str(actor or "").strip(),
            "decision": str(decision or "").strip(),
            "outcome": str(outcome or "").strip(),
            "reason": str(reason or "").strip(),
            "trigger": str(trigger or "").strip() or None,
            "snapshot_id_before": str(snapshot_id or "").strip() or None,
            "snapshot_id_after": str(snapshot_id_after or "").strip() or None,
            "resulting_registry_hash": str(resulting_registry_hash or "").strip() or None,
            "changed_ids": normalized_changed,
        }
        row_id = hashlib.sha256(
            json.dumps(payload_for_hash, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        row = {
            "id": row_id,
            "seq": seq,
            "ts": int(ts),
            "ts_iso": _iso_from_epoch(int(ts)),
            "snapshot_id": payload_for_hash.get("snapshot_id_before"),
            **payload_for_hash,
        }
        entries = state.get("entries") if isinstance(state.get("entries"), list) else []
        entries.append(row)
        entries.sort(key=self._entry_sort_key)
        if len(entries) > self.max_items:
            entries = entries[-self.max_items :]
        state["entries"] = entries
        state["next_seq"] = seq + 1
        self.save(state)
        return row

    def recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.state.get("entries") if isinstance(self.state.get("entries"), list) else []
        sorted_rows = sorted([dict(item) for item in rows if isinstance(item, dict)], key=self._entry_sort_key)
        return list(reversed(sorted_rows))[: max(1, int(limit))]

    def get(self, ledger_id: str) -> dict[str, Any] | None:
        target = str(ledger_id or "").strip()
        if not target:
            return None
        rows = self.state.get("entries") if isinstance(self.state.get("entries"), list) else []
        for item in rows:
            if not isinstance(item, dict):
                continue
            if str(item.get("id") or "").strip() == target:
                return dict(item)
        return None


__all__ = ["ActionLedgerStore"]
