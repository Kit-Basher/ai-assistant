from __future__ import annotations

from functools import wraps
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB

from agent.memory_contract import (
    PENDING_STATUS_ABORTED,
    PENDING_STATUS_DONE,
    PENDING_STATUS_EXPIRED,
    PENDING_STATUS_READY_TO_RESUME,
    PENDING_STATUS_WAITING_FOR_USER,
    build_memory_summary,
    deterministic_memory_snapshot,
    is_meta_action,
    normalize_pending_item,
    normalize_thread_state,
)
from agent.working_memory import (
    WorkingMemoryState,
    build_working_memory_summary,
    normalize_working_memory_state,
    working_memory_state_to_dict,
)


_FOLLOWUP_POSITIVE = {
    "yes",
    "y",
    "ok",
    "okay",
    "sure",
    "do it",
    "go ahead",
    "that one",
    "show me more",
}
_FOLLOWUP_NEGATIVE = {"no", "n", "cancel", "stop", "never mind", "nevermind"}
_NEW_THREAD_PHRASES = {"new topic", "different topic", "switch topic", "separate question"}
_MANAGED_RUNTIME_SUFFIXES = (
    "thread_state",
    "pending_items",
    "last_meaningful_user_request",
    "last_agent_action",
    "working_memory_state",
    "persistence_status",
)


def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _with_runtime_lock(method):
    @wraps(method)
    def _wrapped(self, *args: Any, **kwargs: Any):
        with self._lock:
            return method(self, *args, **kwargs)

    return _wrapped


class MemoryRuntime:
    def __init__(self, db: MemoryDB) -> None:
        self._db = db
        self._lock = threading.RLock()
        self._revision_cache: dict[str, int] = {}

    @staticmethod
    def _thread_key(user_id: str) -> str:
        return f"memory_runtime:{str(user_id).strip()}:thread_state"

    @staticmethod
    def _pending_key(user_id: str) -> str:
        return f"memory_runtime:{str(user_id).strip()}:pending_items"

    @staticmethod
    def _last_request_key(user_id: str) -> str:
        return f"memory_runtime:{str(user_id).strip()}:last_meaningful_user_request"

    @staticmethod
    def _last_action_key(user_id: str) -> str:
        return f"memory_runtime:{str(user_id).strip()}:last_agent_action"

    @staticmethod
    def _working_memory_key(user_id: str) -> str:
        return f"memory_runtime:{str(user_id).strip()}:working_memory_state"

    @staticmethod
    def _persistence_status_key(user_id: str) -> str:
        return f"memory_runtime:{str(user_id).strip()}:persistence_status"

    @staticmethod
    def _managed_runtime_key(user_id: str, suffix: str) -> str:
        normalized_suffix = str(suffix or "").strip()
        return f"memory_runtime:{str(user_id).strip()}:{normalized_suffix}"

    @staticmethod
    def _managed_runtime_prefix() -> str:
        return "memory_runtime:"

    @staticmethod
    def _user_id_from_managed_key(key: str) -> tuple[str | None, str | None]:
        normalized = str(key or "").strip()
        prefix = MemoryRuntime._managed_runtime_prefix()
        if not normalized.startswith(prefix):
            return None, None
        remainder = normalized[len(prefix) :]
        for suffix in _MANAGED_RUNTIME_SUFFIXES:
            marker = f":{suffix}"
            if remainder.endswith(marker):
                user_id = remainder[: -len(marker)]
                return user_id, suffix
        return None, None

    @staticmethod
    def _inspect_raw_json(raw: str | None, *, expected_type: type[Any]) -> dict[str, Any]:
        if raw is None:
            return {
                "status": "missing",
                "healthy": True,
                "error": None,
                "parsed": None,
                "raw_preview": None,
            }
        raw_text = str(raw)
        preview = raw_text if len(raw_text) <= 160 else raw_text[:157] + "..."
        try:
            parsed = json.loads(raw_text)
        except (TypeError, ValueError):
            return {
                "status": "corrupt_json",
                "healthy": False,
                "error": "invalid_json",
                "parsed": None,
                "raw_preview": preview,
            }
        if not isinstance(parsed, expected_type):
            return {
                "status": "invalid_type",
                "healthy": False,
                "error": f"expected_{expected_type.__name__}",
                "parsed": None,
                "raw_preview": preview,
            }
        return {
            "status": "ok",
            "healthy": True,
            "error": None,
            "parsed": parsed,
            "raw_preview": preview,
        }

    def _inspect_json_entry(self, key: str, *, expected_type: type[Any], update_cache: bool = True) -> dict[str, Any]:
        entry = self._db.get_user_pref_entry(key)
        if entry is None:
            if update_cache:
                self._revision_cache[key] = 0
            inspect = self._inspect_raw_json(None, expected_type=expected_type)
            return {**inspect, "revision": 0, "updated_at": None}
        revision = int(entry.get("revision") or 0)
        if update_cache:
            self._revision_cache[key] = revision
        inspect = self._inspect_raw_json(entry.get("value"), expected_type=expected_type)
        return {
            **inspect,
            "revision": revision,
            "updated_at": entry.get("updated_at"),
        }

    def _load_json(self, key: str, default: Any) -> Any:
        entry = self._db.get_user_pref_entry(key)
        if entry is None:
            self._revision_cache[key] = 0
            return default
        raw = entry.get("value")
        self._revision_cache[key] = int(entry.get("revision") or 0)
        if not raw:
            return default
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return default
        return parsed

    def _expected_revision_for_key(self, key: str) -> int:
        if key in self._revision_cache:
            return int(self._revision_cache.get(key) or 0)
        entry = self._db.get_user_pref_entry(key)
        revision = int((entry or {}).get("revision") or 0)
        self._revision_cache[key] = revision
        return revision

    @staticmethod
    def _clean_persistence_event(event: dict[str, Any]) -> dict[str, Any]:
        attempted_at = int(event.get("attempted_at") or _now_epoch())
        stored_revision = event.get("stored_revision")
        normalized_stored_revision = (
            int(stored_revision)
            if isinstance(stored_revision, int)
            or (isinstance(stored_revision, str) and str(stored_revision).isdigit())
            else None
        )
        return {
            "kind": str(event.get("kind") or "").strip() or None,
            "key": str(event.get("key") or "").strip() or None,
            "status": str(event.get("status") or "").strip() or None,
            "reason": str(event.get("reason") or "").strip() or None,
            "expected_revision": int(event.get("expected_revision") or 0),
            "stored_revision": normalized_stored_revision,
            "rejected": bool(event.get("rejected", False)),
            "attempted_at": attempted_at,
        }

    def _persist_json_revisioned(
        self,
        user_id: str,
        *,
        key: str,
        kind: str,
        payload: Any,
        record_status: bool = True,
    ) -> bool:
        expected_revision = self._expected_revision_for_key(key)
        result = self._db.set_user_pref_if_revision(
            key,
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            expected_revision,
        )
        if bool(result.get("ok", False)):
            new_revision = int(result.get("revision") or (expected_revision + 1))
            self._revision_cache[key] = new_revision
            if record_status:
                self._record_persistence_status(
                    user_id,
                    {
                        "kind": kind,
                        "key": key,
                        "status": "ok",
                        "reason": None,
                        "expected_revision": expected_revision,
                        "stored_revision": new_revision,
                        "rejected": False,
                        "attempted_at": _now_epoch(),
                    },
                )
            return True
        if record_status:
            self._record_persistence_status(
                user_id,
                {
                    "kind": kind,
                    "key": key,
                    "status": "revision_conflict",
                    "reason": "stale_write_conflict",
                    "expected_revision": expected_revision,
                    "stored_revision": result.get("revision"),
                    "rejected": True,
                    "attempted_at": _now_epoch(),
                },
            )
        return False

    def _persist_text_revisioned(
        self,
        user_id: str,
        *,
        key: str,
        kind: str,
        value: str,
        record_status: bool = True,
    ) -> bool:
        expected_revision = self._expected_revision_for_key(key)
        result = self._db.set_user_pref_if_revision(key, value, expected_revision)
        if bool(result.get("ok", False)):
            new_revision = int(result.get("revision") or (expected_revision + 1))
            self._revision_cache[key] = new_revision
            if record_status:
                self._record_persistence_status(
                    user_id,
                    {
                        "kind": kind,
                        "key": key,
                        "status": "ok",
                        "reason": None,
                        "expected_revision": expected_revision,
                        "stored_revision": new_revision,
                        "rejected": False,
                        "attempted_at": _now_epoch(),
                    },
                )
            return True
        if record_status:
            self._record_persistence_status(
                user_id,
                {
                    "kind": kind,
                    "key": key,
                    "status": "revision_conflict",
                    "reason": "stale_write_conflict",
                    "expected_revision": expected_revision,
                    "stored_revision": result.get("revision"),
                    "rejected": True,
                    "attempted_at": _now_epoch(),
                },
            )
        return False

    def _record_persistence_status(self, user_id: str, event: dict[str, Any]) -> None:
        key = self._persistence_status_key(user_id)
        cleaned = self._clean_persistence_event(event)
        for _ in range(2):
            current = self._load_json(
                key,
                {
                    "last_attempted_write": None,
                    "last_successful_write": None,
                    "last_conflict": None,
                    "active_conflict": False,
                },
            )
            if not isinstance(current, dict):
                current = {
                    "last_attempted_write": None,
                    "last_successful_write": None,
                    "last_conflict": None,
                    "active_conflict": False,
                }
            current["last_attempted_write"] = cleaned
            if bool(cleaned.get("rejected", False)):
                current["last_conflict"] = cleaned
                current["active_conflict"] = True
            else:
                current["last_successful_write"] = cleaned
                current["active_conflict"] = False
            if self._persist_json_revisioned(
                user_id,
                key=key,
                kind="persistence_status",
                payload=current,
                record_status=False,
            ):
                return

    def _inspect_working_memory_record(self, user_id: str, *, update_cache: bool = True) -> dict[str, Any]:
        key = self._working_memory_key(user_id)
        inspect = self._inspect_json_entry(key, expected_type=dict, update_cache=update_cache)
        if inspect["status"] == "missing":
            return {
                "key": key,
                "status": "missing",
                "healthy": True,
                "error": None,
                "raw_preview": inspect["raw_preview"],
                "revision": int(inspect.get("revision") or 0),
                "state": WorkingMemoryState(),
            }
        if not bool(inspect.get("healthy", False)):
            return {
                "key": key,
                "status": inspect["status"],
                "healthy": False,
                "error": inspect["error"],
                "raw_preview": inspect["raw_preview"],
                "revision": int(inspect.get("revision") or 0),
                "state": None,
            }
        parsed = inspect.get("parsed") if isinstance(inspect.get("parsed"), dict) else {}
        try:
            normalized = normalize_working_memory_state(parsed)
        except Exception as exc:
            return {
                "key": key,
                "status": "invalid_payload",
                "healthy": False,
                "error": exc.__class__.__name__,
                "raw_preview": inspect["raw_preview"],
                "revision": int(inspect.get("revision") or 0),
                "state": None,
            }
        return {
            "key": key,
            "status": "ok",
            "healthy": True,
            "error": None,
            "raw_preview": inspect["raw_preview"],
            "revision": int(inspect.get("revision") or 0),
            "state": normalized,
        }

    @staticmethod
    def _default_thread_id(user_id: str) -> str:
        return f"user:{str(user_id).strip() or 'unknown'}"

    def _persistence_status(self, user_id: str) -> dict[str, Any]:
        inspect = self._inspect_json_entry(self._persistence_status_key(user_id), expected_type=dict)
        if inspect["status"] != "ok":
            return {
                "healthy": True,
                "status": inspect["status"],
                "last_attempted_write": None,
                "last_successful_write": None,
                "last_conflict": None,
                "active_conflict": False,
            }
        parsed = inspect.get("parsed") if isinstance(inspect.get("parsed"), dict) else {}
        last_attempted_write = (
            parsed.get("last_attempted_write") if isinstance(parsed.get("last_attempted_write"), dict) else None
        )
        last_successful_write = (
            parsed.get("last_successful_write") if isinstance(parsed.get("last_successful_write"), dict) else None
        )
        last_conflict = parsed.get("last_conflict") if isinstance(parsed.get("last_conflict"), dict) else None
        return {
            "healthy": True,
            "status": "ok",
            "last_attempted_write": (
                dict(last_attempted_write) if isinstance(last_attempted_write, dict) else None
            ),
            "last_successful_write": (
                dict(last_successful_write) if isinstance(last_successful_write, dict) else None
            ),
            "last_conflict": dict(last_conflict) if isinstance(last_conflict, dict) else None,
            "active_conflict": bool(parsed.get("active_conflict", False)),
        }

    @_with_runtime_lock
    def get_thread_state(self, user_id: str) -> dict[str, Any]:
        parsed = self._load_json(self._thread_key(user_id), {})
        if not isinstance(parsed, dict):
            parsed = {}
        normalized = normalize_thread_state(
            parsed,
            user_id=user_id,
            default_thread_id=self._default_thread_id(user_id),
        )
        return normalized

    @_with_runtime_lock
    def set_thread_state(self, user_id: str, **updates: Any) -> dict[str, Any]:
        current = self.get_thread_state(user_id)
        merged = dict(current)
        for key, value in updates.items():
            merged[str(key)] = value
        merged["user_id"] = str(user_id)
        merged["updated_at"] = _now_epoch()
        normalized = normalize_thread_state(
            merged,
            user_id=user_id,
            default_thread_id=self._default_thread_id(user_id),
        )
        if not self._persist_json_revisioned(
            user_id,
            key=self._thread_key(user_id),
            kind="thread_state",
            payload=normalized,
        ):
            return self.get_thread_state(user_id)
        return normalized

    @_with_runtime_lock
    def set_current_topic(
        self,
        user_id: str,
        *,
        topic: str | None,
        runtime_mode: str | None = None,
        last_tool: str | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {"current_topic": topic, "status": status}
        if runtime_mode is not None:
            updates["runtime_mode"] = runtime_mode
        if last_tool is not None:
            updates["last_tool"] = last_tool
        return self.set_thread_state(user_id, **updates)

    @_with_runtime_lock
    def set_last_tool(self, user_id: str, tool_name: str | None) -> dict[str, Any]:
        return self.set_thread_state(user_id, last_tool=(str(tool_name).strip().lower() or None))

    @_with_runtime_lock
    def record_user_request(self, user_id: str, text: str) -> None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        self._persist_text_revisioned(
            user_id,
            key=self._last_request_key(user_id),
            kind="last_meaningful_user_request",
            value=cleaned,
        )

    @_with_runtime_lock
    def record_agent_action(self, user_id: str, text: str, *, action_kind: str | None = None) -> bool:
        if is_meta_action(action_kind):
            return False
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return False
        return self._persist_text_revisioned(
            user_id,
            key=self._last_action_key(user_id),
            kind="last_agent_action",
            value=cleaned,
        )

    def _load_pending_items(self, user_id: str) -> list[dict[str, Any]]:
        parsed = self._load_json(self._pending_key(user_id), [])
        if not isinstance(parsed, list):
            parsed = []
        thread_id = self.get_thread_state(user_id).get("thread_id") or self._default_thread_id(user_id)
        normalized = [
            normalize_pending_item(item, default_thread_id=str(thread_id))
            for item in parsed
            if isinstance(item, dict)
        ]
        normalized.sort(key=lambda row: (int(row.get("created_at") or 0), str(row.get("pending_id") or "")))
        return normalized

    def _save_pending_items(self, user_id: str, rows: list[dict[str, Any]]) -> bool:
        ordered = sorted(
            [normalize_pending_item(row, default_thread_id=self._default_thread_id(user_id)) for row in rows],
            key=lambda row: (int(row.get("created_at") or 0), str(row.get("pending_id") or "")),
        )
        return self._persist_json_revisioned(
            user_id,
            key=self._pending_key(user_id),
            kind="pending_items",
            payload=ordered,
        )

    @_with_runtime_lock
    def clear_expired_pending_items(self, user_id: str, *, now_ts: int | None = None) -> int:
        timestamp = int(now_ts if isinstance(now_ts, int) else _now_epoch())
        changed = 0
        rows = self._load_pending_items(user_id)
        for row in rows:
            if row["status"] in {PENDING_STATUS_DONE, PENDING_STATUS_ABORTED, PENDING_STATUS_EXPIRED}:
                continue
            if int(row.get("expires_at") or 0) <= timestamp:
                row["status"] = PENDING_STATUS_EXPIRED
                changed += 1
        if changed:
            self._save_pending_items(user_id, rows)
        return changed

    @_with_runtime_lock
    def list_pending_items(
        self,
        user_id: str,
        *,
        thread_id: str | None = None,
        include_expired: bool = False,
        now_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        timestamp = int(now_ts if isinstance(now_ts, int) else _now_epoch())
        self.clear_expired_pending_items(user_id, now_ts=timestamp)
        rows = self._load_pending_items(user_id)
        out: list[dict[str, Any]] = []
        for row in rows:
            if thread_id and str(row.get("thread_id") or "").strip() != str(thread_id).strip():
                continue
            if not include_expired and row["status"] == PENDING_STATUS_EXPIRED:
                continue
            out.append(dict(row))
        out.sort(key=lambda row: (int(row.get("created_at") or 0), str(row.get("pending_id") or "")))
        return out

    @_with_runtime_lock
    def add_pending_item(self, user_id: str, item: dict[str, Any]) -> dict[str, Any]:
        rows = self._load_pending_items(user_id)
        thread_id = str(item.get("thread_id") or "").strip() or self.get_thread_state(user_id)["thread_id"]
        normalized = normalize_pending_item(
            {
                "pending_id": item.get("pending_id") or f"pending-{uuid.uuid4().hex[:10]}",
                "kind": item.get("kind"),
                "origin_tool": item.get("origin_tool"),
                "question": item.get("question"),
                "options": item.get("options") if isinstance(item.get("options"), list) else [],
                "created_at": item.get("created_at") or _now_epoch(),
                "expires_at": item.get("expires_at") or (_now_epoch() + 600),
                "thread_id": thread_id,
                "status": item.get("status") or PENDING_STATUS_WAITING_FOR_USER,
            },
            default_thread_id=thread_id,
        )
        context = item.get("context") if isinstance(item.get("context"), dict) else {}
        if context:
            normalized["context"] = {str(k): context[k] for k in sorted(context.keys())}
        rows = [row for row in rows if str(row.get("pending_id") or "") != normalized["pending_id"]]
        rows.append(normalized)
        self._save_pending_items(user_id, rows)
        return normalized

    @_with_runtime_lock
    def set_pending_status(self, user_id: str, pending_id: str, status: str) -> bool:
        rows = self._load_pending_items(user_id)
        changed = False
        normalized_status = str(status or "").strip().upper()
        if normalized_status not in {
            PENDING_STATUS_WAITING_FOR_USER,
            PENDING_STATUS_READY_TO_RESUME,
            PENDING_STATUS_EXPIRED,
            PENDING_STATUS_DONE,
            PENDING_STATUS_ABORTED,
        }:
            normalized_status = PENDING_STATUS_ABORTED
        for row in rows:
            if str(row.get("pending_id") or "") != str(pending_id or ""):
                continue
            row["status"] = normalized_status
            changed = True
        if changed:
            self._save_pending_items(user_id, rows)
        return changed

    @_with_runtime_lock
    def abort_pending_for_thread(self, user_id: str, thread_id: str) -> int:
        rows = self._load_pending_items(user_id)
        changed = 0
        for row in rows:
            if str(row.get("thread_id") or "").strip() != str(thread_id or "").strip():
                continue
            if row["status"] in {PENDING_STATUS_DONE, PENDING_STATUS_ABORTED, PENDING_STATUS_EXPIRED}:
                continue
            row["status"] = PENDING_STATUS_ABORTED
            changed += 1
        if changed:
            self._save_pending_items(user_id, rows)
        return changed

    @_with_runtime_lock
    def remove_pending_item(self, user_id: str, pending_id: str) -> bool:
        rows = self._load_pending_items(user_id)
        kept = [row for row in rows if str(row.get("pending_id") or "") != str(pending_id or "")]
        if len(kept) == len(rows):
            return False
        self._save_pending_items(user_id, kept)
        return True

    @_with_runtime_lock
    def resolve_followup(self, user_id: str, text: str, current_thread_id: str) -> dict[str, Any]:
        normalized = _normalize_text(text)
        if not normalized:
            return {"type": "none", "reason": "empty"}
        kind: str | None = None
        if normalized in _FOLLOWUP_POSITIVE:
            kind = "accept"
        elif normalized in _FOLLOWUP_NEGATIVE:
            kind = "decline"
        elif normalized in {"show me more", "details", "that one"}:
            kind = "details"
        if kind is None:
            return {"type": "none", "reason": "not_followup"}

        rows = [
            row
            for row in self.list_pending_items(
                user_id,
                thread_id=current_thread_id,
                include_expired=True,
            )
            if row["status"] in {PENDING_STATUS_WAITING_FOR_USER, PENDING_STATUS_READY_TO_RESUME}
        ]
        expired_rows = [
            row
            for row in self.list_pending_items(
                user_id,
                thread_id=current_thread_id,
                include_expired=True,
            )
            if row["status"] == PENDING_STATUS_EXPIRED
        ]
        if not rows:
            if expired_rows:
                return {
                    "type": "expired",
                    "reason": "pending_expired",
                    "intent": kind,
                    "pending_ids": [str(row.get("pending_id") or "") for row in expired_rows],
                }
            return {"type": "none", "reason": "no_pending", "intent": kind}
        if len(rows) > 1:
            return {
                "type": "ambiguous",
                "reason": "multiple_pending",
                "intent": kind,
                "pending_ids": [str(row.get("pending_id") or "") for row in rows],
            }
        return {"type": "match", "reason": "single_pending", "intent": kind, "pending_item": dict(rows[0])}

    @_with_runtime_lock
    def is_ambiguous_followup(self, user_id: str, text: str, current_thread_id: str) -> bool:
        result = self.resolve_followup(user_id, text, current_thread_id)
        return str(result.get("type") or "") == "ambiguous"

    @_with_runtime_lock
    def should_start_new_thread(self, user_id: str, text: str, current_thread_id: str) -> bool:
        _ = current_thread_id
        normalized = _normalize_text(text)
        if not normalized:
            return False
        if any(phrase in normalized for phrase in _NEW_THREAD_PHRASES):
            return True
        thread = self.get_thread_state(user_id)
        topic = _normalize_text(str(thread.get("current_topic") or ""))
        if not topic:
            return False
        if len(normalized.split()) < 4:
            return False
        topic_tokens = {token for token in re.split(r"[^a-z0-9]+", topic) if token}
        text_tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
        if not topic_tokens or not text_tokens:
            return False
        overlap = topic_tokens.intersection(text_tokens)
        return len(overlap) == 0 and any(trigger in normalized for trigger in {"instead", "different", "new"})

    @_with_runtime_lock
    def get_memory_summary(self, user_id: str, thread_id: str | None = None) -> dict[str, Any]:
        thread = self.get_thread_state(user_id)
        if thread_id and str(thread_id).strip() and str(thread.get("thread_id") or "").strip() != str(thread_id).strip():
            thread = self.set_thread_state(user_id, thread_id=str(thread_id).strip())
        pending = self.list_pending_items(user_id, thread_id=str(thread.get("thread_id") or "").strip())
        return build_memory_summary(
            thread_state=thread,
            pending_items=pending,
            last_meaningful_user_request=self._db.get_user_pref(self._last_request_key(user_id)),
            last_agent_action=self._db.get_user_pref(self._last_action_key(user_id)),
        )

    @_with_runtime_lock
    def deterministic_snapshot(self, user_id: str, thread_id: str | None = None) -> dict[str, Any]:
        thread = self.get_thread_state(user_id)
        if thread_id and str(thread_id).strip():
            thread = self.set_thread_state(user_id, thread_id=str(thread_id).strip())
        thread_id_value = str(thread.get("thread_id") or "").strip() or self._default_thread_id(user_id)
        pending = self.list_pending_items(user_id, thread_id=thread_id_value, include_expired=True)
        return deterministic_memory_snapshot(
            thread_state=thread,
            pending_items=pending,
            last_meaningful_user_request=self._db.get_user_pref(self._last_request_key(user_id)),
            last_agent_action=self._db.get_user_pref(self._last_action_key(user_id)),
        )

    @_with_runtime_lock
    def load_working_memory_state(self, user_id: str) -> tuple[WorkingMemoryState, dict[str, Any] | None]:
        inspect = self._inspect_working_memory_record(user_id)
        if inspect["status"] == "missing":
            return WorkingMemoryState(), None
        if not bool(inspect.get("healthy", False)):
            return WorkingMemoryState(), {
                "key": inspect["key"],
                "status": inspect["status"],
                "error": inspect["error"],
                "raw_preview": inspect["raw_preview"],
                "revision": int(inspect.get("revision") or 0),
            }
        state = inspect.get("state")
        if isinstance(state, WorkingMemoryState):
            return state, None
        return WorkingMemoryState(), {
            "key": inspect["key"],
            "status": "invalid_payload",
            "error": "unknown",
            "raw_preview": inspect["raw_preview"],
            "revision": int(inspect.get("revision") or 0),
        }

    @_with_runtime_lock
    def save_working_memory_state(
        self,
        user_id: str,
        state: WorkingMemoryState,
        *,
        refuse_if_corrupt: bool = True,
    ) -> bool:
        if refuse_if_corrupt:
            inspect = self._inspect_working_memory_record(user_id, update_cache=False)
            if inspect["status"] not in {"missing", "ok"}:
                self._record_persistence_status(
                    user_id,
                    {
                        "kind": "working_memory_state",
                        "key": self._working_memory_key(user_id),
                        "status": inspect["status"],
                        "reason": "refused_corrupt_existing_state",
                        "expected_revision": int(inspect.get("revision") or 0),
                        "stored_revision": inspect.get("revision"),
                        "rejected": True,
                        "updated_at": _now_epoch(),
                    },
                )
                return False
        key = self._working_memory_key(user_id)
        return self._persist_json_revisioned(
            user_id,
            key=key,
            kind="working_memory_state",
            payload=working_memory_state_to_dict(state),
        )

    @_with_runtime_lock
    def get_working_memory_summary(self, user_id: str) -> dict[str, Any]:
        state, issue = self.load_working_memory_state(user_id)
        if issue is not None:
            return {
                "healthy": False,
                "status": str(issue.get("status") or "corrupt_json"),
                "error": str(issue.get("error") or "invalid_json"),
                "summary": None,
            }
        return {
            "healthy": True,
            "status": "ok",
            "error": None,
            "summary": build_working_memory_summary(state),
        }

    @_with_runtime_lock
    def inspect_user_state(self, user_id: str) -> dict[str, Any]:
        thread_key = self._thread_key(user_id)
        pending_key = self._pending_key(user_id)
        last_request_key = self._last_request_key(user_id)
        last_action_key = self._last_action_key(user_id)

        thread_inspect = self._inspect_json_entry(thread_key, expected_type=dict)
        pending_inspect = self._inspect_json_entry(pending_key, expected_type=list)
        working_memory_inspect = self._inspect_working_memory_record(user_id)
        persistence_status = self._persistence_status(user_id)
        last_request_entry = self._db.get_user_pref_entry(last_request_key)
        last_action_entry = self._db.get_user_pref_entry(last_action_key)
        self._revision_cache[last_request_key] = int((last_request_entry or {}).get("revision") or 0)
        self._revision_cache[last_action_key] = int((last_action_entry or {}).get("revision") or 0)
        thread_parsed = thread_inspect.get("parsed") if isinstance(thread_inspect.get("parsed"), dict) else None
        pending_parsed = pending_inspect.get("parsed") if isinstance(pending_inspect.get("parsed"), list) else None
        active_thread_id = None
        pending_count = 0
        working_memory_summary = None
        if thread_parsed is not None:
            active_thread_id = str(thread_parsed.get("thread_id") or "").strip() or None
        if pending_parsed is not None:
            pending_count = len([row for row in pending_parsed if isinstance(row, dict)])
        if bool(working_memory_inspect.get("healthy", False)) and isinstance(working_memory_inspect.get("state"), WorkingMemoryState):
            working_memory_summary = build_working_memory_summary(
                working_memory_inspect["state"]
            )
        entries = [
            {
                "key": thread_key,
                "kind": "thread_state",
                "status": thread_inspect["status"],
                "healthy": bool(thread_inspect["healthy"]),
                "error": thread_inspect["error"],
                "raw_preview": thread_inspect["raw_preview"],
                "revision": int(thread_inspect.get("revision") or 0),
            },
            {
                "key": pending_key,
                "kind": "pending_items",
                "status": pending_inspect["status"],
                "healthy": bool(pending_inspect["healthy"]),
                "error": pending_inspect["error"],
                "raw_preview": pending_inspect["raw_preview"],
                "revision": int(pending_inspect.get("revision") or 0),
            },
            {
                "key": last_request_key,
                "kind": "last_meaningful_user_request",
                "status": "ok" if last_request_entry is not None else "missing",
                "healthy": True,
                "error": None,
                "raw_preview": None,
                "revision": int((last_request_entry or {}).get("revision") or 0),
            },
            {
                "key": last_action_key,
                "kind": "last_agent_action",
                "status": "ok" if last_action_entry is not None else "missing",
                "healthy": True,
                "error": None,
                "raw_preview": None,
                "revision": int((last_action_entry or {}).get("revision") or 0),
            },
            {
                "key": working_memory_inspect["key"],
                "kind": "working_memory_state",
                "status": working_memory_inspect["status"],
                "healthy": bool(working_memory_inspect["healthy"]),
                "error": working_memory_inspect["error"],
                "raw_preview": working_memory_inspect["raw_preview"],
                "revision": int(working_memory_inspect.get("revision") or 0),
            },
        ]
        corrupt_entries = [dict(entry) for entry in entries if not bool(entry.get("healthy", False))]
        return {
            "user_id": str(user_id),
            "healthy": not corrupt_entries,
            "active_thread_id": active_thread_id,
            "pending_count": pending_count,
            "working_memory": {
                "healthy": bool(working_memory_inspect["healthy"]),
                "status": working_memory_inspect["status"],
                "error": working_memory_inspect["error"],
                "revision": int(working_memory_inspect.get("revision") or 0),
                "summary": working_memory_summary,
            },
            "persistence": {
                "contract": {
                    "mode": "per_key_optimistic_concurrency_control",
                    "storage_model": "full_record_replace",
                    "merge_on_write": False,
                    "cross_key_atomic_snapshot": False,
                },
                "current_revisions": {
                    "thread_state": int(thread_inspect.get("revision") or 0),
                    "pending_items": int(pending_inspect.get("revision") or 0),
                    "last_meaningful_user_request": int((last_request_entry or {}).get("revision") or 0),
                    "last_agent_action": int((last_action_entry or {}).get("revision") or 0),
                    "working_memory_state": int(working_memory_inspect.get("revision") or 0),
                },
                "last_attempted_write": persistence_status.get("last_attempted_write"),
                "last_successful_write": persistence_status.get("last_successful_write"),
                "last_conflict": persistence_status.get("last_conflict"),
                "active_conflict": bool(persistence_status.get("active_conflict", False)),
            },
            "entries": entries,
            "corrupt_entries": corrupt_entries,
        }

    @_with_runtime_lock
    def inspect_all_state(self) -> dict[str, Any]:
        rows = self._db.list_user_prefs()
        by_user: dict[str, dict[str, Any]] = {}
        unknown_keys: list[str] = []
        for row in rows:
            key = str((row or {}).get("key") or "").strip()
            user_id, suffix = self._user_id_from_managed_key(key)
            if user_id is None or suffix is None:
                if key.startswith(self._managed_runtime_prefix()):
                    unknown_keys.append(key)
                continue
            if user_id not in by_user:
                by_user[user_id] = self.inspect_user_state(user_id)
        users = [by_user[user_id] for user_id in sorted(by_user.keys())]
        degraded_users = [row for row in users if not bool(row.get("healthy", False))]
        return {
            "users": users,
            "user_count": len(users),
            "healthy": not degraded_users and not unknown_keys,
            "degraded_user_count": len(degraded_users),
            "unknown_keys": sorted(unknown_keys),
        }

    @_with_runtime_lock
    def reset_user_state(self, user_id: str) -> dict[str, Any]:
        deleted = 0
        deleted_keys: list[str] = []
        for suffix in _MANAGED_RUNTIME_SUFFIXES:
            key = self._managed_runtime_key(user_id, suffix)
            if self._db.delete_user_pref(key):
                deleted += 1
                deleted_keys.append(key)
        return {
            "user_id": str(user_id),
            "deleted_keys": deleted_keys,
            "deleted_count": deleted,
        }

    @_with_runtime_lock
    def reset_all_state(self) -> dict[str, Any]:
        rows = self._db.list_user_prefs()
        keys = [
            str((row or {}).get("key") or "").strip()
            for row in rows
            if str((row or {}).get("key") or "").strip().startswith(self._managed_runtime_prefix())
        ]
        deleted = 0
        for key in keys:
            if self._db.delete_user_pref(key):
                deleted += 1
        return {
            "deleted_count": deleted,
            "deleted_keys": sorted(keys),
        }
