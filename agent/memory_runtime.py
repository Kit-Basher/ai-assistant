from __future__ import annotations

import json
import re
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
    normalize_pending_item,
    normalize_thread_state,
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


def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


class MemoryRuntime:
    def __init__(self, db: MemoryDB) -> None:
        self._db = db

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

    def _load_json(self, key: str, default: Any) -> Any:
        raw = self._db.get_user_pref(key)
        if not raw:
            return default
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return default
        return parsed

    def _save_json(self, key: str, payload: Any) -> None:
        self._db.set_user_pref(
            key,
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        )

    @staticmethod
    def _default_thread_id(user_id: str) -> str:
        return f"user:{str(user_id).strip() or 'unknown'}"

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
        self._save_json(self._thread_key(user_id), normalized)
        return normalized

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

    def set_last_tool(self, user_id: str, tool_name: str | None) -> dict[str, Any]:
        return self.set_thread_state(user_id, last_tool=(str(tool_name).strip().lower() or None))

    def record_user_request(self, user_id: str, text: str) -> None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        self._db.set_user_pref(self._last_request_key(user_id), cleaned)

    def record_agent_action(self, user_id: str, text: str) -> None:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return
        self._db.set_user_pref(self._last_action_key(user_id), cleaned)

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

    def _save_pending_items(self, user_id: str, rows: list[dict[str, Any]]) -> None:
        ordered = sorted(
            [normalize_pending_item(row, default_thread_id=self._default_thread_id(user_id)) for row in rows],
            key=lambda row: (int(row.get("created_at") or 0), str(row.get("pending_id") or "")),
        )
        self._save_json(self._pending_key(user_id), ordered)

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

    def remove_pending_item(self, user_id: str, pending_id: str) -> bool:
        rows = self._load_pending_items(user_id)
        kept = [row for row in rows if str(row.get("pending_id") or "") != str(pending_id or "")]
        if len(kept) == len(rows):
            return False
        self._save_pending_items(user_id, kept)
        return True

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

    def is_ambiguous_followup(self, user_id: str, text: str, current_thread_id: str) -> bool:
        result = self.resolve_followup(user_id, text, current_thread_id)
        return str(result.get("type") or "") == "ambiguous"

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
