from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


@dataclass
class KnowledgeCacheEntry:
    user_id: str
    created_at: datetime
    query: str
    facts: dict[str, Any]
    intent: dict[str, Any] | None
    facts_hash: str


def facts_hash(facts: dict[str, Any]) -> str:
    payload = json.dumps(facts, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class KnowledgeQueryCache:
    def __init__(self, ttl_minutes: int = 15, now_fn: Callable[[], datetime] | None = None) -> None:
        self._ttl = timedelta(minutes=int(ttl_minutes))
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._entries: dict[str, KnowledgeCacheEntry] = {}

    def set(
        self,
        user_id: str,
        query: str,
        facts: dict[str, Any],
        intent: dict[str, Any] | None,
        now_dt: datetime | None = None,
    ) -> KnowledgeCacheEntry:
        created_at = now_dt or self._now_fn()
        entry = KnowledgeCacheEntry(
            user_id=user_id,
            created_at=created_at,
            query=query,
            facts=facts,
            intent=intent,
            facts_hash=facts_hash(facts),
        )
        self._entries[user_id] = entry
        return entry

    def get_recent(self, user_id: str, now_dt: datetime | None = None) -> KnowledgeCacheEntry | None:
        entry = self._entries.get(user_id)
        if not entry:
            return None
        now_val = now_dt or self._now_fn()
        if now_val - entry.created_at > self._ttl:
            self._entries.pop(user_id, None)
            return None
        return entry
