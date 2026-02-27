from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Protocol

from agent.memory_v2.types import MemoryItem, MemoryLevel


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    tags_json TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memory_items_level_updated
    ON memory_items(level, updated_at DESC, id ASC);

CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    tags_json TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memory_events_created
    ON memory_events(created_at DESC, id ASC);

CREATE TABLE IF NOT EXISTS bootstrap_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bootstrap_state_updated
    ON bootstrap_state(updated_at DESC, key ASC);
"""


class MemoryStorage(Protocol):
    def list_memory_items(self, *, level: MemoryLevel) -> list[MemoryItem]:
        ...

    def list_episodic_events(self, *, limit: int = 500) -> list[MemoryItem]:
        ...


class SQLiteMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _normalize_tags(tags: dict[str, str] | None) -> dict[str, str]:
        if not isinstance(tags, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in sorted(tags.items(), key=lambda item: str(item[0])):
            key_norm = str(key).strip().lower()
            value_norm = str(value).strip()
            if key_norm and value_norm:
                out[key_norm] = value_norm
        return out

    @staticmethod
    def _tags_to_json(tags: dict[str, str] | None) -> str:
        normalized = SQLiteMemoryStore._normalize_tags(tags)
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _tags_from_json(raw: str | None) -> dict[str, str]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return SQLiteMemoryStore._normalize_tags({str(key): str(value) for key, value in parsed.items()})

    @staticmethod
    def _coerce_level(value: str | MemoryLevel) -> MemoryLevel:
        if isinstance(value, MemoryLevel):
            return value
        normalized = str(value or "").strip().lower()
        if normalized == MemoryLevel.EPISODIC.value:
            return MemoryLevel.EPISODIC
        if normalized == MemoryLevel.SEMANTIC.value:
            return MemoryLevel.SEMANTIC
        if normalized == MemoryLevel.PROCEDURAL.value:
            return MemoryLevel.PROCEDURAL
        raise ValueError(f"unsupported memory level: {value}")

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        level = SQLiteMemoryStore._coerce_level(str(row["level"]))
        return MemoryItem(
            id=str(row["id"]),
            level=level,
            text=str(row["text"]),
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
            tags=SQLiteMemoryStore._tags_from_json(str(row["tags_json"])),
            source_kind=str(row["source_kind"]),
            source_ref=str(row["source_ref"]),
            pinned=bool(int(row["pinned"])),
        )

    def upsert_memory_item(self, item: MemoryItem) -> MemoryItem:
        level = self._coerce_level(item.level)
        if level == MemoryLevel.EPISODIC:
            raise ValueError("use append_episodic_event for episodic rows")
        created_at = int(item.created_at)
        updated_at = int(item.updated_at)
        if updated_at < created_at:
            updated_at = created_at
        payload = (
            str(item.id),
            level.value,
            str(item.text),
            created_at,
            updated_at,
            self._tags_to_json(item.tags),
            str(item.source_kind or "unknown"),
            str(item.source_ref or ""),
            1 if bool(item.pinned) else 0,
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO memory_items (
                    id, level, text, created_at, updated_at, tags_json, source_kind, source_ref, pinned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    level = excluded.level,
                    text = excluded.text,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    tags_json = excluded.tags_json,
                    source_kind = excluded.source_kind,
                    source_ref = excluded.source_ref,
                    pinned = excluded.pinned
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()
        return MemoryItem(
            id=str(item.id),
            level=level,
            text=str(item.text),
            created_at=created_at,
            updated_at=updated_at,
            tags=self._normalize_tags(item.tags),
            source_kind=str(item.source_kind or "unknown"),
            source_ref=str(item.source_ref or ""),
            pinned=bool(item.pinned),
        )

    def append_episodic_event(
        self,
        *,
        text: str,
        tags: dict[str, str] | None,
        source_kind: str,
        source_ref: str,
        pinned: bool = False,
        event_id: str | None = None,
        created_at: int | None = None,
    ) -> MemoryItem:
        created = int(created_at) if created_at is not None else int(time.time())
        stable_seed = f"{text}|{source_kind}|{source_ref}|{created}"
        generated_id = "E-" + hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:12]
        identifier = str(event_id or generated_id)
        tags_json = self._tags_to_json(tags)
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_events (
                    id, text, created_at, tags_json, source_kind, source_ref, pinned
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    str(text),
                    created,
                    tags_json,
                    str(source_kind or "unknown"),
                    str(source_ref or ""),
                    1 if pinned else 0,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return MemoryItem(
            id=identifier,
            level=MemoryLevel.EPISODIC,
            text=str(text),
            created_at=created,
            updated_at=created,
            tags=self._tags_from_json(tags_json),
            source_kind=str(source_kind or "unknown"),
            source_ref=str(source_ref or ""),
            pinned=bool(pinned),
        )

    def list_episodic_events(self, *, limit: int = 500) -> list[MemoryItem]:
        bounded_limit = max(1, int(limit))
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT id, text, created_at, tags_json, source_kind, source_ref, pinned
                FROM memory_events
                ORDER BY created_at DESC, id ASC
                LIMIT ?
                """,
                (bounded_limit,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        output: list[MemoryItem] = []
        for row in rows:
            output.append(
                MemoryItem(
                    id=str(row["id"]),
                    level=MemoryLevel.EPISODIC,
                    text=str(row["text"]),
                    created_at=int(row["created_at"]),
                    updated_at=int(row["created_at"]),
                    tags=self._tags_from_json(str(row["tags_json"])),
                    source_kind=str(row["source_kind"]),
                    source_ref=str(row["source_ref"]),
                    pinned=bool(int(row["pinned"])),
                )
            )
        return output

    def list_memory_items(self, *, level: MemoryLevel) -> list[MemoryItem]:
        normalized = self._coerce_level(level)
        if normalized == MemoryLevel.EPISODIC:
            return self.list_episodic_events(limit=500)
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT id, level, text, created_at, updated_at, tags_json, source_kind, source_ref, pinned
                FROM memory_items
                WHERE level = ?
                ORDER BY pinned DESC, updated_at DESC, id ASC
                """,
                (normalized.value,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return [self._row_to_item(row) for row in rows]

    def get_state(self, key: str) -> str | None:
        normalized = str(key or "").strip()
        if not normalized:
            return None
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT value FROM bootstrap_state WHERE key = ? LIMIT 1",
                (normalized,),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return str(row["value"])

    def set_state(self, key: str, value: str, *, updated_at: int | None = None) -> None:
        normalized_key = str(key or "").strip()
        if not normalized_key:
            raise ValueError("state key is required")
        normalized_value = str(value)
        ts = int(updated_at) if updated_at is not None else int(time.time())
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO bootstrap_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (normalized_key, normalized_value, ts),
            )
            conn.commit()
        finally:
            conn.close()

    def get_bool_state(self, key: str) -> bool:
        value = str(self.get_state(key) or "").strip().lower()
        return value in {"1", "true", "yes", "y", "on"}
