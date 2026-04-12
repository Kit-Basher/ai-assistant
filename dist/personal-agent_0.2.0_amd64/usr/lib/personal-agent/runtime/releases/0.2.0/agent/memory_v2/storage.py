from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Protocol

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
    pinned INTEGER NOT NULL DEFAULT 0,
    fact_key TEXT,
    fact_group TEXT,
    superseded_at INTEGER,
    is_current INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_memory_items_level_updated
    ON memory_items(level, updated_at DESC, id ASC);
CREATE INDEX IF NOT EXISTS idx_memory_items_semantic_current
    ON memory_items(level, fact_group, fact_key, is_current, updated_at DESC, id ASC);

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
    def list_memory_items(self, *, level: MemoryLevel, include_history: bool = False) -> list[MemoryItem]:
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
            self._ensure_memory_items_columns(conn)
            conn.commit()
        finally:
            conn.close()

    def _ensure_memory_items_columns(self, conn: sqlite3.Connection) -> None:
        cur = conn.execute("PRAGMA table_info(memory_items)")
        rows = cur.fetchall()
        existing = {str(row["name"]) for row in rows}
        if "fact_key" not in existing:
            conn.execute("ALTER TABLE memory_items ADD COLUMN fact_key TEXT")
        if "fact_group" not in existing:
            conn.execute("ALTER TABLE memory_items ADD COLUMN fact_group TEXT")
        if "superseded_at" not in existing:
            conn.execute("ALTER TABLE memory_items ADD COLUMN superseded_at INTEGER")
        if "is_current" not in existing:
            conn.execute("ALTER TABLE memory_items ADD COLUMN is_current INTEGER NOT NULL DEFAULT 1")

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
        fact_key_raw = row["fact_key"] if "fact_key" in row.keys() else None
        fact_group_raw = row["fact_group"] if "fact_group" in row.keys() else None
        superseded_raw = row["superseded_at"] if "superseded_at" in row.keys() else None
        is_current_raw = row["is_current"] if "is_current" in row.keys() else 1
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
            fact_key=str(fact_key_raw).strip() if fact_key_raw is not None and str(fact_key_raw).strip() else None,
            fact_group=(
                str(fact_group_raw).strip() if fact_group_raw is not None and str(fact_group_raw).strip() else None
            ),
            superseded_at=int(superseded_raw) if superseded_raw is not None else None,
            is_current=bool(int(is_current_raw)),
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
            str(item.fact_key).strip() if item.fact_key is not None and str(item.fact_key).strip() else None,
            str(item.fact_group).strip() if item.fact_group is not None and str(item.fact_group).strip() else None,
            int(item.superseded_at) if item.superseded_at is not None else None,
            1 if bool(item.is_current) else 0,
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO memory_items (
                    id, level, text, created_at, updated_at, tags_json, source_kind, source_ref, pinned,
                    fact_key, fact_group, superseded_at, is_current
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    level = excluded.level,
                    text = excluded.text,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    tags_json = excluded.tags_json,
                    source_kind = excluded.source_kind,
                    source_ref = excluded.source_ref,
                    pinned = excluded.pinned,
                    fact_key = excluded.fact_key,
                    fact_group = excluded.fact_group,
                    superseded_at = excluded.superseded_at,
                    is_current = excluded.is_current
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
            fact_key=str(item.fact_key).strip() if item.fact_key is not None and str(item.fact_key).strip() else None,
            fact_group=(
                str(item.fact_group).strip() if item.fact_group is not None and str(item.fact_group).strip() else None
            ),
            superseded_at=int(item.superseded_at) if item.superseded_at is not None else None,
            is_current=bool(item.is_current),
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

    def list_memory_items(self, *, level: MemoryLevel, include_history: bool = False) -> list[MemoryItem]:
        normalized = self._coerce_level(level)
        if normalized == MemoryLevel.EPISODIC:
            return self.list_episodic_events(limit=500)
        where = "WHERE level = ?"
        params: list[Any] = [normalized.value]
        if normalized == MemoryLevel.SEMANTIC and not include_history:
            where += " AND is_current = 1"
        conn = self._connect()
        try:
            cur = conn.execute(
                f"""
                SELECT
                    id, level, text, created_at, updated_at, tags_json, source_kind, source_ref, pinned,
                    fact_key, fact_group, superseded_at, is_current
                FROM memory_items
                {where}
                ORDER BY pinned DESC, updated_at DESC, id ASC
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return [self._row_to_item(row) for row in rows]

    def get_current_semantic(self, *, fact_group: str, fact_key: str) -> MemoryItem | None:
        group = str(fact_group or "").strip()
        key = str(fact_key or "").strip()
        if not group or not key:
            return None
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT
                    id, level, text, created_at, updated_at, tags_json, source_kind, source_ref, pinned,
                    fact_key, fact_group, superseded_at, is_current
                FROM memory_items
                WHERE level = ? AND fact_group = ? AND fact_key = ? AND is_current = 1
                ORDER BY updated_at DESC, id ASC
                LIMIT 1
                """,
                (MemoryLevel.SEMANTIC.value, group, key),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_item(row)

    def supersede_semantic(self, *, item_id: str, superseded_at: int) -> None:
        identifier = str(item_id or "").strip()
        if not identifier:
            return
        ts = int(superseded_at)
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE memory_items
                SET is_current = 0,
                    superseded_at = ?,
                    updated_at = CASE
                        WHEN updated_at < ? THEN ?
                        ELSE updated_at
                    END
                WHERE id = ? AND level = ?
                """,
                (ts, ts, ts, identifier, MemoryLevel.SEMANTIC.value),
            )
            conn.commit()
        finally:
            conn.close()

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

    def stats(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM memory_items) AS memory_items_count,
                    (SELECT COUNT(*) FROM memory_items WHERE level = ?) AS semantic_items_count,
                    (SELECT COUNT(*) FROM memory_items WHERE level = ?) AS procedural_items_count,
                    (SELECT COUNT(*) FROM memory_events) AS episodic_events_count,
                    (SELECT COUNT(*) FROM bootstrap_state) AS bootstrap_state_count
                """,
                (
                    MemoryLevel.SEMANTIC.value,
                    MemoryLevel.PROCEDURAL.value,
                ),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {
                "memory_items_count": 0,
                "semantic_items_count": 0,
                "procedural_items_count": 0,
                "episodic_events_count": 0,
                "bootstrap_state_count": 0,
            }
        return {
            "memory_items_count": int(row["memory_items_count"]),
            "semantic_items_count": int(row["semantic_items_count"]),
            "procedural_items_count": int(row["procedural_items_count"]),
            "episodic_events_count": int(row["episodic_events_count"]),
            "bootstrap_state_count": int(row["bootstrap_state_count"]),
        }

    def clear_all(self) -> dict[str, int]:
        before = self.stats()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memory_items")
            conn.execute("DELETE FROM memory_events")
            conn.execute("DELETE FROM bootstrap_state")
            conn.commit()
        finally:
            conn.close()
        return before
