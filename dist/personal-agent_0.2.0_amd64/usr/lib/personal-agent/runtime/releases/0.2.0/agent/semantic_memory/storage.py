from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from agent.semantic_memory.types import SemanticChunkRecord, SemanticIndexState, SemanticSourceKind, SemanticSourceRecord


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS semantic_sources (
    id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    scope TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    title TEXT,
    thread_id TEXT,
    project_id TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ready',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_semantic_sources_scope_updated
    ON semantic_sources(scope, updated_at DESC, id ASC);
CREATE INDEX IF NOT EXISTS idx_semantic_sources_kind_scope
    ON semantic_sources(source_kind, scope, updated_at DESC, id ASC);

CREATE TABLE IF NOT EXISTS semantic_chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES semantic_sources(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_semantic_chunks_source
    ON semantic_chunks(source_id, chunk_index ASC, id ASC);
CREATE INDEX IF NOT EXISTS idx_semantic_chunks_hash
    ON semantic_chunks(chunk_hash, id ASC);

CREATE TABLE IF NOT EXISTS semantic_vectors (
    chunk_id TEXT PRIMARY KEY,
    embed_provider TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES semantic_chunks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_semantic_vectors_model_dim
    ON semantic_vectors(embed_model, embedding_dim, chunk_id ASC);

CREATE TABLE IF NOT EXISTS semantic_index_state (
    scope TEXT PRIMARY KEY,
    embed_provider TEXT,
    embed_model TEXT,
    embedding_dim INTEGER,
    status TEXT NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    vector_count INTEGER NOT NULL DEFAULT 0,
    last_indexed_at INTEGER,
    stale_since INTEGER,
    last_error_kind TEXT,
    last_error_message TEXT,
    updated_at INTEGER NOT NULL,
    details_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_semantic_index_state_updated
    ON semantic_index_state(updated_at DESC, scope ASC);
"""


def _now_ts() -> int:
    return int(time.time())


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class SQLiteSemanticStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _normalize_kind(value: SemanticSourceKind | str) -> SemanticSourceKind:
        if isinstance(value, SemanticSourceKind):
            return value
        normalized = str(value or "").strip().lower()
        for kind in SemanticSourceKind:
            if normalized == kind.value:
                return kind
        raise ValueError(f"unsupported semantic source kind: {value}")

    @staticmethod
    def _normalize_scope(scope: str | None) -> str:
        value = str(scope or "").strip()
        return value or "global"

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> SemanticSourceRecord:
        metadata_raw = str(row["metadata_json"] or "").strip()
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return SemanticSourceRecord(
            id=str(row["id"]),
            source_kind=SemanticSourceKind(str(row["source_kind"])),
            source_ref=str(row["source_ref"]),
            scope=str(row["scope"]),
            content_hash=str(row["content_hash"]),
            title=str(row["title"]).strip() if row["title"] is not None and str(row["title"]).strip() else None,
            thread_id=str(row["thread_id"]).strip() if row["thread_id"] is not None and str(row["thread_id"]).strip() else None,
            project_id=str(row["project_id"]).strip() if row["project_id"] is not None and str(row["project_id"]).strip() else None,
            pinned=bool(int(row["pinned"])),
            status=str(row["status"] or "ready"),
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
            metadata=metadata,
        )

    def upsert_source(
        self,
        *,
        source_id: str,
        source_kind: SemanticSourceKind | str,
        source_ref: str,
        scope: str | None,
        content_hash: str,
        title: str | None = None,
        thread_id: str | None = None,
        project_id: str | None = None,
        pinned: bool = False,
        status: str = "ready",
        metadata: dict[str, Any] | None = None,
        created_at: int | None = None,
        updated_at: int | None = None,
    ) -> SemanticSourceRecord:
        kind = self._normalize_kind(source_kind)
        scope_value = self._normalize_scope(scope)
        created = int(created_at) if created_at is not None else _now_ts()
        updated = int(updated_at) if updated_at is not None else created
        if updated < created:
            updated = created
        payload = (
            str(source_id),
            kind.value,
            str(source_ref),
            scope_value,
            str(content_hash),
            str(title).strip() if title is not None and str(title).strip() else None,
            str(thread_id).strip() if thread_id is not None and str(thread_id).strip() else None,
            str(project_id).strip() if project_id is not None and str(project_id).strip() else None,
            1 if pinned else 0,
            str(status or "ready"),
            created,
            updated,
            _stable_json(metadata or {}),
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO semantic_sources (
                    id, source_kind, source_ref, scope, content_hash, title, thread_id, project_id, pinned,
                    status, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_kind = excluded.source_kind,
                    source_ref = excluded.source_ref,
                    scope = excluded.scope,
                    content_hash = excluded.content_hash,
                    title = excluded.title,
                    thread_id = excluded.thread_id,
                    project_id = excluded.project_id,
                    pinned = excluded.pinned,
                    status = excluded.status,
                    created_at = CASE WHEN semantic_sources.created_at > excluded.created_at THEN excluded.created_at ELSE semantic_sources.created_at END,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()
        return SemanticSourceRecord(
            id=str(source_id),
            source_kind=kind,
            source_ref=str(source_ref),
            scope=scope_value,
            content_hash=str(content_hash),
            title=str(title).strip() if title is not None and str(title).strip() else None,
            thread_id=str(thread_id).strip() if thread_id is not None and str(thread_id).strip() else None,
            project_id=str(project_id).strip() if project_id is not None and str(project_id).strip() else None,
            pinned=bool(pinned),
            status=str(status or "ready"),
            created_at=created,
            updated_at=updated,
            metadata=metadata or {},
        )

    def replace_chunks(self, *, source_id: str, chunks: list[dict[str, Any]]) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM semantic_vectors WHERE chunk_id IN (SELECT id FROM semantic_chunks WHERE source_id = ?)", (str(source_id),))
            conn.execute("DELETE FROM semantic_chunks WHERE source_id = ?", (str(source_id),))
            for row in chunks:
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                conn.execute(
                    """
                    INSERT INTO semantic_chunks (
                        id, source_id, chunk_index, text, chunk_hash, char_start, char_end, created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row["id"]),
                        str(source_id),
                        int(row["chunk_index"]),
                        str(row["text"]),
                        str(row["chunk_hash"]),
                        int(row["char_start"]) if row.get("char_start") is not None else None,
                        int(row["char_end"]) if row.get("char_end") is not None else None,
                        int(row["created_at"]),
                        int(row["updated_at"]),
                        _stable_json(metadata),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def upsert_vector(
        self,
        *,
        chunk_id: str,
        embed_provider: str,
        embed_model: str,
        embedding_dim: int,
        vector: tuple[float, ...],
        created_at: int | None = None,
        updated_at: int | None = None,
    ) -> None:
        created = int(created_at) if created_at is not None else _now_ts()
        updated = int(updated_at) if updated_at is not None else created
        if updated < created:
            updated = created
        vector_values = tuple(float(item) for item in vector)
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive")
        if len(vector_values) != int(embedding_dim):
            raise ValueError("embedding vector length mismatch")
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO semantic_vectors (
                    chunk_id, embed_provider, embed_model, embedding_dim, vector_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    embed_provider = excluded.embed_provider,
                    embed_model = excluded.embed_model,
                    embedding_dim = excluded.embedding_dim,
                    vector_json = excluded.vector_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(chunk_id),
                    str(embed_provider),
                    str(embed_model),
                    int(embedding_dim),
                    _stable_json([float(item) for item in vector_values]),
                    created,
                    updated,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_joined_rows(
        self,
        *,
        scopes: list[str] | None = None,
        source_kinds: list[SemanticSourceKind] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        scope_values = [self._normalize_scope(scope) for scope in scopes or [] if str(scope or "").strip()]
        kind_values = [kind.value for kind in source_kinds or []]
        where: list[str] = ["sv.vector_json IS NOT NULL"]
        params: list[Any] = []
        if scope_values:
            placeholders = ",".join("?" for _ in scope_values)
            where.append(f"ss.scope IN ({placeholders})")
            params.extend(scope_values)
        if kind_values:
            placeholders = ",".join("?" for _ in kind_values)
            where.append(f"ss.source_kind IN ({placeholders})")
            params.extend(kind_values)
        sql = f"""
            SELECT
                ss.id AS source_id,
                ss.source_kind,
                ss.source_ref,
                ss.scope,
                ss.content_hash,
                ss.title,
                ss.thread_id,
                ss.project_id,
                ss.pinned,
                ss.status,
                ss.created_at AS source_created_at,
                ss.updated_at AS source_updated_at,
                ss.metadata_json AS source_metadata_json,
                sc.id AS chunk_id,
                sc.chunk_index,
                sc.text AS chunk_text,
                sc.chunk_hash,
                sc.char_start,
                sc.char_end,
                sc.created_at AS chunk_created_at,
                sc.updated_at AS chunk_updated_at,
                sc.metadata_json AS chunk_metadata_json,
                sv.embed_provider,
                sv.embed_model,
                sv.embedding_dim,
                sv.vector_json,
                sv.created_at AS vector_created_at,
                sv.updated_at AS vector_updated_at
            FROM semantic_sources ss
            JOIN semantic_chunks sc ON sc.source_id = ss.id
            JOIN semantic_vectors sv ON sv.chunk_id = sc.id
            {"WHERE " + " AND ".join(where) if where else ""}
            ORDER BY ss.pinned DESC, ss.updated_at DESC, sc.chunk_index ASC, sc.id ASC
            LIMIT ?
        """
        params.append(max(1, int(limit)))
        conn = self._connect()
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        finally:
            conn.close()
        output: list[dict[str, Any]] = []
        for row in rows:
            try:
                source_metadata = json.loads(str(row["source_metadata_json"] or "") or "{}")
            except json.JSONDecodeError:
                source_metadata = {}
            if not isinstance(source_metadata, dict):
                source_metadata = {}
            try:
                chunk_metadata = json.loads(str(row["chunk_metadata_json"] or "") or "{}")
            except json.JSONDecodeError:
                chunk_metadata = {}
            if not isinstance(chunk_metadata, dict):
                chunk_metadata = {}
            output.append(
                {
                    "source_id": str(row["source_id"]),
                    "source_kind": str(row["source_kind"]),
                    "source_ref": str(row["source_ref"]),
                    "scope": str(row["scope"]),
                    "content_hash": str(row["content_hash"]),
                    "title": str(row["title"]).strip() if row["title"] is not None and str(row["title"]).strip() else None,
                    "thread_id": str(row["thread_id"]).strip() if row["thread_id"] is not None and str(row["thread_id"]).strip() else None,
                    "project_id": str(row["project_id"]).strip() if row["project_id"] is not None and str(row["project_id"]).strip() else None,
                    "pinned": bool(int(row["pinned"])),
                    "status": str(row["status"] or "ready"),
                    "source_created_at": int(row["source_created_at"]),
                    "source_updated_at": int(row["source_updated_at"]),
                    "source_metadata": source_metadata,
                    "chunk_id": str(row["chunk_id"]),
                    "chunk_index": int(row["chunk_index"]),
                    "chunk_text": str(row["chunk_text"]),
                    "chunk_hash": str(row["chunk_hash"]),
                    "char_start": int(row["char_start"]) if row["char_start"] is not None else None,
                    "char_end": int(row["char_end"]) if row["char_end"] is not None else None,
                    "chunk_created_at": int(row["chunk_created_at"]),
                    "chunk_updated_at": int(row["chunk_updated_at"]),
                    "chunk_metadata": chunk_metadata,
                    "embed_provider": str(row["embed_provider"]),
                    "embed_model": str(row["embed_model"]),
                    "embedding_dim": int(row["embedding_dim"]),
                    "vector_json": str(row["vector_json"]),
                    "vector_created_at": int(row["vector_created_at"]),
                    "vector_updated_at": int(row["vector_updated_at"]),
                }
            )
        return output

    def list_sources(self, *, scope: str | None = None, source_kind: SemanticSourceKind | None = None) -> list[SemanticSourceRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(self._normalize_scope(scope))
        if source_kind is not None:
            clauses.append("source_kind = ?")
            params.append(source_kind.value)
        sql = "SELECT * FROM semantic_sources"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY pinned DESC, updated_at DESC, id ASC"
        conn = self._connect()
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        finally:
            conn.close()
        return [self._record_from_row(row) for row in rows]

    def get_source(self, source_id: str) -> SemanticSourceRecord | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM semantic_sources WHERE id = ? LIMIT 1", (str(source_id),)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._record_from_row(row)

    def list_chunks(self, *, source_id: str | None = None) -> list[SemanticChunkRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(str(source_id))
        sql = "SELECT * FROM semantic_chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY source_id ASC, chunk_index ASC, id ASC"
        conn = self._connect()
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        finally:
            conn.close()
        output: list[SemanticChunkRecord] = []
        for row in rows:
            metadata_raw = str(row["metadata_json"] or "").strip()
            try:
                metadata = json.loads(metadata_raw) if metadata_raw else {}
            except json.JSONDecodeError:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            output.append(
                SemanticChunkRecord(
                    id=str(row["id"]),
                    source_id=str(row["source_id"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    chunk_hash=str(row["chunk_hash"]),
                    char_start=int(row["char_start"]) if row["char_start"] is not None else None,
                    char_end=int(row["char_end"]) if row["char_end"] is not None else None,
                    created_at=int(row["created_at"]),
                    updated_at=int(row["updated_at"]),
                    metadata=metadata,
                )
            )
        return output

    def count_contents(
        self,
        *,
        scope: str | None = None,
        source_kind: SemanticSourceKind | None = None,
        source_kinds: list[SemanticSourceKind] | None = None,
    ) -> dict[str, int]:
        scope_value = self._normalize_scope(scope) if scope is not None else None
        kind_values = [kind.value for kind in source_kinds or [] if kind is not None]
        kind_value = source_kind.value if source_kind is not None else None
        where: list[str] = []
        params: list[Any] = []
        if scope_value is not None:
            where.append("ss.scope = ?")
            params.append(scope_value)
        if kind_values:
            placeholders = ",".join("?" for _ in kind_values)
            where.append(f"ss.source_kind IN ({placeholders})")
            params.extend(kind_values)
        elif kind_value is not None:
            where.append("ss.source_kind = ?")
            params.append(kind_value)
        filter_sql = "WHERE " + " AND ".join(where) if where else ""
        conn = self._connect()
        try:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(DISTINCT ss.id) AS source_count,
                    COUNT(DISTINCT sc.id) AS chunk_count,
                    COUNT(DISTINCT sv.chunk_id) AS vector_count,
                    COUNT(DISTINCT CASE WHEN sv.chunk_id IS NULL THEN sc.id END) AS missing_vector_count
                FROM semantic_sources ss
                LEFT JOIN semantic_chunks sc ON sc.source_id = ss.id
                LEFT JOIN semantic_vectors sv ON sv.chunk_id = sc.id
                {filter_sql}
                """,
                tuple(params),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {"source_count": 0, "chunk_count": 0, "vector_count": 0, "missing_vector_count": 0}
        return {
            "source_count": int(row["source_count"] or 0),
            "chunk_count": int(row["chunk_count"] or 0),
            "vector_count": int(row["vector_count"] or 0),
            "missing_vector_count": int(row["missing_vector_count"] or 0),
        }

    def get_index_state(self, scope: str = "global") -> SemanticIndexState | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM semantic_index_state WHERE scope = ? LIMIT 1", (self._normalize_scope(scope),)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._state_from_row(row)

    @staticmethod
    def _state_from_row(row: sqlite3.Row) -> SemanticIndexState:
        details_raw = str(row["details_json"] or "").strip()
        try:
            details = json.loads(details_raw) if details_raw else {}
        except json.JSONDecodeError:
            details = {}
        if not isinstance(details, dict):
            details = {}
        return SemanticIndexState(
            scope=str(row["scope"]),
            embed_provider=str(row["embed_provider"]) if row["embed_provider"] is not None else None,
            embed_model=str(row["embed_model"]) if row["embed_model"] is not None else None,
            embedding_dim=int(row["embedding_dim"]) if row["embedding_dim"] is not None else None,
            status=str(row["status"] or "unknown"),
            source_count=int(row["source_count"]),
            chunk_count=int(row["chunk_count"]),
            vector_count=int(row["vector_count"]),
            last_indexed_at=int(row["last_indexed_at"]) if row["last_indexed_at"] is not None else None,
            stale_since=int(row["stale_since"]) if row["stale_since"] is not None else None,
            last_error_kind=str(row["last_error_kind"]) if row["last_error_kind"] is not None else None,
            last_error_message=str(row["last_error_message"]) if row["last_error_message"] is not None else None,
            updated_at=int(row["updated_at"]),
            details=details,
        )

    def set_index_state(
        self,
        *,
        scope: str = "global",
        embed_provider: str | None,
        embed_model: str | None,
        embedding_dim: int | None,
        status: str,
        source_count: int,
        chunk_count: int,
        vector_count: int,
        last_indexed_at: int | None = None,
        stale_since: int | None = None,
        last_error_kind: str | None = None,
        last_error_message: str | None = None,
        details: dict[str, Any] | None = None,
        updated_at: int | None = None,
    ) -> SemanticIndexState:
        ts = int(updated_at) if updated_at is not None else _now_ts()
        scope_value = self._normalize_scope(scope)
        payload = (
            scope_value,
            str(embed_provider) if embed_provider is not None else None,
            str(embed_model) if embed_model is not None else None,
            int(embedding_dim) if embedding_dim is not None else None,
            str(status or "unknown"),
            max(0, int(source_count)),
            max(0, int(chunk_count)),
            max(0, int(vector_count)),
            int(last_indexed_at) if last_indexed_at is not None else None,
            int(stale_since) if stale_since is not None else None,
            str(last_error_kind) if last_error_kind is not None else None,
            str(last_error_message) if last_error_message is not None else None,
            ts,
            _stable_json(details or {}),
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO semantic_index_state (
                    scope, embed_provider, embed_model, embedding_dim, status, source_count, chunk_count, vector_count,
                    last_indexed_at, stale_since, last_error_kind, last_error_message, updated_at, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    embed_provider = excluded.embed_provider,
                    embed_model = excluded.embed_model,
                    embedding_dim = excluded.embedding_dim,
                    status = excluded.status,
                    source_count = excluded.source_count,
                    chunk_count = excluded.chunk_count,
                    vector_count = excluded.vector_count,
                    last_indexed_at = excluded.last_indexed_at,
                    stale_since = excluded.stale_since,
                    last_error_kind = excluded.last_error_kind,
                    last_error_message = excluded.last_error_message,
                    updated_at = excluded.updated_at,
                    details_json = excluded.details_json
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()
        return SemanticIndexState(
            scope=scope_value,
            embed_provider=payload[1],
            embed_model=payload[2],
            embedding_dim=payload[3],
            status=payload[4],
            source_count=payload[5],
            chunk_count=payload[6],
            vector_count=payload[7],
            last_indexed_at=payload[8],
            stale_since=payload[9],
            last_error_kind=payload[10],
            last_error_message=payload[11],
            updated_at=ts,
            details=details or {},
        )

    def delete_source(self, source_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM semantic_sources WHERE id = ?", (str(source_id),))
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict[str, int]:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM semantic_sources) AS source_count,
                    (SELECT COUNT(*) FROM semantic_chunks) AS chunk_count,
                    (SELECT COUNT(*) FROM semantic_vectors) AS vector_count,
                    (SELECT COUNT(*) FROM semantic_index_state) AS index_state_count
                """
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {
                "source_count": 0,
                "chunk_count": 0,
                "vector_count": 0,
                "index_state_count": 0,
            }
        return {
            "source_count": int(row["source_count"]),
            "chunk_count": int(row["chunk_count"]),
            "vector_count": int(row["vector_count"]),
            "index_state_count": int(row["index_state_count"]),
        }

    def clear_all(self) -> dict[str, int]:
        before = self.stats()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM semantic_vectors")
            conn.execute("DELETE FROM semantic_chunks")
            conn.execute("DELETE FROM semantic_sources")
            conn.execute("DELETE FROM semantic_index_state")
            conn.commit()
        finally:
            conn.close()
        return before
