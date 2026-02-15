from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


@dataclass
class NoteRecord:
    id: int
    project_id: int | None
    content: str
    tags: str | None
    created_at: str


@dataclass
class ProjectRecord:
    id: int
    name: str
    pitch: str | None
    status: str
    priority: int | None
    tags: str | None
    created_at: str
    updated_at: str


@dataclass
class ReminderRecord:
    id: int
    when_ts: str
    text: str
    status: str
    created_at: str
    sent_at: str | None = None
    last_error: str | None = None


@dataclass
class PendingClarificationRecord:
    id: str
    user_id: str
    chat_id: str
    intent_type: str
    partial_args_json: str
    question: str
    options_json: str
    expires_at: str
    created_at: str


class MemoryDB:
    SCHEMA_VERSION = 2
    GRAPH_IMPORT_MAX_NODES = 200
    GRAPH_IMPORT_MAX_EDGES = 500
    GRAPH_IMPORT_MAX_ALIASES = 300
    GRAPH_PACK_MAX_THREADS = 10

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._tx_depth = 0

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Any:
        try:
            self._tx_depth += 1
            self._conn.execute("BEGIN")
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            if self._tx_depth > 0:
                self._tx_depth -= 1

    def _commit_if_needed(self) -> None:
        if self._tx_depth == 0:
            self._conn.commit()

    def init_schema(self, schema_path: str) -> None:
        with open(schema_path, "r", encoding="utf-8") as handle:
            script = handle.read()
        self._conn.executescript(script)
        self._ensure_reminder_columns()
        self._ensure_preference_columns()
        self._ensure_user_prefs_table()
        self._ensure_thread_prefs_table()
        self._ensure_thread_anchors_table()
        self._ensure_thread_labels_table()
        self._ensure_graph_tables()
        self._ensure_open_loop_columns()
        self._ensure_schema_meta()
        self._conn.commit()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_note(self, content: str, project_id: int | None, tags: str | None) -> int:
        created_at = self._now_iso()
        cur = self._conn.execute(
            "INSERT INTO notes (project_id, content, tags, created_at) VALUES (?, ?, ?, ?)",
            (project_id, content, tags, created_at),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def list_projects(self) -> list[ProjectRecord]:
        cur = self._conn.execute(
            "SELECT id, name, pitch, status, priority, tags, created_at, updated_at FROM projects ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
        return [ProjectRecord(**dict(row)) for row in rows]

    def find_project_by_name(self, name: str) -> ProjectRecord | None:
        cur = self._conn.execute(
            "SELECT id, name, pitch, status, priority, tags, created_at, updated_at FROM projects WHERE lower(name) = lower(?)",
            (name,),
        )
        row = cur.fetchone()
        return ProjectRecord(**dict(row)) if row else None

    def add_project(self, name: str, pitch: str | None) -> int:
        now = self._now_iso()
        cur = self._conn.execute(
            "INSERT INTO projects (name, pitch, status, priority, tags, created_at, updated_at) VALUES (?, ?, 'active', 3, NULL, ?, ?)",
            (name, pitch, now, now),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def add_task(
        self,
        project_id: int | None,
        title: str,
        effort_mins: int | None,
        impact_1to5: int | None,
    ) -> int:
        now = self._now_iso()
        cur = self._conn.execute(
            "INSERT INTO tasks (project_id, title, effort_mins, impact_1to5, status, created_at, updated_at) VALUES (?, ?, ?, ?, 'todo', ?, ?)",
            (project_id, title, effort_mins, impact_1to5, now, now),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def add_open_loop(self, title: str, due_date: str | None = None, priority: int = 3) -> int:
        now = self._now_iso()
        bounded_priority = max(1, min(3, int(priority)))
        cur = self._conn.execute(
            "INSERT INTO open_loops (title, due_date, priority, status, created_at, completed_at) VALUES (?, ?, ?, 'open', ?, NULL)",
            (title, due_date, bounded_priority, now),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def list_open_loops(
        self,
        status: str = "open",
        limit: int = 20,
        order: str = "due",
    ) -> list[dict[str, Any]]:
        order_sql = (
            "ORDER BY COALESCE(due_date, '9999-12-31') ASC, priority ASC, created_at DESC"
            if order == "due"
            else "ORDER BY priority ASC, COALESCE(due_date, '9999-12-31') ASC, created_at DESC"
            if order == "important"
            else "ORDER BY created_at DESC"
        )
        if status == "all":
            cur = self._conn.execute(
                f"""
                SELECT id, title, due_date, priority, status, created_at, completed_at
                FROM open_loops
                {order_sql}
                LIMIT ?
                """,
                (limit,),
            )
        else:
            cur = self._conn.execute(
                f"""
                SELECT id, title, due_date, priority, status, created_at, completed_at
                FROM open_loops
                WHERE status = ?
                {order_sql}
                LIMIT ?
                """,
                (status, limit),
            )
        return [dict(row) for row in cur.fetchall()]

    def complete_open_loop_by_title(self, title_fragment: str) -> int:
        now = self._now_iso()
        cur = self._conn.execute(
            """
            UPDATE open_loops
            SET status = 'done', completed_at = ?
            WHERE id = (
                SELECT id FROM open_loops
                WHERE status = 'open' AND lower(title) LIKE lower(?)
                ORDER BY created_at DESC
                LIMIT 1
            )
            """,
            (now, f"%{title_fragment}%"),
        )
        self._commit_if_needed()
        return int(cur.rowcount)

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT id, project_id, title, details, effort_mins, impact_1to5, status, due_date, created_at, updated_at
            FROM tasks
            ORDER BY
                CASE status WHEN 'todo' THEN 0 WHEN 'doing' THEN 1 ELSE 2 END,
                updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, project_id, title, details, effort_mins, impact_1to5, status, due_date, created_at, updated_at
            FROM tasks
            WHERE id = ?
            """,
            (int(task_id),),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def mark_task_done(self, task_id: int) -> bool:
        now = self._now_iso()
        cur = self._conn.execute(
            "UPDATE tasks SET status = 'done', updated_at = ? WHERE id = ?",
            (now, int(task_id)),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def add_reminder(self, when_ts: str, text: str) -> int:
        created_at = self._now_iso()
        cur = self._conn.execute(
            "INSERT INTO reminders (when_ts, text, status, created_at) VALUES (?, ?, 'pending', ?)",
            (when_ts, text, created_at),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def list_due_reminders(self, now_ts: str) -> list[ReminderRecord]:
        cur = self._conn.execute(
            """
            SELECT id, when_ts, text, status, created_at, sent_at, last_error
            FROM reminders
            WHERE status = 'pending' AND when_ts <= ?
            ORDER BY when_ts ASC
            """,
            (now_ts,),
        )
        rows = cur.fetchall()
        return [ReminderRecord(**dict(row)) for row in rows]

    def claim_reminder_sent(self, reminder_id: int, sent_at: str) -> bool:
        cur = self._conn.execute(
            "UPDATE reminders SET status = 'sent', sent_at = ? WHERE id = ? AND status = 'pending'",
            (sent_at, reminder_id),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def mark_reminder_failed(self, reminder_id: int, error: str | None = None) -> None:
        self._conn.execute(
            "UPDATE reminders SET status = 'failed', last_error = ? WHERE id = ?",
            (error, reminder_id),
        )
        self._commit_if_needed()

    def _ensure_reminder_columns(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(reminders)")
        cols = {row["name"] for row in cur.fetchall()}
        if "sent_at" not in cols:
            self._conn.execute("ALTER TABLE reminders ADD COLUMN sent_at TEXT")
        if "last_error" not in cols:
            self._conn.execute("ALTER TABLE reminders ADD COLUMN last_error TEXT")

    def _ensure_preference_columns(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(preferences)")
        cols = {row["name"] for row in cur.fetchall()}
        if "updated_at" not in cols:
            self._conn.execute("ALTER TABLE preferences ADD COLUMN updated_at TEXT")

    def _ensure_user_prefs_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_prefs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def _ensure_thread_prefs_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_prefs (
                thread_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, key)
            )
            """
        )

    def _ensure_thread_anchors_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_anchors (
                thread_id TEXT NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                bullets TEXT NOT NULL,
                open_line TEXT NOT NULL
            )
            """
        )

    def _ensure_thread_labels_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_labels (
                thread_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def _ensure_graph_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                thread_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, node_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                thread_id TEXT NOT NULL,
                from_node TEXT NOT NULL,
                to_node TEXT NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, from_node, to_node, relation)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_aliases (
                thread_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                node_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, alias)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_focus (
                thread_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_relation_types (
                thread_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, relation)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_relation_mode (
                thread_id TEXT PRIMARY KEY,
                strict INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_relation_constraints (
                thread_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                "constraint" TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, relation, "constraint")
            )
            """
        )

    def _ensure_open_loop_columns(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(open_loops)")
        cols = {row["name"] for row in cur.fetchall()}
        if "priority" not in cols:
            self._conn.execute("ALTER TABLE open_loops ADD COLUMN priority INTEGER NOT NULL DEFAULT 3")

    def _ensure_schema_meta(self) -> None:
        self._conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.execute(
            """
            INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(self.SCHEMA_VERSION),),
        )

    def get_schema_version(self) -> int:
        cur = self._conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
        row = cur.fetchone()
        if not row:
            return 0
        try:
            return int(row["value"])
        except Exception:
            return 0

    def set_preference(self, key: str, value: str) -> None:
        now = self._now_iso()
        self._conn.execute(
            "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, now),
        )
        self._commit_if_needed()

    def get_preference(self, key: str) -> str | None:
        cur = self._conn.execute(
            "SELECT value FROM preferences WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        return row["value"] if row else None

    def list_preferences(self) -> list[dict[str, Any]]:
        cur = self._conn.execute("SELECT key, value, updated_at FROM preferences ORDER BY key ASC")
        return [dict(row) for row in cur.fetchall()]

    def set_user_pref(self, key: str, value: str) -> None:
        now = self._now_iso()
        self._conn.execute(
            """
            INSERT INTO user_prefs (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, now),
        )
        self._commit_if_needed()

    def get_user_pref(self, key: str) -> str | None:
        cur = self._conn.execute(
            "SELECT value FROM user_prefs WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        return row["value"] if row else None

    def list_user_prefs(self) -> list[dict[str, Any]]:
        cur = self._conn.execute("SELECT key, value, updated_at FROM user_prefs ORDER BY key ASC")
        return [dict(row) for row in cur.fetchall()]

    def clear_user_prefs(self) -> None:
        self._conn.execute("DELETE FROM user_prefs")
        self._commit_if_needed()

    def set_thread_pref(self, thread_id: str, key: str, value: str) -> None:
        now = self._now_iso()
        self._conn.execute(
            """
            INSERT INTO thread_prefs (thread_id, key, value, updated_at) VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (thread_id, key, value, now),
        )
        self._commit_if_needed()

    def get_thread_pref(self, thread_id: str, key: str) -> str | None:
        cur = self._conn.execute(
            "SELECT value FROM thread_prefs WHERE thread_id = ? AND key = ?",
            (thread_id, key),
        )
        row = cur.fetchone()
        return row["value"] if row else None

    def list_thread_prefs(self, thread_id: str) -> dict[str, str]:
        cur = self._conn.execute(
            "SELECT key, value FROM thread_prefs WHERE thread_id = ? ORDER BY key ASC",
            (thread_id,),
        )
        return {str(row["key"]): str(row["value"]) for row in cur.fetchall()}

    def clear_thread_prefs(self, thread_id: str) -> None:
        self._conn.execute("DELETE FROM thread_prefs WHERE thread_id = ?", (thread_id,))
        self._commit_if_needed()

    def add_thread_anchor(self, thread_id: str, title: str, bullets_json: str, open_line: str) -> int:
        created_at = self._now_iso()
        cur = self._conn.execute(
            """
            INSERT INTO thread_anchors (thread_id, created_at, title, bullets, open_line)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, created_at, title, bullets_json, open_line),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def list_thread_anchors(self, thread_id: str, limit: int = 10) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT thread_id, id, created_at, title, bullets, open_line
            FROM thread_anchors
            WHERE thread_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (thread_id, int(limit)),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_latest_anchor_title(self, thread_id: str) -> str | None:
        cur = self._conn.execute(
            """
            SELECT title
            FROM thread_anchors
            WHERE thread_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (thread_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        title = str(row["title"] or "").strip()
        return title if title else None

    def set_thread_label(self, thread_id: str, label: str) -> None:
        now = self._now_iso()
        self._conn.execute(
            """
            INSERT INTO thread_labels (thread_id, label, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                label = excluded.label,
                updated_at = excluded.updated_at
            """,
            (thread_id, label, now),
        )
        self._commit_if_needed()

    def get_thread_label(self, thread_id: str) -> str | None:
        cur = self._conn.execute(
            "SELECT label FROM thread_labels WHERE thread_id = ?",
            (thread_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        value = str(row["label"] or "").strip()
        return value if value else None

    def clear_thread_label(self, thread_id: str) -> None:
        self._conn.execute("DELETE FROM thread_labels WHERE thread_id = ?", (thread_id,))
        self._commit_if_needed()

    def list_thread_labels(self, thread_ids: list[str]) -> dict[str, str]:
        normalized = sorted({str(thread_id).strip() for thread_id in thread_ids if str(thread_id).strip()})
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        cur = self._conn.execute(
            f"""
            SELECT thread_id, label
            FROM thread_labels
            WHERE thread_id IN ({placeholders})
            ORDER BY thread_id ASC
            """,
            tuple(normalized),
        )
        return {
            str(row["thread_id"]): str(row["label"])
            for row in cur.fetchall()
            if str(row["thread_id"]).strip() and str(row["label"]).strip()
        }

    @staticmethod
    def _normalize_graph_node_id(node_id: str) -> str:
        raw = (node_id or "").strip().lower()
        normalized = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
        return normalized

    @staticmethod
    def _normalize_graph_label(label: str) -> str:
        cleaned = " ".join((label or "").replace("?", "").split()).strip()
        if len(cleaned) > 80:
            cleaned = cleaned[:80].rstrip()
        return cleaned

    @staticmethod
    def _normalize_graph_relation(relation: str) -> str:
        cleaned = "_".join((relation or "").replace("?", "").split()).strip().lower()
        cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
        if len(cleaned) > 40:
            cleaned = cleaned[:40].rstrip()
        return cleaned

    def create_graph_node(self, thread_id: str, node_id: str, label: str) -> bool:
        tid = str(thread_id or "").strip()
        nid = self._normalize_graph_node_id(node_id)
        normalized_label = self._normalize_graph_label(label)
        if not tid or not nid or not normalized_label:
            return False
        now = self._now_iso()
        self._conn.execute(
            """
            INSERT INTO graph_nodes (thread_id, node_id, label, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id, node_id) DO UPDATE SET
                label = excluded.label,
                created_at = excluded.created_at
            """,
            (tid, nid, normalized_label, now),
        )
        self._commit_if_needed()
        return True

    def set_graph_node_label(self, thread_id: str, node_id: str, new_label: str) -> bool:
        tid = str(thread_id or "").strip()
        nid = self._normalize_graph_node_id(node_id)
        label = self._normalize_graph_label(new_label)
        if not tid or not nid or not label:
            return False
        cur = self._conn.execute(
            "UPDATE graph_nodes SET label = ? WHERE thread_id = ? AND node_id = ?",
            (label, tid, nid),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def list_graph_nodes(self, thread_id: str) -> list[dict[str, Any]]:
        tid = str(thread_id or "").strip()
        if not tid:
            return []
        cur = self._conn.execute(
            """
            SELECT thread_id, node_id, label, created_at
            FROM graph_nodes
            WHERE thread_id = ?
            ORDER BY node_id ASC
            """,
            (tid,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_graph_node(self, thread_id: str, node_id: str) -> dict[str, Any] | None:
        tid = str(thread_id or "").strip()
        nid = self._normalize_graph_node_id(node_id)
        if not tid or not nid:
            return None
        cur = self._conn.execute(
            """
            SELECT thread_id, node_id, label, created_at
            FROM graph_nodes
            WHERE thread_id = ? AND node_id = ?
            LIMIT 1
            """,
            (tid, nid),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def create_graph_edge(self, thread_id: str, from_node: str, to_node: str, relation: str) -> bool:
        tid = str(thread_id or "").strip()
        src = self._normalize_graph_node_id(from_node)
        dst = self._normalize_graph_node_id(to_node)
        rel = self._normalize_graph_relation(relation)
        if not tid or not src or not dst or not rel:
            return False
        cur = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM graph_nodes
            WHERE thread_id = ? AND node_id IN (?, ?)
            """,
            (tid, src, dst),
        )
        row = cur.fetchone()
        if int(row["cnt"] or 0) < 2:
            return False
        now = self._now_iso()
        self._conn.execute(
            """
            INSERT INTO graph_edges (thread_id, from_node, to_node, relation, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id, from_node, to_node, relation) DO UPDATE SET
                created_at = excluded.created_at
            """,
            (tid, src, dst, rel, now),
        )
        self._commit_if_needed()
        return True

    def list_graph_edges(self, thread_id: str) -> list[dict[str, Any]]:
        tid = str(thread_id or "").strip()
        if not tid:
            return []
        cur = self._conn.execute(
            """
            SELECT thread_id, from_node, to_node, relation, created_at
            FROM graph_edges
            WHERE thread_id = ?
            ORDER BY from_node ASC, relation ASC, to_node ASC
            """,
            (tid,),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_out_edges(self, thread_id: str, node_id: str) -> list[tuple[str, str]]:
        tid = str(thread_id or "").strip()
        nid = self._normalize_graph_node_id(node_id)
        if not tid or not nid:
            return []
        cur = self._conn.execute(
            """
            SELECT relation, to_node
            FROM graph_edges
            WHERE thread_id = ? AND from_node = ?
            ORDER BY relation ASC, to_node ASC
            """,
            (tid, nid),
        )
        return [
            (
                self._normalize_graph_relation(str(row["relation"] or "")),
                self._normalize_graph_node_id(str(row["to_node"] or "")),
            )
            for row in cur.fetchall()
            if self._normalize_graph_relation(str(row["relation"] or ""))
            and self._normalize_graph_node_id(str(row["to_node"] or ""))
        ]

    def list_in_edges(self, thread_id: str, node_id: str) -> list[tuple[str, str]]:
        tid = str(thread_id or "").strip()
        nid = self._normalize_graph_node_id(node_id)
        if not tid or not nid:
            return []
        cur = self._conn.execute(
            """
            SELECT relation, from_node
            FROM graph_edges
            WHERE thread_id = ? AND to_node = ?
            ORDER BY relation ASC, from_node ASC
            """,
            (tid, nid),
        )
        return [
            (
                self._normalize_graph_relation(str(row["relation"] or "")),
                self._normalize_graph_node_id(str(row["from_node"] or "")),
            )
            for row in cur.fetchall()
            if self._normalize_graph_relation(str(row["relation"] or ""))
            and self._normalize_graph_node_id(str(row["from_node"] or ""))
        ]

    def list_all_edges(self, thread_id: str) -> list[tuple[str, str, str]]:
        tid = str(thread_id or "").strip()
        if not tid:
            return []
        cur = self._conn.execute(
            """
            SELECT from_node, relation, to_node
            FROM graph_edges
            WHERE thread_id = ?
            ORDER BY from_node ASC, relation ASC, to_node ASC
            """,
            (tid,),
        )
        return [
            (
                self._normalize_graph_node_id(str(row["from_node"] or "")),
                self._normalize_graph_relation(str(row["relation"] or "")),
                self._normalize_graph_node_id(str(row["to_node"] or "")),
            )
            for row in cur.fetchall()
            if self._normalize_graph_node_id(str(row["from_node"] or ""))
            and self._normalize_graph_relation(str(row["relation"] or ""))
            and self._normalize_graph_node_id(str(row["to_node"] or ""))
        ]

    def get_graph_node_label(self, thread_id: str, node_id: str) -> str | None:
        node = self.get_graph_node(thread_id, node_id)
        if not node:
            return None
        value = self._normalize_graph_label(str(node.get("label") or ""))
        return value if value else None

    def normalize_relation(self, relation: str) -> str:
        return self._normalize_graph_relation(relation)

    def add_relation_type(self, thread_id: str, relation: str) -> bool:
        tid = str(thread_id or "").strip()
        normalized = self._normalize_graph_relation(relation)
        if not tid or not normalized:
            return False
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO graph_relation_types (thread_id, relation, created_at)
            VALUES (?, ?, ?)
            """,
            (tid, normalized, self._now_iso()),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def remove_relation_type(self, thread_id: str, relation: str) -> bool:
        tid = str(thread_id or "").strip()
        normalized = self._normalize_graph_relation(relation)
        if not tid or not normalized:
            return False
        cur = self._conn.execute(
            """
            DELETE FROM graph_relation_types
            WHERE thread_id = ? AND relation = ?
            """,
            (tid, normalized),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def list_relation_types(self, thread_id: str) -> list[str]:
        tid = str(thread_id or "").strip()
        if not tid:
            return []
        cur = self._conn.execute(
            """
            SELECT relation
            FROM graph_relation_types
            WHERE thread_id = ?
            ORDER BY relation ASC
            """,
            (tid,),
        )
        return [str(row["relation"]) for row in cur.fetchall() if str(row["relation"]).strip()]

    def set_relation_strict_mode(self, thread_id: str, strict: bool) -> None:
        tid = str(thread_id or "").strip()
        if not tid:
            return
        strict_int = 1 if strict else 0
        self._conn.execute(
            """
            INSERT INTO graph_relation_mode (thread_id, strict, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                strict = excluded.strict,
                updated_at = excluded.updated_at
            """,
            (tid, strict_int, self._now_iso()),
        )
        self._commit_if_needed()

    def get_relation_strict_mode(self, thread_id: str) -> bool:
        tid = str(thread_id or "").strip()
        if not tid:
            return False
        cur = self._conn.execute(
            "SELECT strict FROM graph_relation_mode WHERE thread_id = ? LIMIT 1",
            (tid,),
        )
        row = cur.fetchone()
        if row is None:
            return False
        try:
            return int(row["strict"]) == 1
        except Exception:
            return False

    def validate_relation_allowed(self, thread_id: str, relation: str) -> bool:
        tid = str(thread_id or "").strip()
        normalized = self._normalize_graph_relation(relation)
        if not tid or not normalized:
            return False
        if not self.get_relation_strict_mode(tid):
            return True
        cur = self._conn.execute(
            """
            SELECT 1
            FROM graph_relation_types
            WHERE thread_id = ? AND relation = ?
            LIMIT 1
            """,
            (tid, normalized),
        )
        return cur.fetchone() is not None

    def add_relation_constraint(self, thread_id: str, relation: str, constraint: str) -> bool:
        tid = str(thread_id or "").strip()
        normalized_relation = self._normalize_graph_relation(relation)
        normalized_constraint = str(constraint or "").strip().lower()
        if not tid or not normalized_relation or normalized_constraint != "acyclic":
            return False
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO graph_relation_constraints (thread_id, relation, "constraint", created_at)
            VALUES (?, ?, ?, ?)
            """,
            (tid, normalized_relation, normalized_constraint, self._now_iso()),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def remove_relation_constraint(self, thread_id: str, relation: str, constraint: str) -> bool:
        tid = str(thread_id or "").strip()
        normalized_relation = self._normalize_graph_relation(relation)
        normalized_constraint = str(constraint or "").strip().lower()
        if not tid or not normalized_relation or normalized_constraint != "acyclic":
            return False
        cur = self._conn.execute(
            """
            DELETE FROM graph_relation_constraints
            WHERE thread_id = ? AND relation = ? AND "constraint" = ?
            """,
            (tid, normalized_relation, normalized_constraint),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def list_relation_constraints(self, thread_id: str) -> list[tuple[str, str]]:
        tid = str(thread_id or "").strip()
        if not tid:
            return []
        cur = self._conn.execute(
            """
            SELECT relation, "constraint" AS constraint_value
            FROM graph_relation_constraints
            WHERE thread_id = ?
            ORDER BY relation ASC, "constraint" ASC
            """,
            (tid,),
        )
        return [
            (str(row["relation"]).strip(), str(row["constraint_value"]).strip())
            for row in cur.fetchall()
            if str(row["relation"]).strip() and str(row["constraint_value"]).strip()
        ]

    def has_relation_constraint(self, thread_id: str, relation: str, constraint: str) -> bool:
        tid = str(thread_id or "").strip()
        normalized_relation = self._normalize_graph_relation(relation)
        normalized_constraint = str(constraint or "").strip().lower()
        if not tid or not normalized_relation or normalized_constraint != "acyclic":
            return False
        cur = self._conn.execute(
            """
            SELECT 1
            FROM graph_relation_constraints
            WHERE thread_id = ? AND relation = ? AND "constraint" = ?
            LIMIT 1
            """,
            (tid, normalized_relation, normalized_constraint),
        )
        return cur.fetchone() is not None

    def would_create_cycle(self, thread_id: str, relation: str, from_node: str, to_node: str) -> bool:
        tid = str(thread_id or "").strip()
        rel = self._normalize_graph_relation(relation)
        src = self._normalize_graph_node_id(from_node)
        dst = self._normalize_graph_node_id(to_node)
        if not tid or not rel or not src or not dst:
            return True
        if src == dst:
            return True

        cur = self._conn.execute(
            """
            SELECT from_node, to_node
            FROM graph_edges
            WHERE thread_id = ? AND relation = ?
            ORDER BY from_node ASC, to_node ASC
            """,
            (tid, rel),
        )
        adjacency: dict[str, list[str]] = {}
        for row in cur.fetchall():
            edge_src = self._normalize_graph_node_id(str(row["from_node"] or ""))
            edge_dst = self._normalize_graph_node_id(str(row["to_node"] or ""))
            if not edge_src or not edge_dst:
                continue
            adjacency.setdefault(edge_src, []).append(edge_dst)

        visited: set[str] = {dst}
        queue: list[str] = [dst]
        visited_count = 1
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor == src:
                    return True
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                visited_count += 1
                if visited_count > 500:
                    return True
                queue.append(neighbor)
        return False

    def validate_acyclic_constraints(self, thread_id: str) -> bool:
        tid = str(thread_id or "").strip()
        if not tid:
            return True
        constrained_relations = [
            relation
            for relation, constraint in self.list_relation_constraints(tid)
            if constraint == "acyclic"
        ]
        if not constrained_relations:
            return True

        for relation in constrained_relations:
            rel = self._normalize_graph_relation(relation)
            if not rel:
                return False
            cur = self._conn.execute(
                """
                SELECT from_node, to_node
                FROM graph_edges
                WHERE thread_id = ? AND relation = ?
                ORDER BY from_node ASC, to_node ASC
                """,
                (tid, rel),
            )
            nodes: set[str] = set()
            adjacency: dict[str, list[str]] = {}
            indegree: dict[str, int] = {}
            edge_count = 0
            for row in cur.fetchall():
                edge_count += 1
                if edge_count > 1000:
                    return False
                src = self._normalize_graph_node_id(str(row["from_node"] or ""))
                dst = self._normalize_graph_node_id(str(row["to_node"] or ""))
                if not src or not dst:
                    continue
                nodes.add(src)
                nodes.add(dst)
                indegree.setdefault(src, 0)
                indegree[dst] = indegree.get(dst, 0) + 1
                adjacency.setdefault(src, []).append(dst)

            if len(nodes) > 500:
                return False
            if not nodes:
                continue

            for node in nodes:
                indegree.setdefault(node, 0)
                adjacency.setdefault(node, [])

            queue = sorted([node for node in nodes if indegree.get(node, 0) == 0])
            processed = 0
            while queue:
                node = queue.pop(0)
                processed += 1
                for neighbor in adjacency.get(node, []):
                    indegree[neighbor] = indegree.get(neighbor, 0) - 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
                queue.sort()

            if processed != len(nodes):
                return False

        return True

    def clear_graph(self, thread_id: str) -> None:
        tid = str(thread_id or "").strip()
        if not tid:
            return
        self._conn.execute("DELETE FROM graph_edges WHERE thread_id = ?", (tid,))
        self._conn.execute("DELETE FROM graph_aliases WHERE thread_id = ?", (tid,))
        self._conn.execute("DELETE FROM graph_nodes WHERE thread_id = ?", (tid,))
        self._commit_if_needed()

    def delete_graph_node(self, thread_id: str, node_id: str) -> bool:
        tid = str(thread_id or "").strip()
        nid = self._normalize_graph_node_id(node_id)
        if not tid or not nid:
            return False
        cur = self._conn.execute(
            "DELETE FROM graph_nodes WHERE thread_id = ? AND node_id = ?",
            (tid, nid),
        )
        if cur.rowcount != 1:
            self._commit_if_needed()
            return False
        self._conn.execute(
            "DELETE FROM graph_edges WHERE thread_id = ? AND (from_node = ? OR to_node = ?)",
            (tid, nid, nid),
        )
        self._conn.execute(
            "DELETE FROM graph_aliases WHERE thread_id = ? AND node_id = ?",
            (tid, nid),
        )
        self._commit_if_needed()
        return True

    def add_graph_alias(self, thread_id: str, alias: str, node_id: str) -> bool:
        tid = str(thread_id or "").strip()
        normalized_alias = self._normalize_graph_node_id(alias)
        nid = self._normalize_graph_node_id(node_id)
        if not tid or not normalized_alias or not nid:
            return False
        cur = self._conn.execute(
            """
            SELECT 1
            FROM graph_nodes
            WHERE thread_id = ? AND node_id = ?
            LIMIT 1
            """,
            (tid, nid),
        )
        if cur.fetchone() is None:
            return False
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO graph_aliases (thread_id, alias, node_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (tid, normalized_alias, nid, self._now_iso()),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def remove_graph_alias(self, thread_id: str, alias: str) -> bool:
        tid = str(thread_id or "").strip()
        normalized_alias = self._normalize_graph_node_id(alias)
        if not tid or not normalized_alias:
            return False
        cur = self._conn.execute(
            "DELETE FROM graph_aliases WHERE thread_id = ? AND alias = ?",
            (tid, normalized_alias),
        )
        self._commit_if_needed()
        return cur.rowcount == 1

    def resolve_graph_ref(self, thread_id: str, ref: str) -> str | None:
        tid = str(thread_id or "").strip()
        normalized = self._normalize_graph_node_id(ref)
        if not tid or not normalized:
            return None
        cur = self._conn.execute(
            """
            SELECT node_id
            FROM graph_nodes
            WHERE thread_id = ? AND node_id = ?
            LIMIT 1
            """,
            (tid, normalized),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row["node_id"])
        cur = self._conn.execute(
            """
            SELECT node_id
            FROM graph_aliases
            WHERE thread_id = ? AND alias = ?
            LIMIT 1
            """,
            (tid, normalized),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return str(row["node_id"])

    def list_graph_aliases(self, thread_id: str) -> list[tuple[str, str]]:
        tid = str(thread_id or "").strip()
        if not tid:
            return []
        cur = self._conn.execute(
            """
            SELECT alias, node_id
            FROM graph_aliases
            WHERE thread_id = ?
            ORDER BY alias ASC
            """,
            (tid,),
        )
        return [(str(row["alias"]), str(row["node_id"])) for row in cur.fetchall()]

    def set_thread_focus_node(self, thread_id: str, node_id: str) -> bool:
        tid = str(thread_id or "").strip()
        nid = self._normalize_graph_node_id(node_id)
        if not tid or not nid:
            return False
        exists = self.get_graph_node(tid, nid)
        if exists is None:
            return False
        now = self._now_iso()
        self._conn.execute(
            """
            INSERT INTO thread_focus (thread_id, node_id, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                node_id = excluded.node_id,
                updated_at = excluded.updated_at
            """,
            (tid, nid, now),
        )
        self._commit_if_needed()
        return True

    def clear_thread_focus_node(self, thread_id: str) -> bool:
        tid = str(thread_id or "").strip()
        if not tid:
            return False
        cur = self._conn.execute("DELETE FROM thread_focus WHERE thread_id = ?", (tid,))
        self._commit_if_needed()
        return cur.rowcount == 1

    def get_thread_focus_node(self, thread_id: str) -> str | None:
        tid = str(thread_id or "").strip()
        if not tid:
            return None
        cur = self._conn.execute(
            "SELECT node_id FROM thread_focus WHERE thread_id = ? LIMIT 1",
            (tid,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        value = self._normalize_graph_node_id(str(row["node_id"] or ""))
        return value if value else None

    def list_related_nodes(self, thread_id: str, node_id: str, limit: int = 3) -> list[str]:
        tid = str(thread_id or "").strip()
        nid = self._normalize_graph_node_id(node_id)
        max_rows = max(1, int(limit))
        if not tid or not nid:
            return []
        cur = self._conn.execute(
            """
            SELECT relation, to_node
            FROM graph_edges
            WHERE thread_id = ? AND from_node = ?
            ORDER BY relation ASC, to_node ASC
            """,
            (tid, nid),
        )
        results: list[str] = []
        seen: set[str] = set()
        for row in cur.fetchall():
            to_node = self._normalize_graph_node_id(str(row["to_node"] or ""))
            if not to_node or to_node in seen:
                continue
            seen.add(to_node)
            results.append(to_node)
            if len(results) >= max_rows:
                break
        return results

    def export_graph(self, thread_id: str) -> dict[str, Any]:
        tid = str(thread_id or "").strip()
        if not tid:
            return {
                "thread_id": "",
                "exported_at": self._now_iso(),
                "nodes": [],
                "aliases": [],
                "edges": [],
                "focus_node": None,
            }

        cur_nodes = self._conn.execute(
            """
            SELECT node_id, label, created_at
            FROM graph_nodes
            WHERE thread_id = ?
            ORDER BY node_id ASC
            """,
            (tid,),
        )
        nodes = [
            {
                "node_id": str(row["node_id"]),
                "label": str(row["label"]),
                "created_at": str(row["created_at"]),
            }
            for row in cur_nodes.fetchall()
        ]

        cur_aliases = self._conn.execute(
            """
            SELECT alias, node_id, created_at
            FROM graph_aliases
            WHERE thread_id = ?
            ORDER BY alias ASC
            """,
            (tid,),
        )
        aliases = [
            {
                "alias": str(row["alias"]),
                "node_id": str(row["node_id"]),
                "created_at": str(row["created_at"]),
            }
            for row in cur_aliases.fetchall()
        ]

        cur_edges = self._conn.execute(
            """
            SELECT from_node, relation, to_node, created_at
            FROM graph_edges
            WHERE thread_id = ?
            ORDER BY from_node ASC, relation ASC, to_node ASC
            """,
            (tid,),
        )
        edges = [
            {
                "from": str(row["from_node"]),
                "relation": str(row["relation"]),
                "to": str(row["to_node"]),
                "created_at": str(row["created_at"]),
            }
            for row in cur_edges.fetchall()
        ]

        cur_focus = self._conn.execute(
            """
            SELECT node_id, updated_at
            FROM thread_focus
            WHERE thread_id = ?
            LIMIT 1
            """,
            (tid,),
        )
        focus_row = cur_focus.fetchone()
        focus_node = (
            {
                "node_id": str(focus_row["node_id"]),
                "updated_at": str(focus_row["updated_at"]),
            }
            if focus_row is not None
            else None
        )

        return {
            "thread_id": tid,
            "exported_at": self._now_iso(),
            "nodes": nodes,
            "aliases": aliases,
            "edges": edges,
            "focus_node": focus_node,
        }

    def _validate_graph_import_payload(
        self,
        payload: Any,
        *,
        allowed_existing_node_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        for key in ("nodes", "aliases", "edges", "focus_node"):
            if key not in payload:
                return None

        raw_nodes = payload.get("nodes")
        raw_aliases = payload.get("aliases")
        raw_edges = payload.get("edges")
        raw_focus = payload.get("focus_node")
        if not isinstance(raw_nodes, list) or not isinstance(raw_aliases, list) or not isinstance(raw_edges, list):
            return None

        nodes: list[tuple[str, str, str]] = []
        node_ids: set[str] = set()
        for item in raw_nodes:
            if not isinstance(item, dict):
                return None
            node_id = self._normalize_graph_node_id(str(item.get("node_id") or ""))
            label = self._normalize_graph_label(str(item.get("label") or ""))
            created_at = str(item.get("created_at") or "").strip()
            if not node_id or not label or not created_at:
                return None
            if node_id in node_ids:
                return None
            node_ids.add(node_id)
            nodes.append((node_id, label, created_at))
        nodes.sort(key=lambda row: row[0])

        aliases: list[tuple[str, str, str]] = []
        alias_keys: set[str] = set()
        for item in raw_aliases:
            if not isinstance(item, dict):
                return None
            alias = self._normalize_graph_node_id(str(item.get("alias") or ""))
            node_id = self._normalize_graph_node_id(str(item.get("node_id") or ""))
            created_at = str(item.get("created_at") or "").strip()
            if not alias or not node_id or not created_at:
                return None
            if alias in alias_keys:
                return None
            alias_keys.add(alias)
            aliases.append((alias, node_id, created_at))
        aliases.sort(key=lambda row: row[0])

        edges: list[tuple[str, str, str, str]] = []
        edge_keys: set[tuple[str, str, str]] = set()
        for item in raw_edges:
            if not isinstance(item, dict):
                return None
            from_node = self._normalize_graph_node_id(str(item.get("from") or ""))
            relation = self._normalize_graph_relation(str(item.get("relation") or ""))
            to_node = self._normalize_graph_node_id(str(item.get("to") or ""))
            created_at = str(item.get("created_at") or "").strip()
            if not from_node or not relation or not to_node or not created_at:
                return None
            edge_key = (from_node, relation, to_node)
            if edge_key in edge_keys:
                return None
            edge_keys.add(edge_key)
            edges.append((from_node, relation, to_node, created_at))
        edges.sort(key=lambda row: (row[0], row[1], row[2]))

        focus_node: tuple[str, str] | None
        if raw_focus is None:
            focus_node = None
        else:
            if not isinstance(raw_focus, dict):
                return None
            focus_id = self._normalize_graph_node_id(str(raw_focus.get("node_id") or ""))
            updated_at = str(raw_focus.get("updated_at") or "").strip()
            if not focus_id or not updated_at:
                return None
            focus_node = (focus_id, updated_at)

        allowed_node_ids = set(node_ids)
        if allowed_existing_node_ids:
            allowed_node_ids.update(allowed_existing_node_ids)

        for _, node_id, _ in aliases:
            if node_id not in allowed_node_ids:
                return None
        for from_node, _, to_node, _ in edges:
            if from_node not in allowed_node_ids or to_node not in allowed_node_ids:
                return None
        if focus_node is not None and focus_node[0] not in allowed_node_ids:
            return None

        return {
            "nodes": nodes,
            "aliases": aliases,
            "edges": edges,
            "focus_node": focus_node,
        }

    def _graph_import_within_caps(self, normalized_payload: dict[str, Any]) -> bool:
        return (
            len(normalized_payload["nodes"]) <= self.GRAPH_IMPORT_MAX_NODES
            and len(normalized_payload["edges"]) <= self.GRAPH_IMPORT_MAX_EDGES
            and len(normalized_payload["aliases"]) <= self.GRAPH_IMPORT_MAX_ALIASES
        )

    def _graph_existing_node_ids(self, thread_id: str) -> set[str]:
        tid = str(thread_id or "").strip()
        if not tid:
            return set()
        cur_existing_nodes = self._conn.execute(
            "SELECT node_id FROM graph_nodes WHERE thread_id = ? ORDER BY node_id ASC",
            (tid,),
        )
        return {
            self._normalize_graph_node_id(str(row["node_id"] or ""))
            for row in cur_existing_nodes.fetchall()
            if self._normalize_graph_node_id(str(row["node_id"] or ""))
        }

    def _apply_graph_replace_normalized(self, thread_id: str, normalized: dict[str, Any]) -> None:
        tid = str(thread_id or "").strip()
        self._conn.execute("DELETE FROM thread_focus WHERE thread_id = ?", (tid,))
        self._conn.execute("DELETE FROM graph_edges WHERE thread_id = ?", (tid,))
        self._conn.execute("DELETE FROM graph_aliases WHERE thread_id = ?", (tid,))
        self._conn.execute("DELETE FROM graph_nodes WHERE thread_id = ?", (tid,))
        for node_id, label, created_at in normalized["nodes"]:
            self._conn.execute(
                """
                INSERT INTO graph_nodes (thread_id, node_id, label, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (tid, node_id, label, created_at),
            )
        for alias, node_id, created_at in normalized["aliases"]:
            self._conn.execute(
                """
                INSERT INTO graph_aliases (thread_id, alias, node_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (tid, alias, node_id, created_at),
            )
        for from_node, relation, to_node, created_at in normalized["edges"]:
            self._conn.execute(
                """
                INSERT INTO graph_edges (thread_id, from_node, to_node, relation, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tid, from_node, to_node, relation, created_at),
            )
        focus_node = normalized["focus_node"]
        if focus_node is not None:
            self._conn.execute(
                """
                INSERT INTO thread_focus (thread_id, node_id, updated_at)
                VALUES (?, ?, ?)
                """,
                (tid, focus_node[0], focus_node[1]),
            )

    def _apply_graph_merge_normalized(self, thread_id: str, normalized: dict[str, Any]) -> None:
        tid = str(thread_id or "").strip()
        for node_id, label, created_at in normalized["nodes"]:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO graph_nodes (thread_id, node_id, label, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (tid, node_id, label, created_at),
            )
        for alias, node_id, created_at in normalized["aliases"]:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO graph_aliases (thread_id, alias, node_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (tid, alias, node_id, created_at),
            )
        for from_node, relation, to_node, created_at in normalized["edges"]:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO graph_edges (thread_id, from_node, to_node, relation, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tid, from_node, to_node, relation, created_at),
            )
        focus_node = normalized["focus_node"]
        if focus_node is not None:
            cur_focus = self._conn.execute(
                "SELECT 1 FROM thread_focus WHERE thread_id = ? LIMIT 1",
                (tid,),
            )
            if cur_focus.fetchone() is None:
                self._conn.execute(
                    """
                    INSERT INTO thread_focus (thread_id, node_id, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (tid, focus_node[0], focus_node[1]),
                )

    def import_graph_replace(self, thread_id: str, payload_dict: dict[str, Any]) -> bool:
        tid = str(thread_id or "").strip()
        if not tid:
            return False
        normalized = self._validate_graph_import_payload(payload_dict, allowed_existing_node_ids=None)
        if normalized is None:
            return False
        if not self._graph_import_within_caps(normalized):
            return False
        try:
            with self.transaction():
                self._apply_graph_replace_normalized(tid, normalized)
                if not self.validate_acyclic_constraints(tid):
                    raise ValueError("acyclic_constraint_violation")
        except Exception:
            return False
        return True

    def import_graph_merge(self, thread_id: str, payload_dict: dict[str, Any]) -> bool:
        tid = str(thread_id or "").strip()
        if not tid:
            return False
        existing_node_ids = self._graph_existing_node_ids(tid)
        normalized = self._validate_graph_import_payload(payload_dict, allowed_existing_node_ids=existing_node_ids)
        if normalized is None:
            return False
        if not self._graph_import_within_caps(normalized):
            return False
        try:
            with self.transaction():
                self._apply_graph_merge_normalized(tid, normalized)
                if not self.validate_acyclic_constraints(tid):
                    raise ValueError("acyclic_constraint_violation")
        except Exception:
            return False
        return True

    def thread_exists_for_graph_ops(self, thread_id: str, recent_thread_ids: set[str] | None = None) -> bool:
        tid = str(thread_id or "").strip()
        if not tid:
            return False
        recents = recent_thread_ids
        if recents is None:
            recents = {
                str(row.get("thread_id") or "").strip()
                for row in self.list_recent_threads(limit=5000)
                if isinstance(row, dict) and str(row.get("thread_id") or "").strip()
            }
        if tid in recents:
            return True
        cur = self._conn.execute(
            "SELECT 1 FROM thread_anchors WHERE thread_id = ? LIMIT 1",
            (tid,),
        )
        if cur.fetchone() is not None:
            return True
        for table_name in ("graph_nodes", "graph_edges", "graph_aliases", "thread_focus"):
            cur = self._conn.execute(f"SELECT 1 FROM {table_name} WHERE thread_id = ? LIMIT 1", (tid,))
            if cur.fetchone() is not None:
                return True
        return False

    def export_graph_pack(self, thread_ids: list[str]) -> dict[str, Any]:
        normalized_ids = sorted({str(thread_id or "").strip() for thread_id in thread_ids if str(thread_id or "").strip()})
        threads: list[dict[str, Any]] = []
        for thread_id in normalized_ids:
            graph_export = self.export_graph(thread_id)
            graph = {
                "exported_at": graph_export["exported_at"],
                "nodes": graph_export["nodes"],
                "aliases": graph_export["aliases"],
                "edges": graph_export["edges"],
                "focus_node": graph_export["focus_node"],
            }
            threads.append(
                {
                    "thread_id": thread_id,
                    "graph": graph,
                }
            )
        return {
            "pack_version": 1,
            "exported_at": self._now_iso(),
            "threads": threads,
        }

    def _validate_graph_pack(
        self,
        pack_dict: Any,
        *,
        merge: bool,
    ) -> list[tuple[str, dict[str, Any]]] | None:
        if not isinstance(pack_dict, dict):
            return None
        try:
            pack_version = int(pack_dict.get("pack_version") or 0)
        except Exception:
            return None
        if pack_version != 1:
            return None
        raw_threads = pack_dict.get("threads")
        if not isinstance(raw_threads, list):
            return None
        if not raw_threads or len(raw_threads) > self.GRAPH_PACK_MAX_THREADS:
            return None

        seen_thread_ids: set[str] = set()
        normalized_entries: list[tuple[str, dict[str, Any]]] = []
        for entry in raw_threads:
            if not isinstance(entry, dict):
                return None
            thread_id = str(entry.get("thread_id") or "").strip()
            graph_payload = entry.get("graph")
            if not thread_id or not isinstance(graph_payload, dict):
                return None
            if thread_id in seen_thread_ids:
                return None
            seen_thread_ids.add(thread_id)
            allowed_existing = self._graph_existing_node_ids(thread_id) if merge else None
            normalized_graph = self._validate_graph_import_payload(
                graph_payload,
                allowed_existing_node_ids=allowed_existing,
            )
            if normalized_graph is None:
                return None
            if not self._graph_import_within_caps(normalized_graph):
                return None
            normalized_entries.append((thread_id, normalized_graph))
        normalized_entries.sort(key=lambda row: row[0])
        return normalized_entries

    def import_graph_pack_replace(self, pack_dict: dict[str, Any]) -> bool:
        normalized_entries = self._validate_graph_pack(pack_dict, merge=False)
        if normalized_entries is None:
            return False
        try:
            with self.transaction():
                for thread_id, normalized_graph in normalized_entries:
                    self._apply_graph_replace_normalized(thread_id, normalized_graph)
                    if not self.validate_acyclic_constraints(thread_id):
                        raise ValueError("acyclic_constraint_violation")
        except Exception:
            return False
        return True

    def import_graph_pack_merge(self, pack_dict: dict[str, Any]) -> bool:
        normalized_entries = self._validate_graph_pack(pack_dict, merge=True)
        if normalized_entries is None:
            return False
        try:
            with self.transaction():
                for thread_id, normalized_graph in normalized_entries:
                    self._apply_graph_merge_normalized(thread_id, normalized_graph)
                    if not self.validate_acyclic_constraints(thread_id):
                        raise ValueError("acyclic_constraint_violation")
        except Exception:
            return False
        return True

    def clone_graph(self, from_thread_id: str, to_thread_id: str, merge: bool) -> bool:
        src = str(from_thread_id or "").strip()
        dst = str(to_thread_id or "").strip()
        if not src or not dst:
            return False
        if not self.thread_exists_for_graph_ops(src):
            return False
        exported = self.export_graph(src)
        payload = {
            "nodes": exported.get("nodes", []),
            "aliases": exported.get("aliases", []),
            "edges": exported.get("edges", []),
            "focus_node": exported.get("focus_node"),
        }
        if merge:
            return self.import_graph_merge(dst, payload)
        return self.import_graph_replace(dst, payload)

    def list_recent_threads(self, limit: int = 10) -> list[dict[str, Any]]:
        max_rows = max(1, int(limit))
        latest_by_thread: dict[str, str] = {}

        try:
            cur = self._conn.execute(
                """
                SELECT
                    json_extract(payload_json, '$.thread_id') AS thread_id,
                    MAX(ts) AS last_ts
                FROM activity_log
                WHERE type = 'epistemic_turn'
                  AND json_extract(payload_json, '$.thread_id') IS NOT NULL
                  AND TRIM(json_extract(payload_json, '$.thread_id')) <> ''
                GROUP BY json_extract(payload_json, '$.thread_id')
                """
            )
            for row in cur.fetchall():
                thread_id = str(row["thread_id"] or "").strip()
                last_ts = str(row["last_ts"] or "").strip()
                if thread_id and last_ts:
                    latest_by_thread[thread_id] = last_ts
        except Exception:
            cur = self._conn.execute(
                """
                SELECT ts, payload_json
                FROM activity_log
                WHERE type = 'epistemic_turn'
                ORDER BY ts DESC, id DESC
                LIMIT 5000
                """
            )
            for row in cur.fetchall():
                ts = str(row["ts"] or "").strip()
                if not ts:
                    continue
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except Exception:
                    payload = {}
                thread_id = str(payload.get("thread_id") or "").strip() if isinstance(payload, dict) else ""
                if not thread_id:
                    continue
                if thread_id not in latest_by_thread:
                    latest_by_thread[thread_id] = ts

        cur = self._conn.execute(
            """
            SELECT thread_id, MAX(created_at) AS last_ts
            FROM thread_anchors
            GROUP BY thread_id
            """
        )
        for row in cur.fetchall():
            thread_id = str(row["thread_id"] or "").strip()
            last_ts = str(row["last_ts"] or "").strip()
            if not thread_id or not last_ts:
                continue
            existing = latest_by_thread.get(thread_id)
            if existing is None or last_ts > existing:
                latest_by_thread[thread_id] = last_ts

        records = [{"thread_id": thread_id, "last_ts": last_ts} for thread_id, last_ts in latest_by_thread.items()]
        records.sort(key=lambda row: str(row.get("thread_id") or ""))
        records.sort(key=lambda row: str(row.get("last_ts") or ""), reverse=True)
        return records[:max_rows]

    def clear_thread_anchors(self, thread_id: str) -> None:
        self._conn.execute("DELETE FROM thread_anchors WHERE thread_id = ?", (thread_id,))
        self._commit_if_needed()

    def log_activity(self, event_type: str, payload: dict[str, Any]) -> None:
        ts = self._now_iso()
        self._conn.execute(
            "INSERT INTO activity_log (ts, type, payload_json) VALUES (?, ?, ?)",
            (ts, event_type, json.dumps(payload, ensure_ascii=True)),
        )
        self._commit_if_needed()

    def audit_log_create(
        self,
        user_id: str,
        action_type: str,
        action_id: str,
        status: str,
        details_json: dict[str, Any] | str | None = None,
        error: str | None = None,
        created_at: str | None = None,
        **kwargs: Any,
    ) -> int:
        created_at = created_at or self._now_iso()
        if details_json is None and "details" in kwargs:
            details_json = kwargs.get("details")
        if details_json is None:
            details_json = {}
        if isinstance(details_json, str):
            details_payload = details_json
        else:
            details_payload = json.dumps(details_json, ensure_ascii=True)
        cur = self._conn.execute(
            """
            INSERT INTO audit_log (
                created_at,
                user_id,
                action_type,
                action_id,
                status,
                details_json,
                error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                user_id,
                action_type,
                action_id,
                status,
                details_payload,
                error,
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def audit_log_update_status(self, *args: Any, **kwargs: Any) -> bool:
        if not args:
            raise TypeError("audit_log_update_status requires arguments")

        if isinstance(args[0], int):
            audit_id = int(args[0])
            status = args[1] if len(args) > 1 else None
            error = args[2] if len(args) > 2 else kwargs.get("error")
            details = args[3] if len(args) > 3 else kwargs.get("details")
            if status is None:
                raise TypeError("status is required")
            if details is None:
                cur = self._conn.execute(
                    "UPDATE audit_log SET status = ?, error = ? WHERE id = ?",
                    (status, error, audit_id),
                )
            else:
                cur = self._conn.execute(
                    "UPDATE audit_log SET status = ?, error = ?, details_json = ? WHERE id = ?",
                    (status, error, json.dumps(details, ensure_ascii=True), audit_id),
                )
        else:
            user_id = str(args[0])
            action_id = str(args[1]) if len(args) > 1 else None
            status = args[2] if len(args) > 2 else None
            error = args[3] if len(args) > 3 else kwargs.get("error")
            if action_id is None or status is None:
                raise TypeError("action_id and status are required")
            cur = self._conn.execute(
                "UPDATE audit_log SET status = ?, error = ? WHERE user_id = ? AND action_id = ?",
                (status, error, user_id, action_id),
            )

        self._commit_if_needed()
        return cur.rowcount > 0

    def audit_log_list_recent(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT id, created_at, user_id, action_type, action_id, status, details_json, error
            FROM audit_log
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            try:
                record["details"] = json.loads(record.pop("details_json") or "{}")
            except json.JSONDecodeError:
                record["details"] = {}
                record.pop("details_json", None)
            results.append(record)
        return results

    def audit_log_latest_by_type(self, action_type: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, created_at, user_id, action_type, action_id, status, details_json, error
            FROM audit_log
            WHERE action_type = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (action_type,),
        )
        row = cur.fetchone()
        if not row:
            return None
        record = dict(row)
        try:
            record["details"] = json.loads(record.pop("details_json") or "{}")
        except json.JSONDecodeError:
            record["details"] = {}
            record.pop("details_json", None)
        return record

    def disk_baseline_set(
        self,
        user_id: str,
        snapshot_json: dict[str, Any] | str,
        snapshot_hash: str,
        created_at: str | None = None,
    ) -> None:
        created_at = created_at or self._now_iso()
        if isinstance(snapshot_json, str):
            payload = snapshot_json
        else:
            payload = json.dumps(snapshot_json, ensure_ascii=True)
        self._conn.execute(
            """
            INSERT INTO disk_baselines (user_id, snapshot_json, snapshot_hash, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                snapshot_hash = excluded.snapshot_hash,
                created_at = excluded.created_at
            """,
            (user_id, payload, snapshot_hash, created_at),
        )
        self._commit_if_needed()

    def disk_baseline_get(self, user_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT user_id, snapshot_json, snapshot_hash, created_at FROM disk_baselines WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        record = dict(row)
        try:
            record["snapshot"] = json.loads(record.get("snapshot_json") or "{}")
        except json.JSONDecodeError:
            record["snapshot"] = {}
        return record

    def activity_log_latest(self, event_type: str) -> str | None:
        cur = self._conn.execute(
            "SELECT ts FROM activity_log WHERE type = ? ORDER BY ts DESC LIMIT 1",
            (event_type,),
        )
        row = cur.fetchone()
        if row:
            return row["ts"]
        return None

    def activity_log_list_recent(self, event_type: str, limit: int = 5) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT ts, type, payload_json
            FROM activity_log
            WHERE type = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (event_type, limit),
        )
        rows = cur.fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            try:
                record["payload"] = json.loads(record.pop("payload_json") or "{}")
            except json.JSONDecodeError:
                record["payload"] = {}
                record.pop("payload_json", None)
            results.append(record)
        return results

    def insert_disk_snapshot(
        self,
        taken_at: str,
        snapshot_local_date: str,
        hostname: str,
        mountpoint: str,
        filesystem: str | None,
        total_bytes: int,
        used_bytes: int,
        free_bytes: int,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO disk_snapshots (
                taken_at,
                snapshot_local_date,
                hostname,
                mountpoint,
                filesystem,
                total_bytes,
                used_bytes,
                free_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_local_date, mountpoint) DO UPDATE SET
                taken_at = excluded.taken_at,
                snapshot_local_date = excluded.snapshot_local_date,
                hostname = excluded.hostname,
                filesystem = excluded.filesystem,
                total_bytes = excluded.total_bytes,
                used_bytes = excluded.used_bytes,
                free_bytes = excluded.free_bytes
            """,
            (
                taken_at,
                snapshot_local_date,
                hostname,
                mountpoint,
                filesystem,
                total_bytes,
                used_bytes,
                free_bytes,
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def insert_dir_size_samples(self, taken_at: str, scope: str, samples: list[tuple[str, int]]) -> None:
        self._conn.execute(
            "DELETE FROM dir_size_samples WHERE taken_at = ? AND scope = ?",
            (taken_at, scope),
        )
        self._conn.executemany(
            "INSERT INTO dir_size_samples (taken_at, scope, path, bytes) VALUES (?, ?, ?, ?)",
            [(taken_at, scope, path, int(bytes_val)) for path, bytes_val in samples],
        )
        self._commit_if_needed()

    def insert_storage_scan_stats(
        self, taken_at: str, scope: str, dirs_scanned: int, errors_skipped: int
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO storage_scan_stats (taken_at, scope, dirs_scanned, errors_skipped)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(taken_at, scope) DO UPDATE SET
                dirs_scanned = excluded.dirs_scanned,
                errors_skipped = excluded.errors_skipped
            """,
            (taken_at, scope, int(dirs_scanned), int(errors_skipped)),
        )
        self._commit_if_needed()

    def insert_resource_snapshot(
        self,
        taken_at: str,
        snapshot_local_date: str,
        hostname: str,
        load_1m: float,
        load_5m: float,
        load_15m: float,
        mem_total: int,
        mem_used: int,
        mem_free: int,
        swap_total: int,
        swap_used: int,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO resource_snapshots (
                taken_at,
                snapshot_local_date,
                hostname,
                load_1m,
                load_5m,
                load_15m,
                mem_total,
                mem_used,
                mem_free,
                swap_total,
                swap_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_local_date, hostname) DO UPDATE SET
                taken_at = excluded.taken_at,
                hostname = excluded.hostname,
                load_1m = excluded.load_1m,
                load_5m = excluded.load_5m,
                load_15m = excluded.load_15m,
                mem_total = excluded.mem_total,
                mem_used = excluded.mem_used,
                mem_free = excluded.mem_free,
                swap_total = excluded.swap_total,
                swap_used = excluded.swap_used
            """,
            (
                taken_at,
                snapshot_local_date,
                hostname,
                float(load_1m),
                float(load_5m),
                float(load_15m),
                int(mem_total),
                int(mem_used),
                int(mem_free),
                int(swap_total),
                int(swap_used),
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def replace_resource_process_samples(
        self,
        taken_at: str,
        category: str,
        samples: list[tuple[int, str, int, int]],
    ) -> None:
        self._conn.execute(
            "DELETE FROM resource_process_samples WHERE taken_at = ? AND category = ?",
            (taken_at, category),
        )
        self._conn.executemany(
            """
            INSERT INTO resource_process_samples (taken_at, category, pid, name, cpu_ticks, rss_bytes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (taken_at, category, int(pid), name, int(cpu_ticks), int(rss_bytes))
                for pid, name, cpu_ticks, rss_bytes in samples
            ],
        )
        self._commit_if_needed()

    def insert_resource_scan_stats(
        self, taken_at: str, scope: str, procs_scanned: int, errors_skipped: int
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO resource_scan_stats (taken_at, scope, procs_scanned, errors_skipped)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(taken_at, scope) DO UPDATE SET
                procs_scanned = excluded.procs_scanned,
                errors_skipped = excluded.errors_skipped
            """,
            (taken_at, scope, int(procs_scanned), int(errors_skipped)),
        )
        self._commit_if_needed()

    def get_latest_resource_snapshot(self) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, taken_at, snapshot_local_date, hostname, load_1m, load_5m, load_15m,
                   mem_total, mem_used, mem_free, swap_total, swap_used
            FROM resource_snapshots
            ORDER BY taken_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_previous_resource_snapshot(self, before_taken_at: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, taken_at, snapshot_local_date, hostname, load_1m, load_5m, load_15m,
                   mem_total, mem_used, mem_free, swap_total, swap_used
            FROM resource_snapshots
            WHERE taken_at < ?
            ORDER BY taken_at DESC
            LIMIT 1
            """,
            (before_taken_at,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_resource_process_samples(self, taken_at: str, category: str) -> list[dict[str, Any]]:
        order_by = "cpu_ticks DESC" if category == "cpu" else "rss_bytes DESC"
        query = (
            "SELECT pid, name, cpu_ticks, rss_bytes "
            "FROM resource_process_samples "
            "WHERE taken_at = ? AND category = ? "
            f"ORDER BY {order_by}"
        )
        cur = self._conn.execute(query, (taken_at, category))
        return [dict(row) for row in cur.fetchall()]

    def get_latest_resource_scan_stats(self, scope: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT taken_at, scope, procs_scanned, errors_skipped
            FROM resource_scan_stats
            WHERE scope = ?
            ORDER BY taken_at DESC
            LIMIT 1
            """,
            (scope,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def insert_network_snapshot(
        self,
        taken_at: str,
        snapshot_local_date: str,
        hostname: str,
        default_iface: str,
        default_gateway: str,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO network_snapshots (
                taken_at,
                snapshot_local_date,
                hostname,
                default_iface,
                default_gateway
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_local_date, hostname) DO UPDATE SET
                taken_at = excluded.taken_at,
                default_iface = excluded.default_iface,
                default_gateway = excluded.default_gateway
            """,
            (taken_at, snapshot_local_date, hostname, default_iface, default_gateway),
        )
        self._commit_if_needed()

    def insert_anomaly_events(
        self,
        user_id: str,
        observed_at: str,
        events: list[dict[str, Any]],
    ) -> int:
        if not events:
            return 0
        rows = []
        for event in events:
            rows.append(
                (
                    user_id,
                    observed_at,
                    event.get("snapshot_id"),
                    event.get("source", ""),
                    event.get("anomaly_key", ""),
                    event.get("severity", "info"),
                    event.get("message", ""),
                    event.get("metric_name"),
                    event.get("metric_value"),
                    event.get("metric_unit"),
                    json.dumps(event.get("context", {}), ensure_ascii=True),
                )
            )
        with self.transaction():
            cur = self._conn.executemany(
                """
                INSERT OR IGNORE INTO anomaly_events (
                    user_id,
                    observed_at,
                    snapshot_id,
                    source,
                    anomaly_key,
                    severity,
                    message,
                    metric_name,
                    metric_value,
                    metric_unit,
                    context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return cur.rowcount if cur is not None else 0

    def get_anomalies(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT observed_at, source, anomaly_key, severity, message,
                   metric_name, metric_value, metric_unit, context_json
            FROM anomaly_events
            WHERE user_id = ?
              AND substr(observed_at, 1, 10) BETWEEN ? AND ?
            ORDER BY observed_at ASC
            LIMIT ?
            """,
            (user_id, start_date, end_date, int(limit)),
        )
        rows = []
        for row in cur.fetchall():
            payload = dict(row)
            try:
                payload["context"] = json.loads(payload.get("context_json") or "{}")
            except Exception:
                payload["context"] = {}
            payload.pop("context_json", None)
            rows.append(payload)
        return rows
        return int(cur.lastrowid)

    def replace_network_interfaces(
        self,
        taken_at: str,
        samples: list[tuple[str, str, int, int, int, int]],
    ) -> None:
        self._conn.execute(
            "DELETE FROM network_interfaces WHERE taken_at = ?",
            (taken_at,),
        )
        self._conn.executemany(
            """
            INSERT INTO network_interfaces (taken_at, name, state, rx_bytes, tx_bytes, rx_errors, tx_errors)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (taken_at, name, state, int(rx_bytes), int(tx_bytes), int(rx_errors), int(tx_errors))
                for name, state, rx_bytes, tx_bytes, rx_errors, tx_errors in samples
            ],
        )
        self._commit_if_needed()

    def replace_network_nameservers(self, taken_at: str, nameservers: list[str]) -> None:
        self._conn.execute(
            "DELETE FROM network_nameservers WHERE taken_at = ?",
            (taken_at,),
        )
        self._conn.executemany(
            "INSERT INTO network_nameservers (taken_at, nameserver) VALUES (?, ?)",
            [(taken_at, ns) for ns in nameservers],
        )
        self._commit_if_needed()

    def get_latest_network_snapshot(self) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, taken_at, snapshot_local_date, hostname, default_iface, default_gateway
            FROM network_snapshots
            ORDER BY taken_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_previous_network_snapshot(self, before_taken_at: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, taken_at, snapshot_local_date, hostname, default_iface, default_gateway
            FROM network_snapshots
            WHERE taken_at < ?
            ORDER BY taken_at DESC
            LIMIT 1
            """,
            (before_taken_at,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_network_interfaces(self, taken_at: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT name, state, rx_bytes, tx_bytes, rx_errors, tx_errors
            FROM network_interfaces
            WHERE taken_at = ?
            ORDER BY name ASC
            """,
            (taken_at,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_network_nameservers(self, taken_at: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT nameserver
            FROM network_nameservers
            WHERE taken_at = ?
            ORDER BY nameserver ASC
            """,
            (taken_at,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_latest_snapshot_local_date_any(self) -> str | None:
        cur = self._conn.execute(
            """
            SELECT MAX(snapshot_local_date) AS latest_date FROM (
                SELECT snapshot_local_date FROM disk_snapshots
                UNION ALL
                SELECT snapshot_local_date FROM resource_snapshots
                UNION ALL
                SELECT snapshot_local_date FROM network_snapshots
            )
            """
        )
        row = cur.fetchone()
        if not row:
            return None
        return row["latest_date"]

    def get_latest_snapshot_taken_at_any(self) -> str | None:
        cur = self._conn.execute(
            """
            SELECT MAX(taken_at) AS latest_ts FROM (
                SELECT taken_at FROM disk_snapshots
                UNION ALL
                SELECT taken_at FROM resource_snapshots
                UNION ALL
                SELECT taken_at FROM network_snapshots
            )
            """
        )
        row = cur.fetchone()
        if not row:
            return None
        return row["latest_ts"]

    def list_disk_snapshots_between(
        self, mountpoint: str, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT snapshot_local_date, taken_at, used_bytes
            FROM disk_snapshots
            WHERE mountpoint = ?
              AND snapshot_local_date BETWEEN ? AND ?
            ORDER BY snapshot_local_date ASC
            """,
            (mountpoint, start_date, end_date),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_disk_snapshot_window_stats(self, start_date: str, end_date: str) -> dict[str, Any]:
        cur = self._conn.execute(
            """
            SELECT
                COUNT(DISTINCT snapshot_local_date) AS snapshot_count,
                MIN(snapshot_local_date) AS earliest_snapshot_date,
                MAX(snapshot_local_date) AS latest_snapshot_date,
                MIN(taken_at) AS earliest_taken_at,
                MAX(taken_at) AS latest_taken_at
            FROM disk_snapshots
            WHERE snapshot_local_date BETWEEN ? AND ?
            """,
            (start_date, end_date),
        )
        row = cur.fetchone()
        return dict(row) if row else {
            "snapshot_count": 0,
            "earliest_snapshot_date": None,
            "latest_snapshot_date": None,
            "earliest_taken_at": None,
            "latest_taken_at": None,
        }

    def get_disk_snapshot_for_mount_and_date(
        self, mountpoint: str, snapshot_local_date: str
    ) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, taken_at, snapshot_local_date, hostname, mountpoint, filesystem, total_bytes, used_bytes, free_bytes
            FROM disk_snapshots
            WHERE mountpoint = ? AND snapshot_local_date = ?
            LIMIT 1
            """,
            (mountpoint, snapshot_local_date),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_disk_snapshot_taken_at(self) -> str | None:
        cur = self._conn.execute(
            "SELECT MAX(taken_at) AS latest_ts FROM disk_snapshots"
        )
        row = cur.fetchone()
        if not row:
            return None
        return row["latest_ts"]

    def list_disk_snapshot_mountpoints_for_taken_at(self, taken_at: str) -> list[str]:
        cur = self._conn.execute(
            """
            SELECT DISTINCT mountpoint
            FROM disk_snapshots
            WHERE taken_at = ?
            ORDER BY mountpoint ASC
            """,
            (taken_at,),
        )
        return [row["mountpoint"] for row in cur.fetchall()]

    def list_dir_size_samples_for_date(self, scope: str, date_str: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT path, bytes
            FROM dir_size_samples
            WHERE scope = ? AND substr(taken_at, 1, 10) = ?
            """,
            (scope, date_str),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_dir_size_sample_dates(self, scope: str, start_date: str, end_date: str) -> list[str]:
        cur = self._conn.execute(
            """
            SELECT DISTINCT substr(taken_at, 1, 10) AS day
            FROM dir_size_samples
            WHERE scope = ? AND substr(taken_at, 1, 10) BETWEEN ? AND ?
            ORDER BY day ASC
            """,
            (scope, start_date, end_date),
        )
        return [row["day"] for row in cur.fetchall()]

    def list_storage_scan_stats_between(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT errors_skipped
            FROM storage_scan_stats
            WHERE substr(taken_at, 1, 10) BETWEEN ? AND ?
            """,
            (start_date, end_date),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_resource_snapshots_between(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT snapshot_local_date, taken_at, load_1m, load_5m, load_15m,
                   mem_used, mem_total, swap_used, swap_total
            FROM resource_snapshots
            WHERE snapshot_local_date BETWEEN ? AND ?
            ORDER BY snapshot_local_date ASC
            """,
            (start_date, end_date),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_resource_process_samples_for_date(self, date_str: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT rps.pid, rps.name, rps.rss_bytes
            FROM resource_process_samples rps
            JOIN resource_snapshots rs ON rs.taken_at = rps.taken_at
            WHERE rs.snapshot_local_date = ? AND rps.category = 'rss'
            ORDER BY rps.rss_bytes DESC
            LIMIT 5
            """,
            (date_str,),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_network_snapshots_between(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT snapshot_local_date, taken_at, default_iface, default_gateway
            FROM network_snapshots
            WHERE snapshot_local_date BETWEEN ? AND ?
            ORDER BY snapshot_local_date ASC
            """,
            (start_date, end_date),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_network_nameservers_for_date(self, date_str: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT ns.nameserver
            FROM network_nameservers ns
            JOIN network_snapshots snap ON snap.taken_at = ns.taken_at
            WHERE snap.snapshot_local_date = ?
            ORDER BY ns.nameserver ASC
            """,
            (date_str,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_network_interfaces_for_date(self, date_str: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT ni.name, ni.state, ni.rx_bytes, ni.tx_bytes, ni.rx_errors, ni.tx_errors
            FROM network_interfaces ni
            JOIN network_snapshots snap ON snap.taken_at = ni.taken_at
            WHERE snap.snapshot_local_date = ?
            ORDER BY ni.name ASC
            """,
            (date_str,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_latest_disk_snapshot(self, mountpoint: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, taken_at, snapshot_local_date, hostname, mountpoint, filesystem, total_bytes, used_bytes, free_bytes
            FROM disk_snapshots
            WHERE mountpoint = ?
            ORDER BY taken_at DESC
            LIMIT 1
            """,
            (mountpoint,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_previous_disk_snapshot(self, mountpoint: str, before_taken_at: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT id, taken_at, snapshot_local_date, hostname, mountpoint, filesystem, total_bytes, used_bytes, free_bytes
            FROM disk_snapshots
            WHERE mountpoint = ? AND taken_at < ?
            ORDER BY taken_at DESC
            LIMIT 1
            """,
            (mountpoint, before_taken_at),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_dir_size_samples(self, scope: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT taken_at FROM dir_size_samples WHERE scope = ? ORDER BY taken_at DESC LIMIT 1",
            (scope,),
        )
        row = cur.fetchone()
        if not row:
            return None
        taken_at = row["taken_at"]
        cur = self._conn.execute(
            "SELECT path, bytes FROM dir_size_samples WHERE scope = ? AND taken_at = ? ORDER BY bytes DESC",
            (scope, taken_at),
        )
        samples = [(r["path"], int(r["bytes"])) for r in cur.fetchall()]
        return {"taken_at": taken_at, "samples": samples}

    def get_latest_storage_scan_stats(self, scope: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT taken_at, scope, dirs_scanned, errors_skipped
            FROM storage_scan_stats
            WHERE scope = ?
            ORDER BY taken_at DESC
            LIMIT 1
            """,
            (scope,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_previous_dir_size_samples(self, scope: str, before_taken_at: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT taken_at FROM dir_size_samples
            WHERE scope = ? AND taken_at < ?
            ORDER BY taken_at DESC
            LIMIT 1
            """,
            (scope, before_taken_at),
        )
        row = cur.fetchone()
        if not row:
            return None
        taken_at = row["taken_at"]
        cur = self._conn.execute(
            "SELECT path, bytes FROM dir_size_samples WHERE scope = ? AND taken_at = ? ORDER BY bytes DESC",
            (scope, taken_at),
        )
        samples = [(r["path"], int(r["bytes"])) for r in cur.fetchall()]
        return {"taken_at": taken_at, "samples": samples}

    def get_storage_scan_stats_for_taken_at(self, scope: str, taken_at: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT taken_at, scope, dirs_scanned, errors_skipped
            FROM storage_scan_stats
            WHERE scope = ? AND taken_at = ?
            LIMIT 1
            """,
            (scope, taken_at),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def replace_pending_clarification(
        self,
        pending_id: str,
        user_id: str,
        chat_id: str,
        intent_type: str,
        partial_args_json: str,
        question: str,
        options_json: str,
        expires_at: str,
        created_at: str,
    ) -> None:
        self._conn.execute(
            "DELETE FROM pending_clarifications WHERE user_id = ? AND chat_id = ?",
            (user_id, chat_id),
        )
        self._conn.execute(
            """
            INSERT INTO pending_clarifications (
                id,
                user_id,
                chat_id,
                intent_type,
                partial_args_json,
                question,
                options_json,
                expires_at,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pending_id,
                user_id,
                chat_id,
                intent_type,
                partial_args_json,
                question,
                options_json,
                expires_at,
                created_at,
            ),
        )
        self._commit_if_needed()

    def get_pending_clarification(
        self, user_id: str, chat_id: str, now_ts: str
    ) -> PendingClarificationRecord | None:
        self._conn.execute(
            "DELETE FROM pending_clarifications WHERE user_id = ? AND chat_id = ? AND expires_at <= ?",
            (user_id, chat_id, now_ts),
        )
        cur = self._conn.execute(
            """
            SELECT id, user_id, chat_id, intent_type, partial_args_json, question, options_json, expires_at, created_at
            FROM pending_clarifications
            WHERE user_id = ? AND chat_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, chat_id),
        )
        row = cur.fetchone()
        if row:
            return PendingClarificationRecord(**dict(row))
        return None

    def delete_pending_clarification(self, pending_id: str) -> None:
        self._conn.execute(
            "DELETE FROM pending_clarifications WHERE id = ?",
            (pending_id,),
        )
        self._commit_if_needed()

    # --- Last report registry / report history (for followups) ---

    def upsert_last_report(
        self,
        user_id: str,
        report_key: str,
        taken_at: str,
        payload: dict[str, Any],
        audit_ref: str | None = None,
    ) -> None:
        created_at = self._now_iso()
        payload_json = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO report_history (user_id, report_key, taken_at, payload_json, audit_ref, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, report_key, taken_at, payload_json, audit_ref, created_at),
            )
            self._conn.execute(
                """
                INSERT INTO last_report_registry (user_id, report_key, taken_at, payload_json, audit_ref, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, report_key) DO UPDATE SET
                    taken_at = excluded.taken_at,
                    payload_json = excluded.payload_json,
                    audit_ref = excluded.audit_ref,
                    created_at = excluded.created_at
                """,
                (user_id, report_key, taken_at, payload_json, audit_ref, created_at),
            )

    def get_last_report(self, user_id: str, report_key: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT user_id, report_key, taken_at, payload_json, audit_ref, created_at
            FROM last_report_registry
            WHERE user_id = ? AND report_key = ?
            """,
            (user_id, report_key),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_report_history(self, user_id: str, report_key: str, limit: int = 2) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT user_id, report_key, taken_at, payload_json, audit_ref, created_at
            FROM report_history
            WHERE user_id = ? AND report_key = ?
            ORDER BY taken_at DESC
            LIMIT ?
            """,
            (user_id, report_key, int(limit)),
        )
        return [dict(row) for row in cur.fetchall()]

    # --- System facts snapshots (used by /brief) ---

    def insert_system_facts_snapshot(
        self,
        id: str,
        user_id: str,
        taken_at: str,
        boot_id: str,
        schema_version: int,
        facts_json: str,
        content_hash_sha256: str,
        partial: bool,
        errors_json: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO system_facts_snapshots (
                id, user_id, taken_at, boot_id, schema_version,
                facts_json, content_hash_sha256, partial, errors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id = excluded.user_id,
                taken_at = excluded.taken_at,
                boot_id = excluded.boot_id,
                schema_version = excluded.schema_version,
                facts_json = excluded.facts_json,
                content_hash_sha256 = excluded.content_hash_sha256,
                partial = excluded.partial,
                errors_json = excluded.errors_json
            """,
            (
                id,
                user_id,
                taken_at,
                boot_id,
                int(schema_version),
                facts_json,
                content_hash_sha256,
                1 if partial else 0,
                errors_json,
            ),
        )
        self._commit_if_needed()

    def list_system_facts_snapshots(self, user_id: str, limit: int = 2) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT id, user_id, taken_at, boot_id, schema_version, facts_json, content_hash_sha256, partial, errors_json
            FROM system_facts_snapshots
            WHERE user_id = ?
            ORDER BY taken_at DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        )
        return [dict(row) for row in cur.fetchall()]
