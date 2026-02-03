from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
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
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def init_schema(self, schema_path: str) -> None:
        with open(schema_path, "r", encoding="utf-8") as handle:
            script = handle.read()
        self._conn.executescript(script)
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
        self._conn.commit()
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
        self._conn.commit()
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
        self._conn.commit()
        return int(cur.lastrowid)

    def add_reminder(self, when_ts: str, text: str) -> int:
        created_at = self._now_iso()
        cur = self._conn.execute(
            "INSERT INTO reminders (when_ts, text, status, created_at) VALUES (?, ?, 'pending', ?)",
            (when_ts, text, created_at),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_due_reminders(self, now_ts: str) -> list[ReminderRecord]:
        cur = self._conn.execute(
            "SELECT id, when_ts, text, status, created_at FROM reminders WHERE status = 'pending' AND when_ts <= ? ORDER BY when_ts ASC",
            (now_ts,),
        )
        rows = cur.fetchall()
        return [ReminderRecord(**dict(row)) for row in rows]

    def mark_reminder_sent(self, reminder_id: int) -> None:
        self._conn.execute(
            "UPDATE reminders SET status = 'sent' WHERE id = ?",
            (reminder_id,),
        )
        self._conn.commit()

    def set_preference(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO preferences (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_preference(self, key: str) -> str | None:
        cur = self._conn.execute(
            "SELECT value FROM preferences WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        return row["value"] if row else None

    def log_activity(self, event_type: str, payload: dict[str, Any]) -> None:
        ts = self._now_iso()
        self._conn.execute(
            "INSERT INTO activity_log (ts, type, payload_json) VALUES (?, ?, ?)",
            (ts, event_type, json.dumps(payload, ensure_ascii=True)),
        )
        self._conn.commit()

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
        self._conn.commit()

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
        self._conn.commit()
