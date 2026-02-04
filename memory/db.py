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

    @contextmanager
    def transaction(self) -> Any:
        try:
            self._conn.execute("BEGIN")
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _commit_if_needed(self) -> None:
        if not self._conn.in_transaction:
            self._conn.commit()

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
        self._commit_if_needed()

    def set_preference(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO preferences (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._commit_if_needed()

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
