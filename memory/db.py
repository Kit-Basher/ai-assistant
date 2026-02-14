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
