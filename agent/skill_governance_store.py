from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any


class SkillGovernanceStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._ensure_schema()

    @staticmethod
    def _now_ts() -> int:
        return int(time.time())

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_execution_governance (
                    skill_id TEXT PRIMARY KEY,
                    skill_type TEXT NOT NULL,
                    requested_execution_mode TEXT NOT NULL,
                    requested_capabilities_json TEXT NOT NULL,
                    persistence_requested INTEGER NOT NULL DEFAULT 0,
                    allowed INTEGER NOT NULL DEFAULT 0,
                    requires_user_approval INTEGER NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL,
                    source_issues_json TEXT NOT NULL DEFAULT '[]',
                    source_pack TEXT,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS managed_adapters (
                    adapter_id TEXT PRIMARY KEY,
                    adapter_type TEXT NOT NULL,
                    source_skill TEXT,
                    source_package TEXT,
                    approved INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    startup_policy TEXT,
                    health_status TEXT,
                    last_error TEXT,
                    owner TEXT,
                    requested_by TEXT,
                    reason TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS background_tasks (
                    task_id TEXT PRIMARY KEY,
                    source_skill TEXT,
                    source_package TEXT,
                    schedule TEXT,
                    trigger_type TEXT,
                    approved INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    health_status TEXT,
                    last_run_at INTEGER,
                    last_error TEXT,
                    resource_limits_json TEXT NOT NULL DEFAULT '{}',
                    owner TEXT,
                    requested_by TEXT,
                    reason TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            self._conn.commit()

    @staticmethod
    def _decode_json(value: Any, *, default: Any) -> Any:
        try:
            return json.loads(str(value or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            return default

    def record_skill_governance(
        self,
        *,
        skill_id: str,
        skill_type: str,
        requested_execution_mode: str,
        requested_capabilities: list[str],
        persistence_requested: bool,
        allowed: bool,
        requires_user_approval: bool,
        reason: str,
        source_issues: list[str] | None = None,
        source_pack: str | None = None,
    ) -> dict[str, Any]:
        now_ts = self._now_ts()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO skill_execution_governance (
                    skill_id, skill_type, requested_execution_mode, requested_capabilities_json,
                    persistence_requested, allowed, requires_user_approval, reason,
                    source_issues_json, source_pack, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(skill_id) DO UPDATE SET
                    skill_type = excluded.skill_type,
                    requested_execution_mode = excluded.requested_execution_mode,
                    requested_capabilities_json = excluded.requested_capabilities_json,
                    persistence_requested = excluded.persistence_requested,
                    allowed = excluded.allowed,
                    requires_user_approval = excluded.requires_user_approval,
                    reason = excluded.reason,
                    source_issues_json = excluded.source_issues_json,
                    source_pack = excluded.source_pack,
                    updated_at = excluded.updated_at
                """,
                (
                    str(skill_id or "").strip() or "unknown_skill",
                    str(skill_type or "").strip().lower() or "general",
                    str(requested_execution_mode or "").strip().lower() or "in_process",
                    json.dumps(sorted({str(item).strip().lower() for item in requested_capabilities if str(item).strip()}), ensure_ascii=True),
                    1 if persistence_requested else 0,
                    1 if allowed else 0,
                    1 if requires_user_approval else 0,
                    str(reason or "").strip() or "unknown",
                    json.dumps(sorted({str(item).strip().lower() for item in (source_issues or []) if str(item).strip()}), ensure_ascii=True),
                    str(source_pack or "").strip() or None,
                    now_ts,
                ),
            )
            self._conn.commit()
            return self.get_skill_governance(skill_id) or {}

    def get_skill_governance(self, skill_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT skill_id, skill_type, requested_execution_mode, requested_capabilities_json,
                       persistence_requested, allowed, requires_user_approval, reason,
                       source_issues_json, source_pack, updated_at
                FROM skill_execution_governance
                WHERE skill_id = ?
                """,
                (skill_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "skill_id": str(row["skill_id"]),
            "skill_type": str(row["skill_type"]),
            "requested_execution_mode": str(row["requested_execution_mode"]),
            "requested_capabilities": [
                str(item).strip()
                for item in self._decode_json(row["requested_capabilities_json"], default=[])
                if str(item).strip()
            ],
            "persistence_requested": bool(int(row["persistence_requested"] or 0)),
            "allowed": bool(int(row["allowed"] or 0)),
            "requires_user_approval": bool(int(row["requires_user_approval"] or 0)),
            "reason": str(row["reason"]),
            "source_issues": [
                str(item).strip()
                for item in self._decode_json(row["source_issues_json"], default=[])
                if str(item).strip()
            ],
            "source_pack": str(row["source_pack"] or "").strip() or None,
            "updated_at": int(row["updated_at"] or 0),
        }

    def list_skill_governance(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT skill_id
                FROM skill_execution_governance
                ORDER BY skill_id ASC
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            parsed = self.get_skill_governance(str(row["skill_id"]))
            if parsed is not None:
                out.append(parsed)
        return out

    def register_managed_adapter(
        self,
        *,
        adapter_id: str,
        adapter_type: str,
        source_skill: str | None,
        source_package: str | None,
        approved: bool,
        enabled: bool,
        startup_policy: str | None,
        health_status: str | None,
        last_error: str | None,
        owner: str | None,
        requested_by: str | None,
        reason: str | None,
    ) -> dict[str, Any]:
        now_ts = self._now_ts()
        with self._lock:
            existing = self.get_managed_adapter(adapter_id)
            created_at = int(existing.get("created_at") or now_ts) if existing else now_ts
            self._conn.execute(
                """
                INSERT INTO managed_adapters (
                    adapter_id, adapter_type, source_skill, source_package, approved, enabled,
                    startup_policy, health_status, last_error, owner, requested_by, reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(adapter_id) DO UPDATE SET
                    adapter_type = excluded.adapter_type,
                    source_skill = excluded.source_skill,
                    source_package = excluded.source_package,
                    approved = excluded.approved,
                    enabled = excluded.enabled,
                    startup_policy = excluded.startup_policy,
                    health_status = excluded.health_status,
                    last_error = excluded.last_error,
                    owner = excluded.owner,
                    requested_by = excluded.requested_by,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
                """,
                (
                    adapter_id,
                    adapter_type,
                    (str(source_skill).strip() or None) if source_skill is not None else None,
                    (str(source_package).strip() or None) if source_package is not None else None,
                    1 if approved else 0,
                    1 if enabled else 0,
                    str(startup_policy or "").strip() or None,
                    str(health_status or "").strip() or None,
                    str(last_error or "").strip() or None,
                    str(owner or "").strip() or None,
                    str(requested_by or "").strip() or None,
                    str(reason or "").strip() or None,
                    created_at,
                    now_ts,
                ),
            )
            self._conn.commit()
            return self.get_managed_adapter(adapter_id) or {}

    def get_managed_adapter(self, adapter_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT adapter_id, adapter_type, source_skill, source_package, approved, enabled,
                       startup_policy, health_status, last_error, owner, requested_by, reason,
                       created_at, updated_at
                FROM managed_adapters
                WHERE adapter_id = ?
                """,
                (adapter_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "adapter_id": str(row["adapter_id"]),
            "adapter_type": str(row["adapter_type"]),
            "source_skill": str(row["source_skill"] or "").strip() or None,
            "source_package": str(row["source_package"] or "").strip() or None,
            "approved": bool(int(row["approved"] or 0)),
            "enabled": bool(int(row["enabled"] or 0)),
            "startup_policy": str(row["startup_policy"] or "").strip() or None,
            "health_status": str(row["health_status"] or "").strip() or None,
            "last_error": str(row["last_error"] or "").strip() or None,
            "owner": str(row["owner"] or "").strip() or None,
            "requested_by": str(row["requested_by"] or "").strip() or None,
            "reason": str(row["reason"] or "").strip() or None,
            "created_at": int(row["created_at"] or 0),
            "updated_at": int(row["updated_at"] or 0),
        }

    def list_managed_adapters(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT adapter_id FROM managed_adapters ORDER BY adapter_id ASC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            parsed = self.get_managed_adapter(str(row["adapter_id"]))
            if parsed is not None:
                out.append(parsed)
        return out

    def has_approved_adapter_for_skill(self, skill_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1
                FROM managed_adapters
                WHERE source_skill = ? AND approved = 1
                LIMIT 1
                """,
                (skill_id,),
            ).fetchone()
        return row is not None

    def register_background_task(
        self,
        *,
        task_id: str,
        source_skill: str | None,
        source_package: str | None,
        schedule: str | None,
        trigger_type: str | None,
        approved: bool,
        enabled: bool,
        health_status: str | None,
        last_run_at: int | None,
        last_error: str | None,
        resource_limits: dict[str, Any] | None,
        owner: str | None,
        requested_by: str | None,
        reason: str | None,
    ) -> dict[str, Any]:
        now_ts = self._now_ts()
        with self._lock:
            existing = self.get_background_task(task_id)
            created_at = int(existing.get("created_at") or now_ts) if existing else now_ts
            self._conn.execute(
                """
                INSERT INTO background_tasks (
                    task_id, source_skill, source_package, schedule, trigger_type, approved, enabled,
                    health_status, last_run_at, last_error, resource_limits_json, owner, requested_by,
                    reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    source_skill = excluded.source_skill,
                    source_package = excluded.source_package,
                    schedule = excluded.schedule,
                    trigger_type = excluded.trigger_type,
                    approved = excluded.approved,
                    enabled = excluded.enabled,
                    health_status = excluded.health_status,
                    last_run_at = excluded.last_run_at,
                    last_error = excluded.last_error,
                    resource_limits_json = excluded.resource_limits_json,
                    owner = excluded.owner,
                    requested_by = excluded.requested_by,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    (str(source_skill).strip() or None) if source_skill is not None else None,
                    (str(source_package).strip() or None) if source_package is not None else None,
                    str(schedule or "").strip() or None,
                    str(trigger_type or "").strip() or None,
                    1 if approved else 0,
                    1 if enabled else 0,
                    str(health_status or "").strip() or None,
                    int(last_run_at) if last_run_at is not None else None,
                    str(last_error or "").strip() or None,
                    json.dumps(resource_limits or {}, ensure_ascii=True, sort_keys=True),
                    str(owner or "").strip() or None,
                    str(requested_by or "").strip() or None,
                    str(reason or "").strip() or None,
                    created_at,
                    now_ts,
                ),
            )
            self._conn.commit()
            return self.get_background_task(task_id) or {}

    def get_background_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT task_id, source_skill, source_package, schedule, trigger_type, approved, enabled,
                       health_status, last_run_at, last_error, resource_limits_json, owner, requested_by,
                       reason, created_at, updated_at
                FROM background_tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "task_id": str(row["task_id"]),
            "source_skill": str(row["source_skill"] or "").strip() or None,
            "source_package": str(row["source_package"] or "").strip() or None,
            "schedule": str(row["schedule"] or "").strip() or None,
            "trigger_type": str(row["trigger_type"] or "").strip() or None,
            "approved": bool(int(row["approved"] or 0)),
            "enabled": bool(int(row["enabled"] or 0)),
            "health_status": str(row["health_status"] or "").strip() or None,
            "last_run_at": int(row["last_run_at"] or 0) if row["last_run_at"] is not None else None,
            "last_error": str(row["last_error"] or "").strip() or None,
            "resource_limits": self._decode_json(row["resource_limits_json"], default={}),
            "owner": str(row["owner"] or "").strip() or None,
            "requested_by": str(row["requested_by"] or "").strip() or None,
            "reason": str(row["reason"] or "").strip() or None,
            "created_at": int(row["created_at"] or 0),
            "updated_at": int(row["updated_at"] or 0),
        }

    def list_background_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT task_id FROM background_tasks ORDER BY task_id ASC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            parsed = self.get_background_task(str(row["task_id"]))
            if parsed is not None:
                out.append(parsed)
        return out

    def has_approved_background_task_for_skill(self, skill_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1
                FROM background_tasks
                WHERE source_skill = ? AND approved = 1
                LIMIT 1
                """,
                (skill_id,),
            ).fetchone()
        return row is not None
