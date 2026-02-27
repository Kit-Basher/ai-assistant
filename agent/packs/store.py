from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

from agent.packs.manifest import PackManifest, compute_permissions_hash, normalize_permissions


class PackStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_packs (
                    pack_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    trust TEXT NOT NULL,
                    manifest_path TEXT,
                    permissions_json TEXT NOT NULL,
                    permissions_hash TEXT NOT NULL,
                    approved_permissions_hash TEXT,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    installed_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            self._conn.commit()

    @staticmethod
    def _now_ts() -> int:
        return int(time.time())

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        permissions_raw = str(row["permissions_json"] or "{}")
        try:
            permissions = json.loads(permissions_raw)
        except json.JSONDecodeError:
            permissions = {"ifaces": [], "fs": {"read": [], "write": []}, "net": {"allow_domains": [], "deny": []}, "proc": {"spawn": []}}
        return {
            "pack_id": str(row["pack_id"]),
            "version": str(row["version"]),
            "trust": str(row["trust"]),
            "manifest_path": str(row["manifest_path"] or "").strip() or None,
            "permissions": normalize_permissions(permissions),
            "permissions_hash": str(row["permissions_hash"]),
            "approved_permissions_hash": str(row["approved_permissions_hash"] or "").strip() or None,
            "enabled": bool(int(row["enabled"] or 0)),
            "installed_at": int(row["installed_at"] or 0),
            "updated_at": int(row["updated_at"] or 0),
        }

    def get_pack(self, pack_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT pack_id, version, trust, manifest_path, permissions_json, permissions_hash,
                       approved_permissions_hash, enabled, installed_at, updated_at
                FROM skill_packs
                WHERE pack_id = ?
                """,
                (pack_id,),
            )
            row = cur.fetchone()
            return self._row_to_dict(row)

    def list_packs(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT pack_id, version, trust, manifest_path, permissions_json, permissions_hash,
                       approved_permissions_hash, enabled, installed_at, updated_at
                FROM skill_packs
                ORDER BY pack_id ASC
                """
            )
            rows = cur.fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                parsed = self._row_to_dict(row)
                if parsed is not None:
                    out.append(parsed)
            return out

    def install_pack(
        self,
        manifest: PackManifest,
        *,
        manifest_path: str | None = None,
        enable: bool = False,
    ) -> dict[str, Any]:
        now_ts = self._now_ts()
        pack_id = manifest.pack_id
        permissions = normalize_permissions(manifest.permissions)
        permissions_json = json.dumps(permissions, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        permissions_hash = compute_permissions_hash(permissions)

        with self._lock:
            existing = self.get_pack(pack_id)
            approved_permissions_hash: str | None
            if manifest.trust == "native":
                approved_permissions_hash = permissions_hash
            elif existing is not None:
                approved_permissions_hash = str(existing.get("approved_permissions_hash") or "").strip() or None
            else:
                approved_permissions_hash = None

            enabled_value = bool(enable)
            if manifest.trust == "native":
                enabled_value = True
            elif existing is not None:
                enabled_value = bool(enable) if enable else bool(existing.get("enabled", False))

            installed_at = int(existing.get("installed_at") or now_ts) if existing else now_ts
            self._conn.execute(
                """
                INSERT INTO skill_packs (
                    pack_id, version, trust, manifest_path, permissions_json, permissions_hash,
                    approved_permissions_hash, enabled, installed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pack_id) DO UPDATE SET
                    version = excluded.version,
                    trust = excluded.trust,
                    manifest_path = excluded.manifest_path,
                    permissions_json = excluded.permissions_json,
                    permissions_hash = excluded.permissions_hash,
                    approved_permissions_hash = excluded.approved_permissions_hash,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    pack_id,
                    manifest.version,
                    manifest.trust,
                    (manifest_path or "").strip() or None,
                    permissions_json,
                    permissions_hash,
                    approved_permissions_hash,
                    1 if enabled_value else 0,
                    installed_at,
                    now_ts,
                ),
            )
            self._conn.commit()
            updated = self.get_pack(pack_id)
            assert updated is not None
            return updated

    def ensure_native_pack(
        self,
        *,
        pack_id: str,
        version: str,
        permissions: dict[str, Any],
        manifest_path: str | None = None,
    ) -> dict[str, Any]:
        manifest = PackManifest(
            pack_id=pack_id,
            version=version,
            title=pack_id,
            description="native skill pack",
            entrypoints=(f"skills.{pack_id}:handler",),
            trust="native",
            permissions=permissions,
        )
        return self.install_pack(manifest, manifest_path=manifest_path, enable=True)

    def set_enabled(self, pack_id: str, enabled: bool) -> dict[str, Any] | None:
        now_ts = self._now_ts()
        with self._lock:
            self._conn.execute(
                "UPDATE skill_packs SET enabled = ?, updated_at = ? WHERE pack_id = ?",
                (1 if enabled else 0, now_ts, pack_id),
            )
            self._conn.commit()
            return self.get_pack(pack_id)

    def set_approval_hash(self, pack_id: str, approval_hash: str | None) -> dict[str, Any] | None:
        now_ts = self._now_ts()
        with self._lock:
            self._conn.execute(
                "UPDATE skill_packs SET approved_permissions_hash = ?, updated_at = ? WHERE pack_id = ?",
                ((approval_hash or "").strip() or None, now_ts, pack_id),
            )
            self._conn.commit()
            return self.get_pack(pack_id)
