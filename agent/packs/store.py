from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from agent.packs.diffing import build_pack_diff, build_pack_version_ref
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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_packs (
                    pack_id TEXT PRIMARY KEY,
                    pack_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    pack_type TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    status TEXT NOT NULL,
                    canonical_json TEXT NOT NULL,
                    risk_json TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    quarantine_path TEXT,
                    normalized_path TEXT,
                    installed_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_pack_diffs (
                    from_pack_id TEXT NOT NULL,
                    to_pack_id TEXT NOT NULL,
                    diff_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (from_pack_id, to_pack_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_pack_registry_cache (
                    source_id TEXT PRIMARY KEY,
                    source_json TEXT NOT NULL,
                    listings_json TEXT NOT NULL,
                    fetched_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
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

    @staticmethod
    def _sequence(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    @staticmethod
    def _external_row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        try:
            canonical = json.loads(str(row["canonical_json"] or "{}"))
        except json.JSONDecodeError:
            canonical = {}
        try:
            risk = json.loads(str(row["risk_json"] or "{}"))
        except json.JSONDecodeError:
            risk = {}
        try:
            review = json.loads(str(row["review_json"] or "{}"))
        except json.JSONDecodeError:
            review = {}
        trust = canonical.get("trust") if isinstance(canonical.get("trust"), dict) else {}
        pack_identity = canonical.get("pack_identity") if isinstance(canonical.get("pack_identity"), dict) else {}
        return {
            "pack_id": str(row["pack_id"]),
            "canonical_id": str(pack_identity.get("canonical_id") or row["pack_id"]),
            "content_hash": str(pack_identity.get("content_hash") or "").strip() or None,
            "source_fingerprint": str(pack_identity.get("source_fingerprint") or "").strip() or None,
            "name": str(row["pack_name"]),
            "version": str(row["version"]),
            "type": str(row["pack_type"]),
            "classification": str(row["classification"]),
            "status": str(row["status"]),
            "trust": str(trust.get("level") or "review_required"),
            "risk_score": float(risk.get("score") or 0.0),
            "risk_level": str(risk.get("level") or "unknown"),
            "risk_flags": list(risk.get("flags") if isinstance(risk.get("flags"), list) else []),
            "hard_block_reasons": list(
                risk.get("hard_block_reasons") if isinstance(risk.get("hard_block_reasons"), list) else []
            ),
            "review_required": bool(review.get("review_required", True)),
            "non_executable": True,
            "enabled": False,
            "permissions": canonical.get("permissions") if isinstance(canonical.get("permissions"), dict) else {},
            "capabilities": canonical.get("capabilities") if isinstance(canonical.get("capabilities"), dict) else {},
            "components": list(PackStore._sequence(canonical.get("components"))),
            "assets": list(PackStore._sequence(canonical.get("assets"))),
            "pack_identity": pack_identity,
            "trust_anchor": canonical.get("trust_anchor") if isinstance(canonical.get("trust_anchor"), dict) else {},
            "source_history": list(PackStore._sequence(canonical.get("source_history"))),
            "versions": list(PackStore._sequence(canonical.get("versions"))),
            "source": canonical.get("source") if isinstance(canonical.get("source"), dict) else {},
            "quarantine_path": str(row["quarantine_path"] or "").strip() or None,
            "normalized_path": str(row["normalized_path"] or "").strip() or None,
            "canonical_pack": canonical,
            "risk_report": risk,
            "review_envelope": review,
            "change_summary": review.get("change_summary") if isinstance(review.get("change_summary"), dict) else {},
            "previous_version": review.get("previous_version") if isinstance(review.get("previous_version"), dict) else None,
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
        out = self.list_runtime_packs()
        out.extend(self.list_external_packs())
        return sorted(out, key=lambda item: str(item.get("pack_id") or ""))

    def list_runtime_packs(self) -> list[dict[str, Any]]:
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

    def list_external_packs(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT pack_id, pack_name, version, pack_type, classification, status,
                       canonical_json, risk_json, review_json, quarantine_path, normalized_path,
                       installed_at, updated_at
                FROM external_packs
                ORDER BY pack_id ASC
                """
            )
            rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            parsed = self._external_row_to_dict(row)
            if parsed is not None:
                out.append(parsed)
        return out

    def get_external_pack(self, pack_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT pack_id, pack_name, version, pack_type, classification, status,
                       canonical_json, risk_json, review_json, quarantine_path, normalized_path,
                       installed_at, updated_at
                FROM external_packs
                WHERE pack_id = ?
                """,
                (pack_id,),
            )
            row = cur.fetchone()
        return self._external_row_to_dict(row)

    def _get_external_pack_diff_row(self, from_pack_id: str, to_pack_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT from_pack_id, to_pack_id, diff_json, created_at, updated_at
                FROM external_pack_diffs
                WHERE from_pack_id = ? AND to_pack_id = ?
                """,
                (from_pack_id, to_pack_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        try:
            diff_payload = json.loads(str(row["diff_json"] or "{}"))
        except json.JSONDecodeError:
            diff_payload = {}
        return {
            "from_pack_id": str(row["from_pack_id"]),
            "to_pack_id": str(row["to_pack_id"]),
            "diff": diff_payload,
            "created_at": int(row["created_at"] or 0),
            "updated_at": int(row["updated_at"] or 0),
        }

    def get_registry_source_cache(self, source_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT source_id, source_json, listings_json, fetched_at, expires_at, updated_at
                FROM external_pack_registry_cache
                WHERE source_id = ?
                """,
                (source_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        try:
            source_payload = json.loads(str(row["source_json"] or "{}"))
        except json.JSONDecodeError:
            source_payload = {}
        try:
            listings_payload = json.loads(str(row["listings_json"] or "[]"))
        except json.JSONDecodeError:
            listings_payload = []
        return {
            "source_id": str(row["source_id"]),
            "source": source_payload if isinstance(source_payload, dict) else {},
            "listings": listings_payload if isinstance(listings_payload, list) else [],
            "fetched_at": int(row["fetched_at"] or 0),
            "expires_at": int(row["expires_at"] or 0),
            "updated_at": int(row["updated_at"] or 0),
        }

    def set_registry_source_cache(
        self,
        *,
        source_id: str,
        source_payload: dict[str, Any],
        listings_payload: list[dict[str, Any]],
        ttl_seconds: int,
    ) -> dict[str, Any]:
        now_ts = self._now_ts()
        ttl = max(int(ttl_seconds or 0), 0)
        expires_at = now_ts + ttl if ttl > 0 else now_ts - 1
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO external_pack_registry_cache (
                    source_id, source_json, listings_json, fetched_at, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_json = excluded.source_json,
                    listings_json = excluded.listings_json,
                    fetched_at = excluded.fetched_at,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (
                    source_id,
                    json.dumps(source_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                    json.dumps(listings_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                    now_ts,
                    expires_at,
                    now_ts,
                ),
            )
            self._conn.commit()
        cached = self.get_registry_source_cache(source_id)
        assert cached is not None
        return cached

    def _store_external_pack_diff(self, from_pack_id: str, to_pack_id: str, diff_payload: dict[str, Any]) -> dict[str, Any]:
        existing = self._get_external_pack_diff_row(from_pack_id, to_pack_id)
        now_ts = self._now_ts()
        created_at = int(existing.get("created_at") or now_ts) if existing else now_ts
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO external_pack_diffs (
                    from_pack_id, to_pack_id, diff_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(from_pack_id, to_pack_id) DO UPDATE SET
                    diff_json = excluded.diff_json,
                    updated_at = excluded.updated_at
                """,
                (
                    from_pack_id,
                    to_pack_id,
                    json.dumps(diff_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                    created_at,
                    now_ts,
                ),
            )
            self._conn.commit()
        cached = self._get_external_pack_diff_row(from_pack_id, to_pack_id)
        assert cached is not None
        return cached

    @staticmethod
    def _canonical_id_from_pack(canonical_pack: dict[str, Any]) -> str:
        pack_identity = canonical_pack.get("pack_identity") if isinstance(canonical_pack.get("pack_identity"), dict) else {}
        return str(pack_identity.get("canonical_id") or canonical_pack.get("id") or "").strip() or "external-pack"

    @staticmethod
    def _content_hash_from_pack(canonical_pack: dict[str, Any]) -> str | None:
        pack_identity = canonical_pack.get("pack_identity") if isinstance(canonical_pack.get("pack_identity"), dict) else {}
        return str(pack_identity.get("content_hash") or "").strip() or None

    @staticmethod
    def _source_key_from_pack(canonical_pack: dict[str, Any]) -> str | None:
        pack_identity = canonical_pack.get("pack_identity") if isinstance(canonical_pack.get("pack_identity"), dict) else {}
        return str(pack_identity.get("source_key") or "").strip() or None

    @staticmethod
    def _merge_source_history(existing: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_fingerprint: dict[str, dict[str, Any]] = {}
        for entry in existing + current:
            if not isinstance(entry, dict):
                continue
            fingerprint = str(entry.get("source_fingerprint") or "").strip()
            key = fingerprint or json.dumps(entry, ensure_ascii=True, sort_keys=True)
            by_fingerprint[key] = entry
        return sorted(
            by_fingerprint.values(),
            key=lambda item: str(item.get("fetched_at") or ""),
        )

    @staticmethod
    def _merge_versions(existing: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_version_key: dict[str, dict[str, Any]] = {}
        for entry in existing + current:
            if not isinstance(entry, dict):
                continue
            content_hash = str(entry.get("content_hash") or "").strip()
            canonical_id = str(entry.get("canonical_id") or "").strip()
            version_key = content_hash or canonical_id
            if not version_key:
                continue
            by_version_key[version_key] = entry
        return sorted(
            by_version_key.values(),
            key=lambda item: str(item.get("seen_at") or ""),
        )

    @staticmethod
    def _diff_external_packs(previous_pack: dict[str, Any], current_pack: dict[str, Any]) -> dict[str, Any]:
        def _included_entries(pack: dict[str, Any]) -> dict[str, str]:
            included: dict[str, str] = {}
            for component in PackStore._sequence(pack.get("components")):
                if not isinstance(component, dict) or not bool(component.get("included", False)):
                    continue
                included[str(component.get("path") or "")] = str(component.get("sha256") or "")
            for asset in PackStore._sequence(pack.get("assets")):
                if not isinstance(asset, dict) or not bool(asset.get("included", False)):
                    continue
                included[str(asset.get("path") or "")] = str(asset.get("sha256") or "")
            return included

        previous_entries = _included_entries(previous_pack)
        current_entries = _included_entries(current_pack)
        new_files = sorted(path for path in current_entries if path not in previous_entries)
        removed_files = sorted(path for path in previous_entries if path not in current_entries)
        changed_instructions = sorted(
            path
            for path in current_entries
            if path in previous_entries
            and current_entries[path] != previous_entries[path]
            and str(path).lower().endswith(".md")
        )
        return {
            "new_files": new_files,
            "removed_files": removed_files,
            "changed_instructions": changed_instructions,
            "has_changes": bool(new_files or removed_files or changed_instructions),
        }

    def get_external_pack_history(self, canonical_id: str) -> dict[str, Any] | None:
        current = self.get_external_pack(canonical_id)
        if current is None:
            return None
        source_keys = {
            str(entry.get("source_key") or "").strip()
            for entry in self._sequence(current.get("source_history"))
            if isinstance(entry, dict) and str(entry.get("source_key") or "").strip()
        }
        pack_identity = current.get("pack_identity") if isinstance(current.get("pack_identity"), dict) else {}
        current_source_key = str(pack_identity.get("source_key") or "").strip()
        if current_source_key:
            source_keys.add(current_source_key)
        lineage = []
        for row in self.list_external_packs():
            row_identity = row.get("pack_identity") if isinstance(row.get("pack_identity"), dict) else {}
            row_source_keys = {
                str(entry.get("source_key") or "").strip()
                for entry in self._sequence(row.get("source_history"))
                if isinstance(entry, dict) and str(entry.get("source_key") or "").strip()
            }
            row_identity_key = str(row_identity.get("source_key") or "").strip()
            if row_identity_key:
                row_source_keys.add(row_identity_key)
            if source_keys and row_source_keys.isdisjoint(source_keys):
                continue
            lineage.append(row)
        if not lineage:
            lineage = [current]
        lineage = sorted(
            lineage,
            key=lambda item: (int(item.get("installed_at") or 0), int(item.get("updated_at") or 0), str(item.get("canonical_id") or "")),
        )
        merged_source_history = self._merge_source_history(
            [],
            [
                entry
                for row in lineage
                for entry in self._sequence(row.get("source_history"))
                if isinstance(entry, dict)
            ],
        )
        version_chain = [build_pack_version_ref(row).to_dict() for row in lineage]
        return {
            "pack": current,
            "source_history": merged_source_history,
            "version_chain": version_chain,
            "version_count": len(version_chain),
        }

    def get_or_build_external_pack_diff(self, from_pack_id: str, to_pack_id: str) -> dict[str, Any] | None:
        cached = self._get_external_pack_diff_row(from_pack_id, to_pack_id)
        if cached is not None:
            return cached["diff"] if isinstance(cached.get("diff"), dict) else None
        previous = self.get_external_pack(from_pack_id)
        current = self.get_external_pack(to_pack_id)
        if previous is None or current is None:
            return None
        diff_payload = build_pack_diff(previous, current).to_dict()
        stored = self._store_external_pack_diff(from_pack_id, to_pack_id, diff_payload)
        return stored["diff"] if isinstance(stored.get("diff"), dict) else None

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

    def record_external_pack(
        self,
        *,
        canonical_pack: dict[str, Any],
        classification: str,
        status: str,
        risk_report: dict[str, Any],
        review_envelope: dict[str, Any],
        quarantine_path: str | None,
        normalized_path: str | None,
    ) -> dict[str, Any]:
        now_ts = self._now_ts()
        pack_id = self._canonical_id_from_pack(canonical_pack)
        canonical_pack["id"] = pack_id
        existing = self.get_external_pack(pack_id)
        installed_at = int(existing.get("installed_at") or now_ts) if existing else now_ts
        existing_history = self._sequence(existing.get("source_history")) if isinstance(existing, dict) else []
        current_history = self._sequence(canonical_pack.get("source_history"))
        canonical_pack["source_history"] = self._merge_source_history(existing_history, current_history)
        existing_versions = self._sequence(existing.get("versions")) if isinstance(existing, dict) else []
        current_versions = self._sequence(canonical_pack.get("versions"))
        canonical_pack["versions"] = self._merge_versions(existing_versions, current_versions)
        if existing is not None:
            existing_anchor = existing.get("trust_anchor") if isinstance(existing.get("trust_anchor"), dict) else {}
            anchor = canonical_pack.get("trust_anchor") if isinstance(canonical_pack.get("trust_anchor"), dict) else {}
            canonical_pack["trust_anchor"] = {
                "first_seen_at": str(existing_anchor.get("first_seen_at") or anchor.get("first_seen_at") or ""),
                "first_seen_source": existing_anchor.get("first_seen_source") or anchor.get("first_seen_source"),
                "local_review_status": str(existing_anchor.get("local_review_status") or anchor.get("local_review_status") or "unreviewed"),
                "user_approved_hashes": list(
                    dict.fromkeys(
                        [
                            str(item).strip()
                            for item in (
                                (existing_anchor.get("user_approved_hashes") if isinstance(existing_anchor.get("user_approved_hashes"), list) else [])
                                + (anchor.get("user_approved_hashes") if isinstance(anchor.get("user_approved_hashes"), list) else [])
                            )
                            if str(item).strip()
                        ]
                    )
                ),
            }
        source_key = self._source_key_from_pack(canonical_pack)
        content_hash = self._content_hash_from_pack(canonical_pack)
        prior_same_source: dict[str, Any] | None = None
        if source_key:
            for row in self.list_external_packs():
                row_identity = row.get("pack_identity") if isinstance(row.get("pack_identity"), dict) else {}
                if str(row_identity.get("source_key") or "").strip() != source_key:
                    continue
                row_canonical_id = str(row.get("canonical_id") or row.get("pack_id") or "").strip()
                row_content_hash = str(row.get("content_hash") or "").strip() or None
                if row_canonical_id == pack_id and row_content_hash == content_hash:
                    continue
                if prior_same_source is None or int(row.get("updated_at") or 0) > int(prior_same_source.get("updated_at") or 0):
                    prior_same_source = row
        if prior_same_source is not None:
            risk_flags = list(risk_report.get("flags") if isinstance(risk_report.get("flags"), list) else [])
            if "upstream_content_changed" not in risk_flags:
                risk_flags.append("upstream_content_changed")
            risk_score = min(1.0, float(risk_report.get("score") or 0.0) + 0.2)
            risk_report = {
                **risk_report,
                "score": round(risk_score, 4),
                "flags": sorted(dict.fromkeys(risk_flags)),
            }
            trust = canonical_pack.get("trust") if isinstance(canonical_pack.get("trust"), dict) else {}
            trust_flags = list(trust.get("flags") if isinstance(trust.get("flags"), list) else [])
            if "upstream_content_changed" not in trust_flags:
                trust_flags.append("upstream_content_changed")
            canonical_pack["trust"] = {
                **trust,
                "risk_score": round(risk_score, 4),
                "flags": sorted(dict.fromkeys(trust_flags)),
            }
            diff_summary = self._diff_external_packs(
                prior_same_source.get("canonical_pack") if isinstance(prior_same_source.get("canonical_pack"), dict) else {},
                canonical_pack,
            )
            review_envelope = {
                **review_envelope,
                "why_risk": sorted(
                    dict.fromkeys(
                        list(review_envelope.get("why_risk") if isinstance(review_envelope.get("why_risk"), list) else [])
                        + ["upstream_content_changed"]
                    )
                ),
                "safe_options": [
                    "Review it as a new pack.",
                    "Compare it to the previous version.",
                    "Ignore the update.",
                ],
                "summary": (
                    "This pack has changed since the last time it was seen. "
                    + (
                        "I treat it as a new version, not the same pack. "
                        if str(prior_same_source.get("canonical_id") or prior_same_source.get("pack_id") or "").strip() != pack_id
                        else "It normalized to the same safe pack, but the upstream source changed. "
                    )
                    + "I can review it as a new pack, compare it to the previous version, or ignore the update."
                ),
                "change_summary": diff_summary,
                "previous_version": {
                    "canonical_id": prior_same_source.get("canonical_id") or prior_same_source.get("pack_id"),
                    "content_hash": prior_same_source.get("content_hash"),
                },
            }
            current_versions = self._sequence(canonical_pack.get("versions"))
            previous_versions = self._sequence(prior_same_source.get("versions"))
            canonical_pack["versions"] = self._merge_versions(
                previous_versions,
                current_versions + [
                    {
                        "canonical_id": str(prior_same_source.get("canonical_id") or prior_same_source.get("pack_id") or "").strip(),
                        "content_hash": str(prior_same_source.get("content_hash") or "").strip() or None,
                        "status": str(prior_same_source.get("status") or "").strip() or None,
                        "seen_at": int(prior_same_source.get("updated_at") or 0),
                    }
                ],
            )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO external_packs (
                    pack_id, pack_name, version, pack_type, classification, status,
                    canonical_json, risk_json, review_json, quarantine_path, normalized_path,
                    installed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pack_id) DO UPDATE SET
                    pack_name = excluded.pack_name,
                    version = excluded.version,
                    pack_type = excluded.pack_type,
                    classification = excluded.classification,
                    status = excluded.status,
                    canonical_json = excluded.canonical_json,
                    risk_json = excluded.risk_json,
                    review_json = excluded.review_json,
                    quarantine_path = excluded.quarantine_path,
                    normalized_path = excluded.normalized_path,
                    updated_at = excluded.updated_at
                """,
                (
                    pack_id,
                    str(canonical_pack.get("name") or pack_id),
                    str(canonical_pack.get("version") or "0.1.0"),
                    str(canonical_pack.get("type") or "skill"),
                    str(classification or "").strip() or "unknown_pack",
                    str(status or "").strip() or "blocked",
                    json.dumps(canonical_pack, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                    json.dumps(risk_report, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                    json.dumps(review_envelope, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                    (quarantine_path or "").strip() or None,
                    (normalized_path or "").strip() or None,
                    installed_at,
                    now_ts,
                ),
            )
            self._conn.commit()
        updated = self.get_external_pack(pack_id)
        assert updated is not None
        if prior_same_source is not None:
            previous_id = str(prior_same_source.get("canonical_id") or prior_same_source.get("pack_id") or "").strip()
            if previous_id:
                self._store_external_pack_diff(previous_id, pack_id, build_pack_diff(prior_same_source, updated).to_dict())
        return updated

    def set_external_pack_review_status(
        self,
        canonical_id: str,
        *,
        local_review_status: str,
        approve_current_hash: bool = False,
    ) -> dict[str, Any] | None:
        current = self.get_external_pack(canonical_id)
        if current is None:
            return None
        canonical_pack = current.get("canonical_pack") if isinstance(current.get("canonical_pack"), dict) else {}
        trust_anchor = canonical_pack.get("trust_anchor") if isinstance(canonical_pack.get("trust_anchor"), dict) else {}
        content_hash = self._content_hash_from_pack(canonical_pack)
        approved_hashes = [
            str(item).strip()
            for item in (trust_anchor.get("user_approved_hashes") if isinstance(trust_anchor.get("user_approved_hashes"), list) else [])
            if str(item).strip()
        ]
        if approve_current_hash and content_hash and content_hash not in approved_hashes:
            approved_hashes.append(content_hash)
        canonical_pack["trust_anchor"] = {
            "first_seen_at": str(trust_anchor.get("first_seen_at") or ""),
            "first_seen_source": trust_anchor.get("first_seen_source"),
            "local_review_status": str(local_review_status or "reviewed").strip() or "reviewed",
            "user_approved_hashes": approved_hashes,
        }
        with self._lock:
            self._conn.execute(
                """
                UPDATE external_packs
                SET canonical_json = ?, updated_at = ?
                WHERE pack_id = ?
                """,
                (
                    json.dumps(canonical_pack, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                    self._now_ts(),
                    canonical_id,
                ),
            )
            self._conn.commit()
        return self.get_external_pack(canonical_id)

    def external_storage_root(self) -> str:
        parent = Path(self.db_path).expanduser().resolve().parent
        return str((parent / "external_packs").resolve())
