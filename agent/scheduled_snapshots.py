from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from memory.db import MemoryDB
from skills.disk_report import safe_disk
from skills.storage_governor import collector
from skills.resource_governor import collector as resource_collector
from skills.network_governor import collector as network_collector


def run_scheduled_snapshot(db: MemoryDB, home_path: str, root_path: str) -> None:
    snapshot = safe_disk.build_snapshot(home_path, root_path)
    snapshot_json = json.dumps(snapshot, ensure_ascii=True, sort_keys=True)
    snapshot_hash = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
    db.log_activity(
        "disk_report",
        {
            "snapshot": snapshot,
            "snapshot_hash": snapshot_hash,
            "source": "scheduled",
        },
    )


def safe_run_scheduled_snapshot(db: MemoryDB, home_path: str, root_path: str) -> None:
    try:
        run_scheduled_snapshot(db, home_path, root_path)
    except Exception:  # pragma: no cover - logging only
        logging.getLogger(__name__).exception("Scheduled disk snapshot failed")


def run_storage_snapshot(db: MemoryDB, timezone: str, home_path: str, user_id: str) -> dict[str, Any]:
    try:
        with db.transaction():
            payload = collector.collect_and_persist_snapshot(
                db,
                timezone=timezone,
                home_path=home_path,
            )
            try:
                db.log_activity(
                    "storage_snapshot",
                    {
                        "event_type": "storage_snapshot",
                        "mode": "observe",
                        "status": "executed",
                        "actor_id": user_id,
                        "timezone": timezone,
                        "source": "scheduled",
                        "taken_at": payload.get("taken_at"),
                    },
                )
            except Exception as exc:
                raise RuntimeError("audit_log_failed") from exc
    except Exception as exc:
        if str(exc) == "audit_log_failed":
            raise RuntimeError("Audit logging failed. Operation aborted.") from exc
        raise
    return payload


def safe_run_storage_snapshot(db: MemoryDB, timezone: str, home_path: str, user_id: str) -> None:
    try:
        run_storage_snapshot(db, timezone, home_path, user_id)
    except Exception:  # pragma: no cover - logging only
        logging.getLogger(__name__).exception("Scheduled storage snapshot failed")


def run_resource_snapshot(db: MemoryDB, timezone: str, user_id: str) -> dict[str, Any]:
    try:
        with db.transaction():
            payload = resource_collector.collect_and_persist_snapshot(
                db,
                timezone=timezone,
            )
            try:
                db.log_activity(
                    "resource_snapshot",
                    {
                        "event_type": "resource_snapshot",
                        "mode": "observe",
                        "status": "executed",
                        "actor_id": user_id,
                        "timezone": timezone,
                        "source": "scheduled",
                        "taken_at": payload.get("taken_at"),
                    },
                )
            except Exception as exc:
                raise RuntimeError("audit_log_failed") from exc
    except Exception as exc:
        if str(exc) == "audit_log_failed":
            raise RuntimeError("Audit logging failed. Operation aborted.") from exc
        raise
    return payload


def safe_run_resource_snapshot(db: MemoryDB, timezone: str, user_id: str) -> None:
    try:
        run_resource_snapshot(db, timezone, user_id)
    except Exception:  # pragma: no cover - logging only
        logging.getLogger(__name__).exception("Scheduled resource snapshot failed")


def run_network_snapshot(db: MemoryDB, timezone: str, user_id: str) -> dict[str, Any]:
    try:
        with db.transaction():
            payload = network_collector.collect_and_persist_snapshot(
                db,
                timezone=timezone,
            )
            try:
                db.log_activity(
                    "network_snapshot",
                    {
                        "event_type": "network_snapshot",
                        "mode": "observe",
                        "status": "executed",
                        "actor_id": user_id,
                        "timezone": timezone,
                        "source": "scheduled",
                        "taken_at": payload.get("taken_at"),
                    },
                )
            except Exception as exc:
                raise RuntimeError("audit_log_failed") from exc
    except Exception as exc:
        if str(exc) == "audit_log_failed":
            raise RuntimeError("Audit logging failed. Operation aborted.") from exc
        raise
    return payload


def safe_run_network_snapshot(db: MemoryDB, timezone: str, user_id: str) -> None:
    try:
        run_network_snapshot(db, timezone, user_id)
    except Exception:  # pragma: no cover - logging only
        logging.getLogger(__name__).exception("Scheduled network snapshot failed")
