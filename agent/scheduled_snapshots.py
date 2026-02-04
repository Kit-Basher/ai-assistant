from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from memory.db import MemoryDB
from skills.disk_report import safe_disk
from skills.storage_governor import collector


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
    details = {
        "event_type": "storage_snapshot",
        "mode": "observe",
        "timezone": timezone,
        "source": "scheduled",
    }
    audit_id = db.audit_log_create(
        user_id=user_id,
        action_type="storage_snapshot",
        action_id="storage.snapshot",
        status="started",
        details=details,
    )
    try:
        with db.transaction():
            payload = collector.collect_and_persist_snapshot(
                db,
                timezone=timezone,
                home_path=home_path,
            )
            try:
                db.audit_log_update_status(
                    audit_id,
                    "executed",
                    details={
                        "event_type": "storage_snapshot",
                        "taken_at": payload.get("taken_at"),
                        "source": "scheduled",
                    },
                )
            except Exception as exc:
                raise RuntimeError("audit_update_failed") from exc
    except Exception as exc:
        if str(exc) == "audit_update_failed":
            raise RuntimeError("Audit logging failed. Operation aborted.") from exc
        try:
            db.audit_log_update_status(audit_id, "failed", str(exc))
        except Exception as audit_exc:
            raise RuntimeError("Audit logging failed. Operation aborted.") from audit_exc
        raise
    return payload


def safe_run_storage_snapshot(db: MemoryDB, timezone: str, home_path: str, user_id: str) -> None:
    try:
        run_storage_snapshot(db, timezone, home_path, user_id)
    except Exception:  # pragma: no cover - logging only
        logging.getLogger(__name__).exception("Scheduled storage snapshot failed")
