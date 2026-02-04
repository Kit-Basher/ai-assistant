from __future__ import annotations

from typing import Any

from skills.storage_governor import collector

AUDIT_HARD_FAIL_MSG = "Audit logging failed. Operation aborted."


def _bytes_to_human(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0B"
    units = ["B", "K", "M", "G", "T", "P", "E"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}B"
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}{unit}"
        value /= 1024
    return f"{int(value)}B"


def _format_delta(delta_bytes: int | None, prev_bytes: int | None = None) -> str:
    if delta_bytes is None:
        return "n/a"
    sign = "+" if delta_bytes >= 0 else "-"
    magnitude = _bytes_to_human(abs(delta_bytes))
    if prev_bytes and prev_bytes > 0:
        pct = (delta_bytes / prev_bytes) * 100.0
        pct_str = f"{pct:+.1f}%"
        return f"{sign}{magnitude} ({pct_str})"
    return f"{sign}{magnitude}"


def _blocked(message: str) -> dict[str, Any]:
    return {"status": "blocked", "message": message, "text": message}


def storage_snapshot(
    context: dict[str, Any],
    top_n: int | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    db = context.get("db") if context else None
    timezone = (context or {}).get("timezone") or "UTC"
    actor_id = user_id or (context or {}).get("user_id") or "system"

    if not db:
        return _blocked("Database not available.")

    details = {
        "event_type": "storage_snapshot",
        "mode": "observe",
        "top_n": int(top_n) if top_n is not None else collector.TOP_N_DEFAULT,
        "timezone": timezone,
    }
    try:
        audit_id = db.audit_log_create(
            user_id=actor_id,
            action_type="storage_snapshot",
            action_id="storage.snapshot",
            status="started",
            details=details,
        )
    except Exception:
        return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}

    try:
        with db.transaction():
            payload = collector.collect_and_persist_snapshot(
                db,
                timezone=timezone,
                top_n=details["top_n"],
            )
            try:
                db.audit_log_update_status(
                    audit_id,
                    "executed",
                    details={
                        "event_type": "storage_snapshot",
                        "taken_at": payload.get("taken_at"),
                        "mounts": payload.get("mounts", []),
                        "root_top_count": len(payload.get("root_top", [])),
                        "home_top_count": len(payload.get("home_top", [])),
                    },
                )
            except Exception as exc:
                raise RuntimeError("audit_update_failed") from exc
    except Exception as exc:
        if str(exc) == "audit_update_failed":
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        try:
            db.audit_log_update_status(audit_id, "failed", str(exc))
        except Exception:
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        return {"status": "failed", "message": "Snapshot failed.", "text": "Snapshot failed."}

    text = "Snapshot stored: {} ({})".format(payload.get("taken_at", ""), timezone)
    return {"status": "ok", "text": text, "payload": payload}


def storage_report(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    db = context.get("db") if context else None
    timezone = (context or {}).get("timezone") or "UTC"
    actor_id = user_id or (context or {}).get("user_id") or "system"

    if not db:
        return _blocked("Database not available.")

    details = {"event_type": "storage_report", "mode": "observe", "timezone": timezone}
    try:
        audit_id = db.audit_log_create(
            user_id=actor_id,
            action_type="storage_report",
            action_id="storage.report",
            status="started",
            details=details,
        )
    except Exception:
        return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}

    try:
        with db.transaction():
            report = _build_storage_report(db, timezone)
            try:
                db.audit_log_update_status(
                    audit_id,
                    "executed",
                    details={"event_type": "storage_report", "taken_at": report.get("taken_at")},
                )
            except Exception as exc:
                raise RuntimeError("audit_update_failed") from exc
    except Exception as exc:
        if str(exc) == "audit_update_failed":
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        try:
            db.audit_log_update_status(audit_id, "failed", str(exc))
        except Exception:
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        return {"status": "failed", "message": "Report failed.", "text": "Report failed."}
    return report


def _build_storage_report(db: Any, timezone: str) -> dict[str, Any]:
    mountpoints = collector.DEFAULT_MOUNTPOINTS
    mount_reports: list[dict[str, Any]] = []
    latest_times: list[str] = []

    for mountpoint in mountpoints:
        latest = db.get_latest_disk_snapshot(mountpoint)
        if not latest:
            continue
        latest_times.append(latest.get("taken_at", ""))
        prev = db.get_previous_disk_snapshot(mountpoint, latest.get("taken_at"))
        delta_used = None
        if prev:
            delta_used = int(latest["used_bytes"]) - int(prev["used_bytes"])
        used_pct = 0.0
        if int(latest["total_bytes"]) > 0:
            used_pct = (int(latest["used_bytes"]) / int(latest["total_bytes"])) * 100.0
        mount_reports.append(
            {
                "mountpoint": mountpoint,
                "taken_at": latest.get("taken_at"),
                "total_bytes": int(latest["total_bytes"]),
                "used_bytes": int(latest["used_bytes"]),
                "free_bytes": int(latest["free_bytes"]),
                "used_pct": used_pct,
                "prev_taken_at": prev.get("taken_at") if prev else None,
                "delta_used": delta_used,
                "prev_used_bytes": int(prev["used_bytes"]) if prev else None,
            }
        )

    root_latest = db.get_latest_dir_size_samples("root_top")
    home_latest = db.get_latest_dir_size_samples("home_top")
    root_stats = None
    home_stats = None
    root_prev = None
    home_prev = None
    if root_latest:
        root_prev = db.get_previous_dir_size_samples("root_top", root_latest["taken_at"])
        root_stats = db.get_storage_scan_stats_for_taken_at("root_top", root_latest["taken_at"])
    if home_latest:
        home_prev = db.get_previous_dir_size_samples("home_top", home_latest["taken_at"])
        home_stats = db.get_storage_scan_stats_for_taken_at("home_top", home_latest["taken_at"])

    if not mount_reports and not root_latest and not home_latest:
        text = "No storage snapshots found yet."
        return {"status": "ok", "text": text, "payload": {"message": text}}

    taken_at = max([t for t in latest_times if t] or [""], default="")
    lines: list[str] = []
    if taken_at:
        lines.append(f"Snapshot taken: {taken_at} ({timezone})")

    if mount_reports:
        lines.append("Mount totals:")
        for mount in mount_reports:
            used = _bytes_to_human(mount["used_bytes"])
            total = _bytes_to_human(mount["total_bytes"])
            free = _bytes_to_human(mount["free_bytes"])
            used_pct = mount["used_pct"]
            delta = _format_delta(mount["delta_used"], mount.get("prev_used_bytes"))
            prev_label = mount.get("prev_taken_at") or "previous snapshot"
            lines.append(
                f"- {mount['mountpoint']} used: {used} / {total} ({used_pct:.1f}%), free {free}; delta used {delta} since {prev_label}"
            )

    if root_latest:
        lines.append("Top / dirs:")
        if root_stats:
            lines.append(
                f"- scan stats: dirs_scanned={root_stats.get('dirs_scanned')}, errors_skipped={root_stats.get('errors_skipped')}"
            )
        prev_map = {path: bytes_val for path, bytes_val in (root_prev or {}).get("samples", [])}
        for path, bytes_val in root_latest.get("samples", []):
            delta = None
            if path in prev_map:
                delta = int(bytes_val) - int(prev_map[path])
            lines.append(
                f"- {path}: {_bytes_to_human(int(bytes_val))} (delta {_format_delta(delta, prev_map.get(path))})"
            )

    if home_latest:
        lines.append("Top home dirs:")
        if home_stats:
            lines.append(
                f"- scan stats: dirs_scanned={home_stats.get('dirs_scanned')}, errors_skipped={home_stats.get('errors_skipped')}"
            )
        prev_map = {path: bytes_val for path, bytes_val in (home_prev or {}).get("samples", [])}
        for path, bytes_val in home_latest.get("samples", []):
            delta = None
            if path in prev_map:
                delta = int(bytes_val) - int(prev_map[path])
            lines.append(
                f"- {path}: {_bytes_to_human(int(bytes_val))} (delta {_format_delta(delta, prev_map.get(path))})"
            )

    payload = {
        "taken_at": taken_at,
        "mounts": mount_reports,
        "root_top": root_latest or {},
        "home_top": home_latest or {},
        "root_stats": root_stats or {},
        "home_stats": home_stats or {},
        "timezone": timezone,
    }
    return {"status": "ok", "text": "\n".join(lines), "payload": payload}
