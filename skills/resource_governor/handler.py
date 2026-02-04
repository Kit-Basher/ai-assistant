from __future__ import annotations

from typing import Any

from skills.resource_governor import collector

AUDIT_HARD_FAIL_MSG = "Audit logging failed. Operation aborted."


def _blocked(message: str) -> dict[str, Any]:
    return {"status": "blocked", "message": message, "text": message}


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


def _format_delta(delta: int | float | None, base: int | float | None = None, suffix: str = "") -> str:
    if delta is None:
        return "n/a"
    sign = "+" if delta >= 0 else "-"
    magnitude = abs(delta)
    if isinstance(delta, float):
        value = f"{sign}{magnitude:.2f}{suffix}"
    else:
        value = f"{sign}{_bytes_to_human(int(magnitude))}"
    if base:
        pct = (delta / base) * 100.0
        return f"{value} ({pct:+.1f}%)"
    return value


def _log_audit(db: Any, event_type: str, payload: dict[str, Any]) -> None:
    try:
        db.log_activity(event_type, payload)
    except Exception as exc:
        raise RuntimeError("audit_log_failed") from exc


def resource_snapshot(
    context: dict[str, Any],
    top_n: int | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    db = context.get("db") if context else None
    timezone = (context or {}).get("timezone") or "UTC"
    actor_id = user_id or (context or {}).get("user_id") or "system"

    if not db:
        return _blocked("Database not available.")

    top_n_value = int(top_n) if top_n is not None else collector.TOP_N_DEFAULT
    try:
        with db.transaction():
            payload = collector.collect_and_persist_snapshot(
                db,
                timezone=timezone,
                top_n=top_n_value,
            )
            _log_audit(
                db,
                "resource_snapshot",
                {
                    "event_type": "resource_snapshot",
                    "mode": "observe",
                    "status": "executed",
                    "actor_id": actor_id,
                    "timezone": timezone,
                    "top_n": top_n_value,
                    "taken_at": payload.get("taken_at"),
                },
            )
    except Exception as exc:
        if str(exc) == "audit_log_failed":
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        return {"status": "failed", "message": "Snapshot failed.", "text": "Snapshot failed."}

    text = "Snapshot stored: {} ({})".format(payload.get("taken_at", ""), timezone)
    return {"status": "ok", "text": text, "payload": payload}


def resource_report(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    db = context.get("db") if context else None
    timezone = (context or {}).get("timezone") or "UTC"
    actor_id = user_id or (context or {}).get("user_id") or "system"

    if not db:
        return _blocked("Database not available.")

    try:
        with db.transaction():
            report = _build_resource_report(db, timezone)
            _log_audit(
                db,
                "resource_report",
                {
                    "event_type": "resource_report",
                    "mode": "observe",
                    "status": "executed",
                    "actor_id": actor_id,
                    "timezone": timezone,
                    "taken_at": report.get("taken_at"),
                },
            )
    except Exception as exc:
        if str(exc) == "audit_log_failed":
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        return {"status": "failed", "message": "Report failed.", "text": "Report failed."}

    return report


def _build_resource_report(db: Any, timezone: str) -> dict[str, Any]:
    latest = db.get_latest_resource_snapshot()
    if not latest:
        text = "No resource snapshots found yet."
        return {"status": "ok", "text": text, "payload": {"message": text}}

    previous = db.get_previous_resource_snapshot(latest.get("taken_at"))
    lines: list[str] = []
    taken_at = latest.get("taken_at", "")
    if taken_at:
        lines.append(f"Snapshot taken: {taken_at} ({timezone})")

    def delta_value(field: str) -> int | None:
        if not previous:
            return None
        return int(latest[field]) - int(previous[field])

    load_delta = None
    if previous:
        load_delta = (
            float(latest["load_1m"]) - float(previous["load_1m"]),
            float(latest["load_5m"]) - float(previous["load_5m"]),
            float(latest["load_15m"]) - float(previous["load_15m"]),
        )

    lines.append(
        "Load avg: 1m={:.2f}, 5m={:.2f}, 15m={:.2f}".format(
            float(latest["load_1m"]),
            float(latest["load_5m"]),
            float(latest["load_15m"]),
        )
    )
    if load_delta:
        lines.append(
            "Load delta: 1m={}, 5m={}, 15m={}".format(
                _format_delta(load_delta[0], None, ""),
                _format_delta(load_delta[1], None, ""),
                _format_delta(load_delta[2], None, ""),
            )
        )

    mem_total = int(latest["mem_total"])
    mem_used = int(latest["mem_used"])
    mem_free = int(latest["mem_free"])
    lines.append(
        "Memory: used {} / total {} (free {})".format(
            _bytes_to_human(mem_used),
            _bytes_to_human(mem_total),
            _bytes_to_human(mem_free),
        )
    )
    mem_used_delta = delta_value("mem_used")
    if mem_used_delta is not None:
        lines.append("Memory delta: {}".format(_format_delta(mem_used_delta, mem_used)))

    swap_total = int(latest["swap_total"])
    swap_used = int(latest["swap_used"])
    if swap_total > 0:
        lines.append(
            "Swap: used {} / total {}".format(
                _bytes_to_human(swap_used),
                _bytes_to_human(swap_total),
            )
        )
        swap_used_delta = delta_value("swap_used")
        if swap_used_delta is not None:
            lines.append("Swap delta: {}".format(_format_delta(swap_used_delta, swap_used)))

    cpu_samples = db.get_resource_process_samples(latest.get("taken_at"), "cpu")
    rss_samples = db.get_resource_process_samples(latest.get("taken_at"), "rss")
    if cpu_samples:
        lines.append("Top processes by CPU ticks (since boot):")
        for row in cpu_samples:
            lines.append(f"- pid={row['pid']} {row['name']}: {row['cpu_ticks']}")
    if rss_samples:
        lines.append("Top processes by RSS:")
        for row in rss_samples:
            lines.append(
                f"- pid={row['pid']} {row['name']}: {_bytes_to_human(int(row['rss_bytes']))}"
            )

    stats = db.get_latest_resource_scan_stats("processes")
    if stats and stats.get("taken_at") == latest.get("taken_at"):
        lines.append(
            "Process scan stats: scanned={}, errors_skipped={}".format(
                stats.get("procs_scanned"), stats.get("errors_skipped")
            )
        )

    payload = {
        "taken_at": latest.get("taken_at"),
        "snapshot_local_date": latest.get("snapshot_local_date"),
        "loads": {
            "1m": float(latest["load_1m"]),
            "5m": float(latest["load_5m"]),
            "15m": float(latest["load_15m"]),
        },
        "memory": {
            "total": mem_total,
            "used": mem_used,
            "free": mem_free,
        },
        "swap": {"total": swap_total, "used": swap_used},
        "cpu_samples": cpu_samples,
        "rss_samples": rss_samples,
    }
    return {"status": "ok", "text": "\n".join(lines), "payload": payload}
