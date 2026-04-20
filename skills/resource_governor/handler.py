from __future__ import annotations

from typing import Any

from agent.cards import normalize_card
from agent.resource_insights import summarize_resource_report
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


def _loadavg_from_observation(observation: dict[str, Any]) -> tuple[float, float, float]:
    loadavg = observation.get("loadavg")
    if isinstance(loadavg, tuple) and len(loadavg) >= 3:
        return float(loadavg[0]), float(loadavg[1]), float(loadavg[2])
    return 0.0, 0.0, 0.0


def _proc_rows(rows: list[Any], *, is_rss: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "pid"):
            pid = int(getattr(row, "pid"))
            name = str(getattr(row, "name", ""))
            cpu_ticks = int(getattr(row, "cpu_ticks", 0))
            rss_bytes = int(getattr(row, "rss_bytes", 0))
        elif isinstance(row, dict):
            pid = int(row.get("pid") or 0)
            name = str(row.get("name") or "")
            cpu_ticks = int(row.get("cpu_ticks") or 0)
            rss_bytes = int(row.get("rss_bytes") or 0)
        else:
            continue
        if pid <= 0 or not name:
            continue
        out.append(
            {
                "pid": pid,
                "name": name,
                "cpu_ticks": cpu_ticks,
                "rss_bytes": rss_bytes,
            }
        )
    if is_rss:
        out.sort(key=lambda item: item["rss_bytes"], reverse=True)
    else:
        out.sort(key=lambda item: item["cpu_ticks"], reverse=True)
    return out


def _memory_note(memory: dict[str, Any]) -> str | None:
    parts: list[str] = []
    buffers = int(memory.get("buffers") or 0)
    cached = int(memory.get("cached") or 0)
    shared = int(memory.get("shared") or 0)
    if buffers > 0:
        parts.append(f"buffers {_bytes_to_human(buffers)}")
    if cached > 0:
        parts.append(f"cached {_bytes_to_human(cached)}")
    if shared > 0:
        parts.append(f"shared {_bytes_to_human(shared)}")
    if not parts:
        return None
    return "MemAvailable includes reclaimable cache/buffers/shared memory: " + ", ".join(parts)


def _live_report(live: dict[str, Any], timezone: str) -> dict[str, Any]:
    mem = live.get("mem") if isinstance(live.get("mem"), dict) else {}
    swap = live.get("swap") if isinstance(live.get("swap"), dict) else {}
    mem_total = int(mem.get("total") or 0)
    mem_used = int(mem.get("used") or 0)
    mem_free = int(mem.get("free") or 0)
    mem_available = int(mem.get("available") or 0)
    mem_used_pct = float(mem.get("used_pct") or 0.0)
    taken_at = str(live.get("taken_at") or "")
    hostname = str(live.get("hostname") or "")
    load_1m, load_5m, load_15m = _loadavg_from_observation(live)
    cpu_samples = _proc_rows(list(live.get("top_cpu") or []), is_rss=False)
    rss_samples = _proc_rows(list(live.get("top_rss") or []), is_rss=True)
    proc_stats = live.get("proc_stats") if isinstance(live.get("proc_stats"), dict) else {}
    note = _memory_note(mem)

    if mem_total <= 0 or mem_used < 0 or mem_available < 0:
        text = "Live memory probe returned no usable memory data."
        return {
            "status": "ok",
            "text": text,
            "payload": {
                "source": "live_invalid",
                "message": text,
                "taken_at": taken_at,
                "hostname": hostname,
                "loads": {"1m": load_1m, "5m": load_5m, "15m": load_15m},
                "memory": mem,
                "swap": swap,
                "cpu_samples": cpu_samples,
                "rss_samples": rss_samples,
                "proc_stats": proc_stats,
            },
            "cards_payload": {
                "cards": [
                    {
                        "title": "Memory",
                        "lines": [text],
                        "severity": "warn",
                    }
                ],
                "raw_available": False,
                "summary": text,
                "confidence": 0.0,
                "next_questions": ["Retry the live memory probe", "Show hardware inventory"],
            },
        }

    lines: list[str] = []
    if taken_at:
        lines.append(f"Live memory probe taken: {taken_at} ({timezone})")
    lines.append(
        "Load avg: 1m={:.2f}, 5m={:.2f}, 15m={:.2f}".format(load_1m, load_5m, load_15m)
    )
    lines.append(
        "Memory used: {} / total {} ({:.1f}%)".format(
            _bytes_to_human(mem_used),
            _bytes_to_human(mem_total),
            mem_used_pct,
        )
    )
    lines.append(
        "Memory available: {} (free {})".format(
            _bytes_to_human(mem_available),
            _bytes_to_human(mem_free),
        )
    )
    if note:
        lines.append(note)

    swap_total = int(swap.get("total") or 0)
    swap_used = int(swap.get("used") or 0)
    if swap_total > 0:
        lines.append(
            "Swap: used {} / total {}".format(
                _bytes_to_human(swap_used),
                _bytes_to_human(swap_total),
            )
        )

    if cpu_samples:
        lines.append("Top CPU processes:")
        for row in cpu_samples[:5]:
            lines.append(f"- pid={row['pid']} {row['name']}: {row['cpu_ticks']} ticks")
    if rss_samples:
        lines.append("Top memory processes:")
        for row in rss_samples[:5]:
            lines.append(f"- pid={row['pid']} {row['name']}: {_bytes_to_human(int(row['rss_bytes']))}")
    if proc_stats:
        lines.append(
            "Process scan stats: scanned={}, errors_skipped={}".format(
                proc_stats.get("procs_scanned"), proc_stats.get("errors_skipped")
            )
        )

    summary = f"Live memory probe: {_bytes_to_human(mem_used)} used / {_bytes_to_human(mem_total)} total ({mem_used_pct:.1f}%)."
    if note:
        summary += f" {note}"

    cards = [
        normalize_card(
            {
                "title": "Memory",
                "lines": [
                    f"Used: {_bytes_to_human(mem_used)} / {_bytes_to_human(mem_total)} ({mem_used_pct:.1f}%)",
                    f"Available: {_bytes_to_human(mem_available)}",
                    note or "Cache/buffers/shared memory note unavailable.",
                ],
                "severity": "ok",
            },
            0,
        )
    ]
    if rss_samples:
        cards.append(
            normalize_card(
                {
                    "title": "Top memory processes",
                    "lines": [f"pid={row['pid']} {row['name']}: {_bytes_to_human(int(row['rss_bytes']))}" for row in rss_samples[:5]],
                    "severity": "ok",
                },
                len(cards),
            )
        )
    if cpu_samples:
        cards.append(
            normalize_card(
                {
                    "title": "Top CPU processes",
                    "lines": [f"pid={row['pid']} {row['name']}: {row['cpu_ticks']} ticks" for row in cpu_samples[:5]],
                    "severity": "ok",
                },
                len(cards),
            )
        )

    payload = {
        "source": "live",
        "taken_at": taken_at,
        "snapshot_local_date": taken_at.split("T")[0] if taken_at else None,
        "hostname": hostname,
        "cpu_count": int(live.get("cpu_count") or 0),
        "loads": {"1m": load_1m, "5m": load_5m, "15m": load_15m},
        "memory": {
            "total": mem_total,
            "used": mem_used,
            "free": mem_free,
            "available": mem_available,
            "buffers": int(mem.get("buffers") or 0),
            "cached": int(mem.get("cached") or 0),
            "shared": int(mem.get("shared") or 0),
            "used_pct": mem_used_pct,
        },
        "swap": {"total": swap_total, "used": swap_used},
        "cpu_samples": cpu_samples,
        "rss_samples": rss_samples,
        "proc_stats": proc_stats,
        "memory_note": note,
    }
    analysis = summarize_resource_report(payload)
    payload["cause_analysis"] = analysis
    return {
        "status": "ok",
        "text": "\n".join(lines),
        "payload": payload,
        "cards_payload": {
            "cards": cards,
            "raw_available": True,
            "summary": analysis.get("summary") or summary,
            "confidence": 1.0,
            "next_questions": analysis.get("followups") or ["Show only the biggest memory users", "Show only CPU deltas"],
        },
    }


def _snapshot_report(db: Any, timezone: str) -> dict[str, Any]:
    latest = db.get_latest_resource_snapshot()
    if not latest:
        text = "No resource snapshots found yet."
        return {
            "status": "ok",
            "text": text,
            "payload": {"message": text, "source": "snapshot_empty"},
            "cards_payload": {
                "cards": [
                    {
                        "title": "Memory",
                        "lines": [text],
                        "severity": "warn",
                    }
                ],
                "raw_available": False,
                "summary": text,
                "confidence": 0.0,
                "next_questions": ["Retry the live memory probe", "Show hardware inventory"],
            },
        }

    mem_total = int(latest.get("mem_total") or 0)
    mem_used = int(latest.get("mem_used") or 0)
    mem_free = int(latest.get("mem_free") or 0)
    if mem_total <= 0 or mem_used < 0 or mem_free < 0:
        text = "The latest stored resource snapshot is incomplete or invalid."
        return {
            "status": "ok",
            "text": text,
            "payload": {
                "message": text,
                "source": "snapshot_invalid",
                "taken_at": latest.get("taken_at"),
            },
            "cards_payload": {
                "cards": [
                    {
                        "title": "Memory",
                        "lines": [text],
                        "severity": "warn",
                    }
                ],
                "raw_available": False,
                "summary": text,
                "confidence": 0.0,
                "next_questions": ["Retry the live memory probe", "Show hardware inventory"],
            },
        }

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

    lines.append(
        "Memory: used {} / total {} (free {})".format(
            _bytes_to_human(mem_used),
            _bytes_to_human(mem_total),
            _bytes_to_human(mem_free),
        )
    )
    lines.append(
        "Memory percent used: {:.1f}%".format((mem_used / float(mem_total)) * 100.0 if mem_total > 0 else 0.0)
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

    latest_taken_at = latest.get("taken_at")
    previous_taken_at = previous.get("taken_at") if previous else None
    cpu_samples = db.get_resource_process_samples(latest_taken_at, "cpu")
    rss_samples = db.get_resource_process_samples(latest_taken_at, "rss")
    prev_cpu_samples = db.get_resource_process_samples(previous_taken_at, "cpu") if previous_taken_at else []
    prev_rss_samples = db.get_resource_process_samples(previous_taken_at, "rss") if previous_taken_at else []
    prev_cpu_map = {int(row["pid"]): int(row["cpu_ticks"]) for row in prev_cpu_samples if "pid" in row}
    prev_rss_map = {int(row["pid"]): int(row["rss_bytes"]) for row in prev_rss_samples if "pid" in row}
    cpu_delta_rows: list[dict[str, Any]] = []
    for row in cpu_samples:
        pid = int(row["pid"])
        delta = int(row["cpu_ticks"]) - int(prev_cpu_map.get(pid, 0))
        cpu_delta_rows.append({"pid": pid, "name": row["name"], "cpu_ticks_delta": delta})
    cpu_delta_rows.sort(key=lambda item: item["cpu_ticks_delta"], reverse=True)
    rss_delta_rows: list[dict[str, Any]] = []
    for row in rss_samples:
        pid = int(row["pid"])
        delta = int(row["rss_bytes"]) - int(prev_rss_map.get(pid, 0))
        rss_delta_rows.append({"pid": pid, "name": row["name"], "rss_bytes_delta": delta})
    rss_delta_rows.sort(key=lambda item: item["rss_bytes_delta"], reverse=True)
    if cpu_samples:
        lines.append("Top processes by CPU ticks (since boot):")
        for row in cpu_samples:
            lines.append(f"- pid={row['pid']} {row['name']}: {row['cpu_ticks']}")
    if rss_samples:
        lines.append("Top processes by RSS:")
        for row in rss_samples:
            lines.append(f"- pid={row['pid']} {row['name']}: {_bytes_to_human(int(row['rss_bytes']))}")

    stats = db.get_latest_resource_scan_stats("processes")
    if stats and stats.get("taken_at") == latest.get("taken_at"):
        lines.append(
            "Process scan stats: scanned={}, errors_skipped={}".format(
                stats.get("procs_scanned"), stats.get("errors_skipped")
            )
        )

    payload = {
        "source": "snapshot",
        "taken_at": latest_taken_at,
        "previous_taken_at": previous_taken_at,
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
        "cpu_deltas": cpu_delta_rows[:5],
        "rss_deltas": rss_delta_rows[:5],
    }
    analysis = summarize_resource_report(payload)
    payload["cause_analysis"] = analysis
    cards = [
        normalize_card(
            {
                "title": "CPU load",
                "lines": [
                    "1m={:.2f}, 5m={:.2f}, 15m={:.2f}".format(
                        float(latest["load_1m"]),
                        float(latest["load_5m"]),
                        float(latest["load_15m"]),
                    )
                ],
                "severity": "warn" if float(latest["load_1m"]) >= 2.0 else "ok",
            },
            0,
        ),
        normalize_card(
            {
                "title": "Memory",
                "lines": [
                    "used {} / total {} (free {})".format(
                        _bytes_to_human(mem_used),
                        _bytes_to_human(mem_total),
                        _bytes_to_human(mem_free),
                    )
                ],
                "severity": "warn" if (mem_total and (mem_used / mem_total) >= 0.85) else "ok",
            },
            1,
        ),
    ]
    if cpu_samples:
        cards.append(
            normalize_card(
                {
                    "title": "Top CPU processes",
                    "lines": [f"pid={row['pid']} {row['name']}: {row['cpu_ticks']}" for row in cpu_samples[:5]],
                    "severity": "ok",
                },
                len(cards),
            )
        )
    if rss_samples:
        cards.append(
            normalize_card(
                {
                    "title": "Top memory processes",
                    "lines": [
                        f"pid={row['pid']} {row['name']}: {_bytes_to_human(int(row['rss_bytes']))}"
                        for row in rss_samples[:5]
                    ],
                    "severity": "ok",
                },
                len(cards),
            )
        )
    if previous_taken_at:
        cards.append(
            normalize_card(
                {
                    "title": "CPU delta vs previous snapshot",
                    "lines": [
                        f"pid={row['pid']} {row['name']}: {row['cpu_ticks_delta']:+d} ticks"
                        for row in cpu_delta_rows[:5]
                        if int(row["cpu_ticks_delta"]) != 0
                    ]
                    or ["No material CPU delta detected."],
                    "severity": "ok",
                },
                len(cards),
            )
        )
        cards.append(
            normalize_card(
                {
                    "title": "Memory delta vs previous snapshot",
                    "lines": [
                        f"pid={row['pid']} {row['name']}: {_format_delta(int(row['rss_bytes_delta']))}"
                        for row in rss_delta_rows[:5]
                        if int(row["rss_bytes_delta"]) != 0
                    ]
                    or ["No material memory delta detected."],
                    "severity": "ok",
                },
                len(cards),
            )
        )
    else:
        cards.append(
            normalize_card(
                {
                    "title": "Comparison",
                    "lines": ["No previous snapshot to compare; here's current status."],
                    "severity": "ok",
                },
                len(cards),
            )
        )
    return {
        "status": "ok",
        "text": "\n".join(lines),
        "payload": payload,
        "cards_payload": {
            "cards": cards,
            "raw_available": True,
            "summary": analysis.get("summary") or "Stored resource snapshot.",
            "confidence": 0.5,
            "next_questions": analysis.get("followups") or ["Show only CPU deltas", "Show only memory deltas"],
        },
    }


def resource_report(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    db = context.get("db") if context else None
    timezone = (context or {}).get("timezone") or "UTC"
    actor_id = user_id or (context or {}).get("user_id") or "system"
    read_only_mode = bool((context or {}).get("read_only_mode"))

    if not db:
        return _blocked("Database not available.")

    live_observation = None
    live_error: Exception | None = None
    try:
        live_observation = collector.collect_live_snapshot(timezone=timezone, top_n=collector.TOP_N_DEFAULT)
    except Exception as exc:
        live_error = exc
        live_observation = None

    live_mem_total = int(((live_observation or {}).get("mem") or {}).get("total") or 0) if isinstance(live_observation, dict) else 0
    if isinstance(live_observation, dict) and live_mem_total > 0:
        report = _live_report(live_observation, timezone)
        if read_only_mode:
            return report
        try:
            with db.transaction():
                _log_audit(
                    db,
                    "resource_report",
                    {
                        "event_type": "resource_report",
                        "mode": "observe",
                        "status": "executed",
                        "actor_id": actor_id,
                        "timezone": timezone,
                        "taken_at": report.get("payload", {}).get("taken_at"),
                        "source": "live",
                    },
                )
        except Exception as exc:
            if str(exc) == "audit_log_failed":
                return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
            return {"status": "failed", "message": "Report failed.", "text": "Report failed."}
        return report

    report = _snapshot_report(db, timezone)
    payload = report.get("payload") if isinstance(report.get("payload"), dict) else {}
    source = str(payload.get("source") or "").strip().lower()

    live_unusable = live_error is not None or live_mem_total <= 0
    if live_unusable and source == "snapshot_empty":
        text = "I couldn't get a live memory probe, and there are no stored resource snapshots yet."
        report["text"] = text
        report["payload"] = {
            **payload,
            "source": "unavailable",
            "message": text,
        }
        report["cards_payload"] = {
            "cards": [
                {
                    "title": "Memory",
                    "lines": [text],
                    "severity": "warn",
                }
            ],
            "raw_available": False,
            "summary": text,
            "confidence": 0.0,
            "next_questions": ["Retry the live memory probe", "Show hardware inventory"],
        }

    if read_only_mode:
        return report

    try:
        with db.transaction():
            _log_audit(
                db,
                "resource_report",
                {
                    "event_type": "resource_report",
                    "mode": "observe",
                    "status": "executed",
                    "actor_id": actor_id,
                    "timezone": timezone,
                    "taken_at": report.get("payload", {}).get("taken_at"),
                    "source": report.get("payload", {}).get("source"),
                },
            )
    except Exception as exc:
        if str(exc) == "audit_log_failed":
            return {"status": "failed", "message": AUDIT_HARD_FAIL_MSG, "text": AUDIT_HARD_FAIL_MSG}
        return {"status": "failed", "message": "Report failed.", "text": "Report failed."}

    return report
