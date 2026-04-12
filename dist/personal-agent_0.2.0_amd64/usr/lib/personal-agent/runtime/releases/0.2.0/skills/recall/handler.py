from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


ADVICE_PATTERNS = (
    "should i",
    "what should i do",
    "recommend",
    "recommendation",
    "fix",
    "optimize",
    "how do i",
    "how to",
    "should we",
    "best way",
    "can you",
    "please advise",
    "suggest",
)


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


def _format_delta_bytes(delta: int) -> str:
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{_bytes_to_human(abs(delta))}"


def _min_avg_max(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    total = sum(values)
    return min(values), total / len(values), max(values)


def _question_has_advice(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ADVICE_PATTERNS)


def _filter_rows_by_ts(
    rows: list[dict[str, Any]], start_ts: str | None, end_ts: str | None
) -> list[dict[str, Any]]:
    if not start_ts or not end_ts:
        return rows
    try:
        start_dt = datetime.fromisoformat(start_ts)
        end_dt = datetime.fromisoformat(end_ts)
    except ValueError:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        ts = row.get("taken_at")
        if not ts:
            filtered.append(row)
            continue
        try:
            row_dt = datetime.fromisoformat(str(ts))
        except ValueError:
            filtered.append(row)
            continue
        if start_dt <= row_dt <= end_dt:
            filtered.append(row)
    return filtered


def ask_query(context: dict[str, Any], question: str, timeframe: dict[str, Any]) -> dict[str, Any]:
    db = context.get("db") if context else None
    tz_name = (context or {}).get("timezone") or "UTC"
    if not db:
        return _blocked("Database not available.")

    if _question_has_advice(question):
        refusal = (
            "I can only provide factual recall from existing snapshots. "
            "Please ask for observations, not advice or actions."
        )
        return {"status": "refused", "text": refusal, "message": refusal}

    start_date = timeframe.get("start_date")
    end_date = timeframe.get("end_date")
    start_ts = timeframe.get("start_ts")
    end_ts = timeframe.get("end_ts")
    label = timeframe.get("label") or ""

    domains = _infer_domains(question)
    if not domains:
        domains = ["storage", "resources", "network"]

    if not start_date or not end_date:
        text = "No snapshots found yet."
        try:
            db.audit_log_create(
                user_id=str(timeframe.get("user_id") or "unknown"),
                action_type="ask_query",
                action_id="ask_query",
                status="failed",
                details={
                    "command": "/ask",
                    "question": question.strip()[:200],
                    "reason": "no_snapshots",
                    "clarification_required": bool(timeframe.get("clarification_required")),
                },
            )
        except Exception:
            return {
                "status": "failed",
                "message": "Audit logging failed. Operation aborted.",
                "text": "Audit logging failed. Operation aborted.",
            }
        return {"status": "ok", "text": text, "message": text}

    lines: list[str] = []
    lines.append(f"Question Restated: {question.strip()}")
    lines.append(f"Timeframe: {label} ({start_date} to {end_date}, {tz_name})")

    row_counts: dict[str, int] = {}

    if "storage" in domains:
        storage_lines, storage_counts = _storage_section(db, start_date, end_date, start_ts, end_ts)
        row_counts.update(storage_counts)
        lines.extend(storage_lines)

    if "resources" in domains:
        resource_lines, resource_counts = _resource_section(db, start_date, end_date, start_ts, end_ts)
        row_counts.update(resource_counts)
        lines.extend(resource_lines)

    if "network" in domains:
        network_lines, network_counts = _network_section(db, start_date, end_date, start_ts, end_ts)
        row_counts.update(network_counts)
        lines.extend(network_lines)

    if "weekly_reflection" in domains:
        reflection_lines, reflection_counts = _weekly_reflection_section(db, tz_name)
        row_counts.update(reflection_counts)
        lines.extend(reflection_lines)

    timeline = _cross_domain_timeline(db, domains, start_date, end_date, start_ts, end_ts)
    if timeline:
        lines.append("Cross-Domain Timeline:")
        for item in timeline:
            lines.append(f"- {item}")

    lines.append("Limits:")
    lines.append("- Facts are based on stored snapshots only.")
    lines.append("- Missing days or domains reduce coverage for this timeframe.")

    audit_details = {
        "command": "/ask",
        "question": question.strip()[:200],
        "timeframe": {
            "label": label,
            "start_date": start_date,
            "end_date": end_date,
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
        "domains": domains,
        "row_counts": row_counts,
        "clarification_required": bool(timeframe.get("clarification_required")),
    }
    try:
        db.audit_log_create(
            user_id=str(timeframe.get("user_id") or "unknown"),
            action_type="ask_query",
            action_id="ask_query",
            status="executed",
            details=audit_details,
        )
    except Exception:
        return {
            "status": "failed",
            "message": "Audit logging failed. Operation aborted.",
            "text": "Audit logging failed. Operation aborted.",
        }

    return {"status": "ok", "text": "\n".join(lines), "payload": {"row_counts": row_counts}}


def _infer_domains(question: str) -> list[str]:
    lowered = question.lower()
    domains: list[str] = []
    if any(word in lowered for word in ("disk", "storage", "ssd", "space", "directory", "folder")):
        domains.append("storage")
    if any(word in lowered for word in ("cpu", "memory", "ram", "load", "swap", "process")):
        domains.append("resources")
    if any(word in lowered for word in ("network", "dns", "gateway", "interface", "rx", "tx")):
        domains.append("network")
    if any(word in lowered for word in ("weekly", "reflection", "rollup")):
        domains.append("weekly_reflection")
    return domains


def _storage_section(
    db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None
) -> tuple[list[str], dict[str, int]]:
    lines = ["Storage:"]
    counts: dict[str, int] = {}
    mountpoints = ["/", "/data", "/data2"]
    any_data = False
    for mount in mountpoints:
        rows = db.list_disk_snapshots_between(mount, start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        counts[f"disk_snapshots:{mount}"] = len(rows)
        if len(rows) >= 2:
            any_data = True
            break
    if not any_data:
        lines.append("- insufficient data")
        return lines, counts

    for mount in mountpoints:
        rows = db.list_disk_snapshots_between(mount, start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            lines.append(f"- {mount}: insufficient data")
            continue
        first = rows[0]
        last = rows[-1]
        delta = int(last["used_bytes"]) - int(first["used_bytes"])
        lines.append(f"- {mount} used change: {delta} ({_format_delta_bytes(delta)})")

    root_growth = _largest_dir_growth(db, "root_top", start, end)
    home_growth = _largest_dir_growth(db, "home_top", start, end)
    if root_growth:
        lines.append(f"- / largest dir growth: {root_growth[0]} ({root_growth[1]})")
    else:
        lines.append("- / largest dir growth: insufficient data")

    if home_growth:
        lines.append(f"- home largest dir growth: {home_growth[0]} ({home_growth[1]})")
    else:
        lines.append("- home largest dir growth: insufficient data")

    stats = db.list_storage_scan_stats_between(start, end)
    counts["storage_scan_stats"] = len(stats)
    total_errors = sum(int(row["errors_skipped"]) for row in stats)
    lines.append(f"- storage scan errors_skipped (window): {total_errors}")

    return lines, counts


def _largest_dir_growth(db: Any, scope: str, start: str, end: str) -> tuple[str, str] | None:
    dates = db.list_dir_size_sample_dates(scope, start, end)
    if len(dates) < 2:
        return None
    first = db.list_dir_size_samples_for_date(scope, dates[0])
    last = db.list_dir_size_samples_for_date(scope, dates[-1])
    if not first or not last:
        return None
    first_map = {row["path"]: int(row["bytes"]) for row in first}
    last_map = {row["path"]: int(row["bytes"]) for row in last}
    best_path = ""
    best_delta = None
    for path, last_val in last_map.items():
        if path not in first_map:
            continue
        delta = last_val - first_map[path]
        if best_delta is None or delta > best_delta:
            best_delta = delta
            best_path = path
    if best_delta is None:
        return None
    return best_path, _format_delta_bytes(best_delta)


def _resource_section(
    db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None
) -> tuple[list[str], dict[str, int]]:
    lines = ["CPU/Memory:"]
    counts: dict[str, int] = {}
    rows = db.list_resource_snapshots_between(start, end)
    rows = _filter_rows_by_ts(rows, start_ts, end_ts)
    counts["resource_snapshots"] = len(rows)
    if len(rows) < 2:
        lines.append("- insufficient data")
        return lines, counts

    load_1m = [float(r["load_1m"]) for r in rows]
    load_5m = [float(r["load_5m"]) for r in rows]
    load_15m = [float(r["load_15m"]) for r in rows]
    mem_used = [float(r["mem_used"]) for r in rows]
    swap_used = [float(r["swap_used"]) for r in rows]
    swap_total = [float(r["swap_total"]) for r in rows]

    l1_min, l1_avg, l1_max = _min_avg_max(load_1m)
    l5_min, l5_avg, l5_max = _min_avg_max(load_5m)
    l15_min, l15_avg, l15_max = _min_avg_max(load_15m)
    lines.append(f"- load_1m min/avg/max: {l1_min:.2f}/{l1_avg:.2f}/{l1_max:.2f}")
    lines.append(f"- load_5m min/avg/max: {l5_min:.2f}/{l5_avg:.2f}/{l5_max:.2f}")
    lines.append(f"- load_15m min/avg/max: {l15_min:.2f}/{l15_avg:.2f}/{l15_max:.2f}")

    mem_min, mem_avg, mem_max = _min_avg_max(mem_used)
    lines.append(
        "- mem_used min/avg/max: {}/{}/{}".format(
            _bytes_to_human(int(mem_min)),
            _bytes_to_human(int(mem_avg)),
            _bytes_to_human(int(mem_max)),
        )
    )

    if any(val > 0 for val in swap_total):
        swap_min, swap_avg, swap_max = _min_avg_max(swap_used)
        lines.append(
            "- swap_used min/avg/max: {}/{}/{}".format(
                _bytes_to_human(int(swap_min)),
                _bytes_to_human(int(swap_avg)),
                _bytes_to_human(int(swap_max)),
            )
        )
    else:
        lines.append("- swap_used: no swap recorded")

    latest_day = rows[-1]["snapshot_local_date"]
    rss_samples = db.get_resource_process_samples_for_date(latest_day)
    counts["resource_process_samples"] = len(rss_samples)
    if rss_samples:
        lines.append("- top processes by RSS (latest day):")
        for row in rss_samples:
            lines.append(
                f"  pid={row['pid']} {row['name']} rss={_bytes_to_human(int(row['rss_bytes']))}"
            )

    return lines, counts


def _network_section(
    db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None
) -> tuple[list[str], dict[str, int]]:
    lines = ["Network:"]
    counts: dict[str, int] = {}
    rows = db.list_network_snapshots_between(start, end)
    rows = _filter_rows_by_ts(rows, start_ts, end_ts)
    counts["network_snapshots"] = len(rows)
    if len(rows) < 2:
        lines.append("- insufficient data")
        return lines, counts

    changes = []
    prev_gateway = None
    for row in rows:
        day = row["snapshot_local_date"]
        gateway = row["default_gateway"]
        if prev_gateway is not None and gateway != prev_gateway:
            changes.append(f"{day}: {prev_gateway} -> {gateway}")
        prev_gateway = gateway

    if changes:
        lines.append("- default gateway changes:")
        for item in changes:
            lines.append(f"  {item}")
    else:
        lines.append("- default gateway changes: none")

    first_day = rows[0]["snapshot_local_date"]
    last_day = rows[-1]["snapshot_local_date"]
    first_ifaces = db.get_network_interfaces_for_date(first_day)
    last_ifaces = db.get_network_interfaces_for_date(last_day)
    counts["network_interfaces:first"] = len(first_ifaces)
    counts["network_interfaces:last"] = len(last_ifaces)
    last_map = {row["name"]: row for row in last_ifaces}
    if first_ifaces and last_ifaces:
        lines.append("- interface rx/tx change (first -> last):")
        for row in first_ifaces:
            last_row = last_map.get(row["name"])
            if not last_row:
                continue
            rx_delta = int(last_row["rx_bytes"]) - int(row["rx_bytes"])
            tx_delta = int(last_row["tx_bytes"]) - int(row["tx_bytes"])
            lines.append(
                f"  {row['name']} rx={_format_delta_bytes(rx_delta)} tx={_format_delta_bytes(tx_delta)}"
            )
    else:
        lines.append("- interface rx/tx change: insufficient data")

    name_changes = []
    prev_names = None
    for row in rows:
        day = row["snapshot_local_date"]
        names = [ns["nameserver"] for ns in db.get_network_nameservers_for_date(day)]
        if prev_names is not None and names != prev_names:
            name_changes.append(day)
        prev_names = names

    if name_changes:
        lines.append("- nameserver changes: {}".format(", ".join(name_changes)))
    else:
        lines.append("- nameserver changes: none")

    return lines, counts


def _weekly_reflection_section(db: Any, timezone: str) -> tuple[list[str], dict[str, int]]:
    from skills.reflection import handler as reflection_handler

    lines = ["Weekly Reflection:"]
    report = reflection_handler.weekly_reflection({"db": db, "timezone": timezone})
    text = report.get("text", "")
    if not text:
        lines.append("- insufficient data")
        return lines, {"weekly_reflection": 0}
    lines.append("- uses last 7-day rollup ending latest snapshot day")
    for line in text.splitlines():
        lines.append(f"  {line}")
    return lines, {"weekly_reflection": 1}


def _cross_domain_timeline(
    db: Any, domains: list[str], start: str, end: str, start_ts: str | None, end_ts: str | None
) -> list[str]:
    events: list[tuple[str, str]] = []

    if "network" in domains:
        rows = db.list_network_snapshots_between(start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        prev_gateway = None
        for row in rows:
            day = row["snapshot_local_date"]
            gateway = row["default_gateway"]
            if prev_gateway is not None and gateway != prev_gateway:
                events.append((day, f"Network gateway changed: {prev_gateway} -> {gateway}"))
            prev_gateway = gateway

    if len({evt[0] for evt in events}) < 2:
        return []

    events.sort(key=lambda item: item[0])
    return [f"{day}: {text}" for day, text in events]
