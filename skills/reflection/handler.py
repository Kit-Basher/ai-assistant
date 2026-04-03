from __future__ import annotations

from datetime import date, timedelta
from typing import Any


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


def weekly_reflection(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    db = context.get("db") if context else None
    timezone = (context or {}).get("timezone") or "UTC"
    if not db:
        return _blocked("Database not available.")

    end_date_str = db.get_latest_snapshot_local_date_any()
    if not end_date_str:
        text = "No snapshots found yet."
        return {"status": "ok", "text": text, "payload": {"message": text}}

    end_date = date.fromisoformat(end_date_str)
    start_date = end_date - timedelta(days=6)
    start_date_str = start_date.isoformat()

    lines: list[str] = []
    lines.append(
        "Weekly reflection ({} to {}, {})".format(
            start_date_str, end_date_str, timezone
        )
    )

    _append_storage(lines, db, start_date_str, end_date_str)
    _append_resources(lines, db, start_date_str, end_date_str)
    _append_network(lines, db, start_date_str, end_date_str)

    return {"status": "ok", "text": "\n".join(lines), "payload": {"start": start_date_str, "end": end_date_str}}


def _append_storage(lines: list[str], db: Any, start: str, end: str) -> None:
    mountpoints = ["/", "/data", "/data2"]
    has_data = False
    for mount in mountpoints:
        rows = db.list_disk_snapshots_between(mount, start, end)
        if len(rows) >= 2:
            has_data = True
            break
    lines.append("Storage:")
    if not has_data:
        lines.append("- insufficient data")
        return

    for mount in mountpoints:
        rows = db.list_disk_snapshots_between(mount, start, end)
        if len(rows) < 2:
            lines.append(f"- {mount}: insufficient data")
            continue
        first = rows[0]
        last = rows[-1]
        delta = int(last["used_bytes"]) - int(first["used_bytes"])
        lines.append(
            "- {} used change: {} ({})".format(
                mount, delta, _format_delta_bytes(delta)
            )
        )

    root_growth = _largest_dir_growth(db, "root_top", start, end)
    home_growth = _largest_dir_growth(db, "home_top", start, end)
    if root_growth:
        lines.append(
            "- / largest dir growth: {} ({})".format(
                root_growth[0], root_growth[2]
            )
        )
    else:
        lines.append("- / largest dir growth: insufficient data")

    if home_growth:
        lines.append(
            "- home largest dir growth: {} ({})".format(
                home_growth[0], home_growth[2]
            )
        )
    else:
        lines.append("- home largest dir growth: insufficient data")

    stats = db.list_storage_scan_stats_between(start, end)
    total_errors = sum(int(row["errors_skipped"]) for row in stats)
    lines.append(f"- storage scan errors_skipped (7d): {total_errors}")


def _largest_dir_growth(db: Any, scope: str, start: str, end: str) -> tuple[str, int, str] | None:
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
    return best_path, best_delta, _format_delta_bytes(best_delta)


def _append_resources(lines: list[str], db: Any, start: str, end: str) -> None:
    rows = db.list_resource_snapshots_between(start, end)
    lines.append("Resources:")
    if len(rows) < 2:
        lines.append("- insufficient data")
        return

    load_1m = [float(r["load_1m"]) for r in rows]
    load_5m = [float(r["load_5m"]) for r in rows]
    load_15m = [float(r["load_15m"]) for r in rows]
    mem_used = [float(r["mem_used"]) for r in rows]
    swap_used = [float(r["swap_used"]) for r in rows]
    swap_total = [float(r["swap_total"]) for r in rows]

    l1_min, l1_avg, l1_max = _min_avg_max(load_1m)
    l5_min, l5_avg, l5_max = _min_avg_max(load_5m)
    l15_min, l15_avg, l15_max = _min_avg_max(load_15m)
    lines.append("- load_1m min/avg/max: {:.2f}/{:.2f}/{:.2f}".format(l1_min, l1_avg, l1_max))
    lines.append("- load_5m min/avg/max: {:.2f}/{:.2f}/{:.2f}".format(l5_min, l5_avg, l5_max))
    lines.append("- load_15m min/avg/max: {:.2f}/{:.2f}/{:.2f}".format(l15_min, l15_avg, l15_max))

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
    samples = db.get_resource_process_samples_for_date(latest_day)
    if samples:
        lines.append("- top processes by RSS (latest day):")
        for row in samples:
            lines.append(
                "  pid={} {} rss={}".format(
                    row["pid"], row["name"], _bytes_to_human(int(row["rss_bytes"]))
                )
            )


def _append_network(lines: list[str], db: Any, start: str, end: str) -> None:
    rows = db.list_network_snapshots_between(start, end)
    lines.append("Network:")
    if len(rows) < 2:
        lines.append("- insufficient data")
        return

    changes = []
    prev_gateway = None
    prev_day = None
    for row in rows:
        day = row["snapshot_local_date"]
        gateway = row["default_gateway"]
        if prev_gateway is not None and gateway != prev_gateway:
            changes.append(f"{day}: {prev_gateway} -> {gateway}")
        prev_gateway = gateway
        prev_day = day

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
                "  {} rx={} tx={}".format(
                    row["name"], _format_delta_bytes(rx_delta), _format_delta_bytes(tx_delta)
                )
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
