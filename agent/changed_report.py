from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass
class ChangedReport:
    baseline_created: bool
    taken_at: str | None
    prev_taken_at: str | None
    machine_summary: str | None
    delta_lines: list[str]


def _bytes_to_human(num_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T", "P"]
    v = float(max(0, int(num_bytes)))
    for u in units:
        if v < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(v)}B"
            s = f"{v:.1f}".rstrip("0").rstrip(".")
            return f"{s}{u}"
        v /= 1024.0
    return f"{int(v)}B"


def _pct_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return new - old


def _load_facts(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(row.get("facts_json") or "{}")
    except Exception:
        return {}


def _get_last_two_system_facts(db: Any, user_id: str) -> list[dict[str, Any]]:
    fn = getattr(db, "list_system_facts_snapshots", None)
    if callable(fn):
        rows = fn(user_id, limit=2)
        if isinstance(rows, list):
            return rows
    conn = getattr(db, "_conn", None)
    if conn is None:
        return []
    try:
        cur = conn.execute(
            """
            SELECT id, user_id, taken_at, boot_id, schema_version, facts_json, content_hash_sha256, partial, errors_json
            FROM system_facts_snapshots
            WHERE user_id = ?
            ORDER BY taken_at DESC
            LIMIT 2
            """,
            (user_id,),
        )
        return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []


def build_changed_report_from_system_facts(db: Any, user_id: str) -> ChangedReport:
    rows = _get_last_two_system_facts(db, user_id)
    if len(rows) < 2:
        taken_at = rows[0]["taken_at"] if rows else None
        return ChangedReport(
            baseline_created=True,
            taken_at=taken_at,
            prev_taken_at=None,
            machine_summary=None,
            delta_lines=[],
        )

    latest_row, prev_row = rows[0], rows[1]
    latest = _load_facts(latest_row)
    prev = _load_facts(prev_row)

    taken_at = str(latest_row.get("taken_at") or "")
    prev_taken_at = str(prev_row.get("taken_at") or "")

    hostname = (
        (((latest.get("snapshot") or {}).get("collector") or {}).get("hostname"))
        or "host"
    )
    kernel = (((latest.get("os") or {}).get("kernel") or {}).get("release")) or None
    machine_summary = hostname if not kernel else f"{hostname} (kernel {kernel})"

    lines: list[str] = []

    # Disk: root mount used%.
    def _root_mount(facts: dict[str, Any]) -> dict[str, Any] | None:
        mounts = (((facts.get("filesystems") or {}).get("mounts")) or [])
        for m in mounts:
            if (m.get("mountpoint") or "") == "/":
                return m
        return None

    latest_root = _root_mount(latest) or {}
    prev_root = _root_mount(prev) or {}
    latest_used_pct = latest_root.get("used_pct")
    prev_used_pct = prev_root.get("used_pct")
    if isinstance(latest_used_pct, (int, float)) and isinstance(prev_used_pct, (int, float)):
        if abs(float(latest_used_pct) - float(prev_used_pct)) >= 0.1:
            d = _pct_delta(float(latest_used_pct), float(prev_used_pct))
            sign = "+" if (d or 0.0) >= 0 else ""
            lines.append(
                f"- Disk /: {prev_used_pct:.1f}% -> {latest_used_pct:.1f}% ({sign}{(d or 0.0):.1f}%)"
            )

    # RAM available.
    latest_avail = (((latest.get("memory") or {}).get("ram_bytes") or {}).get("available"))
    prev_avail = (((prev.get("memory") or {}).get("ram_bytes") or {}).get("available"))
    if isinstance(latest_avail, int) and isinstance(prev_avail, int):
        if latest_avail != prev_avail:
            delta = latest_avail - prev_avail
            sign = "+" if delta >= 0 else "-"
            lines.append(
                "- RAM available: {} -> {} ({}{})".format(
                    _bytes_to_human(prev_avail),
                    _bytes_to_human(latest_avail),
                    sign,
                    _bytes_to_human(abs(delta)),
                )
            )

    # 1m load (only if it moved enough).
    latest_load = (((latest.get("cpu") or {}).get("load") or {}).get("load_1m"))
    prev_load = (((prev.get("cpu") or {}).get("load") or {}).get("load_1m"))
    if isinstance(latest_load, (int, float)) and isinstance(prev_load, (int, float)):
        if abs(float(latest_load) - float(prev_load)) >= 0.25:
            d = float(latest_load) - float(prev_load)
            sign = "+" if d >= 0 else ""
            lines.append(f"- Load (1m): {prev_load:.2f} -> {latest_load:.2f} ({sign}{d:.2f})")

    # Top RAM processes changed.
    def _top_mem_names(facts: dict[str, Any]) -> list[str]:
        procs = (((facts.get("process_summary") or {}).get("top_mem")) or [])
        names: list[str] = []
        for p in procs:
            name = p.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return names[:5]

    if _top_mem_names(latest) != _top_mem_names(prev):
        lines.append("- Top RAM processes changed.")

    return ChangedReport(
        baseline_created=False,
        taken_at=taken_at,
        prev_taken_at=prev_taken_at,
        machine_summary=machine_summary,
        delta_lines=lines,
    )
