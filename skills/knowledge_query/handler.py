from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from skills.storage_governor import collector as storage_collector


SUPPORTED_EXAMPLES = [
    "latest disk report",
    "what changed this week?",
    "largest directory growth in /home",
    "any anomalies lately?",
]


class ParsedIntent:
    def __init__(
        self,
        intent: str | None,
        reason: str | None,
        start_date: str | None,
        end_date: str | None,
        path_filter: str | None,
        mount_filter: str | None,
    ) -> None:
        self.intent = intent
        self.reason = reason
        self.start_date = start_date
        self.end_date = end_date
        self.path_filter = path_filter
        self.mount_filter = mount_filter


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _now_local(tz_name: str) -> date:
    tz = ZoneInfo(tz_name or "UTC")
    return datetime.now(tz).date()


def _start_of_week(today: date) -> date:
    return today - timedelta(days=today.weekday())


def _parse_date_range(text: str, tz_name: str) -> tuple[str, str] | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|-|–)\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        return None
    start = date.fromisoformat(match.group(1))
    end = date.fromisoformat(match.group(2))
    if start > end:
        start, end = end, start
    return start.isoformat(), end.isoformat()


def _parse_time_window(text: str, tz_name: str) -> tuple[str, str]:
    today = _now_local(tz_name)
    if "today" in text:
        return today.isoformat(), today.isoformat()
    if "yesterday" in text:
        day = today - timedelta(days=1)
        return day.isoformat(), day.isoformat()
    if "this week" in text:
        start = _start_of_week(today)
        return start.isoformat(), today.isoformat()
    if "last week" in text:
        start = _start_of_week(today) - timedelta(days=7)
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat()

    match = re.search(r"last\s+(\d{1,2})\s+days", text)
    if match:
        days = max(1, min(90, int(match.group(1))))
        start = today - timedelta(days=days - 1)
        return start.isoformat(), today.isoformat()

    range_match = _parse_date_range(text, tz_name)
    if range_match:
        return range_match

    start = today - timedelta(days=6)
    return start.isoformat(), today.isoformat()


def _parse_path_filter(text: str) -> tuple[str | None, str | None]:
    if "/var/log" in text:
        return "/var/log", "/"
    if "/home" in text:
        return "/home", "/"
    if "downloads" in text:
        return "Downloads", "/"
    if "/data2" in text:
        return "/data2", "/data2"
    if "/data" in text:
        return "/data", "/data"
    if "/" in text:
        return "/", "/"
    return None, None


def _match_intents(text: str) -> list[str]:
    intents = []
    if any(phrase in text for phrase in ("latest", "last snapshot", "last report", "since last snapshot")):
        intents.append("latest_snapshot_summary")
    if any(phrase in text for phrase in ("what changed", "changes", "change over", "changed")):
        intents.append("time_window_storage_changes")
    if any(phrase in text for phrase in ("grew", "growth", "largest", "biggest")) and any(
        word in text for word in ("folder", "directory", "path", "dirs", "directories")
    ):
        intents.append("top_growth_paths")
    if any(phrase in text for phrase in ("anomal", "flag", "unusual")):
        intents.append("anomalies_in_window")
    return list(dict.fromkeys(intents))


def parse_intent(query: str, tz_name: str) -> ParsedIntent:
    text = _normalize(query)
    intents = _match_intents(text)
    if len(intents) != 1:
        return ParsedIntent(None, "ambiguous_or_unknown", None, None, None, None)

    start_date, end_date = _parse_time_window(text, tz_name)
    path_filter, mount_filter = _parse_path_filter(text)
    return ParsedIntent(intents[0], None, start_date, end_date, path_filter, mount_filter)


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


def _render_report(intent: ParsedIntent, facts: dict[str, Any], limits: dict[str, Any]) -> str:
    lines = []
    if intent.intent:
        lines.append(f"Knowledge query: {intent.intent}")
    if intent.start_date and intent.end_date:
        lines.append(f"Window: {intent.start_date} to {intent.end_date}")

    if facts.get("available") is False:
        lines.append("Data availability: not available")
        reason = facts.get("reason")
        if reason:
            lines.append(f"- reason: {reason}")
        return "\n".join(lines)

    if intent.intent == "latest_snapshot_summary":
        lines.append("Latest disk snapshots:")
        mounts = facts.get("mounts", [])
        if not mounts:
            lines.append("- no snapshots available")
        for row in mounts:
            used = _bytes_to_human(int(row["used_bytes"]))
            total = _bytes_to_human(int(row["total_bytes"]))
            delta = row.get("delta_used")
            delta_str = _bytes_to_human(abs(int(delta))) if delta is not None else "n/a"
            sign = "+" if delta is not None and int(delta) >= 0 else "-"
            delta_out = "n/a" if delta is None else f"{sign}{delta_str}"
            lines.append(
                f"- {row['mountpoint']} used {used} / {total} (delta {delta_out} since previous)"
            )
    elif intent.intent == "time_window_storage_changes":
        lines.append("Storage changes:")
        mounts = facts.get("mounts", [])
        if not mounts:
            lines.append("- no snapshots available")
        for row in mounts:
            delta = row.get("delta_used")
            delta_str = "n/a" if delta is None else _bytes_to_human(abs(int(delta)))
            sign = "+" if delta is not None and int(delta) >= 0 else "-"
            delta_out = "n/a" if delta is None else f"{sign}{delta_str}"
            lines.append(f"- {row['mountpoint']} used change: {delta_out}")
    elif intent.intent == "top_growth_paths":
        lines.append("Top path growth:")
        items = facts.get("top_paths", [])
        if not items:
            lines.append("- no growth data available")
        for item in items:
            delta = _bytes_to_human(abs(int(item["delta_bytes"])))
            sign = "+" if int(item["delta_bytes"]) >= 0 else "-"
            lines.append(f"- {item['path']}: {sign}{delta}")
    elif intent.intent == "anomalies_in_window":
        lines.append("Anomalies:")
        anomalies = facts.get("anomalies", [])
        if not anomalies:
            lines.append("- none recorded")
        for flag in anomalies:
            lines.append(f"- {flag}")

    notes = limits.get("notes") or []
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def _latest_snapshot_summary(db: Any, mount_filter: str | None) -> dict[str, Any]:
    mounts = storage_collector.DEFAULT_MOUNTPOINTS
    if mount_filter:
        mounts = [m for m in mounts if m == mount_filter]
    results = []
    for mount in mounts:
        latest = db.get_latest_disk_snapshot(mount)
        if not latest:
            continue
        prev = db.get_previous_disk_snapshot(mount, latest.get("taken_at"))
        delta = None
        if prev:
            delta = int(latest["used_bytes"]) - int(prev["used_bytes"])
        results.append(
            {
                "mountpoint": mount,
                "taken_at": latest.get("taken_at"),
                "used_bytes": int(latest["used_bytes"]),
                "total_bytes": int(latest["total_bytes"]),
                "delta_used": delta,
            }
        )
    if not results:
        return {"available": False, "reason": "no_snapshots"}
    return {"available": True, "mounts": results}


def _storage_changes_in_window(
    db: Any, start_date: str, end_date: str, mount_filter: str | None
) -> dict[str, Any]:
    mounts = storage_collector.DEFAULT_MOUNTPOINTS
    if mount_filter:
        mounts = [m for m in mounts if m == mount_filter]
    results = []
    for mount in mounts:
        rows = db.list_disk_snapshots_between(mount, start_date, end_date)
        if len(rows) < 2:
            results.append({"mountpoint": mount, "delta_used": None})
            continue
        first = rows[0]
        last = rows[-1]
        delta = int(last["used_bytes"]) - int(first["used_bytes"])
        results.append({"mountpoint": mount, "delta_used": delta})
    if not results:
        return {"available": False, "reason": "no_snapshots"}
    return {"available": True, "mounts": results}


def _top_growth_paths(
    db: Any, start_date: str, end_date: str, path_filter: str | None
) -> dict[str, Any]:
    scopes = ["root_top", "home_top"]
    if path_filter:
        if path_filter.startswith("/home") or "Downloads" in path_filter:
            scopes = ["home_top"]
        elif path_filter.startswith("/"):
            scopes = ["root_top"]
    best: list[dict[str, Any]] = []
    for scope in scopes:
        dates = db.list_dir_size_sample_dates(scope, start_date, end_date)
        if len(dates) < 2:
            continue
        first_date = dates[0]
        last_date = dates[-1]
        first = db.list_dir_size_samples_for_date(scope, first_date)
        last = db.list_dir_size_samples_for_date(scope, last_date)
        if not first or not last:
            continue
        first_map = {row["path"]: int(row["bytes"]) for row in first}
        last_map = {row["path"]: int(row["bytes"]) for row in last}
        for path, last_val in last_map.items():
            if path_filter and path_filter not in path:
                continue
            if path not in first_map:
                continue
            delta = int(last_val) - int(first_map[path])
            best.append({"path": path, "delta_bytes": delta, "scope": scope})
    if not best:
        return {"available": False, "reason": "dir_growth_not_stored"}
    best.sort(key=lambda item: item["delta_bytes"], reverse=True)
    return {"available": True, "top_paths": best[:5]}


def _anomalies_in_window(db: Any, start_date: str, end_date: str, user_id: str | None) -> dict[str, Any]:
    if not user_id:
        return {"available": False, "reason": "missing_user_id"}
    try:
        rows = db.get_anomalies(user_id, start_date, end_date)
    except Exception as exc:
        message = str(exc)
        if "anomaly_events" in message.lower():
            return {"available": False, "reason": "anomaly_events_table_missing"}
        return {"available": False, "reason": "anomaly_query_failed"}
    return {"available": True, "anomalies": rows}


def knowledge_query(context: dict[str, Any], query: str) -> dict[str, Any]:
    db = (context or {}).get("db")
    tz_name = (context or {}).get("timezone") or "UTC"
    user_id = (context or {}).get("user_id")
    if not db:
        text = "Database not available."
        return {"text": text, "data": {"intent": None, "facts": {"available": False}}}

    parsed = parse_intent(query, tz_name)
    if not parsed.intent:
        text = (
            "I can answer a few types of questions. Which one do you mean?\n"
            "Examples: " + "; ".join(SUPPORTED_EXAMPLES)
        )
        return {
            "text": text,
            "data": {"intent": None, "facts": {}, "limits": {"notes": ["clarification_required"]}},
        }

    facts: dict[str, Any]
    if parsed.intent == "latest_snapshot_summary":
        facts = _latest_snapshot_summary(db, parsed.mount_filter)
    elif parsed.intent == "time_window_storage_changes":
        facts = _storage_changes_in_window(db, parsed.start_date or "", parsed.end_date or "", parsed.mount_filter)
    elif parsed.intent == "top_growth_paths":
        facts = _top_growth_paths(db, parsed.start_date or "", parsed.end_date or "", parsed.path_filter)
    else:
        facts = _anomalies_in_window(
            db,
            parsed.start_date or "",
            parsed.end_date or "",
            str(user_id) if user_id is not None else None,
        )

    limits = {
        "time_window": {"start": parsed.start_date, "end": parsed.end_date},
        "rows": len(facts.get("mounts", [])) + len(facts.get("top_paths", [])),
        "notes": [],
    }
    if facts.get("available") is False:
        limits["notes"].append(facts.get("reason"))

    report_text = _render_report(parsed, facts, limits)
    return {
        "text": report_text,
        "data": {
            "intent": {
                "name": parsed.intent,
                "start_date": parsed.start_date,
                "end_date": parsed.end_date,
                "path_filter": parsed.path_filter,
                "mount_filter": parsed.mount_filter,
            },
            "facts": facts,
            "limits": limits,
        },
    }
