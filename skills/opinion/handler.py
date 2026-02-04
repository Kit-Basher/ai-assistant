from __future__ import annotations

from datetime import datetime
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
    "please advise",
    "suggest",
)

OPINION_LABELS = {
    "stable",
    "consistent",
    "variable",
    "elevated",
    "lower than usual",
    "higher than usual",
    "within your normal range",
    "outside your recent baseline",
    "worth monitoring",
    "no clear signal",
}

FORBIDDEN_WORDS = (
    "good",
    "bad",
    "healthy",
    "unhealthy",
    "optimal",
    "suboptimal",
    "problem",
    "issue",
    "fix",
    "resolve",
    "improve",
    "recommend",
    "suggest",
    "should",
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


def ask_opinion(context: dict[str, Any], question: str, timeframe: dict[str, Any], trigger: str) -> dict[str, Any]:
    db = context.get("db") if context else None
    tz_name = (context or {}).get("timezone") or "UTC"
    if not db:
        return _blocked("Database not available.")

    if _question_has_advice(question):
        refusal = (
            "I can provide bounded opinions about historical data, but not advice or actions. "
            "Please ask for observations or opinions only."
        )
        return {"status": "refused", "text": refusal, "message": refusal}

    start_date = timeframe.get("start_date")
    end_date = timeframe.get("end_date")
    start_ts = timeframe.get("start_ts")
    end_ts = timeframe.get("end_ts")
    label = timeframe.get("label") or ""
    if not start_date or not end_date:
        text = "No snapshots found yet."
        return {"status": "ok", "text": text, "message": text}

    lines: list[str] = []
    lines.append(f"Question Restated: {question.strip()}")
    lines.append(f"Timeframe: {label} ({start_date} to {end_date}, {tz_name})")

    facts: list[str] = []
    domains = _infer_domains(question)
    if not domains:
        domains = ["storage", "resources", "network"]

    if "storage" in domains:
        facts.extend(_storage_facts(db, start_date, end_date, start_ts, end_ts))
    if "resources" in domains:
        facts.extend(_resource_facts(db, start_date, end_date, start_ts, end_ts))
    if "network" in domains:
        facts.extend(_network_facts(db, start_date, end_date, start_ts, end_ts))

    lines.append("Factual Summary:")
    if facts:
        for line in facts:
            lines.append(f"- {line}")
    else:
        lines.append("- insufficient data")

    lines.append("Opinionated Assessment:")
    labels_used: list[str] = []
    for domain in ["storage", "resources", "network", "weekly_reflection"]:
        if domain not in domains:
            continue
        label_out, basis = _opinion_for_domain(db, domain, start_date, end_date, start_ts, end_ts)
        label_out = _ensure_label(label_out)
        labels_used.append(label_out)
        lines.append(f"- {domain}: {label_out} (basis: {basis})")

    lines.append("Confidence & Limits:")
    lines.append("- Opinions are bounded labels derived from your historical data only.")
    lines.append("- No advice, instructions, or actions are being provided.")
    lines.append("- If data is sparse, the label defaults to 'no clear signal'.")

    if _contains_forbidden(lines):
        return {"status": "failed", "text": "Opinion vocabulary violation.", "message": "Opinion vocabulary violation."}

    audit_details = {
        "command": "/ask_opinion",
        "trigger": trigger,
        "domains": domains,
        "labels_used": labels_used,
    }
    try:
        db.audit_log_create(
            user_id=str(timeframe.get("user_id") or "unknown"),
            action_type="ask_opinion",
            action_id="ask_opinion",
            status="executed",
            details=audit_details,
        )
    except Exception:
        return {
            "status": "failed",
            "message": "Audit logging failed. Operation aborted.",
            "text": "Audit logging failed. Operation aborted.",
        }

    return {"status": "ok", "text": "\n".join(lines), "payload": {"labels_used": labels_used}}


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


def _filter_rows_by_ts(rows: list[dict[str, Any]], start_ts: str | None, end_ts: str | None) -> list[dict[str, Any]]:
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


def _storage_facts(db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None) -> list[str]:
    lines: list[str] = []
    mountpoints = ["/", "/data", "/data2"]
    for mount in mountpoints:
        rows = db.list_disk_snapshots_between(mount, start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            continue
        delta = int(rows[-1]["used_bytes"]) - int(rows[0]["used_bytes"])
        lines.append(f"{mount} used change: {delta} ({_format_delta_bytes(delta)})")
    return lines


def _resource_facts(db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None) -> list[str]:
    rows = db.list_resource_snapshots_between(start, end)
    rows = _filter_rows_by_ts(rows, start_ts, end_ts)
    if len(rows) < 2:
        return []
    load_1m = [float(r["load_1m"]) for r in rows]
    l1_min, l1_avg, l1_max = _min_avg_max(load_1m)
    return [f"load_1m min/avg/max: {l1_min:.2f}/{l1_avg:.2f}/{l1_max:.2f}"]


def _network_facts(db: Any, start: str, end: str, start_ts: str | None, end_ts: str | None) -> list[str]:
    rows = db.list_network_snapshots_between(start, end)
    rows = _filter_rows_by_ts(rows, start_ts, end_ts)
    if len(rows) < 2:
        return []
    changes = []
    prev_gateway = None
    for row in rows:
        day = row["snapshot_local_date"]
        gateway = row["default_gateway"]
        if prev_gateway is not None and gateway != prev_gateway:
            changes.append(f"{day}: {prev_gateway} -> {gateway}")
        prev_gateway = gateway
    if not changes:
        return ["default gateway changes: none"]
    return [f"default gateway changes: {', '.join(changes)}"]


def _opinion_for_domain(
    db: Any, domain: str, start: str, end: str, start_ts: str | None, end_ts: str | None
) -> tuple[str, str]:
    if domain == "storage":
        rows = db.list_disk_snapshots_between("/", start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            return "no clear signal", "insufficient data (n<2)"
        delta = int(rows[-1]["used_bytes"]) - int(rows[0]["used_bytes"])
        avg = sum(int(r["used_bytes"]) for r in rows) / len(rows)
        label = _label_from_delta(delta, avg)
        return label, f"/ used change {delta} ({_format_delta_bytes(delta)}), avg {int(avg)}"

    if domain == "resources":
        rows = db.list_resource_snapshots_between(start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            return "no clear signal", "insufficient data (n<2)"
        load_1m = [float(r["load_1m"]) for r in rows]
        l1_min, l1_avg, l1_max = _min_avg_max(load_1m)
        range_val = l1_max - l1_min
        label = _label_from_range(range_val, l1_avg)
        return label, f"load_1m range {range_val:.2f}, avg {l1_avg:.2f}"

    if domain == "network":
        rows = db.list_network_snapshots_between(start, end)
        rows = _filter_rows_by_ts(rows, start_ts, end_ts)
        if len(rows) < 2:
            return "no clear signal", "insufficient data (n<2)"
        changes = 0
        prev_gateway = None
        for row in rows:
            gateway = row["default_gateway"]
            if prev_gateway is not None and gateway != prev_gateway:
                changes += 1
            prev_gateway = gateway
        if changes == 0:
            return "stable", "default gateway changes 0"
        return "variable", f"default gateway changes {changes}"

    return "no clear signal", "insufficient data"


def _label_from_delta(delta: int, avg: float) -> str:
    if avg <= 0:
        return "no clear signal"
    ratio = abs(delta) / avg
    if ratio <= 0.05:
        return "stable"
    if ratio <= 0.20:
        return "within your normal range"
    if delta > 0:
        return "higher than usual"
    return "lower than usual"


def _label_from_range(range_val: float, avg: float) -> str:
    if avg <= 0:
        return "no clear signal"
    ratio = range_val / avg
    if ratio <= 0.20:
        return "consistent"
    if ratio <= 0.50:
        return "variable"
    return "outside your recent baseline"


def _ensure_label(label: str) -> str:
    if label not in OPINION_LABELS:
        return "no clear signal"
    return label


def _contains_forbidden(lines: list[str]) -> bool:
    text = "\n".join(lines).lower()
    return any(word in text for word in FORBIDDEN_WORDS)
