from __future__ import annotations

import json
import re
from typing import Any

from agent.resource_insights import summarize_resource_report
from skills.resource_governor import collector

_PID_RE = re.compile(r"\bpid\s*[:=#]?\s*(\d+)\b", re.IGNORECASE)
_STATE_HINT_RE = re.compile(
    r"\b(still running|still alive|still there|still open|gone|killed|exited|stopped|running now|"
    r"is it running|is that running|is that still running|is this still running|process\b|pid\b|"
    r"kill\b|terminate\b|stop\b|close\b|quit\b|end\b)\b",
    re.IGNORECASE,
)


def _load_json(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return None
    return None


def _get_last_report(db: Any, user_id: str, report_key: str) -> dict[str, Any] | None:
    fn = getattr(db, "get_last_report", None)
    if callable(fn):
        row = fn(user_id, report_key)
        if isinstance(row, dict):
            payload = _load_json(row.get("payload") or row.get("payload_json"))
            if payload is not None:
                return payload
        if isinstance(row, tuple) and len(row) >= 2:
            payload = _load_json(row[1])
            if payload is not None:
                return payload

    # Fallback: query the sqlite connection directly if present.
    conn = getattr(db, "_conn", None)
    if conn is None:
        return None
    try:
        cur = conn.execute(
            """
            SELECT payload_json
            FROM last_report_registry
            WHERE user_id = ? AND report_key = ?
            """,
            (user_id, report_key),
        )
        row = cur.fetchone()
        if not row:
            return None
        return _load_json(row["payload_json"] if isinstance(row, dict) else row[0])
    except Exception:
        return None


def _load_process_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("rss_samples", "top_rss", "cpu_samples", "top_cpu"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for row in items:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            pid = int(row.get("pid") or 0)
            if not name or pid <= 0:
                continue
            rows.append(
                {
                    "pid": pid,
                    "name": name,
                    "cpu_ticks": int(row.get("cpu_ticks") or 0),
                    "rss_bytes": int(row.get("rss_bytes") or 0),
                }
            )
    seen: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        pid = int(row["pid"])
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(row)
    return deduped


def _dominant_process(payload: dict[str, Any]) -> dict[str, Any] | None:
    analysis = payload.get("cause_analysis") if isinstance(payload.get("cause_analysis"), dict) else {}
    dominant = analysis.get("dominant_process") if isinstance(analysis, dict) else None
    if isinstance(dominant, dict):
        pid = int(dominant.get("pid") or 0)
        name = str(dominant.get("name") or "").strip()
        if pid > 0 and name:
            return {
                "pid": pid,
                "name": name,
                "cpu_ticks": int(dominant.get("cpu_ticks") or 0),
                "rss_bytes": int(dominant.get("rss_bytes") or 0),
            }
    for row in _load_process_rows(payload):
        return row
    return None


def _question_refers_to_process_state(question: str) -> bool:
    return bool(_STATE_HINT_RE.search(str(question or "")))


def _process_reference_from_question(question: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    text = str(question or "")
    pid_match = _PID_RE.search(text)
    if pid_match:
        pid = int(pid_match.group(1))
        if pid > 0:
            return {"pid": pid, "name": None, "label": f"PID {pid}"}
    if _question_refers_to_process_state(question):
        dominant = _dominant_process(payload)
        if dominant:
            return {
                "pid": int(dominant.get("pid") or 0) or None,
                "name": str(dominant.get("name") or "").strip() or None,
                "label": f"{str(dominant.get('name') or 'process').strip()} (PID {int(dominant.get('pid') or 0)})",
            }
    return None


def _find_live_match(reference: dict[str, Any] | None, live_processes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not reference:
        return None
    ref_pid = int(reference.get("pid") or 0)
    ref_name = str(reference.get("name") or "").strip().lower()
    if ref_pid > 0:
        for row in live_processes:
            if int(row.get("pid") or 0) == ref_pid:
                return row
    if ref_name:
        for row in live_processes:
            name = str(row.get("name") or "").strip().lower()
            if name == ref_name or ref_name in name or name in ref_name:
                return row
    return None


def _load_live_snapshot() -> dict[str, Any] | None:
    try:
        live = collector.collect_live_snapshot()
    except Exception:
        return None
    if not isinstance(live, dict):
        return None
    memory = live.get("mem") if isinstance(live.get("mem"), dict) else {}
    if int(memory.get("total") or 0) <= 0:
        return None
    return live


def _humanize_live_memory(live: dict[str, Any]) -> str:
    memory = _snapshot_memory(live)
    used = int(memory.get("used") or 0)
    total = int(memory.get("total") or 0)
    available = int(memory.get("available") or 0)
    if total <= 0:
        return "memory is unavailable"
    return f"memory is {used} of {total} bytes used with {available} bytes available"


def _snapshot_memory(snapshot: dict[str, Any]) -> dict[str, Any]:
    memory = snapshot.get("memory") if isinstance(snapshot.get("memory"), dict) else {}
    if memory:
        return memory
    memory = snapshot.get("mem") if isinstance(snapshot.get("mem"), dict) else {}
    if memory:
        return memory
    return {}


def _live_analysis_payload(live: dict[str, Any], live_processes: list[dict[str, Any]]) -> dict[str, Any]:
    memory = _snapshot_memory(live)
    loadavg = live.get("loadavg")
    if isinstance(loadavg, tuple) and len(loadavg) >= 3:
        loads = {"1m": float(loadavg[0]), "5m": float(loadavg[1]), "15m": float(loadavg[2])}
    elif isinstance(loadavg, dict):
        loads = {
            "1m": float(loadavg.get("1m") or 0.0),
            "5m": float(loadavg.get("5m") or 0.0),
            "15m": float(loadavg.get("15m") or 0.0),
        }
    else:
        loads = {"1m": 0.0, "5m": 0.0, "15m": 0.0}
    return {
        "source": "live",
        "loads": loads,
        "memory": {
            "total": int(memory.get("total") or 0),
            "used": int(memory.get("used") or 0),
            "available": int(memory.get("available") or 0),
            "free": int(memory.get("free") or 0),
            "used_pct": float(memory.get("used_pct") or 0.0),
        },
        "swap": {
            "total": int((live.get("swap") or {}).get("total") or 0) if isinstance(live.get("swap"), dict) else 0,
            "used": int((live.get("swap") or {}).get("used") or 0) if isinstance(live.get("swap"), dict) else 0,
        },
        "cpu_count": int(live.get("cpu_count") or 0),
        "cpu_samples": [
            {
                "pid": int(row.get("pid") or 0),
                "name": str(row.get("name") or ""),
                "cpu_ticks": int(row.get("cpu_ticks") or 0),
                "rss_bytes": int(row.get("rss_bytes") or 0),
            }
            for row in live_processes
        ],
        "rss_samples": [
            {
                "pid": int(row.get("pid") or 0),
                "name": str(row.get("name") or ""),
                "cpu_ticks": int(row.get("cpu_ticks") or 0),
                "rss_bytes": int(row.get("rss_bytes") or 0),
            }
            for row in live_processes
        ],
    }


def _summarize_live_snapshot(live: dict[str, Any], question: str) -> dict[str, Any]:
    if isinstance(live.get("memory"), dict):
        return summarize_resource_report(live, text=question)
    live_processes = _load_process_rows(live)
    if not live_processes:
        live_processes = collector.collect_live_process_index()
    return summarize_resource_report(_live_analysis_payload(live, live_processes), text=question)


def _build_process_state_followup(
    *,
    question: str,
    previous_payload: dict[str, Any] | None,
    live: dict[str, Any],
) -> str:
    live_processes = collector.collect_live_process_index()
    reference = _process_reference_from_question(question, previous_payload or {})
    live_match = _find_live_match(reference, live_processes)
    previous_memory = int(((previous_payload or {}).get("memory") or {}).get("used") or 0)
    live_memory = int(((live.get("mem") or {}).get("used") or 0))
    memory_delta = live_memory - previous_memory if previous_memory > 0 else None

    reference_label = str((reference or {}).get("label") or "that process").strip()
    if reference and reference.get("pid"):
        pid = int(reference.get("pid") or 0)
        if live_match:
            live_label = f"PID {int(live_match.get('pid') or 0)} ({str(live_match.get('name') or '').strip()})"
            cause = f"{reference_label} is still running as {live_label}."
        else:
            cause = f"{reference_label} is no longer running."
    elif reference and reference.get("name"):
        if live_match:
            live_label = f"PID {int(live_match.get('pid') or 0)} ({str(live_match.get('name') or '').strip()})"
            cause = f"{reference_label} is still running as {live_label}."
        else:
            cause = f"{reference_label} is no longer running."
    else:
        analysis = summarize_resource_report(_live_analysis_payload(live, live_processes), text=question)
        cause = str(analysis.get("cause") or "I checked the current live state.").strip()

    if live_match:
        normality = "This is normal if you still need that workload; otherwise it is the active thing to stop."
    else:
        normality = "This is normal after a kill or restart, because the process is gone from the live probe."

    evidence_bits: list[str] = []
    if reference and reference.get("pid"):
        evidence_bits.append(f"fresh live probe checked PID {int(reference.get('pid') or 0)}")
    if live_match:
        evidence_bits.append(f"live process list still shows PID {int(live_match.get('pid') or 0)} as {str(live_match.get('name') or '').strip()}")
    else:
        evidence_bits.append("live process list does not show that PID/name anymore")
    if previous_memory > 0:
        if memory_delta is not None:
            direction = "down" if memory_delta < 0 else "up" if memory_delta > 0 else "unchanged"
            evidence_bits.append(
                f"memory changed from {previous_memory} bytes to {live_memory} bytes ({direction})"
            )
        else:
            evidence_bits.append(f"previous memory was {previous_memory} bytes")
    evidence_bits.append(_humanize_live_memory(live))

    if live_match:
        safe_action = "If you still need it, leave it alone; otherwise stop the parent app or service that launched it."
    else:
        safe_action = "No further action is needed unless you want me to check what replaced it."

    return (
        f"Likely cause: {cause} "
        f"Normality: {normality} "
        f"Evidence: {'; '.join(evidence_bits)}. "
        f"Safe next action: {safe_action}"
    )


def resource_followup(db: Any, user_id: str, kind: str, tz: str, question: str | None = None) -> str:
    _ = tz
    previous_payload = _get_last_report(db, user_id, "resource_report")
    live = _load_live_snapshot()

    if kind == "process_state":
        if live is None:
            if previous_payload:
                return "I could not get a fresh live probe to verify that process."
            return "I could not get a fresh live probe to verify the current process state."
        return _build_process_state_followup(
            question=str(question or ""),
            previous_payload=previous_payload,
            live=live,
        )

    if live is None:
        if previous_payload:
            live = previous_payload
        else:
            return "I could not get a fresh live probe, so I cannot verify the current system state."

    if kind == "top_memory":
        analysis = _summarize_live_snapshot(live, str(question or ""))
        rss = live.get("rss_samples") or analysis.get("top_memory") or []
        lines = [str(analysis.get("summary") or "").strip() or "Here is the current memory picture."]
        lines.append("")
        lines.append("Top memory processes:")
        for p in rss:
            name = p.get("name", "unknown")
            rss_b = int(p.get("rss_bytes") or 0)
            lines.append(f"- {name}: {rss_b}B RSS")
        return "\n".join(lines)

    if kind == "compare":
        if previous_payload and live is not previous_payload:
            newest_used = int(_snapshot_memory(live).get("used") or 0)
            prev_used = int((previous_payload.get("memory") or {}).get("used") or 0)
            return f"Likely cause: current live probe is available. Normality: comparison is grounded in fresh state. Evidence: memory used changed from {prev_used} to {newest_used} bytes. Safe next action: no action needed unless you want the previous snapshot compared in more detail."
        mem_used = int(_snapshot_memory(live).get("used") or 0)
        return f"Likely cause: fresh live probe shows the current memory state. Normality: comparison to a previous snapshot is unavailable. Evidence: current_used={mem_used}. Safe next action: ask again after another snapshot if you want a delta."

    if kind == "changed":
        analysis = _summarize_live_snapshot(live, str(question or ""))
        return str(analysis.get("summary") or "I checked the current live state.")

    analysis = _summarize_live_snapshot(live, str(question or ""))
    return str(analysis.get("summary") or "I checked the current live state.")
