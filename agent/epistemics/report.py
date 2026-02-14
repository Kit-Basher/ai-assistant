from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from memory.db import MemoryDB


def _safe_env_int(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return max(low, min(high, value))


def _safe_env_float(name: str, default: float, low: float, high: float) -> float:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return max(low, min(high, value))


def build_epistemics_report(
    db: Any,
    window_size: int | None = None,
    spike_threshold: float | None = None,
) -> str:
    win = int(window_size) if window_size is not None else _safe_env_int("ROLLING_WINDOW_SIZE", 50, 1, 1000)
    spike = (
        float(spike_threshold)
        if spike_threshold is not None
        else _safe_env_float("SPIKE_THRESHOLD", 0.35, 0.0, 1.0)
    )
    rows = db.activity_log_list_recent("epistemic_gate", limit=win)
    total = len(rows)
    intercepts = 0
    reason_counter: Counter[str] = Counter()
    for row in rows:
        payload = row.get("payload") or {}
        if bool(payload.get("intercepted")):
            intercepts += 1
        reasons = payload.get("reasons") or []
        for reason in reasons:
            if isinstance(reason, str) and reason.strip():
                reason_counter[reason.strip()] += 1
    passes = total - intercepts
    rate = (float(intercepts) / float(total)) if total else 0.0
    top_reasons = sorted(reason_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    spike_flag = bool(total >= 10 and rate >= spike)

    lines = [
        "Epistemics report",
        f"window_size: {win}",
        f"passes: {passes}",
        f"intercepts: {intercepts}",
        f"rolling_uncertain_rate: {rate:.3f}",
        f"spike_flag: {'true' if spike_flag else 'false'}",
        "top_reasons:",
    ]
    if not top_reasons:
        lines.append("- none")
    else:
        for reason, count in top_reasons:
            lines.append(f"- {reason}: {count}")
    return "\n".join(lines)


def _default_db_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "memory" / "agent.db")


def _schema_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "memory" / "schema.sql")


def main() -> None:
    db_path = os.getenv("AGENT_DB_PATH", _default_db_path())
    db = MemoryDB(db_path)
    try:
        db.init_schema(_schema_path())
        print(build_epistemics_report(db))
    finally:
        db.close()


if __name__ == "__main__":
    main()
