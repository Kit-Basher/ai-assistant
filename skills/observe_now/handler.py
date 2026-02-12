from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from skills.resource_governor import handler as resource_handler


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _read_boot_id() -> str:
    boot_id = (_read_text("/proc/sys/kernel/random/boot_id") or "").strip()
    return boot_id or "unknown"


def _network_basics() -> dict[str, Any]:
    # Observe-only: read kernel-provided state; no external commands.
    interfaces: list[str] = []
    try:
        interfaces = sorted(
            [
                name
                for name in os.listdir("/sys/class/net")
                if name and name not in (".", "..")
            ]
        )
    except OSError:
        interfaces = []

    default_route_iface: str | None = None
    route = _read_text("/proc/net/route") or ""
    for line in route.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        iface, destination, flags = parts[0], parts[1], parts[3] if len(parts) > 3 else "0"
        if destination != "00000000":
            continue
        try:
            flags_int = int(flags, 16)
        except ValueError:
            flags_int = 0
        # RTF_UP (0x1) indicates route is usable.
        if flags_int & 0x1:
            default_route_iface = iface
            break

    return {
        "interfaces_count": len(interfaces),
        "default_route_iface": default_route_iface,
    }


def observe_now(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    """
    Explicit capture + persist. This delegates to the resource_governor snapshot path.
    """
    # 1) Collect/persist resource snapshot (existing observe-only path).
    result = resource_handler.resource_snapshot(context, user_id=user_id)

    # 2) Persist a normalized system_facts snapshot for /brief delta comparison.
    db = (context or {}).get("db")
    timezone = (context or {}).get("timezone") or "UTC"
    actor_id = user_id or (context or {}).get("user_id") or "system"
    if not db:
        return result

    # Use microseconds to avoid accidental taken_at collisions in fast repeated calls/tests.
    taken_at = datetime.now(ZoneInfo(timezone)).isoformat(timespec="microseconds")

    latest = None
    try:
        latest = db.get_latest_resource_snapshot()
    except Exception:
        latest = None

    loads = {
        "1m": float(latest["load_1m"]) if latest and "load_1m" in latest else 0.0,
        "5m": float(latest["load_5m"]) if latest and "load_5m" in latest else 0.0,
        "15m": float(latest["load_15m"]) if latest and "load_15m" in latest else 0.0,
    }
    mem_total = int(latest["mem_total"]) if latest and "mem_total" in latest else 0
    mem_used = int(latest["mem_used"]) if latest and "mem_used" in latest else 0

    # Prefer sampling tied to the latest resource snapshot taken_at (seconds resolution).
    resource_taken_at = (latest or {}).get("taken_at")
    top_cpu_rows: list[dict[str, Any]] = []
    top_mem_rows: list[dict[str, Any]] = []
    try:
        if resource_taken_at:
            top_cpu_rows = db.get_resource_process_samples(resource_taken_at, "cpu") or []
            top_mem_rows = db.get_resource_process_samples(resource_taken_at, "rss") or []
    except Exception:
        top_cpu_rows = []
        top_mem_rows = []

    facts: dict[str, Any] = {
        "taken_at": taken_at,
        "boot_id": _read_boot_id(),
        "actor_id": str(actor_id),
        "timezone": str(timezone),
        "loads": loads,
        "memory": {"used": mem_used, "total": mem_total},
        "top_cpu": [
            {
                "pid": int(r.get("pid")),
                "name": str(r.get("name") or ""),
                "cpu_ticks": int(r.get("cpu_ticks") or 0),
            }
            for r in top_cpu_rows
            if isinstance(r, dict) and isinstance(r.get("pid"), int)
        ],
        "top_mem": [
            {
                "pid": int(r.get("pid")),
                "name": str(r.get("name") or ""),
                "rss_bytes": int(r.get("rss_bytes") or 0),
            }
            for r in top_mem_rows
            if isinstance(r, dict) and isinstance(r.get("pid"), int)
        ],
        "network": _network_basics(),
    }

    facts_json = json.dumps(
        facts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    content_hash_sha256 = hashlib.sha256(facts_json.encode("utf-8")).hexdigest()
    snapshot_id = uuid.uuid4().hex  # must be unique per call; do not reuse boot_id/hash

    try:
        db.insert_system_facts_snapshot(
            id=snapshot_id,
            user_id=str(actor_id),
            taken_at=taken_at,
            boot_id=str(facts["boot_id"]),
            schema_version=1,
            facts_json=facts_json,
            content_hash_sha256=content_hash_sha256,
            partial=False,
            errors_json="[]",
        )
    except Exception:
        # /brief should still work even if this insert fails; it will act like baseline.
        return result

    return result
