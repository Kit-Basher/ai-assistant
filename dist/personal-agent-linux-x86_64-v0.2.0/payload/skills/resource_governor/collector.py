from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from memory.db import MemoryDB

TOP_N_DEFAULT = 5


@dataclass
class ProcSample:
    pid: int
    name: str
    cpu_ticks: int
    rss_bytes: int


@dataclass
class ProcStats:
    procs_scanned: int = 0
    errors_skipped: int = 0


def _now_local_iso(tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).isoformat(timespec="seconds")


def _read_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _parse_loadavg() -> tuple[float, float, float] | None:
    content = _read_file("/proc/loadavg")
    if not content:
        return None
    parts = content.split()
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _parse_meminfo() -> dict[str, int]:
    content = _read_file("/proc/meminfo")
    if not content:
        return {}
    data: dict[str, int] = {}
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            value_kb = int(parts[0])
        except ValueError:
            continue
        data[key] = value_kb * 1024
    return data


def _collect_memory() -> dict[str, Any]:
    meminfo = _parse_meminfo()
    mem_total = int(meminfo.get("MemTotal", 0))
    mem_free = int(meminfo.get("MemFree", 0))
    mem_available = int(meminfo.get("MemAvailable", mem_free))
    mem_buffers = int(meminfo.get("Buffers", 0))
    mem_cached = int(meminfo.get("Cached", 0))
    mem_shared = int(meminfo.get("Shmem", 0))
    mem_used = max(0, mem_total - mem_available)
    used_pct = round((mem_used / float(mem_total)) * 100.0, 2) if mem_total > 0 else 0.0
    return {
        "total": mem_total,
        "used": mem_used,
        "available": mem_available,
        "free": mem_free,
        "buffers": mem_buffers,
        "cached": mem_cached,
        "shared": mem_shared,
        "used_pct": used_pct,
    }


def _read_page_size() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return 4096


def _read_proc_stat_fields(pid: str) -> tuple[str, int, int] | None:
    content = _read_file(f"/proc/{pid}/stat")
    if not content:
        return None
    start = content.find("(")
    end = content.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return None
    name = content[start + 1 : end]
    rest = content[end + 2 :].split()
    if len(rest) < 15:
        return None
    try:
        utime = int(rest[11])
        stime = int(rest[12])
    except ValueError:
        return None
    return name, utime, stime


def _read_proc_rss(pid: str, page_size: int) -> int:
    content = _read_file(f"/proc/{pid}/statm")
    if not content:
        return 0
    parts = content.split()
    if len(parts) < 2:
        return 0
    try:
        rss_pages = int(parts[1])
    except ValueError:
        return 0
    return rss_pages * page_size


def _list_pids() -> list[str]:
    try:
        return [entry for entry in os.listdir("/proc") if entry.isdigit()]
    except OSError:
        return []


def _collect_process_samples(top_n: int) -> tuple[list[ProcSample], list[ProcSample], ProcStats]:
    stats = ProcStats()
    page_size = _read_page_size()

    samples: list[ProcSample] = []
    for pid in _list_pids():
        stats.procs_scanned += 1
        fields = _read_proc_stat_fields(pid)
        if not fields:
            stats.errors_skipped += 1
            continue
        name, utime, stime = fields
        rss_bytes = _read_proc_rss(pid, page_size)
        proc_ticks = utime + stime
        samples.append(
            ProcSample(
                pid=int(pid),
                name=name,
                cpu_ticks=proc_ticks,
                rss_bytes=rss_bytes,
            )
        )

    top_cpu = sorted(samples, key=lambda s: s.cpu_ticks, reverse=True)[: max(0, int(top_n))]
    top_rss = sorted(samples, key=lambda s: s.rss_bytes, reverse=True)[: max(0, int(top_n))]
    return top_cpu, top_rss, stats


def collect_live_process_index() -> list[dict[str, Any]]:
    process_index: list[dict[str, Any]] = []
    for pid in _list_pids():
        fields = _read_proc_stat_fields(pid)
        if not fields:
            continue
        name, _, _ = fields
        if not name:
            continue
        process_index.append({"pid": int(pid), "name": name})
    return process_index


def collect_live_snapshot(*, timezone: str = "UTC", top_n: int = TOP_N_DEFAULT) -> dict[str, Any]:
    taken_at = _now_local_iso(timezone)
    snapshot_local_date = taken_at.split("T")[0]
    hostname = socket.gethostname()
    cpu_count = int(os.cpu_count() or 0)

    loadavg = _parse_loadavg() or (0.0, 0.0, 0.0)
    mem = _collect_memory()
    swap_total = 0
    swap_used = 0
    meminfo = _parse_meminfo()
    swap_total = int(meminfo.get("SwapTotal", 0))
    swap_free = int(meminfo.get("SwapFree", 0))
    swap_used = max(0, swap_total - swap_free)

    cpu_samples, rss_samples, stats = _collect_process_samples(top_n)
    return {
        "taken_at": taken_at,
        "snapshot_local_date": snapshot_local_date,
        "hostname": hostname,
        "cpu_count": cpu_count,
        "loadavg": loadavg,
        "mem": mem,
        "swap": {"total": swap_total, "used": swap_used},
        "top_cpu": cpu_samples,
        "top_rss": rss_samples,
        "proc_stats": {
            "procs_scanned": stats.procs_scanned,
            "errors_skipped": stats.errors_skipped,
        },
    }


def collect_and_persist_snapshot(
    db: MemoryDB,
    *,
    timezone: str,
    top_n: int = TOP_N_DEFAULT,
) -> dict[str, Any]:
    live = collect_live_snapshot(timezone=timezone, top_n=top_n)
    taken_at = _now_local_iso(timezone)
    snapshot_local_date = taken_at.split("T")[0]
    hostname = socket.gethostname()
    loadavg = tuple(float(value) for value in live.get("loadavg") or (0.0, 0.0, 0.0))
    mem = live.get("mem") if isinstance(live.get("mem"), dict) else {}
    mem_total = int(mem.get("total", 0))
    mem_free = int(mem.get("free", 0))
    mem_available = int(mem.get("available", mem_free))
    mem_used = int(mem.get("used", max(0, mem_total - mem_available)))
    swap = live.get("swap") if isinstance(live.get("swap"), dict) else {}
    swap_total = int(swap.get("total", 0))
    swap_used = int(swap.get("used", 0))

    db.insert_resource_snapshot(
        taken_at=taken_at,
        snapshot_local_date=snapshot_local_date,
        hostname=hostname,
        load_1m=loadavg[0],
        load_5m=loadavg[1],
        load_15m=loadavg[2],
        mem_total=mem_total,
        mem_used=mem_used,
        mem_free=mem_free,
        swap_total=swap_total,
        swap_used=swap_used,
    )

    cpu_samples = live.get("top_cpu") if isinstance(live.get("top_cpu"), list) else []
    rss_samples = live.get("top_rss") if isinstance(live.get("top_rss"), list) else []
    proc_stats = live.get("proc_stats") if isinstance(live.get("proc_stats"), dict) else {}
    db.replace_resource_process_samples(
        taken_at,
        "cpu",
        [(s.pid, s.name, s.cpu_ticks, s.rss_bytes) for s in cpu_samples],
    )
    db.replace_resource_process_samples(
        taken_at,
        "rss",
        [(s.pid, s.name, s.cpu_ticks, s.rss_bytes) for s in rss_samples],
    )
    db.insert_resource_scan_stats(
        taken_at,
        "processes",
        int(proc_stats.get("procs_scanned", 0)),
        int(proc_stats.get("errors_skipped", 0)),
    )

    return {
        "taken_at": taken_at,
        "snapshot_local_date": snapshot_local_date,
        "hostname": hostname,
        "cpu_count": int(live.get("cpu_count") or 0),
        "loadavg": loadavg,
        "mem": {
            "total": mem_total,
            "used": mem_used,
            "free": mem_free,
            "available": mem_available,
            "buffers": int(mem.get("buffers", 0)),
            "cached": int(mem.get("cached", 0)),
            "shared": int(mem.get("shared", 0)),
            "used_pct": float(mem.get("used_pct", 0.0)),
        },
        "swap": {"total": swap_total, "used": swap_used},
        "top_cpu": cpu_samples,
        "top_rss": rss_samples,
        "proc_stats": {
            "procs_scanned": int(proc_stats.get("procs_scanned", 0)),
            "errors_skipped": int(proc_stats.get("errors_skipped", 0)),
        },
    }
