from __future__ import annotations

import os
import platform
from typing import Any


def summarize_inventory(snapshot: dict[str, Any], roots: list[str] | tuple[str, ...]) -> dict[str, Any]:
    cpu = snapshot.get("cpu") if isinstance(snapshot.get("cpu"), dict) else {}
    memory = snapshot.get("memory") if isinstance(snapshot.get("memory"), dict) else {}
    disk = snapshot.get("disk") if isinstance(snapshot.get("disk"), dict) else {}
    root_disk = disk.get("root") if isinstance(disk.get("root"), dict) else {}
    gpu = snapshot.get("gpu") if isinstance(snapshot.get("gpu"), dict) else {}

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_count_logical": int(os.cpu_count() or 0),
        "cpu_freq_mhz": float(cpu.get("freq_mhz") or 0.0),
        "cpu_load_1m": float((cpu.get("load_avg") or {}).get("1m") or 0.0),
        "mem_total_bytes": int(memory.get("total") or 0),
        "swap_total_bytes": int(memory.get("swap_total") or 0),
        "root_disk_total_bytes": int(root_disk.get("total") or 0),
        "root_disk_used_pct": float(root_disk.get("used_pct") or 0.0),
        "gpu_present": bool(gpu.get("available")),
        "roots": list(roots),
        "top_dirs": list(disk.get("top_dirs") or []),
    }
