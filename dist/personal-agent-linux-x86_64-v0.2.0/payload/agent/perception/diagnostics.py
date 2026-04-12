from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ONE_GIB = 1024 * 1024 * 1024


@dataclass(frozen=True)
class Event:
    kind: str
    severity: str
    summary: str
    evidence_json: dict[str, Any]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def analyze_snapshot(snapshot: dict[str, Any]) -> list[Event]:
    cpu = snapshot.get("cpu") if isinstance(snapshot.get("cpu"), dict) else {}
    memory = snapshot.get("memory") if isinstance(snapshot.get("memory"), dict) else {}
    disk = snapshot.get("disk") if isinstance(snapshot.get("disk"), dict) else {}
    root_disk = disk.get("root") if isinstance(disk.get("root"), dict) else {}
    gpu = snapshot.get("gpu") if isinstance(snapshot.get("gpu"), dict) else {}

    cpu_usage = _to_float(cpu.get("usage_pct"))
    gpu_usage = _to_float(gpu.get("usage_pct"))
    gpu_temp = _to_float(gpu.get("temperature_c"), default=-1.0)
    cpu_freq = _to_float(cpu.get("freq_mhz"))
    cpu_freq_max = _to_float(cpu.get("freq_max_mhz"))
    mem_available = _to_int(memory.get("available"))
    root_disk_used_pct = _to_float(root_disk.get("used_pct"))

    events: list[Event] = []

    if cpu_usage > 85.0 and gpu_usage < 40.0:
        events.append(
            Event(
                kind="cpu_bound",
                severity="warning",
                summary="CPU-bound workload detected.",
                evidence_json={
                    "cpu_usage_pct": cpu_usage,
                    "gpu_usage_pct": gpu_usage,
                },
            )
        )

    if gpu_usage > 90.0:
        events.append(
            Event(
                kind="gpu_bound",
                severity="warning",
                summary="GPU-bound workload detected.",
                evidence_json={
                    "gpu_usage_pct": gpu_usage,
                },
            )
        )

    thermal_freq_drop = cpu_usage > 70.0 and cpu_freq_max > 0 and cpu_freq < (cpu_freq_max * 0.65)
    if gpu_temp > 83.0 or thermal_freq_drop:
        events.append(
            Event(
                kind="thermal_throttle_suspected",
                severity="critical" if gpu_temp > 90.0 else "warning",
                summary="Thermal throttling is suspected.",
                evidence_json={
                    "gpu_temp_c": gpu_temp if gpu_temp >= 0 else None,
                    "cpu_usage_pct": cpu_usage,
                    "cpu_freq_mhz": cpu_freq,
                    "cpu_freq_max_mhz": cpu_freq_max,
                    "cpu_freq_drop_under_load": bool(thermal_freq_drop),
                },
            )
        )

    if root_disk_used_pct > 85.0:
        events.append(
            Event(
                kind="disk_pressure",
                severity="critical" if root_disk_used_pct > 95.0 else "warning",
                summary="Root filesystem is under pressure.",
                evidence_json={
                    "root_disk_used_pct": root_disk_used_pct,
                },
            )
        )

    if mem_available < ONE_GIB:
        events.append(
            Event(
                kind="oom_risk",
                severity="critical",
                summary="Available memory is below 1 GiB.",
                evidence_json={
                    "mem_available_bytes": mem_available,
                    "threshold_bytes": ONE_GIB,
                },
            )
        )

    return events
