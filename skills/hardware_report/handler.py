from __future__ import annotations

from pathlib import Path
import platform
from typing import Any

from agent.skills.system_health import collect_system_health


def _format_bytes(value: int | float | None) -> str:
    amount = float(value or 0)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_index = 0
    while amount >= 1024.0 and unit_index < len(units) - 1:
        amount /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(amount)} {units[unit_index]}"
    return f"{amount:.1f} {units[unit_index]}"


def _read_cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    try:
        for raw_line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, _, value = raw_line.partition(":")
            if key.strip().lower() in {"model name", "hardware"}:
                model = value.strip()
                if model:
                    return model
    except OSError:
        return None
    return None


def _read_uptime_label() -> str | None:
    uptime_path = Path("/proc/uptime")
    if not uptime_path.is_file():
        return None
    try:
        raw = uptime_path.read_text(encoding="utf-8").split()[0]
        total_seconds = int(float(raw))
    except (OSError, ValueError, IndexError):
        return None
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def _gpu_summary(gpu: dict[str, Any]) -> tuple[str, list[str]]:
    if bool(gpu.get("available", False)):
        rows = gpu.get("gpus") if isinstance(gpu.get("gpus"), list) else []
        gpu_lines = []
        for row in rows[:2]:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "GPU").strip()
            util = float(row.get("utilization_gpu_pct") or 0.0)
            used = int(row.get("memory_used_mb") or 0)
            total = int(row.get("memory_total_mb") or 0)
            gpu_lines.append(f"{name}: {util:.1f}% util, {used}/{total} MiB")
        if gpu_lines:
            return gpu_lines[0], gpu_lines
    if bool(gpu.get("expected", False)):
        return (
            "A GPU appears to be present, but I could not read detailed metrics right now.",
            ["GPU visibility is limited right now."],
        )
    return ("I could not detect a readable GPU right now.", ["No readable GPU was detected."])


def hardware_report(context: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    _ = context
    _ = user_id
    observed = collect_system_health()
    cpu = observed.get("cpu") if isinstance(observed.get("cpu"), dict) else {}
    memory = observed.get("memory") if isinstance(observed.get("memory"), dict) else {}
    disk_rows = observed.get("disk") if isinstance(observed.get("disk"), list) else []
    gpu = observed.get("gpu") if isinstance(observed.get("gpu"), dict) else {}

    cpu_model = _read_cpu_model() or "CPU model unavailable"
    cpu_count = int(cpu.get("cpu_count") or 0)
    ram_total = _format_bytes(memory.get("total_bytes"))
    ram_used = _format_bytes(memory.get("used_bytes"))
    ram_pct = float(memory.get("used_pct") or 0.0)
    os_label = " ".join(
        part
        for part in (str(platform.system() or "").strip(), str(platform.release() or "").strip(), str(platform.machine() or "").strip())
        if part
    ).strip()
    uptime_label = _read_uptime_label()
    gpu_summary, gpu_lines = _gpu_summary(gpu)

    storage_lines: list[str] = []
    for row in disk_rows[:3]:
        if not isinstance(row, dict):
            continue
        mountpoint = str(row.get("mountpoint") or "?").strip()
        device = str(row.get("device") or "?").strip()
        total = _format_bytes(row.get("total_bytes"))
        used_pct = float(row.get("used_pct") or 0.0)
        storage_lines.append(f"{mountpoint} on {device}: {used_pct:.1f}% used of {total}")
    if not storage_lines:
        storage_lines.append("I could not read mounted storage details right now.")

    hardware_lines = [
        f"CPU: {cpu_model}" + (f" ({cpu_count} logical cores visible)" if cpu_count > 0 else ""),
        f"RAM: {ram_total} total",
        f"OS: {os_label}" if os_label else "OS: unavailable",
    ]
    if uptime_label:
        hardware_lines.append(f"Uptime: {uptime_label}")

    live_lines = [
        "CPU load: {one:.2f} / {five:.2f} / {fifteen:.2f}".format(
            one=float(((cpu.get("load_average") or {}).get("1m") or 0.0)),
            five=float(((cpu.get("load_average") or {}).get("5m") or 0.0)),
            fifteen=float(((cpu.get("load_average") or {}).get("15m") or 0.0)),
        ),
        f"Memory in use: {ram_used} ({ram_pct:.1f}%)",
    ]

    summary = (
        f"I can see {cpu_model}, {gpu_summary.lower()}, {ram_total} of RAM, "
        f"and {len(storage_lines)} mounted storage view{'s' if len(storage_lines) != 1 else ''}."
    )
    cards_payload = {
        "cards": [
            {
                "title": "Hardware inventory",
                "lines": hardware_lines,
                "severity": "ok",
            },
            {
                "title": "GPU visibility",
                "lines": gpu_lines,
                "severity": "ok" if bool(gpu.get("available", False)) else "warn",
            },
            {
                "title": "Live machine stats",
                "lines": live_lines,
                "severity": "ok",
            },
            {
                "title": "Storage",
                "lines": storage_lines,
                "severity": "ok",
            },
        ],
        "raw_available": True,
        "summary": summary,
        "confidence": 1.0,
        "next_questions": [
            "How much memory am I using?",
            "How is my storage?",
            "Can you see the GPU?",
        ],
    }
    return {
        "status": "ok",
        "text": summary,
        "payload": {
            "cpu_model": cpu_model,
            "cpu_count": cpu_count,
            "gpu": gpu,
            "memory": memory,
            "disk": disk_rows,
            "os": os_label or None,
            "uptime": uptime_label,
        },
        "cards_payload": cards_payload,
    }
