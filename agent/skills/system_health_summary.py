from __future__ import annotations

from typing import Any


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


def render_system_health_summary(data: dict[str, Any]) -> str:
    cpu = data.get("cpu") if isinstance(data.get("cpu"), dict) else {}
    memory = data.get("memory") if isinstance(data.get("memory"), dict) else {}
    disk_rows = data.get("disk") if isinstance(data.get("disk"), list) else []
    gpu = data.get("gpu") if isinstance(data.get("gpu"), dict) else {}
    services = data.get("services") if isinstance(data.get("services"), dict) else {}
    network = data.get("network") if isinstance(data.get("network"), dict) else {}
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []

    disk_by_mount = {
        str(row.get("mountpoint") or ""): row
        for row in disk_rows
        if isinstance(row, dict) and str(row.get("mountpoint") or "")
    }
    root_disk = disk_by_mount.get("/") or (disk_rows[0] if disk_rows else {})
    high_usage_mounts = [
        f"{str(row.get('mountpoint') or '?')} {float(row.get('used_pct') or 0.0):.1f}%"
        for row in disk_rows
        if isinstance(row, dict) and bool(row.get("high_usage", False))
    ]

    ollama = services.get("ollama") if isinstance(services.get("ollama"), dict) else {}
    personal_agent = services.get("personal_agent") if isinstance(services.get("personal_agent"), dict) else {}
    service_line = (
        f"Ollama {str(ollama.get('service_state') or 'unknown')}/"
        f"{'reachable' if bool(ollama.get('reachable', False)) else 'unreachable'}; "
        f"Personal Agent {str(personal_agent.get('service_state') or 'unknown')}/"
        f"{'reachable' if bool(personal_agent.get('reachable', False)) else 'unreachable'}"
    )

    gpu_line = "unavailable"
    gpu_rows = gpu.get("gpus") if isinstance(gpu.get("gpus"), list) else []
    if bool(gpu.get("available", False)) and gpu_rows:
        first_gpu = gpu_rows[0] if isinstance(gpu_rows[0], dict) else {}
        gpu_line = (
            f"{str(first_gpu.get('name') or 'GPU')} "
            f"{float(first_gpu.get('utilization_gpu_pct') or 0.0):.1f}% util, "
            f"{int(first_gpu.get('memory_used_mb') or 0)}/{int(first_gpu.get('memory_total_mb') or 0)} MiB, "
            f"{int(first_gpu.get('temperature_c') or 0)}C"
        )
        if gpu.get("driver_version"):
            gpu_line += f", driver {gpu.get('driver_version')}"

    network_interfaces = ", ".join(str(name) for name in network.get("up_interfaces") or []) or "none"
    network_line = (
        f"{str(network.get('state') or 'unknown')}; "
        f"up={network_interfaces}; "
        f"default_route={'yes' if bool(network.get('default_route', False)) else 'no'}; "
        f"dns={'yes' if bool(network.get('dns_configured', False)) else 'no'}"
    )

    warning_line = "none"
    if warnings:
        warning_line = "; ".join(str(item) for item in warnings[:3])

    lines = [
        "System health",
        "CPU: load {one:.2f}/{five:.2f}/{fifteen:.2f}, usage {usage:.1f}%".format(
            one=float(((cpu.get("load_average") or {}).get("1m") or 0.0)),
            five=float(((cpu.get("load_average") or {}).get("5m") or 0.0)),
            fifteen=float(((cpu.get("load_average") or {}).get("15m") or 0.0)),
            usage=float(cpu.get("usage_pct") or 0.0),
        ),
        "Memory: {used} / {total} used ({pct:.1f}%), {available} available".format(
            used=_format_bytes(memory.get("used_bytes")),
            total=_format_bytes(memory.get("total_bytes")),
            pct=float(memory.get("used_pct") or 0.0),
            available=_format_bytes(memory.get("available_bytes")),
        ),
        "Disk: {mount} {pct:.1f}% used".format(
            mount=str(root_disk.get("mountpoint") or "/"),
            pct=float(root_disk.get("used_pct") or 0.0),
        ),
        f"GPU: {gpu_line}",
        f"Services: {service_line}",
        f"Network: {network_line}",
        f"Warnings: {warning_line}",
    ]
    if high_usage_mounts:
        lines.insert(4, "High disk usage: " + ", ".join(high_usage_mounts[:3]))
    return "\n".join(lines)


__all__ = ["render_system_health_summary"]
