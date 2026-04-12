from __future__ import annotations

from typing import Any

from agent.skills.system_health_analyzer import analyze_system_health


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


def render_system_health_summary(observed: dict[str, Any], analysis: dict[str, Any] | None = None) -> str:
    cpu = observed.get("cpu") if isinstance(observed.get("cpu"), dict) else {}
    memory = observed.get("memory") if isinstance(observed.get("memory"), dict) else {}
    disk_rows = observed.get("disk") if isinstance(observed.get("disk"), list) else []
    gpu = observed.get("gpu") if isinstance(observed.get("gpu"), dict) else {}
    services = observed.get("services") if isinstance(observed.get("services"), dict) else {}
    network = observed.get("network") if isinstance(observed.get("network"), dict) else {}
    collector_warnings = observed.get("warnings") if isinstance(observed.get("warnings"), list) else []
    analysis_payload = analysis if isinstance(analysis, dict) else analyze_system_health(observed)
    warnings = analysis_payload.get("warnings") if isinstance(analysis_payload.get("warnings"), list) else []
    suggestions = (
        analysis_payload.get("suggestions")
        if isinstance(analysis_payload.get("suggestions"), list)
        else []
    )
    overall_status = str(analysis_payload.get("status") or "ok").strip().upper() or "OK"

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
    ]
    if high_usage_mounts:
        lines.insert(4, "High disk usage: " + ", ".join(high_usage_mounts[:3]))
    lines.append(f"Overall: {overall_status}")
    if warnings:
        lines.append("Warnings:")
        for item in warnings:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- {message} {details}".format(
                    message=str(item.get("message") or "").strip(),
                    details=str(item.get("details") or "").strip(),
                ).strip()
            )
    else:
        lines.append("Warnings: none")
        if collector_warnings:
            lines.append("Observed warnings: " + "; ".join(str(item) for item in collector_warnings[:3]))
    if suggestions:
        lines.append("Suggestions:")
        for item in suggestions:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- {label}: {next_action}".format(
                    label=str(item.get("label") or "").strip(),
                    next_action=str(item.get("next_action") or "").strip(),
                ).strip()
            )
    return "\n".join(lines)


__all__ = ["render_system_health_summary"]
