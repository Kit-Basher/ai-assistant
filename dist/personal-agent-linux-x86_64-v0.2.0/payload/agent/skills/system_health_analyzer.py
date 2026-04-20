from __future__ import annotations

from typing import Any

from agent.resource_insights import classify_memory_pressure


def _warning(
    *,
    warning_id: str,
    severity: str,
    component: str,
    message: str,
    details: str,
) -> dict[str, str]:
    return {
        "id": str(warning_id),
        "severity": str(severity),
        "component": str(component),
        "message": str(message),
        "details": str(details),
    }


def _add_suggestion(
    suggestions: list[dict[str, str]],
    seen_ids: set[str],
    *,
    suggestion_id: str,
    label: str,
    next_action: str,
) -> None:
    normalized_id = str(suggestion_id).strip().lower()
    if not normalized_id or normalized_id in seen_ids:
        return
    seen_ids.add(normalized_id)
    suggestions.append(
        {
            "id": normalized_id,
            "label": str(label).strip(),
            "next_action": str(next_action).strip(),
        }
    )


def _escalate(current: str, new: str) -> str:
    order = {"ok": 0, "warn": 1, "critical": 2}
    return new if order.get(str(new), 0) > order.get(str(current), 0) else current


def analyze_system_health(observed: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, str]] = []
    suggestions: list[dict[str, str]] = []
    seen_suggestions: set[str] = set()
    status = "ok"

    cpu = observed.get("cpu") if isinstance(observed.get("cpu"), dict) else {}
    memory = observed.get("memory") if isinstance(observed.get("memory"), dict) else {}
    disk_rows = observed.get("disk") if isinstance(observed.get("disk"), list) else []
    gpu = observed.get("gpu") if isinstance(observed.get("gpu"), dict) else {}
    services = observed.get("services") if isinstance(observed.get("services"), dict) else {}
    network = observed.get("network") if isinstance(observed.get("network"), dict) else {}

    cpu_usage = float(cpu.get("usage_pct") or 0.0)
    cpu_count = int(cpu.get("cpu_count") or 0)
    load_1m = float(((cpu.get("load_average") or {}).get("1m") or 0.0))
    if cpu_usage >= 95.0:
        status = _escalate(status, "critical")
        warnings.append(
            _warning(
                warning_id="cpu_usage_critical",
                severity="critical",
                component="cpu",
                message="CPU usage is critically high.",
                details=f"Observed CPU usage was {cpu_usage:.1f}%.",
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="inspect_cpu_processes",
            label="Inspect top CPU processes",
            next_action="Run: ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head",
        )
    elif cpu_usage >= 85.0:
        status = _escalate(status, "warn")
        warnings.append(
            _warning(
                warning_id="cpu_usage_warn",
                severity="warn",
                component="cpu",
                message="CPU usage is high.",
                details=f"Observed CPU usage was {cpu_usage:.1f}%.",
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="inspect_cpu_processes",
            label="Inspect top CPU processes",
            next_action="Run: ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head",
        )
    if cpu_count > 0 and load_1m >= max(1.0, cpu_count * 1.5):
        status = _escalate(status, "warn")
        warnings.append(
            _warning(
                warning_id="cpu_load_warn",
                severity="warn",
                component="cpu",
                message="CPU load is high for this machine.",
                details=f"1-minute load average is {load_1m:.2f} across {cpu_count} CPUs.",
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="inspect_cpu_processes",
            label="Inspect top CPU processes",
            next_action="Run: ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head",
        )

    total_memory = int(memory.get("total_bytes") or 0)
    available_memory = int(memory.get("available_bytes") or 0)
    memory_pressure = classify_memory_pressure(
        total_bytes=total_memory,
        available_bytes=available_memory,
        swap_used_bytes=int(memory.get("swap_used_bytes") or 0),
    )
    available_pct = float(memory_pressure.get("available_pct") or 0.0) * 100.0 if total_memory > 0 else 100.0
    if bool(memory_pressure.get("is_pressure")) and available_pct < 8.0:
        status = _escalate(status, "critical")
        warnings.append(
            _warning(
                warning_id="memory_available_critical",
                severity="critical",
                component="memory",
                message="Available memory is critically low.",
                details=f"Only {available_pct:.1f}% of RAM is available.",
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="inspect_memory_processes",
            label="Inspect top memory processes",
            next_action="Run: ps -eo pid,comm,%mem,%cpu --sort=-%mem | head",
        )
    elif bool(memory_pressure.get("is_pressure")):
        status = _escalate(status, "warn")
        warnings.append(
            _warning(
                warning_id="memory_available_warn",
                severity="warn",
                component="memory",
                message="Available memory is low.",
                details=f"Only {available_pct:.1f}% of RAM is available.",
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="inspect_memory_processes",
            label="Inspect top memory processes",
            next_action="Run: ps -eo pid,comm,%mem,%cpu --sort=-%mem | head",
        )

    for row in sorted(
        (item for item in disk_rows if isinstance(item, dict)),
        key=lambda item: str(item.get("mountpoint") or ""),
    ):
        used_pct = float(row.get("used_pct") or 0.0)
        mountpoint = str(row.get("mountpoint") or "?")
        if used_pct >= 95.0:
            status = _escalate(status, "critical")
            warnings.append(
                _warning(
                    warning_id=f"disk_critical_{mountpoint}",
                    severity="critical",
                    component="disk",
                    message=f"Disk usage is critically high on {mountpoint}.",
                    details=f"The filesystem is {used_pct:.1f}% full.",
                )
            )
            _add_suggestion(
                suggestions,
                seen_suggestions,
                suggestion_id="inspect_disk_usage",
                label="Inspect largest directories",
                next_action="Run: du -sh ~/* 2>/dev/null | sort -h",
            )
        elif used_pct >= 85.0:
            status = _escalate(status, "warn")
            warnings.append(
                _warning(
                    warning_id=f"disk_warn_{mountpoint}",
                    severity="warn",
                    component="disk",
                    message=f"Disk usage is high on {mountpoint}.",
                    details=f"The filesystem is {used_pct:.1f}% full.",
                )
            )
            _add_suggestion(
                suggestions,
                seen_suggestions,
                suggestion_id="inspect_disk_usage",
                label="Inspect largest directories",
                next_action="Run: du -sh ~/* 2>/dev/null | sort -h",
            )

    gpu_expected = bool(gpu.get("expected", False))
    gpu_available = bool(gpu.get("available", False))
    gpu_error_kind = str(gpu.get("error_kind") or "").strip().lower() or None
    if gpu_expected and not gpu_available:
        status = _escalate(status, "warn")
        message = "GPU appears present but is inaccessible."
        if gpu_error_kind == "not_installed":
            message = "GPU driver tooling appears to be missing."
        warnings.append(
            _warning(
                warning_id="gpu_unavailable_warn",
                severity="warn",
                component="gpu",
                message=message,
                details=f"GPU expected={str(gpu_expected).lower()} error_kind={gpu_error_kind or 'unknown'}.",
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="inspect_gpu_status",
            label="Inspect GPU status",
            next_action="Run: nvidia-smi",
        )

    ollama = services.get("ollama") if isinstance(services.get("ollama"), dict) else {}
    if not bool(ollama.get("reachable", False)) or str(ollama.get("service_state") or "").strip().lower() not in {"active", "activating"}:
        status = _escalate(status, "warn")
        warnings.append(
            _warning(
                warning_id="service_ollama_warn",
                severity="warn",
                component="services",
                message="Ollama is not healthy.",
                details=(
                    f"service_state={str(ollama.get('service_state') or 'unknown')} "
                    f"reachable={str(bool(ollama.get('reachable', False))).lower()}."
                ),
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="check_ollama_service",
            label="Check Ollama service",
            next_action="Run: systemctl status ollama.service --no-pager",
        )

    personal_agent = services.get("personal_agent") if isinstance(services.get("personal_agent"), dict) else {}
    if not bool(personal_agent.get("reachable", False)) or str(personal_agent.get("service_state") or "").strip().lower() not in {"active", "activating"}:
        status = _escalate(status, "warn")
        warnings.append(
            _warning(
                warning_id="service_personal_agent_warn",
                severity="warn",
                component="services",
                message="Personal Agent runtime is not healthy.",
                details=(
                    f"service_state={str(personal_agent.get('service_state') or 'unknown')} "
                    f"reachable={str(bool(personal_agent.get('reachable', False))).lower()}."
                ),
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="run_agent_doctor",
            label="Run diagnostics",
            next_action="Run: python -m agent doctor",
        )

    network_state = str(network.get("state") or "").strip().lower() or "unknown"
    up_interfaces = list(network.get("up_interfaces") or [])
    default_route = bool(network.get("default_route", False))
    dns_configured = bool(network.get("dns_configured", False))
    if network_state == "down":
        status = _escalate(status, "critical")
        warnings.append(
            _warning(
                warning_id="network_down_critical",
                severity="critical",
                component="network",
                message="Network appears fully unavailable.",
                details="No active non-loopback interfaces and no default route were detected.",
            )
        )
        _add_suggestion(
            suggestions,
            seen_suggestions,
            suggestion_id="inspect_network_routes",
            label="Inspect network interfaces and routes",
            next_action="Run: ip addr && ip route",
        )
    else:
        if not default_route:
            status = _escalate(status, "warn")
            warnings.append(
                _warning(
                    warning_id="network_default_route_warn",
                    severity="warn",
                    component="network",
                    message="No default route is configured.",
                    details=f"Active interfaces: {', '.join(str(name) for name in up_interfaces) or 'none'}.",
                )
            )
            _add_suggestion(
                suggestions,
                seen_suggestions,
                suggestion_id="inspect_network_routes",
                label="Inspect network routes",
                next_action="Run: ip route",
            )
        if not dns_configured:
            status = _escalate(status, "warn")
            warnings.append(
                _warning(
                    warning_id="network_dns_warn",
                    severity="warn",
                    component="network",
                    message="DNS resolver configuration is missing.",
                    details="No nameserver entries were detected in resolv.conf.",
                )
            )
            _add_suggestion(
                suggestions,
                seen_suggestions,
                suggestion_id="inspect_dns_config",
                label="Inspect DNS configuration",
                next_action="Run: cat /etc/resolv.conf",
            )

    return {
        "status": status,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def build_system_health_report(observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "observed": observed,
        "analysis": analyze_system_health(observed),
    }


__all__ = ["analyze_system_health", "build_system_health_report"]
