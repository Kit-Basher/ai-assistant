from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any, Callable

from agent.config import runtime_port, runtime_service_name


RunCommand = Callable[..., subprocess.CompletedProcess[str]]

_CPU_SAMPLE_SECONDS = 0.05
_DISK_HIGH_USAGE_PCT = 85.0
_MAX_PROCESS_ROWS = 160
_PSEUDO_FS_TYPES = {
    "autofs",
    "bpf",
    "cgroup",
    "cgroup2",
    "configfs",
    "debugfs",
    "devpts",
    "devtmpfs",
    "efivarfs",
    "fusectl",
    "hugetlbfs",
    "mqueue",
    "overlay",
    "proc",
    "pstore",
    "securityfs",
    "squashfs",
    "sysfs",
    "tmpfs",
    "tracefs",
}


def _read_proc_stat() -> tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0] != "cpu" or len(parts) < 5:
                continue
            values = [int(value) for value in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            return total, idle
    return 0, 0


def _sample_cpu_usage_pct(sample_seconds: float) -> float:
    first_total, first_idle = _read_proc_stat()
    if sample_seconds > 0:
        time.sleep(float(sample_seconds))
    second_total, second_idle = _read_proc_stat()
    total_delta = max(0, second_total - first_total)
    idle_delta = max(0, second_idle - first_idle)
    if total_delta <= 0:
        return 0.0
    busy_delta = max(0, total_delta - idle_delta)
    return round((busy_delta / float(total_delta)) * 100.0, 2)


def _collect_cpu(sample_seconds: float) -> dict[str, Any]:
    try:
        load_1, load_5, load_15 = os.getloadavg()
    except OSError:
        load_1 = load_5 = load_15 = 0.0
    return {
        "load_average": {
            "1m": round(float(load_1), 2),
            "5m": round(float(load_5), 2),
            "15m": round(float(load_15), 2),
        },
        "usage_pct": _sample_cpu_usage_pct(sample_seconds),
        "cpu_count": int(os.cpu_count() or 0),
    }


def _collect_memory() -> dict[str, Any]:
    values_kb: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            token = rest.strip().split(" ", 1)[0]
            try:
                values_kb[key] = int(token)
            except ValueError:
                continue
    total = int(values_kb.get("MemTotal", 0) * 1024)
    available = int(values_kb.get("MemAvailable", values_kb.get("MemFree", 0)) * 1024)
    used = max(0, total - available)
    used_pct = round((used / float(total)) * 100.0, 2) if total > 0 else 0.0
    return {
        "total_bytes": total,
        "used_bytes": used,
        "available_bytes": available,
        "used_pct": used_pct,
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_process_name(pid_path: Path) -> str:
    for candidate in ("comm", "status"):
        try:
            text = (pid_path / candidate).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if candidate == "comm":
            value = text.strip()
            if value:
                return value[:80]
        for line in text.splitlines():
            if line.startswith("Name:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    return value[:80]
    return "unknown"


def _read_process_status_values(pid_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = (pid_path / "status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if sep:
            values[key.strip()] = rest.strip()
    return values


def _status_kb(status: dict[str, str], key: str) -> int:
    parts = str(status.get(key) or "").split(maxsplit=1)
    if not parts:
        return 0
    token = parts[0]
    try:
        return int(token) * 1024
    except (TypeError, ValueError):
        return 0


def _process_cpu_ticks(pid_path: Path) -> int | None:
    try:
        stat = (pid_path / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    parts = stat[close + 2 :].split()
    if len(parts) < 15:
        return None
    try:
        # utime/stime are fields 14/15 in proc(5); after removing pid+comm,
        # they land at offsets 11 and 12 in this slice.
        return int(parts[11]) + int(parts[12])
    except (TypeError, ValueError):
        return None


def _friendly_process_label(name: str) -> tuple[str, str]:
    key = str(name or "unknown").strip().lower()
    labels = {
        "ollama": ("Ollama", "Local AI model service."),
        "firefox": ("Firefox", "Browser tabs and extensions."),
        "firefox-esr": ("Firefox", "Browser tabs and extensions."),
        "chrome": ("Chrome", "Browser tabs and extensions."),
        "chromium": ("Chromium", "Browser tabs and extensions."),
        "chromium-browser": ("Chromium", "Browser tabs and extensions."),
        "gnome-shell": ("GNOME Shell", "The desktop environment."),
        "xorg": ("Xorg", "Display system."),
        "xwayland": ("Xwayland", "Display compatibility service."),
        "dockerd": ("Docker services", "Container service."),
        "containerd": ("Docker services", "Container runtime."),
        "python": ("Python program", "Background Python program."),
        "python3": ("Python program", "Background Python program."),
        "personal-agent": ("Personal Agent", "Personal Agent runtime."),
    }
    return labels.get(key, (str(name or "unknown")[:80], "Local process."))


def _process_group_key(name: str) -> str:
    key = str(name or "unknown").strip().lower()
    browser_aliases = {
        "firefox-esr": "firefox",
        "chrome": "chromium",
        "chromium-browser": "chromium",
        "chrome_crashpad": "chromium",
    }
    service_aliases = {
        "dockerd": "docker",
        "containerd": "docker",
        "containerd-shim": "docker",
    }
    return service_aliases.get(key, browser_aliases.get(key, key or "unknown"))


def _collect_processes(*, memory_total_bytes: int, sample_seconds: float = 0.0) -> dict[str, Any]:
    proc_root = Path("/proc")
    rows: list[dict[str, Any]] = []
    first_ticks: dict[int, int] = {}
    clock_ticks = 100
    try:
        clock_ticks = int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    except Exception:
        clock_ticks = 100

    pid_paths = [path for path in proc_root.iterdir() if path.name.isdigit()] if proc_root.is_dir() else []
    for pid_path in pid_paths:
        try:
            pid = int(pid_path.name)
        except ValueError:
            continue
        ticks = _process_cpu_ticks(pid_path)
        if ticks is not None:
            first_ticks[pid] = ticks

    if sample_seconds > 0:
        time.sleep(float(sample_seconds))

    for pid_path in pid_paths:
        try:
            pid = int(pid_path.name)
        except ValueError:
            continue
        status = _read_process_status_values(pid_path)
        if not status:
            continue
        name = _read_process_name(pid_path)
        rss = _status_kb(status, "VmRSS")
        if rss <= 0:
            continue
        user_scope = "other"
        uid_token = str(status.get("Uid") or "").split(maxsplit=1)[0]
        try:
            if int(uid_token) == os.getuid():
                user_scope = "current_user"
            elif int(uid_token) == 0:
                user_scope = "root"
        except (TypeError, ValueError):
            user_scope = "unknown"
        cpu_percent = 0.0
        second_ticks = _process_cpu_ticks(pid_path)
        if sample_seconds > 0 and second_ticks is not None and pid in first_ticks:
            delta = max(0, int(second_ticks) - int(first_ticks[pid]))
            cpu_count = max(1, int(os.cpu_count() or 1))
            cpu_percent = round((delta / float(clock_ticks)) / float(sample_seconds) / cpu_count * 100.0, 2)
        display_name, description = _friendly_process_label(name)
        rows.append(
            {
                "pid": pid,
                "process_name": name,
                "display_name": display_name,
                "description": description,
                "memory_rss_bytes": int(rss),
                "memory_percent": round((rss / float(memory_total_bytes)) * 100.0, 2) if memory_total_bytes > 0 else 0.0,
                "cpu_percent": cpu_percent,
                "user_scope": user_scope,
            }
        )

    rows.sort(key=lambda row: (int(row.get("memory_rss_bytes") or 0), _safe_float(row.get("cpu_percent"))), reverse=True)
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _process_group_key(str(row.get("process_name") or "unknown"))
        group = groups.setdefault(
            key,
            {
                "group_key": key,
                "display_name": row.get("display_name") or row.get("process_name") or "unknown",
                "description": row.get("description") or "Local process.",
                "memory_rss_bytes": 0,
                "memory_percent": 0.0,
                "cpu_percent": 0.0,
                "process_count_group": 0,
                "top_pid": row.get("pid"),
                "user_scope": row.get("user_scope") or "unknown",
            },
        )
        group["memory_rss_bytes"] = int(group.get("memory_rss_bytes") or 0) + int(row.get("memory_rss_bytes") or 0)
        group["cpu_percent"] = round(float(group.get("cpu_percent") or 0.0) + _safe_float(row.get("cpu_percent")), 2)
        group["process_count_group"] = int(group.get("process_count_group") or 0) + 1
        group["memory_percent"] = round(
            (int(group.get("memory_rss_bytes") or 0) / float(memory_total_bytes)) * 100.0,
            2,
        ) if memory_total_bytes > 0 else 0.0

    grouped = sorted(groups.values(), key=lambda row: (int(row.get("memory_rss_bytes") or 0), _safe_float(row.get("cpu_percent"))), reverse=True)
    return {
        "available": True,
        "count": len(rows),
        "top": rows[: min(_MAX_PROCESS_ROWS, 25)],
        "groups": grouped[:25],
        "redaction": {
            "command_lines_included": False,
            "environment_included": False,
        },
    }


def _iter_mount_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    seen_mounts: set[str] = set()
    with open("/proc/mounts", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            device, mountpoint, fstype = parts[:3]
            if fstype in _PSEUDO_FS_TYPES:
                continue
            if mountpoint in seen_mounts:
                continue
            seen_mounts.add(mountpoint)
            rows.append((device, mountpoint, fstype))
    rows.sort(key=lambda item: item[1])
    return rows


def _collect_disk() -> list[dict[str, Any]]:
    disks: list[dict[str, Any]] = []
    for device, mountpoint, fstype in _iter_mount_rows():
        try:
            usage = shutil.disk_usage(mountpoint)
        except OSError:
            continue
        used_pct = round((usage.used / float(usage.total)) * 100.0, 2) if usage.total > 0 else 0.0
        disks.append(
            {
                "device": device,
                "mountpoint": mountpoint,
                "fstype": fstype,
                "total_bytes": int(usage.total),
                "used_bytes": int(usage.used),
                "free_bytes": int(usage.free),
                "used_pct": used_pct,
                "high_usage": bool(used_pct >= _DISK_HIGH_USAGE_PCT),
            }
        )
    return disks


def _run_command(
    args: list[str],
    *,
    run_command: RunCommand,
    timeout_seconds: float = 1.5,
) -> dict[str, Any]:
    try:
        proc = run_command(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except FileNotFoundError:
        return {"ok": False, "stdout": "", "stderr": "", "error_kind": "not_found", "returncode": None}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "", "error_kind": "timeout", "returncode": None}
    except (OSError, subprocess.SubprocessError):
        return {"ok": False, "stdout": "", "stderr": "", "error_kind": "exec_failed", "returncode": None}
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "error_kind": None if proc.returncode == 0 else "bad_exit",
        "returncode": int(proc.returncode),
    }


def _collect_gpu(run_command: RunCommand) -> dict[str, Any]:
    gpu_expected = bool(Path("/dev/nvidiactl").exists() or Path("/proc/driver/nvidia/version").exists())
    if not gpu_expected:
        for vendor_path in sorted(Path("/sys/class/drm").glob("card*/device/vendor")):
            try:
                if vendor_path.read_text(encoding="utf-8").strip().lower() == "0x10de":
                    gpu_expected = True
                    break
            except OSError:
                continue
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "expected": gpu_expected,
            "driver_version": None,
            "gpus": [],
            "error_kind": "not_installed",
        }
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        run_command=run_command,
        timeout_seconds=2.0,
    )
    if not result["ok"]:
        return {
            "available": False,
            "expected": gpu_expected,
            "driver_version": None,
            "gpus": [],
            "error_kind": result["error_kind"] or "command_failed",
        }
    gpus: list[dict[str, Any]] = []
    driver_version: str | None = None
    for raw_line in str(result["stdout"]).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        name, driver_version_value, util_gpu, mem_used, mem_total, temp_gpu = parts[:6]
        driver_version = driver_version or driver_version_value or None
        try:
            gpus.append(
                {
                    "name": name,
                    "utilization_gpu_pct": float(util_gpu),
                    "memory_used_mb": int(float(mem_used)),
                    "memory_total_mb": int(float(mem_total)),
                    "temperature_c": int(float(temp_gpu)),
                }
            )
        except ValueError:
            continue
    if not gpus:
        return {
            "available": False,
            "expected": gpu_expected,
            "driver_version": driver_version,
            "gpus": [],
            "error_kind": "parse_failed",
        }
    return {
        "available": True,
        "expected": gpu_expected,
        "driver_version": driver_version,
        "gpus": gpus,
        "error_kind": None,
    }


def _systemctl_state(unit: str, *, user: bool, run_command: RunCommand) -> str:
    args = ["systemctl"]
    if user:
        args.append("--user")
    args.extend(["is-active", unit])
    result = _run_command(args, run_command=run_command, timeout_seconds=1.0)
    text = str(result["stdout"] or result["stderr"]).strip().lower()
    if text:
        return text
    if result["error_kind"] == "not_found":
        return "systemctl_missing"
    return "unknown"


def _tcp_reachable(host: str, port: int, *, timeout_seconds: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_seconds)):
            return True
    except OSError:
        return False


def _collect_services(run_command: RunCommand) -> dict[str, Any]:
    return {
        "ollama": {
            "service_name": "ollama.service",
            "service_scope": "system",
            "service_state": _systemctl_state("ollama.service", user=False, run_command=run_command),
            "host": "127.0.0.1",
            "port": 11434,
            "reachable": _tcp_reachable("127.0.0.1", 11434),
        },
        "personal_agent": {
            "service_name": runtime_service_name(),
            "service_scope": "user",
            "service_state": _systemctl_state(runtime_service_name(), user=True, run_command=run_command),
            "host": "127.0.0.1",
            "port": runtime_port(),
            "reachable": _tcp_reachable("127.0.0.1", runtime_port()),
        },
    }


def _default_route_present() -> bool:
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                parts = line.split()
                if len(parts) < 4:
                    continue
                destination = parts[1]
                flags = int(parts[3], 16)
                if destination == "00000000" and (flags & 0x2):
                    return True
    except (OSError, ValueError):
        return False
    return False


def _up_interfaces() -> list[str]:
    interfaces: list[str] = []
    sys_class_net = Path("/sys/class/net")
    if not sys_class_net.is_dir():
        return interfaces
    for path in sorted(sys_class_net.iterdir(), key=lambda item: item.name):
        if path.name == "lo":
            continue
        try:
            operstate = (path / "operstate").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if operstate == "up":
            interfaces.append(path.name)
    return interfaces


def _dns_configured() -> bool:
    try:
        text = Path("/etc/resolv.conf").read_text(encoding="utf-8")
    except OSError:
        return False
    return any(line.strip().startswith("nameserver ") for line in text.splitlines())


def _collect_network() -> dict[str, Any]:
    up_interfaces = _up_interfaces()
    default_route = _default_route_present()
    dns_configured = _dns_configured()
    if up_interfaces and default_route:
        state = "up"
    elif up_interfaces or default_route:
        state = "degraded"
    else:
        state = "down"
    return {
        "state": state,
        "up_interfaces": up_interfaces,
        "default_route": default_route,
        "dns_configured": dns_configured,
    }


def _derive_warnings(
    *,
    memory: dict[str, Any],
    disk: list[dict[str, Any]],
    services: dict[str, Any],
    network: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    for row in disk:
        if bool(row.get("high_usage", False)):
            warnings.append(
                f"Disk usage is high on {row.get('mountpoint')}: {float(row.get('used_pct') or 0.0):.1f}% used."
            )
    if float(memory.get("used_pct") or 0.0) >= 90.0:
        warnings.append(f"Memory usage is high: {float(memory.get('used_pct') or 0.0):.1f}% used.")
    ollama = services.get("ollama") if isinstance(services.get("ollama"), dict) else {}
    if not bool(ollama.get("reachable", False)):
        warnings.append("Ollama is unreachable on 127.0.0.1:11434.")
    personal_agent = services.get("personal_agent") if isinstance(services.get("personal_agent"), dict) else {}
    if not bool(personal_agent.get("reachable", False)):
        warnings.append(f"Personal Agent API is unreachable on 127.0.0.1:{runtime_port()}.")
    if str(network.get("state") or "").strip().lower() == "down":
        warnings.append("No active non-loopback network interface was detected.")
    return warnings


def collect_system_health(
    *,
    run_command: RunCommand | None = None,
    sample_seconds: float = _CPU_SAMPLE_SECONDS,
) -> dict[str, Any]:
    runner = run_command or subprocess.run
    cpu = _collect_cpu(sample_seconds=max(0.0, float(sample_seconds)))
    memory = _collect_memory()
    processes = _collect_processes(memory_total_bytes=int(memory.get("total_bytes") or 0), sample_seconds=max(0.0, float(sample_seconds)))
    disk = _collect_disk()
    gpu = _collect_gpu(runner)
    services = _collect_services(runner)
    network = _collect_network()
    warnings = _derive_warnings(memory=memory, disk=disk, services=services, network=network)
    return {
        "cpu": cpu,
        "memory": memory,
        "processes": processes,
        "disk": disk,
        "gpu": gpu,
        "services": services,
        "network": network,
        "warnings": warnings,
        "collected_at": int(time.time()),
    }


__all__ = ["collect_system_health"]
