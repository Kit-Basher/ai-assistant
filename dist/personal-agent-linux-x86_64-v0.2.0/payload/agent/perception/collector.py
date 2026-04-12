from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Callable


DEFAULT_ROOTS: tuple[str, ...] = ("/home", "/data/projects")
_MAX_DIR_CANDIDATES_PER_ROOT = 200
_CPU_SAMPLE_SECONDS = 0.05


def _normalize_roots(roots: list[str] | tuple[str, ...] | None) -> list[str]:
    raw = list(roots) if roots is not None else []
    if not raw:
        env_roots = (os.getenv("PERCEPTION_ROOTS", "") or "").strip()
        if env_roots:
            raw = [part.strip() for part in env_roots.split(",") if part.strip()]
        else:
            raw = list(DEFAULT_ROOTS)
    normalized: list[str] = []
    seen: set[str] = set()
    for root in raw:
        resolved = str(Path(root).expanduser().resolve())
        if resolved in seen:
            continue
        if not os.path.isdir(resolved):
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return normalized


def _read_proc_stat() -> dict[str, tuple[int, int]]:
    rows: dict[str, tuple[int, int]] = {}
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            name = parts[0]
            if not (name == "cpu" or name.startswith("cpu")):
                continue
            values = [int(value) for value in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            rows[name] = (total, idle)
    return rows


def _usage_pct(prev: tuple[int, int] | None, cur: tuple[int, int] | None) -> float:
    if prev is None or cur is None:
        return 0.0
    total_delta = max(0, cur[0] - prev[0])
    idle_delta = max(0, cur[1] - prev[1])
    if total_delta <= 0:
        return 0.0
    busy_delta = max(0, total_delta - idle_delta)
    return round((busy_delta / float(total_delta)) * 100.0, 2)


def _read_cpu_freqs_mhz() -> tuple[list[float], list[float]]:
    cur_values: list[float] = []
    max_values: list[float] = []
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        cur_path = cpu_dir / "cpufreq" / "scaling_cur_freq"
        max_path = cpu_dir / "cpufreq" / "cpuinfo_max_freq"
        if cur_path.is_file():
            try:
                cur_values.append(int(cur_path.read_text(encoding="utf-8").strip()) / 1000.0)
            except (OSError, ValueError, UnicodeError):
                pass
        if max_path.is_file():
            try:
                max_values.append(int(max_path.read_text(encoding="utf-8").strip()) / 1000.0)
            except (OSError, ValueError, UnicodeError):
                pass

    if cur_values:
        return cur_values, max_values

    # Fallback for systems without cpufreq sysfs.
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("cpu mhz"):
                    _, _, raw = line.partition(":")
                    cur_values.append(float(raw.strip()))
    except (OSError, UnicodeError, ValueError):
        pass

    return cur_values, max_values


def _read_meminfo_bytes() -> dict[str, int]:
    values_kb: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if not key:
                continue
            part = rest.strip().split(" ", 1)[0]
            try:
                values_kb[key] = int(part)
            except ValueError:
                continue
    total = int(values_kb.get("MemTotal", 0) * 1024)
    available = int(values_kb.get("MemAvailable", values_kb.get("MemFree", 0)) * 1024)
    free = int(values_kb.get("MemFree", 0) * 1024)
    swap_total = int(values_kb.get("SwapTotal", 0) * 1024)
    swap_free = int(values_kb.get("SwapFree", 0) * 1024)
    used = max(0, total - available)
    swap_used = max(0, swap_total - swap_free)
    return {
        "total": total,
        "used": used,
        "available": available,
        "free": free,
        "swap_total": swap_total,
        "swap_used": swap_used,
    }


def _du_size_bytes(path: str, run: Callable[..., subprocess.CompletedProcess[str]]) -> int | None:
    try:
        proc = run(
            ["du", "-sb", path],
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    token = ((proc.stdout or "").strip().split("\t", 1)[0]).strip()
    try:
        return int(token)
    except ValueError:
        return None


def _collect_top_dirs(
    roots: list[str],
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, str]] = []
    for root in roots:
        candidates.append((root, root))
        try:
            children = sorted(
                entry.path
                for entry in os.scandir(root)
                if entry.is_dir(follow_symlinks=False)
            )
        except OSError:
            children = []
        for child in children[:_MAX_DIR_CANDIDATES_PER_ROOT]:
            candidates.append((root, child))

    rows: list[dict[str, Any]] = []
    for root, candidate in candidates:
        size_bytes = _du_size_bytes(candidate, run)
        if size_bytes is None:
            continue
        rows.append({"path": candidate, "bytes": int(size_bytes), "root": root})
    rows.sort(key=lambda item: (-int(item["bytes"]), str(item["path"])))
    return rows[:10]


def _run_lines(
    args: list[str],
    run: Callable[..., subprocess.CompletedProcess[str]],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        proc = run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {"ok": False, "lines": [], "error": f"{args[0]} not found"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "lines": [], "error": str(exc)}

    text = (proc.stdout or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if proc.returncode == 0:
        return {"ok": True, "lines": lines, "error": None}
    stderr = (proc.stderr or "").strip()
    return {"ok": False, "lines": lines, "error": stderr or f"exit={proc.returncode}"}


def _collect_gpu_metrics(run: Callable[..., subprocess.CompletedProcess[str]]) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total,power.draw,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    result = _run_lines(cmd, run, timeout_seconds=3.0)
    if not result["ok"] or not result["lines"]:
        return {
            "available": False,
            "usage_pct": None,
            "memory_usage_pct": None,
            "temperature_c": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "power_draw_w": None,
            "sm_clock_mhz": None,
            "error": result.get("error"),
        }

    samples: list[dict[str, float]] = []
    for line in result["lines"]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            samples.append(
                {
                    "usage_pct": float(parts[0]),
                    "memory_usage_pct": float(parts[1]),
                    "temperature_c": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                    "power_draw_w": float(parts[5]),
                    "sm_clock_mhz": float(parts[6]),
                }
            )
        except ValueError:
            continue
    if not samples:
        return {
            "available": False,
            "usage_pct": None,
            "memory_usage_pct": None,
            "temperature_c": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "power_draw_w": None,
            "sm_clock_mhz": None,
            "error": "nvidia-smi output parse failed",
        }

    # Use max utilization and latest thermal/power values across visible GPUs.
    usage_pct = max(sample["usage_pct"] for sample in samples)
    memory_usage_pct = max(sample["memory_usage_pct"] for sample in samples)
    temperature_c = max(sample["temperature_c"] for sample in samples)
    memory_used_mb = sum(sample["memory_used_mb"] for sample in samples)
    memory_total_mb = sum(sample["memory_total_mb"] for sample in samples)
    power_draw_w = sum(sample["power_draw_w"] for sample in samples)
    sm_clock_mhz = max(sample["sm_clock_mhz"] for sample in samples)
    return {
        "available": True,
        "usage_pct": round(usage_pct, 2),
        "memory_usage_pct": round(memory_usage_pct, 2),
        "temperature_c": round(temperature_c, 2),
        "memory_used_mb": int(round(memory_used_mb)),
        "memory_total_mb": int(round(memory_total_mb)),
        "power_draw_w": round(power_draw_w, 2),
        "sm_clock_mhz": round(sm_clock_mhz, 2),
        "error": None,
    }


def collect_snapshot(
    roots: list[str] | tuple[str, ...] | None = None,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    ts = int(time.time())
    stat_before = _read_proc_stat()
    time.sleep(_CPU_SAMPLE_SECONDS)
    stat_after = _read_proc_stat()

    cpu_usage = _usage_pct(stat_before.get("cpu"), stat_after.get("cpu"))
    per_core_names = sorted(name for name in stat_after.keys() if name.startswith("cpu") and name != "cpu")
    per_core_usage = [_usage_pct(stat_before.get(name), stat_after.get(name)) for name in per_core_names]

    cur_freqs, max_freqs = _read_cpu_freqs_mhz()
    avg_freq_mhz = round(sum(cur_freqs) / len(cur_freqs), 2) if cur_freqs else 0.0
    avg_max_freq_mhz = round(sum(max_freqs) / len(max_freqs), 2) if max_freqs else avg_freq_mhz

    load_1m, load_5m, load_15m = os.getloadavg()
    memory = _read_meminfo_bytes()
    root_disk = shutil.disk_usage("/")
    root_disk_used_pct = round((float(root_disk.used) / float(root_disk.total) * 100.0), 2) if root_disk.total else 0.0

    selected_roots = _normalize_roots(roots)
    top_dirs = _collect_top_dirs(selected_roots, run)
    gpu = _collect_gpu_metrics(run)
    failed_units = _run_lines(["systemctl", "--failed"], run, timeout_seconds=3.0)
    journal_errors = _run_lines(
        ["journalctl", "-p", "err..alert", "-n", "50", "--no-pager"],
        run,
        timeout_seconds=5.0,
    )

    return {
        "ts": ts,
        "hostname": platform.node(),
        "cpu": {
            "usage_pct": cpu_usage,
            "per_core_usage_pct": per_core_usage,
            "freq_mhz": avg_freq_mhz,
            "freq_max_mhz": avg_max_freq_mhz,
            "load_avg": {
                "1m": round(float(load_1m), 2),
                "5m": round(float(load_5m), 2),
                "15m": round(float(load_15m), 2),
            },
        },
        "memory": memory,
        "disk": {
            "root": {
                "total": int(root_disk.total),
                "used": int(root_disk.used),
                "free": int(root_disk.free),
                "used_pct": root_disk_used_pct,
            },
            "roots": selected_roots,
            "top_dirs": top_dirs,
        },
        "gpu": gpu,
        "system_health": {
            "failed_units": {"lines": failed_units["lines"][:50], "error": failed_units.get("error"), "ok": failed_units["ok"]},
            "journal_errors": {
                "lines": journal_errors["lines"][:50],
                "error": journal_errors.get("error"),
                "ok": journal_errors["ok"],
            },
        },
    }
