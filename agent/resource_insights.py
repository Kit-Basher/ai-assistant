from __future__ import annotations

from typing import Any


_VM_HINTS = (
    "qemu",
    "qemu-system",
    "virtualbox",
    "virtualboxvm",
    "vbox",
    "vmware",
    "vmware-vmx",
    "virt-manager",
    "virtqemud",
    "virt-launch",
    "virtio",
    "libvirt",
    "gnome-boxes",
    "hyperv",
    "xen",
    "emulator",
    "genymotion",
    "parallels",
)

_LLM_HINTS = (
    "ollama",
    "llama-server",
    "llamacpp",
    "llama.cpp",
    "vllm",
    "localai",
    "koboldcpp",
    "kobold",
    "lmstudio",
    "text-generation-webui",
    "open-webui",
)

_BROWSER_HINTS = (
    "chrome",
    "chromium",
    "brave",
    "firefox",
    "microsoft-edge",
    "msedge",
    "opera",
    "vivaldi",
)

_SHELL_HINTS = (
    "gnome-shell",
    "mutter",
    "kwin",
    "plasmashell",
    "xfwm4",
    "weston",
    "sway",
    "hyprland",
    "cinnamon",
    "budgie",
    "i3",
    "wayfire",
)

_DATABASE_HINTS = (
    "postgres",
    "postmaster",
    "mysqld",
    "mariadbd",
    "mongod",
    "redis-server",
    "clickhouse",
    "influxd",
    "elasticsearch",
    "opensearch",
    "cassandra",
    "rabbitmq",
)

_DAEMON_HINTS = (
    "dockerd",
    "containerd",
    "systemd",
    "journal",
    "dbus-daemon",
    "networkmanager",
    "pipewire",
    "wireplumber",
    "polkitd",
    "packagekitd",
    "fwupd",
    "tracker-miner",
    "baloo",
    "sshd",
)

_JOB_HINTS = (
    "python",
    "python3",
    "node",
    "nodejs",
    "java",
    "ruby",
    "perl",
    "php",
    "bash",
    "sh",
    "uvicorn",
    "gunicorn",
    "celery",
    "pytest",
    "gradle",
    "cargo",
    "go",
)


def _bytes_to_human(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0B"
    units = ["B", "K", "M", "G", "T", "P", "E"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}B"
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}{unit}"
        value /= 1024
    return f"{int(value)}B"


def _gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3) if num_bytes > 0 else 0.0


def classify_memory_pressure(
    *,
    total_bytes: int,
    available_bytes: int,
    swap_used_bytes: int = 0,
) -> dict[str, Any]:
    total = max(int(total_bytes), 0)
    available = max(int(available_bytes), 0)
    swap_used = max(int(swap_used_bytes), 0)
    available_pct = (available / float(total)) if total > 0 else 0.0
    if total <= 0:
        return {
            "state": "unknown",
            "available_pct": available_pct,
            "is_normal": True,
            "is_pressure": False,
            "reason": "unavailable",
        }
    if available_pct >= 0.25:
        return {
            "state": "normal",
            "available_pct": available_pct,
            "is_normal": True,
            "is_pressure": False,
            "reason": "plenty_available",
        }
    if available_pct <= 0.15:
        return {
            "state": "pressure",
            "available_pct": available_pct,
            "is_normal": False,
            "is_pressure": True,
            "reason": "tight",
        }
    return {
        "state": "normal",
        "available_pct": available_pct,
        "is_normal": True,
        "is_pressure": False,
        "reason": "within_expected_range",
    }


def _normalize_rows(payload: dict[str, Any], key: str, fallback_key: str) -> list[dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list):
        rows = payload.get(fallback_key)
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "pid": int(row.get("pid") or 0),
                "name": name,
                "cpu_ticks": int(row.get("cpu_ticks") or 0),
                "rss_bytes": int(row.get("rss_bytes") or 0),
            }
        )
    return out


def _merge_samples(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            pid = int(row.get("pid") or 0)
            name = str(row.get("name") or "").strip()
            if pid <= 0 or not name:
                continue
            existing = merged.setdefault(
                pid,
                {"pid": pid, "name": name, "cpu_ticks": 0, "rss_bytes": 0},
            )
            if len(name) > len(str(existing.get("name") or "")):
                existing["name"] = name
            existing["cpu_ticks"] = max(int(existing.get("cpu_ticks") or 0), int(row.get("cpu_ticks") or 0))
            existing["rss_bytes"] = max(int(existing.get("rss_bytes") or 0), int(row.get("rss_bytes") or 0))
    return list(merged.values())


def _detect_family(name: str) -> tuple[str | None, str | None]:
    lowered = name.lower()
    if any(hint in lowered for hint in _VM_HINTS):
        return "vm", "virtual machine/emulator"
    if any(hint in lowered for hint in _LLM_HINTS):
        return "llm", "LLM runtime"
    if any(hint in lowered for hint in _BROWSER_HINTS):
        return "browser", "browser session"
    if any(hint in lowered for hint in _SHELL_HINTS):
        return "shell", "desktop shell/compositor"
    if any(hint in lowered for hint in _DATABASE_HINTS):
        return "database", "database or storage daemon"
    if any(hint in lowered for hint in _DAEMON_HINTS):
        return "daemon", "background daemon"
    if any(hint in lowered for hint in _JOB_HINTS):
        return "job", "long-running job"
    return None, None


def _family_label(family_key: str | None) -> str | None:
    mapping = {
        "vm": "virtual machine/emulator",
        "llm": "LLM runtime",
        "browser": "browser session",
        "shell": "desktop shell/compositor",
        "database": "database or storage daemon",
        "daemon": "background daemon",
        "job": "long-running job",
    }
    return mapping.get(family_key or "")


def _family_phrasing(family_key: str | None, row_count: int) -> str:
    if family_key == "browser":
        return "a browser session with multiple tabs/processes" if row_count > 1 else "a browser session"
    if family_key == "vm":
        return "a virtual machine/emulator"
    if family_key == "llm":
        return "an LLM runtime or model server"
    if family_key == "shell":
        return "the desktop shell/compositor"
    if family_key == "database":
        return "a database or storage daemon"
    if family_key == "daemon":
        return "a background daemon or service"
    if family_key == "job":
        return "a long-running background job"
    return "an unmanaged background process"


def _background_process_note(text: str, family_key: str | None) -> str | None:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("nothing open", "no windows", "idle", "background", "not open")) and family_key in {
        "vm",
        "llm",
        "browser",
        "shell",
        "database",
        "daemon",
        "job",
        None,
    }:
        return "Even if nothing is open, this points to background work rather than an active app window."
    return None


def _family_role_note(family_key: str | None, rows: list[dict[str, Any]]) -> str | None:
    count = len(rows)
    if family_key == "llm":
        if count > 1:
            return "With multiple Ollama processes, one is usually the server and another is likely a worker or model helper."
        return "Ollama is a model server, so its main process can also spin up helpers for model serving."
    if family_key == "vm":
        if count > 1:
            return "With multiple VM processes, one is usually the virtual machine itself and another is often a helper or emulator support process."
        return "This is the virtual machine/emulator process itself, so the guest workload is what drives the usage."
    if family_key == "browser":
        if count > 1:
            return "Browsers are multi-process by design, so separate tab, renderer, and helper processes are expected."
        return "Browsers often split tabs and renderers into separate processes."
    return None


def _detect_focus(text: str, payload: dict[str, Any]) -> str:
    lowered = (text or "").lower()
    if any(word in lowered for word in ("cpu", "lag", "slow", "stuck", "unresponsive")):
        return "cpu"
    if any(word in lowered for word in ("ram", "memory", "swap", "using 30%", "using 40%", "using 50%")):
        return "memory"

    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    used_pct = float(memory.get("used_pct") or 0.0)
    available = int(memory.get("available") or 0)
    total = int(memory.get("total") or 0)
    memory_pressure = classify_memory_pressure(total_bytes=total, available_bytes=available)
    available_pct = float(memory_pressure.get("available_pct") or 0.0)
    loads = payload.get("loads") if isinstance(payload.get("loads"), dict) else {}
    load_1m = float(loads.get("1m") or 0.0)
    if used_pct >= 50.0 and bool(memory_pressure.get("is_pressure")):
        return "memory"
    if load_1m >= 1.0:
        return "cpu"
    return "memory"


def _memory_note(payload: dict[str, Any]) -> str | None:
    note = str(payload.get("memory_note") or "").strip()
    if note:
        return note
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    buffers = int(memory.get("buffers") or 0)
    cached = int(memory.get("cached") or 0)
    shared = int(memory.get("shared") or 0)
    parts: list[str] = []
    if buffers > 0:
        parts.append(f"buffers {_bytes_to_human(buffers)}")
    if cached > 0:
        parts.append(f"cached {_bytes_to_human(cached)}")
    if shared > 0:
        parts.append(f"shared {_bytes_to_human(shared)}")
    if not parts:
        return None
    return "Reclaimable memory includes " + ", ".join(parts) + "."


def _safe_action_for_family(family_key: str | None, symptom: str) -> str:
    if family_key == "vm":
        return "If you do not need the VM right now, shut it down or pause it."
    if family_key == "llm":
        return "If the model server is idle, stop it or switch to a smaller model."
    if family_key == "browser":
        return "Close the heaviest tabs or restart the browser if it has been open a long time."
    if family_key == "shell":
        return "If the UI is lagging, restart the shell/session only if you are comfortable doing that."
    if family_key == "database":
        return "Leave it alone if you expect that service; otherwise inspect which daemon started it."
    if family_key == "daemon":
        return "Check whether that background daemon is expected before restarting or stopping it."
    if family_key == "job":
        return "Find the parent script or service before stopping it, especially if you did not start it."
    if symptom == "cpu":
        return "Let the current job finish, or stop the workload if you did not expect it to be running."
    return "No urgent action is obvious; keep the system under observation."


def summarize_resource_report(payload: dict[str, Any], text: str = "") -> dict[str, Any]:
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    loads = payload.get("loads") if isinstance(payload.get("loads"), dict) else {}
    cpu_samples = _normalize_rows(payload, "cpu_samples", "top_cpu")
    rss_samples = _normalize_rows(payload, "rss_samples", "top_rss")

    total = int(memory.get("total") or 0)
    used = int(memory.get("used") or 0)
    available = int(memory.get("available") or 0)
    free = int(memory.get("free") or 0)
    used_pct = float(memory.get("used_pct") or ((used / float(total)) * 100.0 if total > 0 else 0.0))
    swap = payload.get("swap") if isinstance(payload.get("swap"), dict) else {}
    swap_used = int(swap.get("used") or 0)
    swap_total = int(swap.get("total") or 0)
    cpu_count = int(payload.get("cpu_count") or 0)
    load_1m = float(loads.get("1m") or 0.0)
    load_5m = float(loads.get("5m") or 0.0)
    load_15m = float(loads.get("15m") or 0.0)

    focus = _detect_focus(text, payload)
    memory_pressure = classify_memory_pressure(
        total_bytes=total,
        available_bytes=available,
        swap_used_bytes=swap_used,
    )
    memory_note = _memory_note(payload)

    rss_ranked = sorted(rss_samples, key=lambda row: row["rss_bytes"], reverse=True)
    cpu_ranked = sorted(cpu_samples, key=lambda row: row["cpu_ticks"], reverse=True)
    merged_samples = _merge_samples(cpu_ranked, rss_ranked)
    sampled_rss_total = sum(int(row["rss_bytes"]) for row in rss_ranked)
    sampled_cpu_total = sum(int(row["cpu_ticks"]) for row in cpu_ranked)

    family_totals: dict[str, dict[str, Any]] = {}
    for row in merged_samples:
        family_key, family_label = _detect_family(row["name"])
        if not family_key:
            continue
        entry = family_totals.setdefault(
            family_key,
            {"rss_bytes": 0, "cpu_ticks": 0, "label": family_label, "rows": []},
        )
        entry["rss_bytes"] += int(row["rss_bytes"])
        entry["cpu_ticks"] += int(row["cpu_ticks"])
        entry["rows"].append(row)

    def _family_rank(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int, int]:
        stats = item[1]
        rss_bytes = int(stats["rss_bytes"])
        cpu_ticks = int(stats["cpu_ticks"])
        rss_share = (rss_bytes / float(sampled_rss_total)) if sampled_rss_total > 0 else 0.0
        cpu_share = (cpu_ticks / float(sampled_cpu_total)) if sampled_cpu_total > 0 else 0.0
        if focus == "cpu":
            return (cpu_share, rss_share, cpu_ticks, rss_bytes)
        return (rss_share, cpu_share, rss_bytes, cpu_ticks)

    families = sorted(family_totals.items(), key=_family_rank, reverse=True)

    top_memory = rss_ranked[:3]
    top_cpu = cpu_ranked[:3]

    primary_family_key: str | None = None
    primary_family_label: str | None = None
    secondary_family_label: str | None = None
    if families:
        primary_family_key = families[0][0]
        primary_family_label = _family_label(primary_family_key)
        if len(families) > 1:
            secondary_family_label = _family_label(families[1][0])

    dominant_process = top_cpu[0] if focus == "cpu" and top_cpu else (top_memory[0] if top_memory else (top_cpu[0] if top_cpu else None))
    dominant_label = None
    if dominant_process:
        _, dominant_process_family = _detect_family(dominant_process["name"])
        dominant_label = dominant_process_family or "an unmanaged background process"

    cause_bits: list[str] = []
    if families:
        primary_family = families[0][1]
        primary_rows = sorted(primary_family["rows"], key=lambda row: (row["rss_bytes"], row["cpu_ticks"]), reverse=True)
        primary_rss_gib = _gib(int(primary_family["rss_bytes"]))
        primary_cpu_ticks = int(primary_family["cpu_ticks"])
        family_phrase = _family_phrasing(primary_family_key, len(primary_rows))
        if primary_family_key == "browser" and len(primary_rows) > 1:
            cause = f"{family_phrase} looks like the biggest sampled consumer"
        elif primary_family_key == "llm":
            cause = f"{family_phrase} looks like the biggest sampled consumer"
        elif primary_family_key == "vm":
            cause = f"{family_phrase} looks like the biggest sampled consumer"
        elif primary_family_key == "shell":
            cause = f"{family_phrase} looks like the biggest sampled consumer"
        elif primary_family_key == "database":
            cause = f"{family_phrase} looks like the biggest sampled consumer"
        elif primary_family_key == "daemon":
            cause = f"{family_phrase} looks like the biggest sampled consumer"
        elif primary_family_key == "job":
            cause = f"{family_phrase} looks like the biggest sampled consumer"
        else:
            cause = f"{family_phrase} called {primary_rows[0]['name']} looks like the biggest sampled consumer"
        top_names = [row["name"] for row in primary_rows[:2]]
        if top_names:
            cause += f" ({', '.join(top_names)})"
        if primary_rss_gib >= 4.0 and focus == "memory":
            cause += f", and it is holding about {primary_rss_gib:.1f} GiB"
        if focus == "cpu" and primary_cpu_ticks > 0:
            cause += f"; it is also the main CPU contributor in this sample"
        role_note = _family_role_note(primary_family_key, primary_rows)
        if role_note:
            cause += f" {role_note}"
        nothing_open_note = _background_process_note(text, primary_family_key)
        if nothing_open_note:
            cause += f". {nothing_open_note}"
        cause_bits.append(cause)
    elif dominant_process:
        cause = f"{dominant_label} called {dominant_process['name']} looks like the biggest sampled consumer"
        nothing_open_note = _background_process_note(text, None)
        if nothing_open_note:
            cause += f". {nothing_open_note}"
        cause_bits.append(cause)
    else:
        cause_bits.append("no single process stands out as the main culprit")

    if len(families) > 1:
        second_label = _family_label(families[1][0]) or families[1][0]
        if second_label and second_label not in cause_bits[0]:
            cause_bits[0] += f"; {second_label} is the next notable contributor"

    if focus == "cpu" and cpu_count > 0 and load_1m >= float(cpu_count):
        normality = "This looks concerning if you did not expect that workload; the system is busy enough to keep all CPUs occupied."
    elif focus == "cpu" and top_cpu:
        if len(top_cpu) > 1 and int(top_cpu[0]["cpu_ticks"]) >= int(top_cpu[1]["cpu_ticks"]) * 2:
            normality = "This looks busy, but not necessarily broken; one process appears to be doing most of the work."
        else:
            normality = "This looks busy, but the probe does not show a clear crash or runaway service."
    elif bool(memory_pressure.get("is_normal")) and float(memory_pressure.get("available_pct") or 0.0) >= 0.25:
        if used_pct >= 70.0:
            normality = "This still looks normal and not under pressure because plenty of RAM is available and Linux can keep cache in memory."
        else:
            normality = "This looks normal and not under pressure because you still have plenty of available memory."
        if memory_note:
            normality += " Some of the used RAM is reclaimable cache/buffers/shared memory."
    elif bool(memory_pressure.get("is_pressure")):
        normality = "This looks concerning because available memory is getting tight and the system may be under pressure."
    else:
        normality = "This looks broadly normal, but it depends on whether you expected those background processes."

    evidence_parts: list[str] = []
    if total > 0:
        evidence_parts.append(
            f"live probe shows { _gib(used):.1f} GiB used of { _gib(total):.1f} GiB total with { _gib(available):.1f} GiB available"
        )
    if focus == "cpu":
        evidence_parts.append(f"1m load is {load_1m:.2f} (5m {load_5m:.2f}, 15m {load_15m:.2f})")
        if cpu_count > 0:
            evidence_parts.append(f"the machine reports {cpu_count} CPU cores")
    if top_memory:
        evidence_parts.append(
            "top memory processes: "
            + ", ".join(
                f"{row['name']} ({_gib(int(row['rss_bytes'])):.1f} GiB RSS)" for row in top_memory
            )
        )
    if top_cpu and focus != "memory":
        evidence_parts.append(
            "top CPU processes: "
            + ", ".join(f"{row['name']} ({row['cpu_ticks']} ticks)" for row in top_cpu)
        )
    elif top_cpu and focus == "memory":
        evidence_parts.append(
            "top CPU processes: "
            + ", ".join(f"{row['name']} ({row['cpu_ticks']} ticks)" for row in top_cpu)
        )
    if memory_note:
        evidence_parts.append(memory_note)
    if swap_total > 0:
        evidence_parts.append(f"swap is { _gib(swap_used):.1f} GiB used of { _gib(swap_total):.1f} GiB")

    safe_action = _safe_action_for_family(primary_family_key, focus)

    summary = (
        f"Likely cause: {cause_bits[0]}. "
        f"Normality: {normality} "
        f"Evidence: {'; '.join(evidence_parts)}. "
        f"Safe next action: {safe_action}"
    )

    followups = ["Show only the biggest memory users", "Show only CPU deltas"]
    if primary_family_key == "browser":
        followups = ["Show browser tabs/processes only", "Show only the biggest memory users"]
    elif primary_family_key == "vm":
        followups = ["Show only the biggest memory users", "Show only CPU deltas"]
    elif primary_family_key == "llm":
        followups = ["Show Ollama/model-server processes only", "Show only the biggest memory users"]
    elif primary_family_key == "shell":
        followups = ["Show desktop shell processes only", "Show only CPU deltas"]
    elif primary_family_key == "database":
        followups = ["Show database/background daemons only", "Show only the biggest memory users"]
    elif primary_family_key == "job":
        followups = ["Show long-running jobs only", "Show only CPU deltas"]
    elif not families:
        followups = ["Retry the live memory probe", "Show hardware inventory"]

    return {
        "focus": focus,
        "summary": summary,
        "normality": normality,
        "cause": cause_bits[0],
        "evidence": evidence_parts,
        "safe_action": safe_action,
        "followups": followups,
        "primary_family": primary_family_key,
        "primary_family_label": primary_family_label,
        "secondary_family_label": secondary_family_label,
        "dominant_process": dominant_process,
        "memory": {
            "total": total,
            "used": used,
            "available": available,
            "free": free,
            "used_pct": used_pct,
        },
        "cpu": {
            "load_1m": load_1m,
            "load_5m": load_5m,
            "load_15m": load_15m,
            "cpu_count": cpu_count,
        },
        "top_memory": top_memory,
        "top_cpu": top_cpu,
        "memory_note": memory_note,
    }
