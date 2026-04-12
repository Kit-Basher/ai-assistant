from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
from typing import Any

from agent.bootstrap.routes import METHOD_ORDER, extract_routes_from_api_server
from agent.perception import collect_snapshot


@dataclass(frozen=True)
class BootstrapSnapshot:
    created_at_ts: int
    os: dict[str, Any]
    hardware: dict[str, Any]
    interfaces: dict[str, Any]
    providers: dict[str, Any]
    capsules: dict[str, Any]
    routes: dict[str, Any]
    notes: list[str]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _parse_os_release(path: str = "/etc/os-release") -> dict[str, str]:
    output: dict[str, str] = {}
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return output
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key_norm = str(key).strip()
        value_norm = str(value).strip().strip('"')
        if key_norm and value_norm:
            output[key_norm] = value_norm
    return dict(sorted(output.items(), key=lambda item: item[0]))


def _collect_routes() -> dict[str, Any]:
    api_server_path = Path(__file__).resolve().parents[1] / "api_server.py"
    routes_by_method = extract_routes_from_api_server(api_server_path)
    counts = {method: len(routes_by_method.get(method, [])) for method in METHOD_ORDER}
    total = sum(counts.values())
    return {
        "methods": {method: list(routes_by_method.get(method, [])) for method in METHOD_ORDER},
        "counts": counts,
        "total": int(total),
    }


def collect_bootstrap_snapshot(runtime: Any, *, now_ts: int | None = None) -> BootstrapSnapshot:
    created_at = int(now_ts if now_ts is not None else _safe_int(datetime.now(timezone.utc).timestamp(), 0))
    roots = list(getattr(runtime.config, "perception_roots", ()) or ())
    perception_snapshot = collect_snapshot(roots)

    os_release = _parse_os_release()
    os_payload = {
        "name": os_release.get("NAME") or platform.system(),
        "version": os_release.get("VERSION") or platform.version(),
        "pretty_name": os_release.get("PRETTY_NAME") or platform.platform(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "hostname": platform.node(),
        "os_release": os_release,
    }

    cpu_payload = perception_snapshot.get("cpu") if isinstance(perception_snapshot.get("cpu"), dict) else {}
    mem_payload = perception_snapshot.get("memory") if isinstance(perception_snapshot.get("memory"), dict) else {}
    gpu_payload = perception_snapshot.get("gpu") if isinstance(perception_snapshot.get("gpu"), dict) else {}
    hardware_payload = {
        "cpu_count_logical": int(os.cpu_count() or 0),
        "cpu_freq_mhz": float(cpu_payload.get("freq_mhz") or 0.0),
        "cpu_load_1m": float((cpu_payload.get("load_avg") or {}).get("1m") or 0.0),
        "mem_total_bytes": _safe_int(mem_payload.get("total"), 0),
        "swap_total_bytes": _safe_int(mem_payload.get("swap_total"), 0),
        "gpu": {
            "available": bool(gpu_payload.get("available")),
            "memory_total_mb": _safe_int(gpu_payload.get("memory_total_mb"), 0),
            "usage_pct": float(gpu_payload.get("usage_pct") or 0.0),
            "error": str(gpu_payload.get("error") or "") or None,
        },
    }

    telegram_status = runtime.telegram_status() if callable(getattr(runtime, "telegram_status", None)) else {"configured": False}
    interfaces_payload = {
        "api": {
            "listening": str(getattr(runtime, "listening_url", "") or ""),
        },
        "memory_v2_enabled": bool(getattr(runtime.config, "memory_v2_enabled", False)),
        "model_watch_enabled": bool(getattr(runtime.config, "model_watch_enabled", False)),
        "llm_automation_enabled": bool(getattr(runtime.config, "llm_automation_enabled", False)),
        "telegram_configured": bool((telegram_status or {}).get("configured")),
        "webui_dev_proxy": bool(getattr(runtime, "webui_dev_proxy", False)),
    }

    providers_payload = runtime.list_providers() if callable(getattr(runtime, "list_providers", None)) else {"providers": []}
    defaults_payload = runtime.get_defaults() if callable(getattr(runtime, "get_defaults", None)) else {}
    provider_rows = providers_payload.get("providers") if isinstance(providers_payload, dict) else []
    if not isinstance(provider_rows, list):
        provider_rows = []

    provider_summary: list[dict[str, Any]] = []
    enabled_provider_ids: list[str] = []
    for row in sorted(provider_rows, key=lambda item: str(item.get("id") or "")):
        if not isinstance(row, dict):
            continue
        provider_id = str(row.get("id") or "").strip().lower()
        if not provider_id:
            continue
        enabled = bool(row.get("enabled", True))
        if enabled:
            enabled_provider_ids.append(provider_id)
        health = row.get("health") if isinstance(row.get("health"), dict) else {}
        provider_summary.append(
            {
                "id": provider_id,
                "enabled": enabled,
                "local": bool(row.get("local", False)),
                "health": {
                    "status": str(health.get("status") or "unknown"),
                    "last_error_kind": str(health.get("last_error_kind") or "") or None,
                    "status_code": _safe_int(health.get("status_code"), 0) or None,
                },
            }
        )
    providers_struct = {
        "enabled_ids": sorted(enabled_provider_ids),
        "rows": provider_summary,
        "defaults": {
            "default_provider": str(defaults_payload.get("default_provider") or "") or None,
            "default_model": str(defaults_payload.get("default_model") or "") or None,
            "routing_mode": str(defaults_payload.get("routing_mode") or "") or None,
        },
    }

    capsules_dir = Path(__file__).resolve().parents[1] / "capsules"
    installed_capsules: list[str] = []
    if capsules_dir.is_dir():
        installed_capsules = sorted(
            item.name
            for item in capsules_dir.iterdir()
            if item.is_dir() and not item.name.startswith("__")
        )
    capsules_payload = {
        "installed": installed_capsules,
    }

    routes_payload = _collect_routes()

    notes: list[str] = []
    if not hardware_payload["gpu"]["available"]:
        notes.append("gpu_unavailable")
    if not interfaces_payload["telegram_configured"]:
        notes.append("telegram_not_configured")
    if "ollama" in providers_struct["enabled_ids"]:
        ollama = next((row for row in provider_summary if row.get("id") == "ollama"), None)
        ollama_health = ollama.get("health") if isinstance(ollama, dict) else {}
        if str((ollama_health or {}).get("status") or "").lower() == "down":
            notes.append("ollama_provider_down")

    return BootstrapSnapshot(
        created_at_ts=created_at,
        os=os_payload,
        hardware=hardware_payload,
        interfaces=interfaces_payload,
        providers=providers_struct,
        capsules=capsules_payload,
        routes=routes_payload,
        notes=sorted(set(notes)),
    )


def snapshot_to_dict(snapshot: BootstrapSnapshot) -> dict[str, Any]:
    return asdict(snapshot)
