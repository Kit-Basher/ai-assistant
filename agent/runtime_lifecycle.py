from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence


class RuntimeLifecyclePhase(str, Enum):
    BOOT = "boot"
    WARMUP = "warmup"
    READY = "ready"
    DEGRADED = "degraded"
    RECOVERING = "recovering"


def _health_status(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("status") or "").strip().lower()


def derive_runtime_lifecycle_phase(
    *,
    startup_phase: str | None,
    warmup_remaining: Sequence[str] | None,
    startup_warmup_started: bool,
    runtime_ready: bool,
    runtime_mode: str | None,
    active_provider_health: Mapping[str, Any] | None = None,
    active_model_health: Mapping[str, Any] | None = None,
    previous_phase: RuntimeLifecyclePhase | str | None = None,
) -> RuntimeLifecyclePhase:
    startup = str(startup_phase or "").strip().lower()
    remaining = [str(item).strip() for item in (warmup_remaining or []) if str(item).strip()]
    previous = str(previous_phase.value if isinstance(previous_phase, RuntimeLifecyclePhase) else previous_phase or "").strip().lower()
    mode = str(runtime_mode or "").strip().upper()
    provider_status = _health_status(active_provider_health)
    model_status = _health_status(active_model_health)

    if startup == "starting" and not startup_warmup_started and not remaining:
        return RuntimeLifecyclePhase.BOOT
    if startup in {"starting", "listening", "warming"} or remaining:
        return RuntimeLifecyclePhase.WARMUP

    unhealthy = bool(
        not runtime_ready
        or startup == "degraded"
        or mode in {"DEGRADED", "FAILED", "BOOTSTRAP_REQUIRED"}
        or provider_status in {"degraded", "down"}
        or model_status in {"degraded", "down"}
    )
    if unhealthy:
        return RuntimeLifecyclePhase.DEGRADED
    if previous == RuntimeLifecyclePhase.DEGRADED.value:
        return RuntimeLifecyclePhase.RECOVERING
    if previous == RuntimeLifecyclePhase.RECOVERING.value:
        return RuntimeLifecyclePhase.READY
    return RuntimeLifecyclePhase.READY
