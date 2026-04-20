from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from agent.config import runtime_port, runtime_service_name
from agent.diagnostics import run_command
from agent.onboarding_contract import (
    ONBOARDING_DEGRADED,
    ONBOARDING_SERVICES_DOWN,
    ONBOARDING_READY,
    detect_onboarding_state,
    onboarding_next_action,
    onboarding_steps,
    onboarding_summary,
)
from agent.recovery_contract import (
    detect_recovery_mode,
    recovery_next_action,
    recovery_summary,
)
from agent.telegram_runtime_state import get_telegram_runtime_state
from agent.telegram_runtime_state import telegram_control_env


_DEFAULT_API_BASE_URL = f"http://127.0.0.1:{runtime_port()}"


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _telegram_enabled(ready_payload: Mapping[str, Any] | None = None) -> bool:
    payload = ready_payload if isinstance(ready_payload, Mapping) else {}
    telegram = payload.get("telegram") if isinstance(payload.get("telegram"), Mapping) else {}
    raw = telegram.get("enabled") if isinstance(telegram, Mapping) else None
    if isinstance(raw, bool):
        return raw
    normalized = str(raw or "").strip().lower()
    if normalized in {"0", "false", "off", "no"}:
        return False
    if normalized in {"1", "true", "on", "yes"}:
        return True
    try:
        return bool(get_telegram_runtime_state(env=telegram_control_env()).get("enabled", False))
    except Exception:
        return False


def _trace_id() -> str:
    return f"setup-{int(time.time())}-{os.getpid()}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api_get_json(*, base_url: str, path: str, timeout_seconds: float = 1.0) -> tuple[bool, dict[str, Any] | str]:
    url = f"{str(base_url or _DEFAULT_API_BASE_URL).rstrip('/')}{path}"
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            parsed = json.loads(exc.read().decode("utf-8") or "{}")
            if isinstance(parsed, dict):
                return True, parsed
        except Exception:
            pass
        return False, f"http_{int(exc.code)}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}:{exc}"
    try:
        parsed = json.loads(raw or "{}")
    except Exception as exc:
        return False, f"json_error:{exc.__class__.__name__}"
    if isinstance(parsed, dict):
        return True, parsed
    return False, "non_object_json"


@dataclass(frozen=True)
class SetupWizardResult:
    trace_id: str
    generated_at: str
    onboarding_state: str
    recovery_mode: str
    summary: str
    why: str
    next_action: str
    steps: list[str]
    suggestions: list[str]
    dry_run: bool
    api_reachable: bool
    diagnosis_source: str = "unknown"
    raw_check_result: dict[str, Any] = field(default_factory=dict)
    mapped_state: str = ""
    diagnosis_confidence: str = "inferred"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "generated_at": self.generated_at,
            "onboarding_state": self.onboarding_state,
            "recovery_mode": self.recovery_mode,
            "summary": self.summary,
            "why": self.why,
            "next_action": self.next_action,
            "steps": list(self.steps),
            "suggestions": list(self.suggestions),
            "dry_run": bool(self.dry_run),
            "api_reachable": bool(self.api_reachable),
            "diagnosis_source": str(self.diagnosis_source or "unknown"),
            "raw_check_result": dict(self.raw_check_result),
            "mapped_state": str(self.mapped_state or self.onboarding_state),
            "diagnosis_confidence": str(self.diagnosis_confidence or "inferred"),
        }


def _parse_systemctl_show(output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in str(output or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[str(key).strip()] = str(value).strip()
    return data


def probe_api_service_state() -> dict[str, Any]:
    result = run_command(
        [
            "systemctl",
            "--user",
            "show",
            runtime_service_name(),
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "UnitFileState",
        ],
        timeout_s=1.0,
    )
    if result.permission_denied:
        return {
            "checked": False,
            "available": False,
            "active_state": "permission_denied",
            "confidence": "unavailable",
        }
    if result.not_available or result.error:
        return {
            "checked": False,
            "available": False,
            "active_state": "unknown",
            "confidence": "unavailable",
        }
    info = _parse_systemctl_show(result.stdout)
    active_state = _norm(info.get("ActiveState"))
    return {
        "checked": True,
        "available": True,
        "active_state": active_state or "unknown",
        "sub_state": _norm(info.get("SubState")) or None,
        "unit_file_state": _norm(info.get("UnitFileState")) or None,
        "confidence": "confirmed" if active_state in {"active", "inactive", "failed"} else "uncertain",
    }


def _diagnose_api_health(
    *,
    ready_payload: Mapping[str, Any] | None,
    llm_status: Mapping[str, Any] | None,
    startup_report: Mapping[str, Any] | None,
    api_reachable: bool | None,
    transport_error: str | None,
    api_service_state: Mapping[str, Any] | None,
    diagnosis_source: str | None,
) -> dict[str, Any]:
    ready = ready_payload if isinstance(ready_payload, Mapping) else {}
    status = llm_status if isinstance(llm_status, Mapping) else {}
    startup = startup_report if isinstance(startup_report, Mapping) else {}
    service = api_service_state if isinstance(api_service_state, Mapping) else {}
    runtime_status = ready.get("runtime_status") if isinstance(ready.get("runtime_status"), Mapping) else {}
    runtime_mode = _norm(runtime_status.get("runtime_mode") or ready.get("runtime_mode"))
    ready_flag = bool(ready.get("ready", False))
    has_live_truth = bool(ready or status or startup)
    service_active_state = _norm(service.get("active_state"))
    raw_check_result = {
        "api_reachable": api_reachable,
        "transport_error": str(transport_error or "").strip() or None,
        "ready": ready_flag if ready else None,
        "phase": _norm(ready.get("startup_phase") or ready.get("phase")) or None,
        "runtime_mode": runtime_mode or None,
        "service_active_state": service_active_state or None,
        "service_checked": bool(service.get("checked", False)),
    }
    if ready_flag and runtime_mode == "ready":
        return {
            "source": str(diagnosis_source or "runtime_truth"),
            "status": "healthy",
            "confidence": "confirmed",
            "force_state": None,
            "recovery_api_reachable": True,
            "raw_check_result": raw_check_result,
        }
    if api_reachable is True:
        return {
            "source": str(diagnosis_source or "ready_probe"),
            "status": "reachable",
            "confidence": "confirmed",
            "force_state": None,
            "recovery_api_reachable": True,
            "raw_check_result": raw_check_result,
        }
    if service_active_state in {"inactive", "failed"}:
        return {
            "source": "service_check",
            "status": "down",
            "confidence": "confirmed",
            "force_state": ONBOARDING_SERVICES_DOWN,
            "recovery_api_reachable": False,
            "raw_check_result": raw_check_result,
        }
    if api_reachable is False and has_live_truth:
        return {
            "source": str(diagnosis_source or "runtime_truth"),
            "status": "healthy",
            "confidence": "confirmed",
            "force_state": None,
            "recovery_api_reachable": True,
            "raw_check_result": raw_check_result,
        }
    if service_active_state == "active":
        return {
            "source": "service_check",
            "status": "uncertain",
            "confidence": "uncertain",
            "force_state": ONBOARDING_DEGRADED if not has_live_truth else None,
            "recovery_api_reachable": None,
            "raw_check_result": raw_check_result,
        }
    if api_reachable is False:
        return {
            "source": str(diagnosis_source or "ready_probe"),
            "status": "uncertain",
            "confidence": "uncertain",
            "force_state": ONBOARDING_DEGRADED if not has_live_truth else None,
            "recovery_api_reachable": None,
            "raw_check_result": raw_check_result,
        }
    return {
        "source": str(diagnosis_source or "runtime_truth"),
        "status": "unknown",
        "confidence": "inferred",
        "force_state": None,
        "recovery_api_reachable": api_reachable,
        "raw_check_result": raw_check_result,
    }


def _state_why(
    *,
    onboarding_state: str,
    recovery_mode: str,
    ready_payload: Mapping[str, Any] | None = None,
    llm_status: Mapping[str, Any] | None = None,
    transport_error: str | None = None,
    api_diagnosis: Mapping[str, Any] | None = None,
) -> str:
    diagnosis = api_diagnosis if isinstance(api_diagnosis, Mapping) else {}
    diagnosis_status = _norm(diagnosis.get("status"))
    raw_check_result = diagnosis.get("raw_check_result") if isinstance(diagnosis.get("raw_check_result"), Mapping) else {}
    service_active_state = _norm(raw_check_result.get("service_active_state"))
    if diagnosis_status == "down":
        return "API service is down."
    if diagnosis_status == "uncertain":
        if service_active_state == "active":
            return "The API may be restarting or temporarily unavailable."
        return "I couldn't confirm API health from the setup check."
    if transport_error and diagnosis_status != "down":
        return "I couldn't confirm API health from the setup check."
    if onboarding_state == ONBOARDING_READY:
        return "All required services and chat model checks are healthy."
    status = llm_status if isinstance(llm_status, Mapping) else {}
    model = str(
        status.get("resolved_default_model")
        or status.get("chat_model")
        or status.get("default_model")
        or ""
    ).strip()
    if onboarding_state == "TOKEN_MISSING":
        return "Telegram token is missing from secret store or environment."
    if onboarding_state == "LLM_MISSING":
        if model:
            return f"Chat model {model} is not healthy."
        return "No chat model available right now."
    if onboarding_state == "SERVICES_DOWN":
        phase = _norm((ready_payload or {}).get("startup_phase") or (ready_payload or {}).get("phase"))
        if phase:
            return f"Runtime phase is {phase}."
        return recovery_summary(recovery_mode)
    if onboarding_state == "DEGRADED":
        return recovery_summary(recovery_mode)
    return "Setup has not been completed yet."


def _safe_suggestions(
    *,
    onboarding_state: str,
    recovery_mode: str,
    telegram_enabled: bool,
) -> list[str]:
    if onboarding_state == "TOKEN_MISSING":
        return [
            "python -m agent.secrets set telegram:bot_token",
            "systemctl --user restart personal-agent-telegram.service",
            "python -m agent status",
        ]
    if onboarding_state == "SERVICES_DOWN":
        suggestions = [
            f"systemctl --user restart {runtime_service_name()}",
        ]
        if telegram_enabled:
            suggestions.append("systemctl --user restart personal-agent-telegram.service")
        suggestions.append("python -m agent status")
        return suggestions
    if onboarding_state == "LLM_MISSING":
        return [
            "python -m agent setup --dry-run",
            "python -m agent doctor",
            "python -m agent status",
        ]
    if onboarding_state == "DEGRADED":
        if recovery_mode == "LOCK_CONFLICT":
            return [
                "Stop duplicate Telegram pollers",
                "systemctl --user restart personal-agent-telegram.service",
                "python -m agent doctor",
            ]
        return [
            "python -m agent doctor",
            "python -m agent setup --dry-run",
            "python -m agent status",
        ]
    if onboarding_state == "NOT_STARTED":
        return [
            "python -m agent setup --dry-run",
            "python -m agent doctor",
            "python -m agent status",
        ]
    return []


def build_setup_result(
    *,
    ready_payload: Mapping[str, Any] | None = None,
    llm_status: Mapping[str, Any] | None = None,
    startup_report: Mapping[str, Any] | None = None,
    api_reachable: bool | None = None,
    transport_error: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
    api_service_state: Mapping[str, Any] | None = None,
    service_probe_fn: Callable[[], Mapping[str, Any]] | None = None,
    diagnosis_source: str | None = None,
) -> SetupWizardResult:
    service_state = api_service_state if isinstance(api_service_state, Mapping) else {}
    if api_reachable is False and not service_state:
        try:
            probed = (service_probe_fn or probe_api_service_state)()
            service_state = probed if isinstance(probed, Mapping) else {}
        except Exception:
            service_state = {}
    telegram_enabled = _telegram_enabled(ready_payload)
    onboarding_state = detect_onboarding_state(
        ready_payload=ready_payload,
        llm_status=llm_status,
        startup_report=startup_report,
    )
    api_diagnosis = _diagnose_api_health(
        ready_payload=ready_payload,
        llm_status=llm_status,
        startup_report=startup_report,
        api_reachable=api_reachable,
        transport_error=transport_error,
        api_service_state=service_state,
        diagnosis_source=diagnosis_source,
    )
    forced_state = str(api_diagnosis.get("force_state") or "").strip().upper()
    if forced_state:
        onboarding_state = forced_state
    recovery_mode = detect_recovery_mode(
        ready_payload=ready_payload,
        llm_status=llm_status,
        api_reachable=api_diagnosis.get("recovery_api_reachable"),
    )
    if str(api_diagnosis.get("status") or "").strip().lower() == "uncertain":
        recovery_mode = "UNKNOWN_FAILURE"
    next_action = onboarding_next_action(onboarding_state, ready_payload=ready_payload)
    if str(api_diagnosis.get("status") or "").strip().lower() == "down":
        next_action = recovery_next_action(recovery_mode)
    if onboarding_state == "DEGRADED":
        next_action = recovery_next_action(recovery_mode)
        if str(api_diagnosis.get("status") or "").strip().lower() == "uncertain":
            next_action = "Run: python -m agent status"
    summary = onboarding_summary(
        onboarding_state,
        ready_payload=ready_payload,
        llm_status=llm_status,
    )
    why = _state_why(
        onboarding_state=onboarding_state,
        recovery_mode=recovery_mode,
        ready_payload=ready_payload,
        llm_status=llm_status,
        transport_error=transport_error,
        api_diagnosis=api_diagnosis,
    )
    steps = onboarding_steps(onboarding_state)
    suggestions = _safe_suggestions(
        onboarding_state=onboarding_state,
        recovery_mode=recovery_mode,
        telegram_enabled=telegram_enabled,
    )
    if onboarding_state == ONBOARDING_READY:
        suggestions = []
    return SetupWizardResult(
        trace_id=str(trace_id or _trace_id()),
        generated_at=_now_iso(),
        onboarding_state=onboarding_state,
        recovery_mode=recovery_mode,
        summary=summary,
        why=why,
        next_action=next_action,
        steps=steps,
        suggestions=suggestions,
        dry_run=bool(dry_run),
        api_reachable=bool(api_diagnosis.get("recovery_api_reachable") is not False),
        diagnosis_source=str(api_diagnosis.get("source") or diagnosis_source or "unknown"),
        raw_check_result=dict(
            api_diagnosis.get("raw_check_result")
            if isinstance(api_diagnosis.get("raw_check_result"), Mapping)
            else {}
        ),
        mapped_state=onboarding_state,
        diagnosis_confidence=str(api_diagnosis.get("confidence") or "inferred"),
    )


def run_setup_wizard(
    *,
    api_base_url: str = _DEFAULT_API_BASE_URL,
    dry_run: bool = False,
    fetch_json: Callable[..., tuple[bool, dict[str, Any] | str]] | None = None,
    service_probe_fn: Callable[[], Mapping[str, Any]] | None = None,
) -> SetupWizardResult:
    fetch = fetch_json or _api_get_json
    ready_ok, ready_payload_or_error = fetch(base_url=api_base_url, path="/ready", timeout_seconds=1.0)
    ready_payload = dict(ready_payload_or_error) if ready_ok and isinstance(ready_payload_or_error, Mapping) else {}
    try:
        telegram_state = get_telegram_runtime_state(env=telegram_control_env())
    except Exception:
        telegram_state = {}
    if telegram_state:
        ready_payload["telegram"] = {
            "enabled": bool(telegram_state.get("enabled", False)),
            "configured": bool(telegram_state.get("token_configured", False)),
            "state": str(telegram_state.get("ready_state") or "disabled_optional"),
        }
        effective_state = str(telegram_state.get("effective_state") or "")
        if effective_state == "enabled_blocked_by_lock":
            ready_payload["failure_code"] = "lock_conflict"
            ready_payload["ready"] = False
        elif effective_state == "enabled_stopped":
            ready_payload["failure_code"] = "service_down"
            ready_payload["ready"] = False
        elif effective_state == "enabled_misconfigured":
            ready_payload["failure_code"] = "missing_token" if not bool(telegram_state.get("token_configured", False)) else "service_down"
            ready_payload["ready"] = False
    llm_status: Mapping[str, Any] = (
        ready_payload.get("llm") if isinstance(ready_payload.get("llm"), Mapping) else {}
    )
    if ready_ok and not llm_status:
        status_ok, status_payload_or_error = fetch(base_url=api_base_url, path="/llm/status", timeout_seconds=1.0)
        if status_ok and isinstance(status_payload_or_error, Mapping):
            llm_status = status_payload_or_error
        else:
            llm_status = {}
    return build_setup_result(
        ready_payload=ready_payload,
        llm_status=llm_status,
        api_reachable=bool(ready_ok),
        transport_error=(None if ready_ok else str(ready_payload_or_error)),
        dry_run=bool(dry_run),
        service_probe_fn=service_probe_fn,
        diagnosis_source="ready_probe",
    )


def render_setup_text(result: SetupWizardResult) -> str:
    lines = [
        f"1) State: {result.onboarding_state}",
        f"2) Why: {result.why}",
        f"3) Next action: {result.next_action}",
    ]
    if result.suggestions:
        lines.append("4) Safe suggestions:")
        for index, row in enumerate(result.suggestions, start=1):
            lines.append(f"   {index}. {row}")
    else:
        lines.append("4) Safe suggestions: none")
    if result.dry_run:
        lines.append("Dry-run: no changes were applied.")
    lines.append(f"trace_id: {result.trace_id}")
    return "\n".join(lines)


def render_telegram_setup_text(result: SetupWizardResult) -> str:
    if result.onboarding_state == ONBOARDING_READY:
        return "Setup is complete. You can chat normally. Send 'help' for commands."
    return "\n".join(
        [
            f"Setup state: {result.onboarding_state.replace('_', ' ').lower()}",
            result.why,
            f"Next: {result.next_action}",
        ]
    )


__all__ = [
    "SetupWizardResult",
    "build_setup_result",
    "render_setup_text",
    "render_telegram_setup_text",
    "run_setup_wizard",
]
