from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from agent.onboarding_contract import (
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


_DEFAULT_API_BASE_URL = "http://127.0.0.1:8765"


def _norm(value: Any) -> str:
    return str(value or "").strip()


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
        }


def _state_why(
    *,
    onboarding_state: str,
    recovery_mode: str,
    ready_payload: Mapping[str, Any] | None = None,
    llm_status: Mapping[str, Any] | None = None,
    transport_error: str | None = None,
) -> str:
    if transport_error:
        return f"API call failed: {transport_error}"
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
        phase = _norm((ready_payload or {}).get("phase"))
        if phase:
            return f"Runtime phase is {phase}."
        return recovery_summary(recovery_mode)
    if onboarding_state == "DEGRADED":
        return recovery_summary(recovery_mode)
    return "Setup has not been completed yet."


def _safe_suggestions(*, onboarding_state: str, recovery_mode: str) -> list[str]:
    if onboarding_state == "TOKEN_MISSING":
        return [
            "python -m agent.secrets set telegram:bot_token",
            "systemctl --user restart personal-agent-telegram.service",
            "python -m agent status",
        ]
    if onboarding_state == "SERVICES_DOWN":
        return [
            "systemctl --user restart personal-agent-api.service",
            "systemctl --user restart personal-agent-telegram.service",
            "python -m agent status",
        ]
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
) -> SetupWizardResult:
    onboarding_state = detect_onboarding_state(
        ready_payload=ready_payload,
        llm_status=llm_status,
        startup_report=startup_report,
    )
    if api_reachable is False and onboarding_state == "NOT_STARTED":
        onboarding_state = "SERVICES_DOWN"
    recovery_mode = detect_recovery_mode(
        ready_payload=ready_payload,
        llm_status=llm_status,
        api_reachable=api_reachable,
    )
    next_action = onboarding_next_action(onboarding_state, ready_payload=ready_payload)
    if onboarding_state == "DEGRADED":
        next_action = recovery_next_action(recovery_mode)
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
    )
    steps = onboarding_steps(onboarding_state)
    suggestions = _safe_suggestions(onboarding_state=onboarding_state, recovery_mode=recovery_mode)
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
        api_reachable=bool(api_reachable is not False),
    )


def run_setup_wizard(
    *,
    api_base_url: str = _DEFAULT_API_BASE_URL,
    dry_run: bool = False,
    fetch_json: Callable[..., tuple[bool, dict[str, Any] | str]] | None = None,
) -> SetupWizardResult:
    fetch = fetch_json or _api_get_json
    ready_ok, ready_payload_or_error = fetch(base_url=api_base_url, path="/ready", timeout_seconds=1.0)
    ready_payload = ready_payload_or_error if ready_ok and isinstance(ready_payload_or_error, Mapping) else {}
    llm_status: Mapping[str, Any] = {}
    if ready_ok:
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
