from __future__ import annotations

"""Canonical Telegram-to-runtime bridge.

This module owns Telegram-facing product logic while keeping transport mechanics in
`telegram_adapter/bot.py`.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from agent.doctor import run_doctor_report
from agent.error_response_ux import deterministic_error_message
from agent.golden_path import bootstrap_needed
from agent.onboarding_contract import ONBOARDING_READY
from agent.runtime_contract import normalize_user_facing_status
from agent.setup_wizard import (
    SetupWizardResult,
    build_setup_result,
    render_telegram_setup_text,
)

_DEFAULT_API_BASE_URL = "http://127.0.0.1:8765"

_TELEGRAM_HELP_TEXT = (
    "Available commands:\n\n"
    "doctor – diagnostics\n"
    "setup – setup/recovery guidance\n"
    "status – runtime status\n"
    "health – health snapshot\n"
    "brief – system summary\n"
    "memory – what are we doing / resume"
)

_ROTATE_TOKEN_TEXT = (
    "Rotate token:\n"
    "1) python -m agent.secrets set telegram:bot_token\n"
    "2) systemctl --user restart personal-agent-telegram.service"
)


def _safe_text(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "I’m still here. What should I do next?"


def _normalize_user_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _contains_any(normalized_text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in normalized_text for phrase in phrases)


def _format_commit_short(value: str | None) -> str:
    commit = str(value or "").strip()
    if not commit:
        return "unknown"
    return commit[:12]


def _setup_trace_id() -> str:
    return f"tg-setup-{int(time.time())}-{os.getpid()}"


def _api_base_url() -> str:
    configured = str(os.getenv("AGENT_API_BASE_URL") or os.getenv("PERSONAL_AGENT_API_BASE_URL") or "").strip()
    if configured:
        return configured.rstrip("/")
    return _DEFAULT_API_BASE_URL


def _default_fetch_local_api_json(path: str, *, timeout_seconds: float = 0.6) -> dict[str, Any]:
    endpoint = str(path or "").strip()
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    url = f"{_api_base_url()}{endpoint}"
    request = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=float(timeout_seconds)) as response:
            body = response.read()
    except (urllib_error.URLError, TimeoutError, OSError):
        return {}
    except Exception:
        return {}
    try:
        decoded = body.decode("utf-8", errors="replace")
        payload = json.loads(decoded)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _runtime_ready_payload(
    runtime: Any | None,
    *,
    fetch_local_api_json: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if runtime is not None and hasattr(runtime, "ready_status"):
        try:
            row = runtime.ready_status()  # type: ignore[attr-defined]
            if isinstance(row, dict):
                payload = row
        except Exception:
            payload = {}
    if payload:
        return payload
    if fetch_local_api_json is None:
        return {}
    return fetch_local_api_json("/ready")


def _runtime_llm_status_payload(
    runtime: Any | None,
    *,
    fetch_local_api_json: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if runtime is not None and hasattr(runtime, "llm_status"):
        try:
            row = runtime.llm_status()  # type: ignore[attr-defined]
            if isinstance(row, dict):
                payload = row
        except Exception:
            payload = {}
    if payload:
        return payload
    if fetch_local_api_json is None:
        return {}
    return fetch_local_api_json("/llm/status")


def _setup_result_from_runtime(
    runtime: Any | None,
    *,
    fetch_local_api_json: Callable[[str], dict[str, Any]] | None = None,
) -> SetupWizardResult | None:
    ready_payload = _runtime_ready_payload(runtime, fetch_local_api_json=fetch_local_api_json)
    llm_status = _runtime_llm_status_payload(runtime, fetch_local_api_json=fetch_local_api_json)
    onboarding_row = ready_payload.get("onboarding") if isinstance(ready_payload.get("onboarding"), dict) else {}
    onboarding_state = str(onboarding_row.get("state") or "").strip().upper()
    if onboarding_state:
        steps_raw = onboarding_row.get("steps") if isinstance(onboarding_row.get("steps"), list) else []
        steps = [str(item).strip() for item in steps_raw if str(item).strip()]
        recovery_row = ready_payload.get("recovery") if isinstance(ready_payload.get("recovery"), dict) else {}
        why_value = str(onboarding_row.get("summary") or "").strip() or "Setup state reported by runtime."
        return SetupWizardResult(
            trace_id=_setup_trace_id(),
            generated_at=datetime.now(timezone.utc).isoformat(),
            onboarding_state=onboarding_state,
            recovery_mode=str(recovery_row.get("mode") or "UNKNOWN_FAILURE"),
            summary=str(onboarding_row.get("summary") or "").strip() or "Setup state reported by runtime.",
            why=why_value,
            next_action=str(onboarding_row.get("next_action") or "").strip() or "Run: python -m agent setup",
            steps=steps,
            suggestions=[],
            dry_run=True,
            api_reachable=True,
        )
    if ready_payload or llm_status:
        try:
            return build_setup_result(
                ready_payload=ready_payload,
                llm_status=llm_status,
                api_reachable=True,
                dry_run=True,
            )
        except Exception:
            pass
    try:
        # Keep adapter behavior deterministic: when runtime state is absent,
        # return contract-driven setup guidance instead of probing live services.
        return build_setup_result(
            ready_payload={},
            llm_status={},
            api_reachable=False,
            dry_run=True,
        )
    except Exception:
        return None


def build_telegram_error(
    *,
    title: str,
    trace_id: str,
    component: str,
    next_action: str,
    route: str = "error",
) -> dict[str, Any]:
    return {
        "ok": False,
        "text": deterministic_error_message(
            title=title,
            trace_id=trace_id,
            component=component,
            next_action=next_action,
        ),
        "route": route,
        "trace_id": trace_id,
        "next_action": next_action,
    }


def build_telegram_setup(
    *,
    runtime: Any | None,
    trace_id: str,
    fetch_local_api_json: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    setup_result = _setup_result_from_runtime(runtime, fetch_local_api_json=fetch_local_api_json)
    if setup_result is None:
        return build_telegram_error(
            title="❌ Setup status is unavailable",
            trace_id=trace_id,
            component="telegram.setup",
            next_action="run `python -m agent setup --dry-run`",
            route="setup",
        )
    return {
        "ok": True,
        "text": render_telegram_setup_text(setup_result),
        "route": "setup",
        "trace_id": trace_id,
        "next_action": str(setup_result.next_action or "").strip() or None,
    }


def build_telegram_help(
    *,
    runtime: Any | None,
    trace_id: str,
    fetch_local_api_json: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    setup_result = _setup_result_from_runtime(runtime, fetch_local_api_json=fetch_local_api_json)
    if setup_result is None:
        return {
            "ok": True,
            "text": _TELEGRAM_HELP_TEXT,
            "route": "help",
            "trace_id": trace_id,
            "next_action": None,
        }
    if str(setup_result.onboarding_state or "").strip().upper() != ONBOARDING_READY:
        return {
            "ok": True,
            "text": render_telegram_setup_text(setup_result),
            "route": "help",
            "trace_id": trace_id,
            "next_action": str(setup_result.next_action or "").strip() or None,
        }
    return {
        "ok": True,
        "text": _TELEGRAM_HELP_TEXT,
        "route": "help",
        "trace_id": trace_id,
        "next_action": None,
    }


def build_telegram_status(
    *,
    runtime: Any | None,
    trace_id: str,
    runtime_version: str | None = None,
    runtime_git_commit: str | None = None,
    runtime_started_ts: float | int | None = None,
    fetch_local_api_json: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ready_payload = _runtime_ready_payload(runtime, fetch_local_api_json=fetch_local_api_json)
    ready_api = ready_payload.get("api") if isinstance(ready_payload.get("api"), dict) else {}
    llm_status = _runtime_llm_status_payload(runtime, fetch_local_api_json=fetch_local_api_json)

    version = str(
        getattr(runtime, "version", "")
        or ready_api.get("version")
        or runtime_version
        or "0.1.0"
    ).strip() or "0.1.0"
    commit_value = str(
        getattr(runtime, "git_commit", "")
        or ready_api.get("git_commit")
        or runtime_git_commit
        or "unknown"
    ).strip()
    commit_short = _format_commit_short(commit_value)

    started_at = getattr(runtime, "started_at", None)
    uptime_seconds = 0
    if isinstance(started_at, datetime):
        uptime_seconds = max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))
    else:
        try:
            uptime_seconds = max(0, int(ready_api.get("uptime_seconds") or 0))
        except Exception:
            try:
                started_ts = float(runtime_started_ts)
            except Exception:
                started_ts = time.time()
            uptime_seconds = max(0, int(time.time() - started_ts))

    provider = str(llm_status.get("default_provider") or "").strip() or None
    model = (
        str(llm_status.get("resolved_default_model") or "").strip()
        or str(llm_status.get("default_model") or "").strip()
        or None
    )
    provider_state = (
        str(
            ((llm_status.get("active_provider_health") or {}).get("status"))
            if isinstance(llm_status.get("active_provider_health"), dict)
            else ""
        )
        .strip()
        .lower()
    )
    model_state = (
        str(
            ((llm_status.get("active_model_health") or {}).get("status"))
            if isinstance(llm_status.get("active_model_health"), dict)
            else ""
        )
        .strip()
        .lower()
    )
    ready = bool(provider_state == "ok" and model_state == "ok")
    failure_code = None
    if not model:
        failure_code = "no_chat_model"
    elif provider_state != "ok":
        failure_code = "provider_unhealthy"
    elif model_state != "ok":
        failure_code = "model_unhealthy"
    runtime_status = (
        ready_payload.get("runtime_status")
        if isinstance(ready_payload.get("runtime_status"), dict)
        else {}
    )
    if not runtime_status:
        runtime_status = normalize_user_facing_status(
            ready=ready,
            bootstrap_required=bootstrap_needed(llm_status=llm_status),
            failure_code=failure_code,
            provider=provider,
            model=model,
            local_providers={"ollama"},
        )
    summary = str(runtime_status.get("summary") or "").strip() or "Agent is starting or degraded."
    runtime_mode = str(runtime_status.get("runtime_mode") or "DEGRADED").strip().upper() or "DEGRADED"
    telegram_state = (
        str(((ready_payload.get("telegram") or {}).get("state")) if isinstance(ready_payload.get("telegram"), dict) else "")
        .strip()
        .lower()
        or "unknown"
    )
    text = (
        f"✅ Agent is running (v{version}, commit {commit_short}, uptime {uptime_seconds}s).\n"
        f"{summary}\n"
        f"runtime_mode: {runtime_mode}\n"
        f"telegram: {telegram_state}"
    )
    return {
        "ok": True,
        "text": text,
        "route": "status",
        "trace_id": trace_id,
        "next_action": str(runtime_status.get("next_action") or "").strip() or None,
    }


def build_telegram_memory(
    *,
    orchestrator: Any | None,
    chat_id: str,
    trace_id: str,
) -> dict[str, Any]:
    if orchestrator is None:
        return build_telegram_error(
            title="❌ Memory summary is unavailable",
            trace_id=trace_id,
            component="telegram.memory",
            next_action="run `agent doctor`",
            route="memory",
        )
    response = orchestrator.handle_message("/memory", user_id=chat_id)
    return {
        "ok": True,
        "text": _safe_text(getattr(response, "text", "")),
        "route": "memory",
        "trace_id": trace_id,
        "next_action": None,
    }


def _build_telegram_doctor(*, trace_id: str) -> dict[str, Any]:
    report = run_doctor_report(online=False, fix=False)
    pass_count = sum(1 for item in report.checks if item.status == "OK")
    warn_count = sum(1 for item in report.checks if item.status == "WARN")
    fail_count = sum(1 for item in report.checks if item.status == "FAIL")
    next_action = report.next_action or "none"
    return {
        "ok": True,
        "text": (
            f"Doctor: {report.summary_status} (trace {report.trace_id})\\n"
            f"PASS {pass_count} · WARN {warn_count} · FAIL {fail_count}\\n"
            f"Next: {next_action}\\n"
            "Run: python -m agent doctor --json for details."
        ),
        "route": "doctor",
        "trace_id": trace_id,
        "next_action": next_action,
    }


def classify_telegram_text_command(text: str) -> str | None:
    normalized = _normalize_user_text(text)
    if not normalized:
        return None
    if normalized in {"/help", "help", "what can you do", "commands"}:
        return "/help"
    if _contains_any(
        normalized,
        (
            "setup",
            "get started",
            "fix setup",
            "configure openrouter",
            "openrouter setup",
            "repair openrouter",
            "fix openrouter",
            "openrouter down",
            "openrouter broken",
            "openrouter unavailable",
            "configure ollama",
            "ollama setup",
            "model help",
            "provider help",
            "llm help",
            "help with model",
            "help with provider",
            "why isnt this working",
            "why isn't this working",
            "why isn’t this working",
            "what do i do next",
        ),
    ):
        return "/setup"
    if _contains_any(
        normalized,
        (
            "rotate token",
            "rotate telegram token",
            "telegram token rotate",
        ),
    ):
        return "/setup_token"
    if normalized in {"memory", "/memory"}:
        return "/memory"
    if normalized in {"brief", "breif", "/brief", "/breif"}:
        return "/brief"
    if _contains_any(
        normalized,
        (
            "what are we doing",
            "where were we",
            "resume",
            "continue where we left off",
            "continue",
        ),
    ):
        return "/memory"
    if normalized in {"doctor", "fix", "diagnose", "diagnostics", "run doctor"}:
        return "/doctor"
    if _contains_any(
        normalized,
        (
            "health",
            "how is the bot health",
            "bot health",
            "health check",
            "system health",
            "how are you running",
            "running ok",
            "show me the stats",
        ),
    ):
        return "/health"
    if _contains_any(
        normalized,
        (
            "anything new on my pc",
            "what changed on my pc",
            "what changed on my computer",
            "what changed on my system",
            "changed on my pc",
        ),
    ):
        return "/brief"
    if _contains_any(
        normalized,
        (
            "status",
            "state",
            "uptime",
            "agent status",
            "bot status",
        ),
    ):
        return "/status"
    return None


def handle_telegram_command(
    *,
    command: str,
    chat_id: str,
    trace_id: str,
    runtime: Any | None,
    orchestrator: Any | None,
    runtime_version: str | None = None,
    runtime_git_commit: str | None = None,
    runtime_started_ts: float | int | None = None,
    fetch_local_api_json: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = str(command or "").strip().lower()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if normalized == "/help":
        return build_telegram_help(
            runtime=runtime,
            trace_id=trace_id,
            fetch_local_api_json=fetch_local_api_json,
        )
    if normalized == "/setup":
        return build_telegram_setup(
            runtime=runtime,
            trace_id=trace_id,
            fetch_local_api_json=fetch_local_api_json,
        )
    if normalized == "/setup_token":
        return {
            "ok": True,
            "text": _ROTATE_TOKEN_TEXT,
            "route": "setup",
            "trace_id": trace_id,
            "next_action": "systemctl --user restart personal-agent-telegram.service",
        }
    if normalized == "/status":
        return build_telegram_status(
            runtime=runtime,
            trace_id=trace_id,
            runtime_version=runtime_version,
            runtime_git_commit=runtime_git_commit,
            runtime_started_ts=runtime_started_ts,
            fetch_local_api_json=fetch_local_api_json,
        )
    if normalized == "/doctor":
        return _build_telegram_doctor(trace_id=trace_id)

    if normalized in {"/health", "/brief", "/memory"}:
        if orchestrator is None:
            return build_telegram_error(
                title="❌ Runtime unavailable",
                trace_id=trace_id,
                component="telegram.bridge",
                next_action="run `agent doctor`",
                route=normalized.lstrip("/"),
            )
        response = orchestrator.handle_message(normalized, user_id=chat_id)
        return {
            "ok": True,
            "text": _safe_text(getattr(response, "text", "")),
            "route": normalized.lstrip("/"),
            "trace_id": trace_id,
            "next_action": None,
        }

    return {
        "ok": True,
        "handled": False,
        "text": "",
        "route": "chat",
        "trace_id": trace_id,
        "next_action": None,
    }


def handle_telegram_text(
    *,
    text: str,
    chat_id: str,
    trace_id: str,
    runtime: Any | None,
    orchestrator: Any | None,
    runtime_version: str | None = None,
    runtime_git_commit: str | None = None,
    runtime_started_ts: float | int | None = None,
    fetch_local_api_json: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    command = classify_telegram_text_command(text)
    if command is None:
        return {
            "ok": True,
            "handled": False,
            "text": "",
            "route": "chat",
            "trace_id": trace_id,
            "next_action": None,
        }
    result = handle_telegram_command(
        command=command,
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=runtime,
        orchestrator=orchestrator,
        runtime_version=runtime_version,
        runtime_git_commit=runtime_git_commit,
        runtime_started_ts=runtime_started_ts,
        fetch_local_api_json=fetch_local_api_json,
    )
    if "handled" not in result:
        result["handled"] = True
    return result


def build_telegram_memory_response(
    *,
    chat_id: str,
    trace_id: str,
    orchestrator: Any | None,
) -> dict[str, Any]:
    return build_telegram_memory(
        chat_id=chat_id,
        trace_id=trace_id,
        orchestrator=orchestrator,
    )


__all__ = [
    "build_telegram_error",
    "build_telegram_help",
    "build_telegram_memory",
    "build_telegram_memory_response",
    "build_telegram_setup",
    "build_telegram_status",
    "classify_telegram_text_command",
    "handle_telegram_command",
    "handle_telegram_text",
]
