from __future__ import annotations

"""Canonical Telegram-to-runtime bridge.

This module owns Telegram-facing product logic while keeping transport mechanics in
`telegram_adapter/bot.py`.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from agent.doctor import run_doctor_report
from agent.error_response_ux import deterministic_error_message
from agent.golden_path import bootstrap_needed
from agent.onboarding_contract import ONBOARDING_READY
from agent.persona import normalize_persona_text
from agent.public_chat import build_no_llm_public_message, build_public_sentence_text, is_no_llm_error_kind
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
    ready_payload = _runtime_ready_payload(runtime, fetch_local_api_json=fetch_local_api_json)
    llm_from_ready = ready_payload.get("llm") if isinstance(ready_payload.get("llm"), dict) else {}
    if llm_from_ready:
        return dict(llm_from_ready)
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
    if ready_payload or llm_status:
        try:
            live_result = build_setup_result(
                ready_payload=ready_payload,
                llm_status=llm_status,
                api_reachable=True,
                dry_run=True,
                diagnosis_source=("runtime_snapshot" if runtime is not None else "api_probe"),
            )
            onboarding_row = ready_payload.get("onboarding") if isinstance(ready_payload.get("onboarding"), dict) else {}
            onboarding_state = str(onboarding_row.get("state") or "").strip().upper()
            runtime_status = ready_payload.get("runtime_status") if isinstance(ready_payload.get("runtime_status"), dict) else {}
            runtime_mode = str(runtime_status.get("runtime_mode") or ready_payload.get("runtime_mode") or "").strip().upper()
            ready_flag = bool(ready_payload.get("ready", False))
            onboarding_is_contradicted = (
                (ready_flag and runtime_mode == "READY" and onboarding_state and onboarding_state != ONBOARDING_READY)
                or (onboarding_state == "SERVICES_DOWN" and live_result.onboarding_state != "SERVICES_DOWN")
            )
            if onboarding_state and not onboarding_is_contradicted:
                steps_raw = onboarding_row.get("steps") if isinstance(onboarding_row.get("steps"), list) else []
                steps = [str(item).strip() for item in steps_raw if str(item).strip()]
                recovery_row = ready_payload.get("recovery") if isinstance(ready_payload.get("recovery"), dict) else {}
                why_value = str(onboarding_row.get("summary") or "").strip() or "Setup state reported by runtime."
                return SetupWizardResult(
                    trace_id=_setup_trace_id(),
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    onboarding_state=onboarding_state,
                    recovery_mode=str(recovery_row.get("mode") or live_result.recovery_mode),
                    summary=str(onboarding_row.get("summary") or "").strip() or live_result.summary,
                    why=why_value,
                    next_action=str(onboarding_row.get("next_action") or "").strip() or live_result.next_action,
                    steps=steps,
                    suggestions=[],
                    dry_run=True,
                    api_reachable=True,
                    diagnosis_source="runtime_onboarding",
                    raw_check_result=dict(live_result.raw_check_result),
                    mapped_state=onboarding_state,
                    diagnosis_confidence="inferred",
                )
            return live_result
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
            diagnosis_source="api_probe",
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
        "diagnosis": {
            "source": str(setup_result.diagnosis_source or "unknown"),
            "mapped_state": str(setup_result.mapped_state or setup_result.onboarding_state),
            "confidence": str(setup_result.diagnosis_confidence or "inferred"),
            "raw_check_result": dict(setup_result.raw_check_result),
        },
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
    summary = normalize_persona_text(
        str(runtime_status.get("summary") or "").strip() or "Agent is starting or degraded."
    )
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
    if normalized in {"/help", "help", "commands"}:
        return "/help"
    if normalized in {"/setup", "setup"}:
        return "/setup"
    if normalized in {"/setup_token", "setup token", "rotate token"}:
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
    if normalized in {"doctor", "/doctor", "fix", "diagnose", "diagnostics", "run doctor"}:
        return "/doctor"
    if normalized in {"health", "/health"}:
        return "/health"
    if normalized in {"/status", "status", "state", "uptime", "agent status", "bot status"}:
        return "/status"
    return None


def _canonical_chat_result(
    *,
    text: str,
    chat_id: str,
    trace_id: str,
    runtime: Any | None,
    orchestrator: Any | None,
    fetch_local_api_chat_json: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if callable(fetch_local_api_chat_json):
        payload = fetch_local_api_chat_json(
            build_telegram_chat_api_payload(text=text, chat_id=chat_id, trace_id=trace_id)
        )
        proxy_error = payload.get("_proxy_error") if isinstance(payload.get("_proxy_error"), dict) else None
        proxy_meta = payload.get("_proxy_meta") if isinstance(payload.get("_proxy_meta"), dict) else {}
        if isinstance(proxy_error, dict):
            return build_telegram_chat_proxy_error_result(
                proxy_error,
                trace_id=trace_id,
                proxy_meta=proxy_meta,
            )
        if isinstance(payload, dict) and payload:
            return build_telegram_chat_payload_result(
                payload,
                trace_id=trace_id,
                ok=bool(payload.get("ok", True)),
                handler_name="api_chat_proxy",
                legacy_compatibility=False,
            )
        return build_telegram_chat_proxy_error_result({"kind": "invalid_response"}, trace_id=trace_id)
    if runtime is not None and hasattr(runtime, "chat"):
        ok, body = runtime.chat(  # type: ignore[attr-defined]
            build_telegram_chat_api_payload(text=text, chat_id=chat_id, trace_id=trace_id)
        )
        payload = body if isinstance(body, dict) else {}
        return build_telegram_chat_payload_result(
            payload,
            trace_id=trace_id,
            ok=bool(ok),
            handler_name="canonical_runtime_chat",
            legacy_compatibility=True,
        )
    if orchestrator is None:
        error = build_telegram_error(
            title="❌ Runtime unavailable",
            trace_id=trace_id,
            component="telegram.bridge",
            next_action="run `agent doctor`",
            route="chat",
        )
        error.update(
            {
                "handled": True,
                "selected_route": "error",
                "handler_name": "canonical_orchestrator_chat",
                "used_llm": False,
                "used_memory": False,
                "used_runtime_state": False,
                "used_tools": [],
                "legacy_compatibility": False,
                "generic_fallback_used": False,
                "generic_fallback_reason": None,
            }
        )
        return error
    response = orchestrator.handle_message(text, user_id=chat_id)
    response_data = getattr(response, "data", None)
    payload = response_data if isinstance(response_data, dict) else {}
    message = _safe_text(str(getattr(response, "text", "") or "").strip() or _structured_error_text(payload))
    route = str(payload.get("route") or "generic_chat").strip().lower() or "generic_chat"
    used_tools = [
        str(item).strip()
        for item in (payload.get("used_tools") if isinstance(payload.get("used_tools"), list) else [])
        if str(item).strip()
    ]
    return {
        "ok": True,
        "handled": True,
        "text": message,
        "route": route,
        "trace_id": trace_id,
        "next_action": None,
        "selected_route": route,
        "handler_name": "canonical_orchestrator_chat",
        "used_llm": bool(payload.get("used_llm", False)),
        "used_memory": bool(payload.get("used_memory", False)),
        "used_runtime_state": bool(payload.get("used_runtime_state", False)),
        "used_tools": used_tools,
        "legacy_compatibility": True,
        "generic_fallback_used": route == "generic_chat",
        "generic_fallback_reason": str(payload.get("generic_fallback_reason") or "").strip() or None,
        "chat_meta": payload,
    }


def _structured_error_text(payload: dict[str, Any]) -> str | None:
    if payload.get("ok") is not False:
        return None
    if is_no_llm_error_kind(payload.get("error_kind")):
        return build_no_llm_public_message()
    detail = str(payload.get("message") or "").strip() or None
    return build_public_sentence_text(
        "I couldn't finish that request",
        detail,
        "Please try again.",
    )


def build_telegram_chat_api_payload(
    *,
    text: str,
    chat_id: str,
    trace_id: str,
    request_id: str | None = None,
    setup_state_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_request_id = str(request_id or "").strip() or uuid.uuid4().hex
    payload = {
        "messages": [{"role": "user", "content": str(text or "")}],
        "purpose": "chat",
        "task_type": "chat",
        "source_surface": "telegram",
        "user_id": f"telegram:{chat_id}",
        "thread_id": f"telegram:{chat_id}:thread",
        "trace_id": trace_id,
        "request_id": normalized_request_id,
    }
    if isinstance(setup_state_hint, dict) and setup_state_hint:
        payload["setup_state_hint"] = dict(setup_state_hint)
    return payload


def build_telegram_chat_payload_result(
    payload: dict[str, Any],
    *,
    trace_id: str,
    ok: bool,
    handler_name: str,
    legacy_compatibility: bool,
) -> dict[str, Any]:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else {}
    proxy_meta = payload.get("_proxy_meta") if isinstance(payload.get("_proxy_meta"), dict) else {}
    if is_no_llm_error_kind(payload.get("error_kind")):
        message = build_no_llm_public_message()
    else:
        message = str(
            (assistant or {}).get("content")
            or payload.get("message")
            or setup.get("summary")
            or meta.get("summary")
            or ""
        ).strip()
        if not message:
            message = str(_structured_error_text(payload) or "").strip()
    route = str(meta.get("route") or "generic_chat").strip().lower() or "generic_chat"
    used_tools = [
        str(item).strip()
        for item in (meta.get("used_tools") if isinstance(meta.get("used_tools"), list) else [])
        if str(item).strip()
    ]
    return {
        "ok": bool(ok),
        "handled": True,
        "text": _safe_text(message),
        "route": route,
        "trace_id": trace_id,
        "next_action": str(payload.get("next_question") or "").strip() or None,
        "selected_route": route,
        "handler_name": handler_name,
        "used_llm": bool(meta.get("used_llm", False)),
        "used_memory": bool(meta.get("used_memory", False)),
        "used_runtime_state": bool(meta.get("used_runtime_state", False)),
        "used_tools": used_tools,
        "legacy_compatibility": legacy_compatibility,
        "generic_fallback_used": bool(meta.get("generic_fallback_used", False)),
        "generic_fallback_reason": str(meta.get("generic_fallback_reason") or "").strip() or None,
        "chat_meta": {
            **meta,
            "proxy_elapsed_ms": (
                int(proxy_meta.get("elapsed_ms"))
                if isinstance(proxy_meta.get("elapsed_ms"), int)
                else None
            ),
            "proxy_timeout_seconds": (
                float(proxy_meta.get("timeout_seconds"))
                if isinstance(proxy_meta.get("timeout_seconds"), (int, float))
                else None
            ),
            "proxy_execution_mode": str(proxy_meta.get("execution_mode") or "").strip() or None,
            "proxy_chat_lock_wait_ms": (
                int(proxy_meta.get("chat_lock_wait_ms"))
                if isinstance(proxy_meta.get("chat_lock_wait_ms"), int)
                else None
            ),
            "proxy_chat_lock_contended": (
                bool(proxy_meta.get("chat_lock_contended"))
                if isinstance(proxy_meta.get("chat_lock_contended"), bool)
                else None
            ),
            "proxy_in_flight_count": (
                int(proxy_meta.get("in_flight_count"))
                if isinstance(proxy_meta.get("in_flight_count"), int)
                else None
            ),
            "proxy_total_elapsed_ms": (
                int(proxy_meta.get("total_elapsed_ms"))
                if isinstance(proxy_meta.get("total_elapsed_ms"), int)
                else None
            ),
        },
    }


def build_telegram_chat_proxy_error_result(
    proxy_error: dict[str, Any],
    *,
    trace_id: str,
    proxy_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proxy_meta = proxy_meta if isinstance(proxy_meta, dict) else {}
    kind = str(proxy_error.get("kind") or "proxy_error").strip().lower() or "proxy_error"
    detail = str(proxy_error.get("detail") or "").strip() or None
    status_code = (
        int(proxy_error.get("status_code"))
        if isinstance(proxy_error.get("status_code"), int)
        else None
    )
    backend_reachable = bool(proxy_error.get("backend_reachable", False))
    backend_ready = bool(proxy_error.get("backend_ready", False))
    backend_phase = str(proxy_error.get("backend_phase") or "").strip() or None
    elapsed_ms = (
        int(proxy_error.get("elapsed_ms"))
        if isinstance(proxy_error.get("elapsed_ms"), int)
        else None
    )
    timeout_seconds = (
        float(proxy_error.get("timeout_seconds"))
        if isinstance(proxy_error.get("timeout_seconds"), (int, float))
        else None
    )

    if kind == "timeout":
        if backend_ready or backend_reachable:
            message = "The agent is running, but that request timed out. Please try again."
        else:
            message = "I couldn't get a reply from the agent in time. Please try again."
    elif kind == "disconnect":
        if backend_ready or backend_reachable:
            message = "The connection to the agent dropped before the reply finished. Please try again."
        else:
            message = "The connection to the agent dropped. Please try again."
    elif kind == "http_error":
        if backend_ready or backend_reachable:
            message = "The agent returned an error while handling that request. Please try again."
        else:
            message = "I couldn't get a clean response from the agent backend."
    elif kind == "unreachable":
        if backend_ready or backend_reachable:
            message = "The agent looks healthy, but this request could not connect cleanly. Please try again."
        else:
            message = "I couldn't reach the agent backend right now."
    else:
        if backend_ready or backend_reachable:
            message = "The agent hit a transient local connection problem. Please try again."
        else:
            message = "I couldn't talk to the local agent backend right now."

    return {
        "ok": False,
        "handled": True,
        "text": message,
        "route": "chat_proxy_error",
        "trace_id": trace_id,
        "next_action": None,
        "selected_route": "chat_proxy_error",
        "handler_name": "api_chat_proxy",
        "used_llm": False,
        "used_memory": False,
        "used_runtime_state": backend_reachable,
        "used_tools": [],
        "legacy_compatibility": False,
        "generic_fallback_used": False,
        "generic_fallback_reason": kind,
        "chat_meta": {
            "proxy_failure_kind": kind,
            "proxy_failure_detail": detail,
            "proxy_failure_status_code": status_code,
            "proxy_backend_reachable": backend_reachable,
            "proxy_backend_ready": backend_ready,
            "proxy_backend_phase": backend_phase,
            "proxy_elapsed_ms": elapsed_ms,
            "proxy_timeout_seconds": timeout_seconds,
            "proxy_execution_mode": str(proxy_meta.get("execution_mode") or "").strip() or None,
            "proxy_chat_lock_wait_ms": (
                int(proxy_meta.get("chat_lock_wait_ms"))
                if isinstance(proxy_meta.get("chat_lock_wait_ms"), int)
                else None
            ),
            "proxy_chat_lock_contended": (
                bool(proxy_meta.get("chat_lock_contended"))
                if isinstance(proxy_meta.get("chat_lock_contended"), bool)
                else None
            ),
            "proxy_in_flight_count": (
                int(proxy_meta.get("in_flight_count"))
                if isinstance(proxy_meta.get("in_flight_count"), int)
                else None
            ),
            "proxy_total_elapsed_ms": (
                int(proxy_meta.get("total_elapsed_ms"))
                if isinstance(proxy_meta.get("total_elapsed_ms"), int)
                else None
            ),
            "runtime_state_failure_reason": kind,
        },
    }


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
    def _attach_command_meta(
        result: dict[str, Any],
        *,
        route: str | None = None,
        used_llm: bool = False,
        used_memory: bool = False,
        used_runtime_state: bool = True,
        used_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_route = str(route or result.get("route") or "chat").strip().lower() or "chat"
        result.setdefault("handled", True)
        result.setdefault("selected_route", normalized_route)
        result.setdefault("handler_name", "command_bridge")
        result.setdefault("used_llm", used_llm)
        result.setdefault("used_memory", used_memory)
        result.setdefault("used_runtime_state", used_runtime_state)
        result.setdefault("used_tools", list(used_tools or []))
        result.setdefault("legacy_compatibility", False)
        result.setdefault("generic_fallback_used", False)
        result.setdefault("generic_fallback_reason", None)
        return result

    normalized = str(command or "").strip().lower()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if normalized == "/help":
        return _attach_command_meta(
            build_telegram_help(
                runtime=runtime,
                trace_id=trace_id,
                fetch_local_api_json=fetch_local_api_json,
            ),
            route="help",
        )
    if normalized == "/setup":
        return _attach_command_meta(
            build_telegram_setup(
                runtime=runtime,
                trace_id=trace_id,
                fetch_local_api_json=fetch_local_api_json,
            ),
            route="setup",
        )
    if normalized == "/setup_token":
        return _attach_command_meta(
            {
                "ok": True,
                "text": _ROTATE_TOKEN_TEXT,
                "route": "setup",
                "trace_id": trace_id,
                "next_action": "systemctl --user restart personal-agent-telegram.service",
            },
            route="setup",
        )
    if normalized == "/status":
        return _attach_command_meta(
            build_telegram_status(
                runtime=runtime,
                trace_id=trace_id,
                runtime_version=runtime_version,
                runtime_git_commit=runtime_git_commit,
                runtime_started_ts=runtime_started_ts,
                fetch_local_api_json=fetch_local_api_json,
            ),
            route="status",
        )
    if normalized == "/doctor":
        return _attach_command_meta(
            _build_telegram_doctor(trace_id=trace_id),
            route="doctor",
        )

    if normalized in {"/health", "/brief", "/memory"}:
        if orchestrator is None:
            return _attach_command_meta(
                build_telegram_error(
                    title="❌ Runtime unavailable",
                    trace_id=trace_id,
                    component="telegram.bridge",
                    next_action="run `agent doctor`",
                    route=normalized.lstrip("/"),
                ),
                route=normalized.lstrip("/"),
                used_runtime_state=False,
            )
        response = orchestrator.handle_message(normalized, user_id=chat_id)
        payload = getattr(response, "data", None)
        meta = payload if isinstance(payload, dict) else {}
        used_tools = [
            str(item).strip()
            for item in (meta.get("used_tools") if isinstance(meta.get("used_tools"), list) else [])
            if str(item).strip()
        ]
        return _attach_command_meta(
            {
                "ok": True,
                "text": _safe_text(getattr(response, "text", "")),
                "route": normalized.lstrip("/"),
                "trace_id": trace_id,
                "next_action": None,
            },
            route=str(meta.get("route") or normalized.lstrip("/")).strip().lower() or normalized.lstrip("/"),
            used_llm=bool(meta.get("used_llm", False)),
            used_memory=bool(meta.get("used_memory", False)),
            used_runtime_state=bool(meta.get("used_runtime_state", True)),
            used_tools=used_tools,
        )

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
    fetch_local_api_chat_json: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    command = classify_telegram_text_command(text)
    if command is not None:
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
        result.setdefault("selected_route", str(result.get("route") or command.lstrip("/")).strip().lower() or "chat")
        result.setdefault("handler_name", "command_bridge")
        result.setdefault("used_llm", False)
        result.setdefault("used_memory", False)
        result.setdefault("used_runtime_state", True)
        result.setdefault("used_tools", [])
        result.setdefault("legacy_compatibility", False)
        result.setdefault("generic_fallback_used", False)
        result.setdefault("generic_fallback_reason", None)
        if isinstance(result.get("text"), str):
            result["text"] = normalize_persona_text(str(result.get("text") or ""))
        return result
    result = _canonical_chat_result(
        text=text,
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=runtime,
        orchestrator=orchestrator,
        fetch_local_api_chat_json=fetch_local_api_chat_json,
    )
    if isinstance(result.get("text"), str):
        result["text"] = normalize_persona_text(str(result.get("text") or ""))
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
    "build_telegram_chat_api_payload",
    "build_telegram_chat_payload_result",
    "build_telegram_chat_proxy_error_result",
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
