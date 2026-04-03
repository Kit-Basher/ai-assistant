from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agent.config import load_config, resolved_default_log_path
from agent.doctor import main as doctor_main
from agent.error_response_ux import deterministic_error_message
from agent.golden_path import (
    bootstrap_needed,
    next_step_for_failure,
)
from agent.llm.install_approval import validate_install_approval
from agent.llm.install_planner import build_install_plan
from agent.llm.install_planner import build_install_plan_for_model
from agent.llm.model_inventory import build_model_inventory
from agent.llm.model_selector import select_model_for_task
from agent.llm.registry import load_registry
from agent.llm.task_classifier import classify_task_request
from agent.logging_bootstrap import configure_logging_if_needed
from agent.runtime_contract import normalize_user_facing_status
from agent.skills.system_health_analyzer import build_system_health_report
from agent.skills.system_health import collect_system_health
from agent.skills.system_health_summary import render_system_health_summary
from agent.setup_wizard import render_setup_text, run_setup_wizard
from agent.telegram_runtime_state import (
    TELEGRAM_SERVICE_NAME,
    clear_stale_telegram_locks,
    get_telegram_runtime_state,
    resolve_telegram_token_with_source,
    telegram_control_env,
    write_telegram_enablement,
)
from agent.version import read_build_info, read_git_commit


_DEFAULT_API_BASE_URL = "http://127.0.0.1:8765"
_LOGGER = logging.getLogger(__name__)
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_STATUS_DEBUG_ENV = "AGENT_CLI_STATUS_DEBUG"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _trace_id(prefix: str) -> str:
    return f"cli-{prefix}-{int(time.time())}-{os.getpid()}"


def _llm_install_result_from_decision(*, decision: dict[str, Any], trace_id: str) -> dict[str, Any]:
    return {
        "ok": str(decision.get("error_kind") or "").strip() == "already_satisfied",
        "executed": False,
        "model_id": str(decision.get("model_id") or "").strip() or None,
        "install_name": str(decision.get("install_name") or "").strip() or None,
        "trace_id": trace_id,
        "error_kind": None if str(decision.get("error_kind") or "").strip() == "already_satisfied" else str(decision.get("error_kind") or "install_not_allowed"),
        "message": (
            "Model already installed and healthy."
            if str(decision.get("error_kind") or "").strip() == "already_satisfied"
            else str(decision.get("message") or "Install request was rejected.")
        ),
        "verification": {},
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _execute_llm_install_via_model_manager(*, config: Any, plan: dict[str, Any], trace_id: str) -> dict[str, Any]:
    decision = validate_install_approval(plan, approve=True)
    model_id = str(decision.get("model_id") or "").strip()
    if str(decision.get("error_kind") or "").strip() not in {"", "already_satisfied"} and not bool(decision.get("allowed", False)):
        return _llm_install_result_from_decision(decision=decision, trace_id=trace_id)
    if not model_id:
        return _llm_install_result_from_decision(decision=decision, trace_id=trace_id)
    from agent.api_server import AgentRuntime

    runtime = AgentRuntime(config)
    return runtime._model_manager().execute_request(
        {
            "kind": "approved_ollama_pull",
            "model_ref": model_id,
        },
        approve=True,
        trace_id=trace_id,
        timeout_seconds=1800.0,
        source="cli.llm_install",
    )


def _http_json(
    *,
    base_url: str,
    path: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 1.5,
) -> tuple[bool, dict[str, Any] | str]:
    url = f"{base_url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    body = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib.request.Request(url=url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
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


def _status_debug_enabled() -> bool:
    value = str(os.getenv(_STATUS_DEBUG_ENV, "")).strip().lower()
    return value in {"1", "true", "yes", "on", "debug"}


def _emit_status_transport_debug(event: str, **fields: Any) -> None:
    detail = " ".join(f"{key}={fields[key]}" for key in sorted(fields))
    _LOGGER.debug("agent.cli.status %s %s", event, detail)
    if _status_debug_enabled():
        print(f"agent-cli-debug {event} {detail}".rstrip(), file=sys.stderr, flush=True)


def _normalize_loopback_base_url(base_url: str) -> str:
    parts = urlsplit(str(base_url).strip())
    hostname = str(parts.hostname or "").strip().lower()
    if hostname not in _LOOPBACK_HOSTS:
        return str(base_url).rstrip("/")
    netloc = "127.0.0.1"
    if parts.port is not None:
        netloc = f"{netloc}:{int(parts.port)}"
    scheme = str(parts.scheme or "http").strip() or "http"
    return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")


def _is_loopback_base_url(base_url: str) -> bool:
    hostname = str(urlsplit(str(base_url).strip()).hostname or "").strip().lower()
    return hostname in _LOOPBACK_HOSTS


def _build_api_request_target(*, base_url: str, path: str) -> str:
    normalized_base = str(base_url).rstrip("/")
    parts = urlsplit(normalized_base)
    base_path = str(parts.path or "").rstrip("/")
    normalized_path = str(path or "").strip()
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    target = f"{base_path}{normalized_path}" or "/"
    return target if target.startswith("/") else f"/{target}"


def _direct_http_json(
    *,
    base_url: str,
    path: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 1.5,
) -> tuple[bool, dict[str, Any] | str]:
    normalized_base = _normalize_loopback_base_url(base_url) if _is_loopback_base_url(base_url) else str(base_url).rstrip("/")
    parts = urlsplit(normalized_base)
    scheme = str(parts.scheme or "http").strip().lower() or "http"
    host = str(parts.hostname or "127.0.0.1").strip() or "127.0.0.1"
    port = int(parts.port or (443 if scheme == "https" else 80))
    url = f"{normalized_base.rstrip('/')}{path}"
    target = _build_api_request_target(base_url=normalized_base, path=path)
    normalized_method = str(method or "GET").upper()
    body_text: str | None = None
    body_bytes: bytes | None = None
    headers: dict[str, str] = {}
    if payload is not None:
        body_text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        body_bytes = body_text.encode("utf-8")
        headers["Content-Type"] = "application/json"
    connection_cls: type[http.client.HTTPConnection] | type[http.client.HTTPSConnection]
    connection_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
    try:
        start = time.monotonic()
        _emit_status_transport_debug(
            "direct_http_start",
            host=host,
            method=normalized_method,
            port=port,
            target=target,
            timeout_seconds=f"{float(timeout_seconds):.3f}",
            url=url,
        )
        connection = connection_cls(host, port=port, timeout=max(float(timeout_seconds), 0.1))
        connection.request(normalized_method, target, body=body_bytes, headers=headers)
        response = connection.getresponse()
        raw_bytes = response.read()
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        _emit_status_transport_debug(
            "direct_http_exception",
            elapsed_ms=elapsed_ms,
            timeout_seconds=f"{float(timeout_seconds):.3f}",
            url=url,
            error=f"{exc.__class__.__name__}:{exc}",
        )
        return False, f"{exc.__class__.__name__}:{exc}"
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
    elapsed_ms = int((time.monotonic() - start) * 1000)
    _emit_status_transport_debug(
        "direct_http_finish",
        elapsed_ms=elapsed_ms,
        response_bytes=len(raw_bytes),
        status=int(response.status),
        reason=str(getattr(response, "reason", "") or ""),
        url=url,
    )
    raw = raw_bytes.decode("utf-8")
    if int(response.status) >= 400:
        try:
            parsed = json.loads(raw or "{}")
            if isinstance(parsed, dict):
                return True, parsed
        except Exception:
            pass
        return False, f"http_{int(response.status)}"
    try:
        parsed = json.loads(raw or "{}")
    except Exception as exc:
        return False, f"json_error:{exc.__class__.__name__}"
    if isinstance(parsed, dict):
        return True, parsed
    return False, "non_object_json"


def _print_error(*, title: str, component: str, next_action: str) -> int:
    print(
        deterministic_error_message(
            title=title,
            trace_id=_trace_id(component.replace(".", "-")),
            component=component,
            next_action=next_action,
        ),
        flush=True,
    )
    return 1


def _timeout_like_error(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return "timeout" in text or "timed out" in text


def _load_ready_status_payload(*, base_url: str) -> tuple[bool, dict[str, Any] | str]:
    attempts = (6.0, 15.0)
    last_error: dict[str, Any] | str = "ready_unavailable"
    use_direct_http = _is_loopback_base_url(base_url)
    for index, timeout_seconds in enumerate(attempts):
        transport = "urllib"
        if use_direct_http:
            transport = "direct_http_loopback"
            ok, payload_or_error = _direct_http_json(
                base_url=base_url,
                path="/ready",
                timeout_seconds=float(timeout_seconds),
            )
        else:
            ok, payload_or_error = _http_json(
                base_url=base_url,
                path="/ready",
                timeout_seconds=float(timeout_seconds),
            )
        if ok and isinstance(payload_or_error, dict):
            return True, payload_or_error
        last_error = payload_or_error
        _LOGGER.debug(
            "agent.cli.status ready fetch failed transport=%s base_url=%s timeout_seconds=%.3f error=%s",
            transport,
            _normalize_loopback_base_url(base_url) if use_direct_http else str(base_url).rstrip("/"),
            float(timeout_seconds),
            payload_or_error,
        )
        if index + 1 >= len(attempts) or not _timeout_like_error(payload_or_error):
            break
    return False, last_error


def _load_runtime_status_payload(*, base_url: str) -> tuple[bool, dict[str, Any] | str]:
    attempts = (2.0, 4.0)
    last_error: dict[str, Any] | str = "runtime_unavailable"
    use_direct_http = _is_loopback_base_url(base_url)
    for index, timeout_seconds in enumerate(attempts):
        transport = "urllib"
        if use_direct_http:
            transport = "direct_http_loopback"
            ok, payload_or_error = _direct_http_json(
                base_url=base_url,
                path="/runtime",
                timeout_seconds=float(timeout_seconds),
            )
        else:
            ok, payload_or_error = _http_json(
                base_url=base_url,
                path="/runtime",
                timeout_seconds=float(timeout_seconds),
            )
        if ok and isinstance(payload_or_error, dict):
            return True, payload_or_error
        last_error = payload_or_error
        _LOGGER.debug(
            "agent.cli.runtime fetch failed transport=%s base_url=%s timeout_seconds=%.3f error=%s",
            transport,
            _normalize_loopback_base_url(base_url) if use_direct_http else str(base_url).rstrip("/"),
            float(timeout_seconds),
            payload_or_error,
        )
        if index + 1 >= len(attempts) or not _timeout_like_error(payload_or_error):
            break
    return False, last_error


def _load_runtime_history_payload(*, base_url: str) -> tuple[bool, dict[str, Any] | str]:
    attempts = (2.0, 4.0)
    last_error: dict[str, Any] | str = "runtime_history_unavailable"
    use_direct_http = _is_loopback_base_url(base_url)
    for index, timeout_seconds in enumerate(attempts):
        transport = "urllib"
        if use_direct_http:
            transport = "direct_http_loopback"
            ok, payload_or_error = _direct_http_json(
                base_url=base_url,
                path="/runtime/history",
                timeout_seconds=float(timeout_seconds),
            )
        else:
            ok, payload_or_error = _http_json(
                base_url=base_url,
                path="/runtime/history",
                timeout_seconds=float(timeout_seconds),
            )
        if ok and isinstance(payload_or_error, dict):
            return True, payload_or_error
        last_error = payload_or_error
        _LOGGER.debug(
            "agent.cli.runtime_history fetch failed transport=%s base_url=%s timeout_seconds=%.3f error=%s",
            transport,
            _normalize_loopback_base_url(base_url) if use_direct_http else str(base_url).rstrip("/"),
            float(timeout_seconds),
            payload_or_error,
        )
        if index + 1 >= len(attempts) or not _timeout_like_error(payload_or_error):
            break
    return False, last_error


def _format_runtime_event_line(event: dict[str, Any]) -> str | None:
    event_name = str(event.get("event") or "").strip().lower()
    if event_name == "runtime_phase_change":
        phase_from = str(event.get("phase_from") or "unknown").strip()
        phase_to = str(event.get("phase_to") or "unknown").strip()
        return f"{phase_from} -> {phase_to}"
    if event_name == "default_model_change":
        new_model = str(event.get("new_model") or "").strip()
        if new_model:
            return f"default model set: {new_model}"
    if event_name == "provider_health_transition":
        provider = str(event.get("provider") or "provider").strip()
        old_status = str(event.get("old_status") or "unknown").strip()
        new_status = str(event.get("new_status") or "unknown").strip()
        return f"{provider} health: {old_status} -> {new_status}"
    if event_name == "provider_switch":
        new_provider = str(event.get("new_provider") or "").strip()
        if new_provider:
            return f"provider switched: {new_provider}"
    return None


def _cmd_status(args: argparse.Namespace) -> int:
    operator_env = telegram_control_env()
    if _status_debug_enabled():
        _emit_status_transport_debug(
            "status_context",
            api_base_url=str(args.api_base_url),
            cwd=os.getcwd(),
            module_file=__file__,
            python_executable=sys.executable,
        )
    ok, payload_or_error = _load_ready_status_payload(base_url=str(args.api_base_url))
    if not ok or not isinstance(payload_or_error, dict):
        return _print_error(
            title="Agent status unavailable",
            component="agent.cli.status",
            next_action="run `agent doctor`",
        )
    payload = payload_or_error
    runtime_ok, runtime_payload_or_error = _load_runtime_status_payload(base_url=str(args.api_base_url))
    runtime_payload = runtime_payload_or_error if runtime_ok and isinstance(runtime_payload_or_error, dict) else {}
    runtime_history_ok, runtime_history_payload_or_error = _load_runtime_history_payload(base_url=str(args.api_base_url))
    runtime_history_payload = (
        runtime_history_payload_or_error
        if runtime_history_ok and isinstance(runtime_history_payload_or_error, dict)
        else {}
    )
    telegram_runtime = get_telegram_runtime_state(env=operator_env)
    telegram_state = str(telegram_runtime.get("effective_state") or "unknown").strip().lower() or "unknown"
    runtime_status = payload.get("runtime_status") if isinstance(payload.get("runtime_status"), dict) else {}
    if runtime_status:
        summary = str(runtime_status.get("summary") or "").strip() or "Agent is starting or degraded."
        runtime_mode = str(runtime_status.get("runtime_mode") or "DEGRADED").strip().upper() or "DEGRADED"
    else:
        ready = bool(payload.get("ready", False))
        message_reason = str(payload.get("llm_reason") or payload.get("failure_code") or "").strip() or None
        provider = (
            str((payload.get("llm") or {}).get("provider") or "").strip()
            if isinstance(payload.get("llm"), dict)
            else None
        )
        model = (
            str((payload.get("llm") or {}).get("model") or "").strip()
            if isinstance(payload.get("llm"), dict)
            else None
        )
        bootstrap = bootstrap_needed(
            llm_available=bool(payload.get("llm_available", True)) if "llm_available" in payload else None,
            availability_reason=str(payload.get("llm_reason") or "").strip() or None,
        )
        normalized_status = normalize_user_facing_status(
            ready=ready,
            bootstrap_required=bootstrap,
            failure_code=message_reason,
            phase=str(payload.get("phase") or "").strip().lower() or None,
            provider=provider,
            model=model,
        )
        summary = str(normalized_status.get("summary") or "").strip() or "Agent is starting or degraded."
        runtime_mode = str(normalized_status.get("runtime_mode") or "DEGRADED").strip().upper() or "DEGRADED"
    message = str(payload.get("message") or "").strip() or ("ready" if payload.get("ready") else "starting")
    lines = [
        summary,
        f"runtime_mode: {runtime_mode}",
        f"telegram: {telegram_state}",
        f"message: {message}",
    ]
    if runtime_payload:
        phase = str(runtime_payload.get("phase") or "").strip()
        default_chat_model = str(runtime_payload.get("default_chat_model") or "").strip()
        health_summary = (
            runtime_payload.get("health_summary")
            if isinstance(runtime_payload.get("health_summary"), dict)
            else {}
        )
        if phase:
            lines.append(f"phase: {phase}")
        if default_chat_model:
            lines.append(f"default_model: {default_chat_model}")
        lines.append(
            "provider_health: ok={ok} degraded={degraded} down={down}".format(
                ok=int(health_summary.get("ok") or 0),
                degraded=int(health_summary.get("degraded") or 0),
                down=int(health_summary.get("down") or 0),
            )
        )
    history_rows = (
        runtime_history_payload.get("events")
        if isinstance(runtime_history_payload.get("events"), list)
        else []
    )
    rendered_events = [
        line
        for line in (_format_runtime_event_line(row) for row in history_rows[-3:] if isinstance(row, dict))
        if line
    ]
    if rendered_events:
        lines.append("Runtime events:")
        lines.extend(f"- {line}" for line in rendered_events)
    next_action = str(telegram_runtime.get("next_action") or "").strip()
    if telegram_state not in {"enabled_running", "disabled_optional"} and next_action:
        lines.append(f"telegram_next_action: {next_action}")
    print("\n".join(lines), flush=True)
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    ready_ok, ready_payload_or_error = _http_json(
        base_url=str(args.api_base_url),
        path="/ready",
        timeout_seconds=1.2,
    )
    payload = (
        ready_payload_or_error.get("llm")
        if ready_ok and isinstance(ready_payload_or_error, dict) and isinstance(ready_payload_or_error.get("llm"), dict)
        else None
    )
    if not isinstance(payload, dict):
        ok, payload_or_error = _http_json(
            base_url=str(args.api_base_url),
            path="/llm/status",
            timeout_seconds=1.2,
        )
        if not ok or not isinstance(payload_or_error, dict):
            return _print_error(
                title="LLM provider unavailable",
                component="agent.cli.health",
                next_action="run `agent doctor`",
            )
        payload = payload_or_error
    if not isinstance(payload, dict):
        return _print_error(
            title="LLM provider unavailable",
            component="agent.cli.health",
            next_action="run `agent doctor`",
        )
    provider = str(payload.get("default_provider") or "unknown").strip() or "unknown"
    model = (
        str(payload.get("resolved_default_model") or "").strip()
        or str(payload.get("default_model") or "").strip()
        or "unknown"
    )
    provider_health = (
        payload.get("active_provider_health")
        if isinstance(payload.get("active_provider_health"), dict)
        else {}
    )
    model_health = (
        payload.get("active_model_health")
        if isinstance(payload.get("active_model_health"), dict)
        else {}
    )
    provider_status = str(provider_health.get("status") or "unknown").strip().lower() or "unknown"
    model_status = str(model_health.get("status") or "unknown").strip().lower() or "unknown"
    allow_remote = bool(payload.get("allow_remote_fallback", True))
    print(
        "\n".join(
            [
                f"provider={provider} ({provider_status})",
                f"model={model} ({model_status})",
                f"allow_remote_fallback={str(allow_remote).lower()}",
            ]
        ),
        flush=True,
    )
    return 0


def _cmd_health_system(_args: argparse.Namespace) -> int:
    observed = collect_system_health()
    report = build_system_health_report(observed)
    if bool(getattr(_args, "json", False)):
        print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
        return 0
    print(
        render_system_health_summary(
            report.get("observed") if isinstance(report.get("observed"), dict) else {},
            report.get("analysis") if isinstance(report.get("analysis"), dict) else {},
        ),
        flush=True,
    )
    return 0


def _load_llm_control_state(*, task_text: str | None = None) -> dict[str, Any]:
    config = load_config(require_telegram_token=False)
    registry = load_registry(config)
    inventory = build_model_inventory(config=config, registry=registry)
    allow_remote_fallback = bool(registry.defaults.allow_remote_fallback)
    default_policy = config.default_policy if isinstance(config.default_policy, dict) else None
    payload: dict[str, Any] = {
        "config": config,
        "registry": registry,
        "inventory": inventory,
    }
    if task_text is not None:
        task_request = classify_task_request(task_text)
        selection = select_model_for_task(
            inventory,
            task_request,
            allow_remote_fallback=allow_remote_fallback,
            policy_name="default",
            policy=default_policy,
        )
        plan = build_install_plan(
            inventory=inventory,
            task_request=task_request,
            selection_result=selection,
            allow_remote_fallback=allow_remote_fallback,
            policy_name="default",
            policy=default_policy,
        )
        payload.update(
            {
                "task_request": task_request,
                "selection": selection,
                "plan": plan,
            }
        )
    return payload


def _inventory_priority(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if bool(row.get("configured", False)) else 1,
        0 if bool(row.get("local", False)) and bool(row.get("installed", False)) else 1,
        0 if bool(row.get("local", False)) else 1,
        0 if bool(row.get("approved", False)) else 1,
        0 if bool(row.get("healthy", False)) else 1,
        0 if bool(row.get("available", False)) else 1,
        str(row.get("provider") or ""),
        str(row.get("id") or ""),
    )


def _cmd_llm_inventory(args: argparse.Namespace) -> int:
    state = _load_llm_control_state()
    inventory = state.get("inventory") if isinstance(state.get("inventory"), list) else []
    if bool(getattr(args, "json", False)):
        print(json.dumps({"inventory": inventory}, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
        return 0
    ordered_inventory = sorted(
        [row for row in inventory if isinstance(row, dict)],
        key=_inventory_priority,
    )
    visible_rows = ordered_inventory
    hidden_count = 0
    if not bool(getattr(args, "all", False)):
        preferred_rows = [
            row
            for row in ordered_inventory
            if bool(row.get("configured", False))
            or bool(row.get("local", False))
            or bool(row.get("approved", False))
        ]
        if preferred_rows:
            visible_rows = preferred_rows[:12]
            hidden_count = max(0, len(ordered_inventory) - len(visible_rows))
        else:
            visible_rows = ordered_inventory[:12]
            hidden_count = max(0, len(ordered_inventory) - len(visible_rows))
    lines = ["LLM inventory"]
    for row in visible_rows:
        capabilities = ",".join(row.get("capabilities") or []) if isinstance(row.get("capabilities"), list) else "none"
        lines.append(
            "- {id} | provider={provider} | local={local} | installed={installed} | available={available} | healthy={healthy} | approved={approved} | configured={configured} | caps={caps}".format(
                id=str(row.get("id") or "unknown"),
                provider=str(row.get("provider") or "unknown"),
                local=str(bool(row.get("local", False))).lower(),
                installed=str(bool(row.get("installed", False))).lower(),
                available=str(bool(row.get("available", False))).lower(),
                healthy=str(bool(row.get("healthy", False))).lower(),
                approved=str(bool(row.get("approved", False))).lower(),
                configured=str(bool(row.get("configured", False))).lower(),
                caps=capabilities or "none",
            )
        )
    if hidden_count > 0:
        lines.append(f"... {hidden_count} additional rows hidden; use --all to show the full inventory.")
    print("\n".join(lines), flush=True)
    return 0


def _cmd_llm_select(args: argparse.Namespace) -> int:
    state = _load_llm_control_state(task_text=str(args.task or ""))
    task_request = state.get("task_request") if isinstance(state.get("task_request"), dict) else {}
    selection = state.get("selection") if isinstance(state.get("selection"), dict) else {}
    if bool(getattr(args, "json", False)):
        print(
            json.dumps(
                {
                    "task_request": task_request,
                    "selection": selection,
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return 0
    fallbacks = selection.get("fallbacks") if isinstance(selection.get("fallbacks"), list) else []
    lines = [
        "LLM selection",
        f"task_type: {str(task_request.get('task_type') or 'chat')}",
        "requirements: " + ",".join(task_request.get("requirements") or []) if isinstance(task_request.get("requirements"), list) else "requirements: none",
        f"preferred_local: {str(bool(task_request.get('preferred_local', True))).lower()}",
        f"selected_model: {str(selection.get('selected_model') or 'none')}",
        f"provider: {str(selection.get('provider') or 'none')}",
        f"reason: {str(selection.get('reason') or 'no_selection')}",
        "fallbacks: " + (", ".join(str(item) for item in fallbacks) if fallbacks else "none"),
    ]
    print("\n".join(lines), flush=True)
    return 0


def _cmd_llm_plan(args: argparse.Namespace) -> int:
    state = _load_llm_control_state(task_text=str(args.task or ""))
    task_request = state.get("task_request") if isinstance(state.get("task_request"), dict) else {}
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    if bool(getattr(args, "json", False)):
        print(
            json.dumps(
                {
                    "task_request": task_request,
                    "plan": plan,
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return 0
    lines = [
        "LLM install plan",
        f"task_type: {str(task_request.get('task_type') or 'chat')}",
        f"needed: {str(bool(plan.get('needed', False))).lower()}",
        f"approved: {str(bool(plan.get('approved', False))).lower()}",
        f"reason: {str(plan.get('reason') or 'none')}",
        f"install_command: {str(plan.get('install_command') or 'none')}",
        f"next_action: {str(plan.get('next_action') or 'none')}",
    ]
    candidates = plan.get("candidates") if isinstance(plan.get("candidates"), list) else []
    if candidates:
        lines.append("candidates:")
        for row in candidates:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- {model_id} | install={install_name} | size={size_hint} | preferred={preferred} | reason={reason}".format(
                    model_id=str(row.get("model_id") or "unknown"),
                    install_name=str(row.get("install_name") or "unknown"),
                    size_hint=str(row.get("size_hint") or "unknown"),
                    preferred=str(bool(row.get("preferred", False))).lower(),
                    reason=str(row.get("reason") or "unknown"),
                )
            )
    steps = plan.get("plan") if isinstance(plan.get("plan"), list) else []
    if steps:
        lines.append("plan:")
        for row in steps:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {str(row.get('id') or 'step')}: {str(row.get('action') or 'unknown')} {str(row.get('model') or '').strip()}".rstrip()
            )
    print("\n".join(lines), flush=True)
    return 0


def _cmd_llm_install(args: argparse.Namespace) -> int:
    config = load_config(require_telegram_token=False)
    registry = load_registry(config)
    inventory = build_model_inventory(config=config, registry=registry)
    if str(getattr(args, "model", "") or "").strip():
        plan = build_install_plan_for_model(
            inventory=inventory,
            model_ref=str(args.model),
        )
        task_request: dict[str, Any] = {"task_type": "explicit_model", "requirements": [], "preferred_local": True}
    else:
        task_text = str(args.task or "").strip()
        task_request = classify_task_request(task_text)
        selection = select_model_for_task(
            inventory,
            task_request,
            allow_remote_fallback=bool(registry.defaults.allow_remote_fallback),
            policy_name="default",
            policy=config.default_policy if isinstance(config.default_policy, dict) else None,
        )
        plan = build_install_plan(
            inventory=inventory,
            task_request=task_request,
            selection_result=selection,
            allow_remote_fallback=bool(registry.defaults.allow_remote_fallback),
            policy_name="default",
            policy=config.default_policy if isinstance(config.default_policy, dict) else None,
        )
    if bool(getattr(args, "json", False)) and not bool(getattr(args, "approve", False)):
        payload = {
            "task_request": task_request,
            "plan": plan,
            "approval": validate_install_approval(plan, approve=False),
        }
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
        return 0
    if not bool(getattr(args, "approve", False)):
        approval = validate_install_approval(plan, approve=False)
        lines = [
            "LLM install request",
            f"needed: {str(bool(plan.get('needed', False))).lower()}",
            f"approved: {str(bool(plan.get('approved', False))).lower()}",
            f"approval_required: {str(str(approval.get('error_kind') or '') == 'approval_required').lower()}",
            f"model_id: {str(approval.get('model_id') or 'none')}",
            f"install_name: {str(approval.get('install_name') or 'none')}",
            f"next_action: {str(plan.get('next_action') or approval.get('message') or 'none')}",
        ]
        print("\n".join(lines), flush=True)
        return 0

    result = _execute_llm_install_via_model_manager(
        config=config,
        plan=plan,
        trace_id=_trace_id("llm-install"),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    else:
        lines = [
            "LLM install result",
            f"ok: {str(bool(result.get('ok', False))).lower()}",
            f"executed: {str(bool(result.get('executed', False))).lower()}",
            f"model_id: {str(result.get('model_id') or 'none')}",
            f"install_name: {str(result.get('install_name') or 'none')}",
            f"error_kind: {str(result.get('error_kind') or 'none')}",
            f"message: {str(result.get('message') or 'none')}",
            f"trace_id: {str(result.get('trace_id') or 'none')}",
        ]
        verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
        if verification:
            lines.extend(
                [
                    "verification:",
                    f"- found={str(bool(verification.get('found', False))).lower()} installed={str(bool(verification.get('installed', False))).lower()} available={str(bool(verification.get('available', False))).lower()} healthy={str(bool(verification.get('healthy', False))).lower()}",
                ]
            )
        print("\n".join(lines), flush=True)
    return 0 if bool(result.get("ok", False)) else 1


def _cmd_brief(args: argparse.Namespace) -> int:
    ok, payload_or_error = _http_json(
        base_url=str(args.api_base_url),
        path="/chat",
        method="POST",
        payload={"messages": [{"role": "user", "content": "/brief"}]},
        timeout_seconds=2.5,
    )
    if not ok or not isinstance(payload_or_error, dict):
        return _print_error(
            title="Brief generation unavailable",
            component="agent.cli.brief",
            next_action="run `agent doctor`",
        )
    payload = payload_or_error
    message = str(payload.get("message") or "").strip()
    if not message:
        return _print_error(
            title="Brief generation unavailable",
            component="agent.cli.brief",
            next_action="run `agent doctor`",
        )
    print(message, flush=True)
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    ok, payload_or_error = _http_json(
        base_url=str(args.api_base_url),
        path="/chat",
        method="POST",
        payload={"messages": [{"role": "user", "content": "/memory"}]},
        timeout_seconds=2.5,
    )
    if not ok or not isinstance(payload_or_error, dict):
        return _print_error(
            title="Memory summary unavailable",
            component="agent.cli.memory",
            next_action="run `agent doctor`",
        )
    payload = payload_or_error
    message = str(payload.get("message") or "").strip()
    if not message:
        return _print_error(
            title="Memory summary unavailable",
            component="agent.cli.memory",
            next_action="run `agent doctor`",
        )
    print(message, flush=True)
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    result = run_setup_wizard(
        api_base_url=str(args.api_base_url),
        dry_run=bool(args.dry_run),
    )
    if bool(args.json):
        print(json.dumps(result.to_dict(), ensure_ascii=True, sort_keys=True, indent=2), flush=True)
        return 0
    print(render_setup_text(result), flush=True)
    return 0


def _resolve_log_path(explicit: str | None) -> Path:
    if explicit:
        return Path(str(explicit)).expanduser()
    env_path = str(os.getenv("AGENT_LOG_PATH", "")).strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path(resolved_default_log_path()).expanduser()


def _cmd_logs(args: argparse.Namespace) -> int:
    log_path = _resolve_log_path(args.path)
    if not log_path.is_file():
        return _print_error(
            title="Log file unavailable",
            component="agent.cli.logs",
            next_action="run `agent doctor`",
        )
    try:
        rows = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return _print_error(
            title="Log file unavailable",
            component="agent.cli.logs",
            next_action="run `agent doctor`",
        )
    lines = rows[-max(1, int(args.lines)) :]
    header = f"Showing last {max(1, int(args.lines))} lines from {log_path}"
    print("\n".join([header, *lines]), flush=True)
    return 0


def _resolve_git_commit() -> str:
    return str(read_git_commit(repo_root=_repo_root(), timeout_seconds=0.3) or "unknown")


def _cmd_version(_args: argparse.Namespace) -> int:
    build_info = read_build_info(repo_root=_repo_root(), timeout_seconds=0.3)
    print(
        f"version={build_info.version} commit={build_info.git_commit or 'unknown'}",
        flush=True,
    )
    return 0


def _run_systemctl_user(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=2.0,
    )


def _render_telegram_status(state: dict[str, Any]) -> str:
    lines = [
        f"enabled: {str(bool(state.get('enabled', False))).lower()}",
        f"config_source: {str(state.get('config_source') or 'default')}",
        f"service_installed: {str(bool(state.get('service_installed', False))).lower()}",
        f"service_active: {str(bool(state.get('service_active', False))).lower()}",
        f"token_configured: {str(bool(state.get('token_configured', False))).lower()}",
        f"lock_present: {str(bool(state.get('lock_present', False))).lower()}",
        f"effective_state: {str(state.get('effective_state') or 'unknown')}",
        f"next_action: {str(state.get('next_action') or 'No action needed.')}",
    ]
    return "\n".join(lines)


def _cmd_telegram_status(_args: argparse.Namespace) -> int:
    state = get_telegram_runtime_state(env=telegram_control_env())
    print(_render_telegram_status(state), flush=True)
    return 0


def _cmd_telegram_enable(_args: argparse.Namespace) -> int:
    operator_env = telegram_control_env()
    write_telegram_enablement(True, env=operator_env)
    state_before = get_telegram_runtime_state(env=operator_env)
    token_configured = bool(state_before.get("token_configured", False))
    try:
        token, _token_source = resolve_telegram_token_with_source(env=operator_env)
    except Exception:
        token = None
    cleared = clear_stale_telegram_locks(token, env=operator_env)
    try:
        _run_systemctl_user(["daemon-reload"])
    except Exception:
        pass
    if token_configured and bool(state_before.get("service_installed", False)):
        _run_systemctl_user(["restart", TELEGRAM_SERVICE_NAME])
    state = get_telegram_runtime_state(env=operator_env)
    if cleared:
        state = {**state, "next_action": str(state.get("next_action") or "No action needed.")}
    print(_render_telegram_status(state), flush=True)
    return 0


def _cmd_telegram_disable(_args: argparse.Namespace) -> int:
    operator_env = telegram_control_env()
    write_telegram_enablement(False, env=operator_env)
    try:
        _run_systemctl_user(["daemon-reload"])
    except Exception:
        pass
    state_before = get_telegram_runtime_state(env=operator_env)
    if bool(state_before.get("service_installed", False)):
        _run_systemctl_user(["stop", TELEGRAM_SERVICE_NAME])
    state = get_telegram_runtime_state(env=operator_env)
    print(_render_telegram_status(state), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent", description="Personal Agent operator CLI")
    sub = parser.add_subparsers(dest="command")

    doctor_parser = sub.add_parser("doctor", help="Run deterministic diagnostics")
    doctor_parser.add_argument("doctor_args", nargs=argparse.REMAINDER, help="Arguments forwarded to agent.doctor")

    status_parser = sub.add_parser("status", help="Show short runtime status")
    status_parser.add_argument("--api-base-url", default=_DEFAULT_API_BASE_URL)

    brief_parser = sub.add_parser("brief", help="Show short system summary")
    brief_parser.add_argument("--api-base-url", default=_DEFAULT_API_BASE_URL)

    memory_parser = sub.add_parser("memory", help="Show continuity summary")
    memory_parser.add_argument("--api-base-url", default=_DEFAULT_API_BASE_URL)

    setup_parser = sub.add_parser("setup", help="Show deterministic setup guidance")
    setup_parser.add_argument("--api-base-url", default=_DEFAULT_API_BASE_URL)
    setup_parser.add_argument("--json", action="store_true", help="emit JSON output")
    setup_parser.add_argument("--dry-run", action="store_true", help="no mutations (default behavior)")

    health_parser = sub.add_parser("health", help="Show LLM/runtime health snapshot")
    health_parser.add_argument("--api-base-url", default=_DEFAULT_API_BASE_URL)

    health_system_parser = sub.add_parser("health_system", help="Show local PC health summary")
    health_system_parser.add_argument("--json", action="store_true", help="emit observed + analyzed JSON output")

    llm_inventory_parser = sub.add_parser("llm_inventory", help="Show deterministic LLM inventory")
    llm_inventory_parser.add_argument("--json", action="store_true", help="emit JSON output")
    llm_inventory_parser.add_argument("--all", action="store_true", help="show the full inventory in plain-text mode")

    llm_select_parser = sub.add_parser("llm_select", help="Show which model would be selected for a task")
    llm_select_parser.add_argument("--task", required=True, help="task description")
    llm_select_parser.add_argument("--json", action="store_true", help="emit JSON output")

    llm_plan_parser = sub.add_parser("llm_plan", help="Show approved local install plan for a task")
    llm_plan_parser.add_argument("--task", required=True, help="task description")
    llm_plan_parser.add_argument("--json", action="store_true", help="emit JSON output")

    llm_install_parser = sub.add_parser("llm_install", help="Preview or execute an approved local Ollama install")
    llm_install_target = llm_install_parser.add_mutually_exclusive_group(required=True)
    llm_install_target.add_argument("--task", help="task description")
    llm_install_target.add_argument("--model", help="approved local model id, for example ollama:llava:7b")
    llm_install_parser.add_argument("--approve", action="store_true", help="execute the approved local install")
    llm_install_parser.add_argument("--json", action="store_true", help="emit JSON output")

    logs_parser = sub.add_parser("logs", help="Show recent agent logs")
    logs_parser.add_argument("--lines", type=int, default=50)
    logs_parser.add_argument("--path", default=None)

    sub.add_parser("telegram_status", help="Show Telegram optional adapter state")
    sub.add_parser("telegram_enable", help="Enable and start the Telegram optional adapter")
    sub.add_parser("telegram_disable", help="Disable and stop the Telegram optional adapter")
    sub.add_parser("version", help="Show version and git commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging_if_needed()
    if argv is None:
        incoming = list(sys.argv[1:])
    else:
        incoming = list(argv)
    if incoming and str(incoming[0]).strip().lower() == "doctor":
        return int(doctor_main(incoming[1:]))

    parser = build_parser()
    args = parser.parse_args(incoming)
    command = str(args.command or "").strip().lower()
    if not command:
        parser.print_help()
        return 2

    if command == "doctor":
        forwarded = list(args.doctor_args or [])
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return int(doctor_main(forwarded))
    if command == "status":
        return _cmd_status(args)
    if command == "brief":
        return _cmd_brief(args)
    if command == "memory":
        return _cmd_memory(args)
    if command == "setup":
        return _cmd_setup(args)
    if command == "health":
        return _cmd_health(args)
    if command == "health_system":
        return _cmd_health_system(args)
    if command == "llm_inventory":
        return _cmd_llm_inventory(args)
    if command == "llm_select":
        return _cmd_llm_select(args)
    if command == "llm_plan":
        return _cmd_llm_plan(args)
    if command == "llm_install":
        return _cmd_llm_install(args)
    if command == "logs":
        return _cmd_logs(args)
    if command == "telegram_status":
        return _cmd_telegram_status(args)
    if command == "telegram_enable":
        return _cmd_telegram_enable(args)
    if command == "telegram_disable":
        return _cmd_telegram_disable(args)
    if command == "version":
        return _cmd_version(args)

    return _print_error(
        title="Unknown command",
        component="agent.cli",
        next_action=next_step_for_failure("config_invalid"),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
