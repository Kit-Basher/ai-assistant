from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agent.config import load_config
from agent.doctor import main as doctor_main
from agent.error_response_ux import deterministic_error_message
from agent.golden_path import (
    bootstrap_needed,
    next_step_for_failure,
)
from agent.llm.install_planner import build_install_plan
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


_DEFAULT_API_BASE_URL = "http://127.0.0.1:8765"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _trace_id(prefix: str) -> str:
    return f"cli-{prefix}-{int(time.time())}-{os.getpid()}"


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


def _cmd_status(args: argparse.Namespace) -> int:
    ok, payload_or_error = _http_json(
        base_url=str(args.api_base_url),
        path="/ready",
        timeout_seconds=1.0,
    )
    if not ok or not isinstance(payload_or_error, dict):
        return _print_error(
            title="LLM provider unavailable",
            component="agent.cli.status",
            next_action="run `agent doctor`",
        )
    payload = payload_or_error
    telegram = payload.get("telegram") if isinstance(payload.get("telegram"), dict) else {}
    telegram_state = str(telegram.get("state") or "unknown").strip().lower() or "unknown"
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
    print(
        "\n".join(
            [
                summary,
                f"runtime_mode: {runtime_mode}",
                f"telegram: {telegram_state}",
                f"message: {message}",
            ]
        ),
        flush=True,
    )
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
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
    return _repo_root() / "logs" / "agent.jsonl"


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
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_repo_root()),
            check=False,
            capture_output=True,
            text=True,
            timeout=0.3,
        )
    except Exception:
        return "unknown"
    value = (proc.stdout or "").strip()
    return value or "unknown"


def _cmd_version(_args: argparse.Namespace) -> int:
    version_path = _repo_root() / "VERSION"
    try:
        version = version_path.read_text(encoding="utf-8").strip()
    except Exception:
        version = "unknown"
    print(f"version={version} commit={_resolve_git_commit()}", flush=True)
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

    logs_parser = sub.add_parser("logs", help="Show recent agent logs")
    logs_parser.add_argument("--lines", type=int, default=50)
    logs_parser.add_argument("--path", default=None)

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
    if command == "logs":
        return _cmd_logs(args)
    if command == "version":
        return _cmd_version(args)

    return _print_error(
        title="Unknown command",
        component="agent.cli",
        next_action=next_step_for_failure("config_invalid"),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
