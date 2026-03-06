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

from agent.doctor import main as doctor_main
from agent.error_response_ux import deterministic_error_message
from agent.golden_path import (
    bootstrap_needed,
    next_step_for_failure,
)
from agent.logging_bootstrap import configure_logging_if_needed
from agent.runtime_contract import normalize_user_facing_status
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
    data = collect_system_health()
    print(render_system_health_summary(data), flush=True)
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

    sub.add_parser("health_system", help="Show local PC health summary")

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
