#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8765"
SERVICE_NAME = "personal-agent-api.service"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    command: str
    next_action: str | None = None


class ApiRequestError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


def _pass(name: str, detail: str, command: str) -> Check:
    return Check(name=name, status="PASS", detail=detail, command=command)


def _warn(name: str, detail: str, command: str, next_action: str | None = None) -> Check:
    return Check(name=name, status="WARN", detail=detail, command=command, next_action=next_action)


def _fail(name: str, detail: str, command: str, next_action: str | None = None) -> Check:
    return Check(name=name, status="FAIL", detail=detail, command=command, next_action=next_action)


def _run(argv: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)


def _git_short_head() -> str:
    return (_run(["git", "rev-parse", "--short", "HEAD"]).stdout or "").strip()


def _git_status_short() -> str:
    return (_run(["git", "status", "--short"]).stdout or "").strip()


def _json_request(method: str, base_url: str, path: str, *, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {"raw": raw[:500]}
        raise ApiRequestError(f"HTTP {exc.code} {method.upper()} {path}", status=int(exc.code), payload=parsed if isinstance(parsed, dict) else {}) from exc
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise ApiRequestError(f"non-object JSON from {method.upper()} {path}", status=status, payload={"raw": raw[:500]})
    return parsed


def _post_chat(base_url: str, message: str, *, thread_id: str, timeout: float) -> dict[str, Any]:
    now = int(time.time() * 1000)
    return _json_request(
        "POST",
        base_url,
        "/chat",
        payload={
            "messages": [{"role": "user", "content": message}],
            "session_id": f"restart-survival-{thread_id}",
            "thread_id": f"restart-survival-{thread_id}",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"restart-survival-{thread_id}-{now}",
        },
        timeout=timeout,
    )


def _assistant_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    return str(assistant.get("content") or payload.get("message") or "").strip()


def _used_tools(payload: dict[str, Any]) -> list[str]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    raw = meta.get("used_tools") if isinstance(meta.get("used_tools"), list) else []
    return [str(item).strip() for item in raw if str(item).strip()]


def _wait_ready(base_url: str, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "not_checked"
    while time.monotonic() < deadline:
        try:
            payload = _json_request("GET", base_url, "/ready", timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{exc.__class__.__name__}: {exc}"
            time.sleep(0.5)
            continue
        if bool(payload.get("ready") or payload.get("chat_usable")):
            return payload
        last_error = str(payload.get("reason") or payload.get("runtime_mode") or "not_ready")
        time.sleep(0.5)
    raise RuntimeError(f"ready timeout: {last_error}")


def _search_state(status: dict[str, Any]) -> str:
    explicit = str(status.get("search_state") or "").strip()
    if explicit:
        return explicit
    if status.get("enabled") and status.get("endpoint_configured") and status.get("available"):
        return "configured_running"
    if status.get("endpoint_configured") and not status.get("available"):
        return "configured_stopped"
    if not status.get("endpoint_configured"):
        return "never_configured"
    return "unknown"


def capture_baseline(base_url: str, timeout: float) -> tuple[list[Check], dict[str, dict[str, Any]]]:
    checks: list[Check] = []
    captured: dict[str, dict[str, Any]] = {}
    for path in ("/version", "/ready", "/search/status", "/telegram/status"):
        command = f"GET {path}"
        try:
            payload = _json_request("GET", base_url, path, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            checks.append(_fail(f"baseline {path}", f"{exc.__class__.__name__}: {exc}", command))
            continue
        captured[path] = payload
        if path == "/version":
            checks.append(_pass("baseline version", f"git_commit={payload.get('git_commit')} runtime_instance={payload.get('runtime_instance')}", command))
        elif path == "/search/status":
            checks.append(_pass("baseline search status", f"search_state={_search_state(payload)} reason={payload.get('reason')}", command))
        elif path == "/telegram/status":
            checks.append(_pass("baseline telegram status", f"configured={payload.get('configured')} state={payload.get('state')}", command))
        else:
            checks.append(_pass("baseline ready", f"ready={payload.get('ready')} chat_usable={payload.get('chat_usable')}", command))
    return checks, captured


def restart_api_service(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    stop_command = f"systemctl --user stop {SERVICE_NAME}"
    stopped = _run(["systemctl", "--user", "stop", SERVICE_NAME], timeout=timeout)
    if stopped.returncode != 0:
        return [_fail("stop api service", (stopped.stderr or stopped.stdout or "stop failed").strip()[:500], stop_command)]
    checks.append(_pass("stop api service", "systemd stop returned 0", stop_command))

    unavailable = False
    unavailable_detail = ""
    for _ in range(10):
        try:
            _json_request("GET", base_url, "/ready", timeout=0.5)
        except Exception as exc:  # noqa: BLE001
            unavailable = True
            unavailable_detail = f"{exc.__class__.__name__}: {exc}"
            break
        time.sleep(0.25)
    if unavailable:
        checks.append(_pass("ready unavailable while stopped", unavailable_detail[:240], "GET /ready"))
    else:
        checks.append(_warn("ready unavailable while stopped", "/ready stayed reachable immediately after stop", "GET /ready", "Check whether another API process is serving the same port."))

    start_command = f"systemctl --user start {SERVICE_NAME}"
    started = _run(["systemctl", "--user", "start", SERVICE_NAME], timeout=timeout)
    if started.returncode != 0:
        return checks + [_fail("start api service", (started.stderr or started.stdout or "start failed").strip()[:500], start_command)]
    checks.append(_pass("start api service", "systemd start returned 0", start_command))
    try:
        ready = _wait_ready(base_url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        checks.append(_fail("ready after start", f"{exc.__class__.__name__}: {exc}", "GET /ready", "Inspect personal-agent-api.service logs."))
    else:
        checks.append(_pass("ready after start", f"runtime_mode={ready.get('runtime_mode')} chat_usable={ready.get('chat_usable')}", "GET /ready"))
    return checks


def verify_surfaces(base_url: str, timeout: float) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    payloads: dict[str, Any] = {}
    head = _git_short_head()
    for path in ("/version", "/state", "/search/status", "/packs/state", "/telegram/status"):
        command = f"GET {path}"
        try:
            payload = _json_request("GET", base_url, path, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            checks.append(_fail(f"post-restart {path}", f"{exc.__class__.__name__}: {exc}", command))
            continue
        payloads[path] = payload
        if path == "/version":
            commit = str(payload.get("git_commit") or "").strip()
            if head and commit != head:
                checks.append(_fail("post-restart version freshness", f"runtime git_commit={commit} checkout={head}", command, "Promote local stable runtime again."))
            else:
                checks.append(_pass("post-restart version freshness", f"runtime git_commit={commit} checkout={head}", command))
        elif path == "/search/status":
            checks.append(_pass("post-restart search status", f"search_state={_search_state(payload)} reason={payload.get('reason')}", command))
        elif path == "/telegram/status":
            checks.append(_pass("post-restart telegram status", f"configured={payload.get('configured')} state={payload.get('state')} service_active={payload.get('service_active')}", command))
        else:
            checks.append(_pass(f"post-restart {path}", "surface returned coherent JSON", command))
    return checks, payloads


def repair_search_if_needed(base_url: str, status: dict[str, Any], timeout: float) -> list[Check]:
    checks: list[Check] = []
    state = _search_state(status)
    if state == "configured_running":
        checks.append(_pass("search repair not needed", "search_state=configured_running", "GET /search/status"))
        return checks
    if state != "configured_stopped":
        checks.append(_fail("search repair precondition", f"search_state={state} reason={status.get('reason')}", "GET /search/status", "Configure trusted managed SearXNG before restart survival proof."))
        return checks

    prompt = "Search the web for the current Debian stable release and summarize the result."
    thread = "search-repair"
    preview_command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    try:
        preview = _post_chat(base_url, prompt, thread_id=thread, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        checks.append(_fail("configured_stopped search repair preview", f"{exc.__class__.__name__}: {exc}", preview_command))
        return checks
    preview_text = _assistant_text(preview)
    preview_tools = _used_tools(preview)
    if "managed_local_service_setup_preview" not in preview_tools:
        checks.append(_fail("configured_stopped search repair preview", f"tools={preview_tools} text={preview_text[:300]}", preview_command))
        return checks
    if "start or repair" not in preview_text.lower() or "say yes" not in preview_text.lower():
        checks.append(_fail("configured_stopped search repair preview", preview_text[:400], preview_command, "Preview must offer bounded Plan Mode repair."))
        return checks
    checks.append(_pass("configured_stopped search repair preview", preview_text.splitlines()[0][:220], preview_command))

    confirm_command = f"POST /chat {json.dumps({'message': 'yes'}, ensure_ascii=True)}"
    try:
        confirm = _post_chat(base_url, "yes", thread_id=thread, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        checks.append(_fail("configured_stopped search repair approval", f"{exc.__class__.__name__}: {exc}", confirm_command))
        return checks
    confirm_text = _assistant_text(confirm)
    confirm_tools = _used_tools(confirm)
    lowered = confirm_text.lower()
    if "managed_local_service_setup" not in confirm_tools:
        checks.append(_fail("configured_stopped search repair approval", f"tools={confirm_tools} text={confirm_text[:300]}", confirm_command))
    elif "metadata-only" not in lowered or "arbitrary docker commands" not in lowered:
        checks.append(_fail("configured_stopped search repair approval", confirm_text[:400], confirm_command, "Repair result must state safety boundaries."))
    else:
        checks.append(_pass("configured_stopped search repair approval", confirm_text.splitlines()[0][:220], confirm_command))
    return checks


def verify_search_query(base_url: str, timeout: float) -> Check:
    prompt = "Search the web for the current Debian stable release and summarize the result."
    command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    try:
        payload = _post_chat(base_url, prompt, thread_id="search-query", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail("metadata-only search after restart", f"{exc.__class__.__name__}: {exc}", command)
    text = _assistant_text(payload)
    tools = _used_tools(payload)
    lowered = text.lower()
    if "safe_web_search" not in tools:
        return _fail("metadata-only search after restart", f"tools={tools} text={text[:300]}", command)
    if "metadata-only" not in lowered or "untrusted" not in lowered or "did not open pages" not in lowered:
        return _fail("metadata-only search after restart", text[:400], command, "Search response must state metadata-only/untrusted boundaries.")
    return _pass("metadata-only search after restart", text.splitlines()[0][:220], command)


def verify_telegram_status_chat(base_url: str, timeout: float) -> Check:
    prompt = "why is Telegram not responding?"
    command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    try:
        payload = _post_chat(base_url, prompt, thread_id="telegram-status", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail("telegram inactive explanation after restart", f"{exc.__class__.__name__}: {exc}", command)
    text = _assistant_text(payload)
    lowered = text.lower()
    if "token" in lowered and ":" in text:
        return _fail("telegram inactive explanation after restart", "response looked like it may expose token material", command)
    if "optional" not in lowered or ("not currently running" not in lowered and "not running" not in lowered):
        return _fail("telegram inactive explanation after restart", text[:400], command)
    return _pass("telegram inactive explanation after restart", text.splitlines()[0][:220], command)


def verify_stale_confirmation_after_restart(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    thread = "stale-confirmation"
    prompt = "Can you install htop on this machine?"
    command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    try:
        preview = _post_chat(base_url, prompt, thread_id=thread, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return [_fail("stale confirmation preview", f"{exc.__class__.__name__}: {exc}", command)]
    preview_text = _assistant_text(preview)
    if "say yes to continue" not in preview_text.lower() or "mutates the local system" not in preview_text.lower():
        return [_fail("stale confirmation preview", preview_text[:400], command)]
    checks.append(_pass("stale confirmation preview", preview_text.splitlines()[0][:220], command))

    checks.extend(restart_api_service(base_url, timeout))
    confirm_command = f"POST /chat {json.dumps({'message': 'yes'}, ensure_ascii=True)}"
    try:
        confirm = _post_chat(base_url, "yes", thread_id=thread, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return checks + [_fail("stale confirmation after restart", f"{exc.__class__.__name__}: {exc}", confirm_command)]
    confirm_text = _assistant_text(confirm)
    lowered = confirm_text.lower()
    if "installed htop" in lowered or "installing htop" in lowered:
        checks.append(_fail("stale confirmation after restart", confirm_text[:400], confirm_command, "Pending install confirmation survived restart incorrectly."))
    elif "current action" in lowered or "tell me what" in lowered or "preview" in lowered or "no current" in lowered:
        checks.append(_pass("stale confirmation after restart", confirm_text.splitlines()[0][:220], confirm_command))
    else:
        checks.append(_warn("stale confirmation after restart", confirm_text[:400], confirm_command, "Response did not execute mutation, but wording should be clearer."))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Installed Personal Agent restart survival smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args(argv)

    status_before = _git_status_short()
    checks: list[Check] = []
    baseline_checks, _baseline = capture_baseline(args.base_url, args.timeout)
    checks.extend(baseline_checks)
    if any(item.status == "FAIL" for item in baseline_checks):
        print_report(checks)
        return 1

    checks.extend(restart_api_service(args.base_url, args.timeout))
    post_checks, payloads = verify_surfaces(args.base_url, args.timeout)
    checks.extend(post_checks)
    search_status = payloads.get("/search/status") if isinstance(payloads.get("/search/status"), dict) else {}
    checks.extend(repair_search_if_needed(args.base_url, search_status, args.timeout))
    try:
        final_search_status = _json_request("GET", args.base_url, "/search/status", timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001
        checks.append(_fail("final search status", f"{exc.__class__.__name__}: {exc}", "GET /search/status"))
    else:
        checks.append(_pass("final search status", f"search_state={_search_state(final_search_status)} reason={final_search_status.get('reason')}", "GET /search/status"))
    checks.append(verify_search_query(args.base_url, args.timeout))
    checks.append(verify_telegram_status_chat(args.base_url, args.timeout))
    checks.extend(verify_stale_confirmation_after_restart(args.base_url, args.timeout))

    status_after = _git_status_short()
    if status_after == status_before:
        checks.append(_pass("git status unchanged", "working tree status unchanged", "git status --short"))
    else:
        checks.append(_fail("git status unchanged", f"before={status_before!r} after={status_after!r}", "git status --short"))

    print_report(checks)
    return 1 if any(item.status == "FAIL" for item in checks) else 0


def print_report(checks: list[Check]) -> None:
    print("# Personal Agent Restart Survival Smoke")
    print(f"Service: {SERVICE_NAME}")
    print(f"Base URL: {DEFAULT_BASE_URL}")
    print()
    for check in checks:
        print(f"## {check.name}: {check.status}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.detail}")
        if check.next_action:
            print(f"- next action: {check.next_action}")
        print()
    passed = sum(1 for item in checks if item.status == "PASS")
    warned = sum(1 for item in checks if item.status == "WARN")
    failed = sum(1 for item in checks if item.status == "FAIL")
    print("## Summary")
    print(f"PASS={passed} WARN={warned} FAIL={failed}")
    print("RESTART_SURVIVAL_SMOKE:", "pass" if failed == 0 else "fail")


if __name__ == "__main__":
    raise SystemExit(main())

