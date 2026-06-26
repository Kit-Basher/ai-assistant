#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass
class Check:
    name: str
    status: str
    detail: str
    command: str
    next_action: str | None = None


class ApiRequestError(RuntimeError):
    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


def _json_request(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_error = json.loads(raw_error or "{}")
        except json.JSONDecodeError:
            parsed_error = {"raw": raw_error[:500]}
        raise ApiRequestError(f"HTTP {exc.code}", payload=parsed_error if isinstance(parsed_error, dict) else {}) from exc
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError("non_object_json")
    return parsed


def _assistant_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    return str(assistant.get("content") or payload.get("message") or "").strip()


def _used_tools(payload: dict[str, Any]) -> list[str]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return [str(item).strip() for item in meta.get("used_tools", []) if str(item).strip()]


def _post_chat(base_url: str, prompt: str, *, run_id: str, thread_id: str, timeout: float) -> dict[str, Any]:
    return _json_request(
        "POST",
        f"{base_url.rstrip('/')}/chat",
        payload={
            "messages": [{"role": "user", "content": prompt}],
            "session_id": f"daily-driver-smoke-{run_id}",
            "thread_id": f"daily-driver-{thread_id}-{run_id}",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"daily-driver-{thread_id}-{int(time.time())}",
        },
        timeout=timeout,
    )


def _pass(name: str, detail: str, command: str) -> Check:
    return Check(name=name, status="PASS", detail=detail, command=command)


def _fail(name: str, detail: str, command: str, next_action: str | None = None) -> Check:
    return Check(name=name, status="FAIL", detail=detail, command=command, next_action=next_action)


def _blocked(name: str, detail: str, command: str, next_action: str | None = None) -> Check:
    return Check(name=name, status="BLOCKED", detail=detail, command=command, next_action=next_action)


def check_status_surfaces(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    for path in ("/ready", "/state", "/search/status", "/packs/state"):
        command = f"GET {path}"
        try:
            payload = _json_request("GET", f"{base_url.rstrip('/')}{path}", timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - operator smoke reports class only.
            checks.append(_fail(command, f"{exc.__class__.__name__}: {exc}", command, "Check the API service and rerun doctor."))
            continue
        if payload.get("ok") is False:
            checks.append(_fail(command, f"ok=false reason={payload.get('reason') or payload.get('error')}", command))
            continue
        if path == "/ready" and not bool(payload.get("chat_usable") or payload.get("ready")):
            checks.append(_fail(command, "runtime is not chat usable", command, str(payload.get("next_action") or "")))
            continue
        checks.append(_pass(command, "surface returned coherent JSON", command))
    return checks


def _exception_detail(exc: Exception) -> str:
    if isinstance(exc, ApiRequestError) and exc.payload:
        reason = exc.payload.get("error") or exc.payload.get("error_kind") or exc.payload.get("message") or exc.payload
        return f"{exc}: {reason}"
    return f"{exc.__class__.__name__}: {exc}"


def check_search_chat(base_url: str, timeout: float, run_id: str) -> Check:
    status_command = "GET /search/status"
    try:
        status = _json_request("GET", f"{base_url.rstrip('/')}/search/status", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail("chat search", f"/search/status failed: {exc.__class__.__name__}", status_command)
    if not (status.get("enabled") and status.get("endpoint_configured") and status.get("available")):
        reason = str(status.get("reason") or "unknown")
        persistent = status.get("persistent_config") if isinstance(status.get("persistent_config"), dict) else {}
        explicit_state = str(status.get("search_state") or "").strip()
        if explicit_state:
            state = explicit_state
        elif reason == "search_disabled" and not status.get("endpoint_configured"):
            state = "never_configured"
        elif reason == "endpoint_unreachable" and status.get("endpoint_configured"):
            state = "configured_stopped"
        elif reason == "invalid_persisted_search_config" or persistent.get("error"):
            state = "invalid_or_untrusted_config"
        else:
            state = "unavailable"
        return _blocked(
            "chat search",
            f"search_state={state} reason={reason}",
            status_command,
            str(status.get("next_action") or "Configure a trusted SearXNG endpoint, then rerun."),
        )
    prompt = "Search the web for the current Debian stable release and summarize the result."
    command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    try:
        payload = _post_chat(base_url, prompt, run_id=run_id, thread_id="search", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail("chat search", f"chat request failed: {_exception_detail(exc)}", command)
    text = _assistant_text(payload)
    tools = _used_tools(payload)
    lowered = text.lower()
    if "safe_web_search" not in tools:
        return _fail("chat search", f"did not report safe_web_search tool; tools={tools}", command)
    if "did not open pages" not in lowered or "untrusted" not in lowered:
        return _fail("chat search", "response did not clearly state metadata-only/untrusted search boundary", command)
    return _pass("chat search", text.splitlines()[0][:220], command)


def check_linux_pack_chat(base_url: str, timeout: float, run_id: str) -> Check:
    state_command = "GET /packs/state"
    try:
        state = _json_request("GET", f"{base_url.rstrip('/')}/packs/state", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail("linux pack chat", f"/packs/state failed: {exc.__class__.__name__}", state_command)
    packs = state.get("packs") if isinstance(state.get("packs"), list) else []
    linux_pack = next(
        (
            row
            for row in packs
            if isinstance(row, dict)
            and str(row.get("name") or "").strip().lower() == "linux troubleshooting workflow"
        ),
        None,
    )
    if not linux_pack or not bool(linux_pack.get("usable")):
        return _blocked(
            "linux pack chat",
            "Linux Troubleshooting Workflow is not installed and usable.",
            state_command,
            "Preview/import/approve/enable the starter Linux Troubleshooting Workflow through Plan Mode.",
        )
    prompt = "My Linux laptop is slow after resume. Use the Linux troubleshooting workflow and give me a safe diagnostic plan before running commands."
    command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    try:
        payload = _post_chat(base_url, prompt, run_id=run_id, thread_id="linux-pack", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail("linux pack chat", f"chat request failed: {_exception_detail(exc)}", command)
    text = _assistant_text(payload)
    tools = _used_tools(payload)
    lowered = text.lower()
    if "external_pack_lookup" not in tools:
        return _fail("linux pack chat", f"did not report external_pack_lookup tool; tools={tools}", command)
    if "using linux troubleshooting workflow" not in lowered or "did not run commands" not in lowered:
        return _fail("linux pack chat", "response did not clearly state pack use and non-execution boundary", command)
    return _pass("linux pack chat", text.splitlines()[0][:220], command)


def check_install_preview(base_url: str, timeout: float, run_id: str) -> Check:
    prompt = "Can you install htop on this machine?"
    command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    try:
        payload = _post_chat(base_url, prompt, run_id=run_id, thread_id="install-preview", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail("install preview", f"chat request failed: {_exception_detail(exc)}", command)
    text = _assistant_text(payload)
    tools = _used_tools(payload)
    lowered = text.lower()
    if "shell" not in tools:
        return _fail("install preview", f"did not route to the confirmation-gated shell preview; tools={tools}", command)
    if "say yes to continue" not in lowered or "mutates the local system" not in lowered:
        return _fail("install preview", "response did not ask for explicit confirmation before install mutation", command)
    if "installing htop" in lowered:
        return _fail("install preview", "response looked like it installed the package instead of previewing", command)
    return _pass("install preview", text.splitlines()[0][:220], command)


def check_normal_chat(base_url: str, timeout: float, run_id: str) -> Check:
    prompt = "Explain why the sky is blue in two sentences."
    command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    try:
        payload = _post_chat(base_url, prompt, run_id=run_id, thread_id="normal-chat", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail("normal chat", f"chat request failed: {_exception_detail(exc)}", command)
    text = _assistant_text(payload)
    tools = _used_tools(payload)
    lowered = text.lower()
    if tools:
        return _fail("normal chat", f"unexpected tool use for normal chat; tools={tools}", command)
    if "search returned" in lowered or "using linux troubleshooting workflow" in lowered:
        return _fail("normal chat", "normal chat forced search or pack language", command)
    return _pass("normal chat", text[:220], command)


def check_doctor(timeout: float) -> Check:
    command = "python -m agent doctor"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "agent", "doctor"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("doctor", f"{exc.__class__.__name__}: {exc}", command)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return _fail("doctor", f"exit={result.returncode}", command, "Read doctor output and fix FAIL checks.")
    if "Status: FAIL" in output:
        return _fail("doctor", "doctor reported FAIL", command, "Read doctor output and fix FAIL checks.")
    status_line = next((line.strip() for line in output.splitlines() if line.startswith("Status:")), "Status: OK")
    return _pass("doctor", status_line, command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily-driver live smoke for Personal Agent.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--skip-doctor", action="store_true")
    args = parser.parse_args(argv)

    run_id = str(int(time.time()))
    checks: list[Check] = []
    checks.extend(check_status_surfaces(args.base_url, args.timeout))
    checks.append(check_search_chat(args.base_url, args.timeout, run_id))
    checks.append(check_linux_pack_chat(args.base_url, args.timeout, run_id))
    checks.append(check_install_preview(args.base_url, args.timeout, run_id))
    checks.append(check_normal_chat(args.base_url, args.timeout, run_id))
    if not args.skip_doctor:
        checks.append(check_doctor(args.timeout))

    print("# Personal Agent Daily-Driver Smoke")
    print()
    for check in checks:
        print(f"## {check.name}: {check.status}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.detail}")
        if check.next_action:
            print(f"- next action: {check.next_action}")
        print()

    failed = [item for item in checks if item.status == "FAIL"]
    blocked = [item for item in checks if item.status == "BLOCKED"]
    print("## Summary")
    print(f"PASS={sum(1 for item in checks if item.status == 'PASS')} BLOCKED={len(blocked)} FAIL={len(failed)}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
