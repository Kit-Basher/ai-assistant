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


DEFAULT_BASE_URL = "http://127.0.0.1:8765"


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str


def _request_json(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_error = json.loads(body or "{}")
        except json.JSONDecodeError:
            parsed_error = {"body": body}
        if isinstance(parsed_error, dict):
            parsed_error.setdefault("http_status", exc.code)
            return parsed_error
        return {"http_status": exc.code, "body": body}
    parsed = json.loads(body or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _post_chat(base_url: str, message: str, *, thread_id: str, timeout: float = 30.0) -> dict[str, Any]:
    return _request_json(
        "POST",
        f"{base_url.rstrip('/')}/chat",
        payload={"message": message, "user_id": "executor-registry-smoke", "thread_id": thread_id},
        timeout=timeout,
    )


def _assistant_text(payload: dict[str, Any]) -> str:
    for key in ("response", "message", "text", "assistant"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for key in ("response", "message", "text", "summary"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, sort_keys=True)[:1000]


def _runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else {}
    if setup:
        return setup
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    runtime_payload = data.get("runtime_payload") if isinstance(data.get("runtime_payload"), dict) else {}
    if runtime_payload:
        return runtime_payload
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    runtime_payload = meta.get("runtime_payload") if isinstance(meta.get("runtime_payload"), dict) else {}
    return runtime_payload if isinstance(runtime_payload, dict) else {}


def _pass(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=True, evidence=evidence.strip()[:1000], command=command)


def _fail(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=False, evidence=evidence.strip()[:1600], command=command)


def _git_status_short() -> str:
    result = subprocess.run(["git", "status", "--short"], check=False, text=True, capture_output=True, timeout=10)
    return result.stdout.strip()


def _wait_ready(base_url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            payload = _request_json("GET", f"{base_url.rstrip('/')}/ready", timeout=5)
            if bool(payload.get("ready", payload.get("ok", False))):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _restart_api(base_url: str, timeout: float) -> Check:
    command = "systemctl --user restart personal-agent-api.service"
    result = subprocess.run(["systemctl", "--user", "restart", "personal-agent-api.service"], check=False, text=True, capture_output=True, timeout=30)
    if result.returncode != 0:
        return _fail("restart api service", (result.stderr or result.stdout or "").strip(), command)
    if not _wait_ready(base_url, timeout):
        return _fail("restart api service", "service did not become ready in time", command)
    return _pass("restart api service", "service restarted and /ready returned", command)


def _confirm_preview_only(base_url: str, prompt: str, *, thread_id: str, timeout: float) -> tuple[Check, Check]:
    preview = _post_chat(base_url, prompt, thread_id=thread_id, timeout=timeout)
    preview_text = _assistant_text(preview)
    preview_payload = _runtime_payload(preview)
    plan = preview_payload.get("canonical_plan") if isinstance(preview_payload.get("canonical_plan"), dict) else {}
    preview_ok = "Plan Mode v2" in preview_text and plan.get("executor_status") == "preview_only"
    preview_check = (
        _pass(f"{prompt} preview-only plan", f"plan_id={plan.get('plan_id')} action={plan.get('action_type')}", f'POST /chat {{"message": {prompt!r}}}')
        if preview_ok
        else _fail(f"{prompt} preview-only plan", json.dumps(preview, sort_keys=True)[:1200], f'POST /chat {{"message": {prompt!r}}}')
    )
    confirm = _post_chat(base_url, "yes", thread_id=thread_id, timeout=timeout)
    confirm_blob = json.dumps(confirm, sort_keys=True).lower()
    confirm_payload = _runtime_payload(confirm)
    executor_result = confirm_payload.get("executor_result") if isinstance(confirm_payload.get("executor_result"), dict) else {}
    confirm_ok = (
        "executor_not_enabled" in confirm_blob
        and (confirm_payload.get("mutated") is False or executor_result.get("mutated") is False)
        and bool(confirm_payload.get("journal_id") or executor_result.get("journal_id"))
    )
    confirm_check = (
        _pass(f"{prompt} preview-only refuses execution", _assistant_text(confirm), 'POST /chat {"message": "yes"}')
        if confirm_ok
        else _fail(f"{prompt} preview-only refuses execution", json.dumps(confirm, sort_keys=True)[:1400], 'POST /chat {"message": "yes"}')
    )
    return preview_check, confirm_check


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    baseline_git = _git_status_short()

    for prompt, thread_id in (
        ("delete all memory about me", "executor-memory-delete"),
        ("uninstall the assistant", "executor-uninstall"),
    ):
        preview_check, confirm_check = _confirm_preview_only(base_url, prompt, thread_id=thread_id, timeout=timeout)
        checks.extend([preview_check, confirm_check])

    cleanup_preview = _post_chat(base_url, "clean old runtime files", thread_id="executor-cleanup", timeout=timeout)
    cleanup_text = _assistant_text(cleanup_preview)
    cleanup_payload = _runtime_payload(cleanup_preview)
    cleanup_plan = cleanup_payload.get("canonical_plan") if isinstance(cleanup_payload.get("canonical_plan"), dict) else {}
    checks.append(
        _pass("cleanup has enabled executor plan", f"plan_id={cleanup_plan.get('plan_id')} action={cleanup_plan.get('action_type')}", 'POST /chat {"message": "clean old runtime files"}')
        if "Plan Mode v2" in cleanup_text and cleanup_plan.get("executor_status") == "enabled" and cleanup_plan.get("action_type") == "operator.cleanup"
        else _fail("cleanup has enabled executor plan", json.dumps(cleanup_preview, sort_keys=True)[:1400], 'POST /chat {"message": "clean old runtime files"}')
    )
    cleanup_cancel = _post_chat(base_url, "no", thread_id="executor-cleanup", timeout=timeout)
    cleanup_cancel_text = _assistant_text(cleanup_cancel).lower()
    checks.append(
        _pass("cleanup enabled plan cancels without execution", _assistant_text(cleanup_cancel), 'POST /chat {"message": "no"}')
        if "cancel" in cleanup_cancel_text
        else _fail("cleanup enabled plan cancels without execution", json.dumps(cleanup_cancel, sort_keys=True)[:1400], 'POST /chat {"message": "no"}')
    )

    support_preview = _post_chat(base_url, "make a support bundle", thread_id="executor-support", timeout=timeout)
    support_preview_text = _assistant_text(support_preview)
    support_preview_payload = _runtime_payload(support_preview)
    support_plan = support_preview_payload.get("canonical_plan") if isinstance(support_preview_payload.get("canonical_plan"), dict) else {}
    checks.append(
        _pass("support bundle has enabled executor plan", f"plan_id={support_plan.get('plan_id')} action={support_plan.get('action_type')}", 'POST /chat {"message": "make a support bundle"}')
        if "Plan Mode v2" in support_preview_text and support_plan.get("executor_status") == "enabled" and support_plan.get("action_type") == "operator.support_bundle"
        else _fail("support bundle has enabled executor plan", json.dumps(support_preview, sort_keys=True)[:1400], 'POST /chat {"message": "make a support bundle"}')
    )
    support_confirm = _post_chat(base_url, "yes", thread_id="executor-support", timeout=timeout)
    support_payload = _runtime_payload(support_confirm)
    support_result = support_payload.get("executor_result") if isinstance(support_payload.get("executor_result"), dict) else {}
    support_details = support_result.get("details") if isinstance(support_result.get("details"), dict) else {}
    artifact_path = str(support_details.get("artifact_path") or "").strip()
    support_ok = bool(support_result.get("ok")) and bool(support_result.get("mutated")) and bool(support_result.get("journal_id")) and bool(artifact_path)
    if support_ok and artifact_path:
        artifact = Path(artifact_path)
        support_ok = artifact.exists() and (artifact / "support_summary.json").exists()
    checks.append(
        _pass("support bundle executor creates redacted artifact", json.dumps(support_result, sort_keys=True)[:1000], 'POST /chat {"message": "yes"}')
        if support_ok
        else _fail("support bundle executor creates redacted artifact", json.dumps(support_confirm, sort_keys=True)[:1600], 'POST /chat {"message": "yes"}')
    )
    support_blob = json.dumps(support_confirm, sort_keys=True).lower()
    secret_markers = ("telegram_token", "bot token", "bearer ", "api_key=", "password=", "secret=")
    checks.append(
        _pass("executor journal/result redacts secrets", "no obvious secret markers in support result", "inspect support executor result")
        if not any(marker in support_blob for marker in secret_markers)
        else _fail("executor journal/result redacts secrets", support_blob[:1600], "inspect support executor result")
    )

    _post_chat(base_url, "make a support bundle", thread_id="executor-stale", timeout=timeout)
    checks.append(_restart_api(base_url, timeout))
    stale = _post_chat(base_url, "yes", thread_id="executor-stale", timeout=timeout)
    checks.append(
        _pass("stale confirmation after restart does not execute", _assistant_text(stale), 'POST /chat {"message": "yes"} after restart')
        if "current action" in _assistant_text(stale).lower()
        else _fail("stale confirmation after restart does not execute", json.dumps(stale, sort_keys=True)[:1200], 'POST /chat {"message": "yes"} after restart')
    )

    _post_chat(base_url, "make a support bundle", thread_id="executor-thread-a", timeout=timeout)
    wrong_thread = _post_chat(base_url, "yes", thread_id="executor-thread-b", timeout=timeout)
    checks.append(
        _pass("unrelated thread cannot execute support bundle plan", _assistant_text(wrong_thread), 'POST /chat {"message": "yes"} from another thread')
        if "different chat thread/session" in _assistant_text(wrong_thread).lower()
        else _fail("unrelated thread cannot execute support bundle plan", json.dumps(wrong_thread, sort_keys=True)[:1200], 'POST /chat {"message": "yes"} from another thread')
    )
    _post_chat(base_url, "no", thread_id="executor-thread-a", timeout=timeout)

    final_git = _git_status_short()
    checks.append(
        _pass("git status unchanged", "working tree status unchanged", "git status --short")
        if final_git == baseline_git
        else _fail("git status unchanged", f"before={baseline_git!r} after={final_git!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Installed-product Executor Registry v1 smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed

    print("# Personal Agent Executor Registry Smoke")
    for check in checks:
        print()
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
    print()
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"EXECUTOR_REGISTRY_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
