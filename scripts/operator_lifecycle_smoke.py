#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8765"
SECRET_PATTERNS = (
    re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\b(?:sk|sk-proj|xoxb|ghp)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    command: str
    next_action: str | None = None


def _request_json(method: str, base_url: str, path: str, *, payload: dict[str, Any] | None = None, timeout: float) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {"raw": raw[:500]}
        return {"ok": False, "http_status": int(exc.code), "error": parsed}
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        return {"ok": False, "error": "non_object_json", "raw": raw[:500]}
    return parsed


def _post_chat(base_url: str, message: str, *, thread_id: str, timeout: float) -> dict[str, Any]:
    now = int(time.time() * 1000)
    return _request_json(
        "POST",
        base_url,
        "/chat",
        payload={
            "messages": [{"role": "user", "content": message}],
            "session_id": f"operator-lifecycle-smoke-{thread_id}",
            "thread_id": f"operator-lifecycle-smoke-{thread_id}",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"operator-lifecycle-smoke-{thread_id}-{now}",
        },
        timeout=timeout,
    )


def _assistant_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    return str(assistant.get("content") or payload.get("message") or "").strip()


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_flatten(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "\n".join(_flatten(item) for item in value)
    return str(value)


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=False)
    return str(result.stdout or "").strip()


def _pass(name: str, detail: str, command: str) -> Check:
    return Check(name, "PASS", detail, command)


def _fail(name: str, detail: str, command: str, next_action: str | None = None) -> Check:
    return Check(name, "FAIL", detail, command, next_action)


def _check_chat(
    base_url: str,
    timeout: float,
    *,
    name: str,
    prompt: str,
    thread_id: str,
    must_contain: tuple[str, ...] = (),
    must_contain_any: tuple[str, ...] = (),
    must_not_contain: tuple[str, ...] = (),
) -> Check:
    command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
    payload = _post_chat(base_url, prompt, thread_id=thread_id, timeout=timeout)
    text = _assistant_text(payload)
    combined = f"{text}\n{_flatten(payload)}"
    lowered = combined.lower()
    if _contains_secret(combined):
        return _fail(name, "response looked like it exposed a token or secret", command)
    missing = [item for item in must_contain if item.lower() not in lowered]
    if missing:
        return _fail(name, f"missing {missing}; text={text[:500]}", command)
    if must_contain_any and not any(item.lower() in lowered for item in must_contain_any):
        return _fail(name, f"missing one of {list(must_contain_any)}; text={text[:500]}", command)
    banned = [item for item in must_not_contain if item.lower() in lowered]
    if banned:
        return _fail(name, f"banned text {banned}; text={text[:500]}", command)
    return _pass(name, text[:500], command)


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    before = _git_status()
    for path in ("/ready", "/state"):
        command = f"GET {path}"
        payload = _request_json("GET", base_url, path, timeout=timeout)
        if payload.get("ok") is False:
            checks.append(_fail(command, _flatten(payload)[:500], command))
        else:
            checks.append(_pass(command, "surface returned coherent JSON", command))
    checks.extend(
        [
            _check_chat(
                base_url,
                timeout,
                name="assistant health",
                prompt="is the assistant healthy?",
                thread_id="health",
                must_contain_any=("Status:", "Doctor:"),
                must_not_contain=("i don't have direct access", "i do not have direct access"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="broken status",
                prompt="what is broken?",
                thread_id="broken",
                must_contain_any=("Status:", "Doctor:"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="storage usage",
                prompt="how much space is this using?",
                thread_id="storage",
                must_contain=("read-only estimate", "cleanup is a separate confirmation-gated action"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="repair preview",
                prompt="repair the assistant",
                thread_id="repair",
                must_contain=("Repair assistant preview", "explicit confirmation", "rollback scope"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="backup preview",
                prompt="back up the assistant",
                thread_id="backup",
                must_contain=("Backup assistant preview", "include local state", "secrets must remain encrypted or redacted"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="restore preview",
                prompt="restore from backup",
                thread_id="restore",
                must_contain=("Restore from backup preview", "safety snapshot", "Excluded:"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="update preview",
                prompt="update the assistant",
                thread_id="update",
                must_contain=("Update assistant preview", "Trusted source:", "Working tree clean:"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="cleanup preview",
                prompt="clean old runtime files",
                thread_id="cleanup",
                must_contain=("Cleanup old Personal Agent files preview", "show exact paths", "I did not delete anything"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="uninstall preview",
                prompt="uninstall the assistant",
                thread_id="uninstall",
                must_contain=("Uninstall assistant preview", "uninstall is destructive", "backup"),
                must_not_contain=("uninstalled", "removed personal agent"),
            ),
            _check_chat(
                base_url,
                timeout,
                name="support bundle preview",
                prompt="make a support bundle",
                thread_id="support",
                must_contain=("Support bundle preview", "redacted support bundle", "raw tokens"),
            ),
        ]
    )
    uninstall_thread = "uninstall-cancel"
    _post_chat(base_url, "uninstall the assistant", thread_id=uninstall_thread, timeout=timeout)
    cancel_payload = _post_chat(base_url, "no", thread_id=uninstall_thread, timeout=timeout)
    confirm_payload = _post_chat(base_url, "confirm", thread_id=uninstall_thread, timeout=timeout)
    confirm_text = _assistant_text(confirm_payload)
    confirm_combined = f"{confirm_text}\n{_flatten(confirm_payload)}".lower()
    if "uninstalled" in confirm_combined or "removed personal agent" in confirm_combined or "executed" in confirm_combined:
        checks.append(_fail("stale destructive confirmation", confirm_text[:500], 'POST /chat {"message": "confirm"}'))
    elif "no current action" in confirm_combined or "don't have a current action" in confirm_combined or "tell me what you want" in confirm_combined:
        checks.append(_pass("stale destructive confirmation", confirm_text[:500], 'POST /chat {"message": "confirm"}'))
    else:
        checks.append(_fail("stale destructive confirmation", confirm_text[:500], 'POST /chat {"message": "confirm"}'))
    cancel_text = _assistant_text(cancel_payload).lower()
    cancel_normalized = cancel_text.replace("’", "'")
    if "cancel" in cancel_normalized or "didn" in cancel_normalized or "no current action" in cancel_normalized or "don't have a current action" in cancel_normalized:
        checks.append(_pass("destructive preview cancel", _assistant_text(cancel_payload)[:500], 'POST /chat {"message": "no"}'))
    else:
        checks.append(_fail("destructive preview cancel", _assistant_text(cancel_payload)[:500], 'POST /chat {"message": "no"}'))
    after = _git_status()
    if after == before:
        checks.append(_pass("git status unchanged", "working tree status unchanged", "git status --short"))
    else:
        checks.append(_fail("git status unchanged", f"before={before!r} after={after!r}", "git status --short"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Installed Personal Agent operator lifecycle smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    print("# Personal Agent Operator Lifecycle Smoke")
    for check in checks:
        print(f"\n## {check.name}: {check.status}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.detail}")
        if check.next_action:
            print(f"- next action: {check.next_action}")
    pass_count = sum(1 for check in checks if check.status == "PASS")
    fail_count = sum(1 for check in checks if check.status == "FAIL")
    print("\n## Summary")
    print(f"PASS={pass_count} FAIL={fail_count}")
    print("OPERATOR_LIFECYCLE_SMOKE:", "pass" if fail_count == 0 else "fail")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
