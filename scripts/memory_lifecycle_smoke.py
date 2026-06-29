#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
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
        return {"ok": False, "http_status": int(exc.code), "raw": raw[:500]}
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": "non_object_json"}


def _post_chat(base_url: str, message: str, *, thread_id: str, timeout: float) -> dict[str, Any]:
    now = int(time.time() * 1000)
    return _request_json(
        "POST",
        base_url,
        "/chat",
        payload={
            "messages": [{"role": "user", "content": message}],
            "session_id": f"memory-lifecycle-smoke-{thread_id}",
            "thread_id": f"memory-lifecycle-smoke-{thread_id}",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"memory-lifecycle-smoke-{thread_id}-{now}",
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


def _fail(name: str, detail: str, command: str) -> Check:
    return Check(name, "FAIL", detail, command)


def _check_chat(
    base_url: str,
    timeout: float,
    *,
    name: str,
    prompt: str,
    thread_id: str,
    must_contain: tuple[str, ...] = (),
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
    banned = [item for item in must_not_contain if item.lower() in lowered]
    if banned:
        return _fail(name, f"banned text {banned}; text={text[:500]}", command)
    return _pass(name, text[:500], command)


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    before = _git_status()
    checks.append(
        _check_chat(
            base_url,
            timeout,
            name="memory inspect",
            prompt="what do you remember about me?",
            thread_id="inspect",
            must_contain=("saved",),
            must_not_contain=("definitely remember your", "your password", "bearer "),
        )
    )
    checks.append(
        _check_chat(
            base_url,
            timeout,
            name="memory status",
            prompt="show memory status",
            thread_id="status",
            must_contain=("saved long-term memory", "current thread context", "pending confirmations/actions", "external tools"),
        )
    )
    checks.append(
        _check_chat(
            base_url,
            timeout,
            name="current-turn no memory",
            prompt="do not use memory for this",
            thread_id="nomem",
            must_contain=("saved memory", "prior conversation context", "this turn", "current-turn tools"),
            must_not_contain=("external information", "external info", "search", "web"),
        )
    )
    preview_cases = (
        ("disable memory for this thread", "thread-disable", ("Disable memory for this thread preview", "current thread only", "explicit confirmation")),
        ("enable memory for this thread", "thread-enable", ("Enable memory for this thread preview", "current thread only", "explicit confirmation")),
        ("disable memory globally", "global-disable", ("Disable memory globally preview", "all users/threads", "explicit confirmation")),
        ("enable memory globally", "global-enable", ("Enable memory globally preview", "all users/threads", "explicit confirmation")),
        ("forget what you remember about cats", "forget-topic", ("Forget topic memory preview", "matching saved memory records", "explicit confirmation")),
        ("delete all memory about me", "delete-all", ("Delete all memory about me preview", "destructive", "explicit confirmation")),
        ("export my memory", "export", ("Export my memory preview", "secrets redacted", "explicit confirmation")),
        ("redact sensitive memory", "redact", ("Redact sensitive memory preview", "raw secrets must not be printed", "explicit confirmation")),
        ("clean up duplicate memories", "cleanup", ("Clean up duplicate memories preview", "duplicate memory candidates", "explicit confirmation")),
    )
    for prompt, thread, contains in preview_cases:
        checks.append(
            _check_chat(
                base_url,
                timeout,
                name=f"memory preview: {thread}",
                prompt=prompt,
                thread_id=thread,
                must_contain=contains + ("scope distinction",),
                must_not_contain=(
                    "deleted all memory",
                    "memory exported successfully",
                    "memory redacted successfully",
                    "deduplicated memory",
                ),
            )
        )
    delete_thread = "delete-cancel"
    _post_chat(base_url, "delete all memory about me", thread_id=delete_thread, timeout=timeout)
    cancel_payload = _post_chat(base_url, "no", thread_id=delete_thread, timeout=timeout)
    confirm_payload = _post_chat(base_url, "confirm", thread_id=delete_thread, timeout=timeout)
    cancel_text = _assistant_text(cancel_payload)
    confirm_text = _assistant_text(confirm_payload)
    confirm_lower = f"{confirm_text}\n{_flatten(confirm_payload)}".lower()
    if "deleted" in confirm_lower or "exported" in confirm_lower or "redacted" in confirm_lower:
        checks.append(_fail("stale memory confirmation", confirm_text[:500], 'POST /chat {"message": "confirm"}'))
    elif "no current action" in confirm_lower or "don't have a current action" in confirm_lower or "tell me what you want" in confirm_lower:
        checks.append(_pass("stale memory confirmation", confirm_text[:500], 'POST /chat {"message": "confirm"}'))
    else:
        checks.append(_fail("stale memory confirmation", confirm_text[:500], 'POST /chat {"message": "confirm"}'))
    if "cancel" in cancel_text.lower() or "didn" in cancel_text.lower():
        checks.append(_pass("memory preview cancel", cancel_text[:500], 'POST /chat {"message": "no"}'))
    else:
        checks.append(_fail("memory preview cancel", cancel_text[:500], 'POST /chat {"message": "no"}'))
    after = _git_status()
    checks.append(
        _pass("git status unchanged", "working tree status unchanged", "git status --short")
        if before == after
        else _fail("git status unchanged", f"before={before!r} after={after!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Installed Personal Agent memory lifecycle smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    print("# Personal Agent Memory Lifecycle Smoke")
    for check in checks:
        print(f"\n## {check.name}: {check.status}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.detail}")
    pass_count = sum(1 for check in checks if check.status == "PASS")
    fail_count = sum(1 for check in checks if check.status == "FAIL")
    print("\n## Summary")
    print(f"PASS={pass_count} FAIL={fail_count}")
    print("MEMORY_LIFECYCLE_SMOKE:", "pass" if fail_count == 0 else "fail")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
