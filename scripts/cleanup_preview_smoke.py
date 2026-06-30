#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
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
            parsed = json.loads(body or "{}")
        except json.JSONDecodeError:
            parsed = {"body": body}
        if isinstance(parsed, dict):
            parsed.setdefault("http_status", exc.code)
            return parsed
        return {"http_status": exc.code, "body": body}
    parsed = json.loads(body or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _post_chat(base_url: str, message: str, *, thread_id: str = "cleanup-preview", timeout: float = 30.0) -> dict[str, Any]:
    return _request_json(
        "POST",
        f"{base_url.rstrip('/')}/chat",
        payload={"message": message, "user_id": "cleanup-preview-smoke", "thread_id": thread_id},
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


def _has_oversized_backup() -> bool:
    backup_root = Path.home() / ".local/share/personal-agent/backups"
    if not backup_root.is_dir():
        return False
    for child in backup_root.iterdir():
        if not child.name.startswith("personal-agent-backup-"):
            continue
        try:
            size = sum(path.stat().st_size for path in child.glob("*.json"))
        except OSError:
            continue
        if size > 10 * 1024 * 1024:
            return True
    return False


def _has_valid_backup() -> bool:
    backup_root = Path.home() / ".local/share/personal-agent/backups"
    if not backup_root.is_dir():
        return False
    for manifest in backup_root.glob("personal-agent-backup-*/manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("backup_schema_version") == "backup.v1":
            return True
    return False


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    before = _git_status_short()
    expected_oversized = _has_oversized_backup()
    expected_valid_backup = _has_valid_backup()
    current_runtime_exists = (Path.home() / ".local/share/personal-agent/runtime/current").exists()

    preview = _post_chat(base_url, "clean old backup files", timeout=timeout)
    text = _assistant_text(preview)
    runtime = _runtime_payload(preview)
    plan = runtime.get("canonical_plan") if isinstance(runtime.get("canonical_plan"), dict) else {}
    cleanup = runtime.get("cleanup_preview") if isinstance(runtime.get("cleanup_preview"), dict) else {}
    candidates = cleanup.get("candidates") if isinstance(cleanup.get("candidates"), list) else []
    protected = cleanup.get("protected") if isinstance(cleanup.get("protected"), list) else []
    candidate_classes = {str(item.get("classification")) for item in candidates if isinstance(item, dict)}
    protected_classes = {str(item.get("classification")) for item in protected if isinstance(item, dict)}

    checks.append(
        _pass("cleanup preview uses Plan Mode v2", f"plan_id={plan.get('plan_id')} executor_status={plan.get('executor_status')}", 'POST /chat {"message": "clean old backup files"}')
        if "Plan Mode v2" in text and plan.get("action_type") == "operator.cleanup" and plan.get("executor_status") == "preview_only"
        else _fail("cleanup preview uses Plan Mode v2", json.dumps(preview, sort_keys=True)[:1400], 'POST /chat {"message": "clean old backup files"}')
    )
    checks.append(
        _pass("cleanup preview is read-only", text[:1000], 'POST /chat {"message": "clean old backup files"}')
        if "I did not delete anything." in text and runtime.get("mutated") is False
        else _fail("cleanup preview is read-only", text[:1400], 'POST /chat {"message": "clean old backup files"}')
    )
    checks.append(
        _pass("cleanup preview returns candidates or explicit none", f"candidates={len(candidates)} recoverable={cleanup.get('estimated_recoverable')}", "inspect cleanup_preview payload")
        if candidates or "No cleanup candidates found" in text
        else _fail("cleanup preview returns candidates or explicit none", json.dumps(runtime, sort_keys=True)[:1400], "inspect cleanup_preview payload")
    )
    checks.append(
        _pass("oversized backup candidates detected if present", ", ".join(sorted(candidate_classes)) or "no candidate classes", "inspect cleanup candidates")
        if (not expected_oversized) or "oversized backup artifact" in candidate_classes
        else _fail("oversized backup candidates detected if present", json.dumps(candidates, sort_keys=True)[:1600], "inspect cleanup candidates")
    )
    checks.append(
        _pass("latest valid backup is protected", ", ".join(sorted(protected_classes)), "inspect protected cleanup entries")
        if (not expected_valid_backup) or "protected latest valid backup" in protected_classes
        else _fail("latest valid backup is protected", json.dumps(protected, sort_keys=True)[:1600], "inspect protected cleanup entries")
    )
    checks.append(
        _pass("current runtime is protected", ", ".join(sorted(protected_classes)), "inspect protected cleanup entries")
        if (not current_runtime_exists) or "protected current runtime" in protected_classes
        else _fail("current runtime is protected", json.dumps(protected, sort_keys=True)[:1600], "inspect protected cleanup entries")
    )
    checks.append(
        _pass("protected rules include secrets and service files", ", ".join(sorted(protected_classes)), "inspect protected cleanup entries")
        if {"protected secret store", "protected active service files"}.issubset(protected_classes)
        else _fail("protected rules include secrets and service files", json.dumps(protected, sort_keys=True)[:1600], "inspect protected cleanup entries")
    )
    unsafe_safe = [item for item in candidates if isinstance(item, dict) and item.get("classification") == "unknown/unsafe candidate" and item.get("safe_to_delete_later")]
    checks.append(
        _pass("unknown candidates are not marked safe", "unknown/unsafe candidates are protected from future deletion", "inspect cleanup candidates")
        if not unsafe_safe
        else _fail("unknown candidates are not marked safe", json.dumps(unsafe_safe, sort_keys=True)[:1600], "inspect cleanup candidates")
    )

    confirm = _post_chat(base_url, "yes", timeout=timeout)
    confirm_runtime = _runtime_payload(confirm)
    executor_result = confirm_runtime.get("executor_result") if isinstance(confirm_runtime.get("executor_result"), dict) else {}
    checks.append(
        _pass("cleanup confirmation does not mutate", json.dumps(executor_result, sort_keys=True)[:1000], 'POST /chat {"message": "yes"}')
        if executor_result.get("error_code") == "executor_not_enabled" and executor_result.get("mutated") is False
        else _fail("cleanup confirmation does not mutate", json.dumps(confirm, sort_keys=True)[:1400], 'POST /chat {"message": "yes"}')
    )

    after = _git_status_short()
    checks.append(
        _pass("git status unchanged", "working tree status unchanged", "git status --short")
        if after == before
        else _fail("git status unchanged", f"before={before!r} after={after!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Installed-product cleanup preview smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed

    print("# Personal Agent Cleanup Preview Smoke")
    for check in checks:
        print()
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
    print()
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"CLEANUP_PREVIEW_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
