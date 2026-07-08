#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
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


def _post_chat(base_url: str, message: str, *, thread_id: str = "restore-validator", timeout: float = 30.0) -> dict[str, Any]:
    return _request_json(
        "POST",
        f"{base_url.rstrip('/')}/chat",
        payload={"message": message, "user_id": "restore-validator-smoke", "thread_id": thread_id},
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
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    value = assistant.get("content")
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


def _latest_valid_backup_path(payload: dict[str, Any]) -> str:
    runtime = _runtime_payload(payload)
    latest = str(runtime.get("latest_valid_backup") or "").strip()
    if latest.startswith("~/"):
        return str(Path.home() / latest[2:])
    backups = runtime.get("backups") if isinstance(runtime.get("backups"), list) else []
    for item in backups:
        if isinstance(item, dict) and item.get("latest_valid"):
            absolute = str(item.get("absolute_path") or "").strip()
            if absolute:
                return absolute
    return latest


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    before = _git_status_short()

    listed = _post_chat(base_url, "show my backups", timeout=timeout)
    listed_text = _assistant_text(listed)
    listed_runtime = _runtime_payload(listed)
    backups = listed_runtime.get("backups") if isinstance(listed_runtime.get("backups"), list) else []
    latest_path = _latest_valid_backup_path(listed)
    checks.append(
        _pass("show my backups lists artifacts", f"backups={len(backups)} latest={latest_path}", 'POST /chat {"message": "show my backups"}')
        if backups and latest_path
        else _fail("show my backups lists artifacts", json.dumps(listed, sort_keys=True)[:1400], 'POST /chat {"message": "show my backups"}')
    )
    checks.append(
        _pass("show my backups is read-only", listed_text[:1000], 'POST /chat {"message": "show my backups"}')
        if "read-only" in listed_text.lower() and listed_runtime.get("mutated") is False
        else _fail("show my backups is read-only", listed_text[:1400], 'POST /chat {"message": "show my backups"}')
    )

    validated = _post_chat(base_url, f"validate this backup: {latest_path}", thread_id="restore-validator-valid", timeout=timeout)
    validated_text = _assistant_text(validated)
    validated_runtime = _runtime_payload(validated)
    included = validated_runtime.get("included_files") if isinstance(validated_runtime.get("included_files"), list) else []
    checks.append(
        _pass("latest valid backup validates", validated_text[:1000], 'POST /chat {"message": "validate this backup: <latest>"}')
        if validated_runtime.get("valid") is True
        and "Backup validation result: valid" in validated_text
        and "Validation is read-only" in validated_text
        and validated_runtime.get("mutated") is False
        else _fail("latest valid backup validates", json.dumps(validated, sort_keys=True)[:1600], 'POST /chat {"message": "validate this backup: <latest>"}')
    )
    checks.append(
        _pass("validator reports manifest metadata", f"schema={validated_runtime.get('schema_version')} files={len(included)}", "inspect validator payload")
        if validated_runtime.get("schema_version") == "backup.v1"
        and validated_runtime.get("created_at")
        and validated_runtime.get("runtime_commit")
        and "manifest.json" in included
        else _fail("validator reports manifest metadata", json.dumps(validated_runtime, sort_keys=True)[:1600], "inspect validator payload")
    )

    unsafe = _post_chat(base_url, "inspect backup /etc", thread_id="restore-validator-unsafe", timeout=timeout)
    unsafe_text = _assistant_text(unsafe)
    unsafe_runtime = _runtime_payload(unsafe)
    checks.append(
        _pass("validator rejects unsafe outside path", unsafe_text[:1000], 'POST /chat {"message": "inspect backup /etc"}')
        if unsafe_runtime.get("valid") is False
        and unsafe_runtime.get("error") == "backup_path_outside_approved_locations"
        and unsafe_runtime.get("mutated") is False
        else _fail("validator rejects unsafe outside path", json.dumps(unsafe, sort_keys=True)[:1600], 'POST /chat {"message": "inspect backup /etc"}')
    )

    with tempfile.TemporaryDirectory(prefix="personal-agent-restore-validator-") as temp_dir:
        malformed_dir = Path(temp_dir) / "personal-agent-backup-malformed"
        malformed_dir.mkdir()
        malformed = _post_chat(
            base_url,
            f"validate this backup: {malformed_dir}",
            thread_id="restore-validator-malformed",
            timeout=timeout,
        )
    malformed_text = _assistant_text(malformed)
    malformed_runtime = _runtime_payload(malformed)
    checks.append(
        _pass("validator detects missing manifest", malformed_text[:1000], 'POST /chat {"message": "validate this backup: <tmp missing manifest>"}')
        if malformed_runtime.get("valid") is False
        and malformed_runtime.get("error") == "manifest_missing"
        and malformed_runtime.get("mutated") is False
        else _fail("validator detects missing manifest", json.dumps(malformed, sort_keys=True)[:1600], 'POST /chat {"message": "validate this backup: <tmp missing manifest>"}')
    )

    restore_preview = _post_chat(base_url, "restore from backup", thread_id="restore-validator-preview", timeout=timeout)
    restore_runtime = _runtime_payload(restore_preview)
    restore_plan = restore_runtime.get("canonical_plan") if isinstance(restore_runtime.get("canonical_plan"), dict) else {}
    checks.append(
        _pass("restore preview is enabled and validation-gated", f"plan_id={restore_plan.get('plan_id')} executor_status={restore_plan.get('executor_status')}", 'POST /chat {"message": "restore from backup"}')
        if restore_plan.get("action_type") == "operator.restore"
        and restore_plan.get("executor_status") == "enabled"
        and "safety snapshot" in _assistant_text(restore_preview).lower()
        else _fail("restore preview is enabled and validation-gated", json.dumps(restore_preview, sort_keys=True)[:1400], 'POST /chat {"message": "restore from backup"}')
    )
    restore_cancel = _post_chat(base_url, "no", thread_id="restore-validator-preview", timeout=timeout)
    cancel_text = _assistant_text(restore_cancel).lower()
    checks.append(
        _pass("restore preview can be cancelled without mutation", _assistant_text(restore_cancel)[:1000], 'POST /chat {"message": "no"}')
        if "cancel" in cancel_text and "mutated=true" not in cancel_text
        else _fail("restore preview can be cancelled without mutation", json.dumps(restore_cancel, sort_keys=True)[:1400], 'POST /chat {"message": "no"}')
    )

    after = _git_status_short()
    checks.append(
        _pass("git status unchanged", "working tree status unchanged", "git status --short")
        if after == before
        else _fail("git status unchanged", f"before={before!r} after={after!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Installed-product Backup v1 restore validator smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed

    print("# Personal Agent Restore Validator Smoke")
    for check in checks:
        print()
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
    print()
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"RESTORE_VALIDATOR_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
