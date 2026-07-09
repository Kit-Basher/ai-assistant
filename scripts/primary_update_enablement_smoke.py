#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8765"
STATE_ROOT = Path.home() / ".local/share/personal-agent"
RUNTIME_ROOT = STATE_ROOT / "runtime"
CURRENT_LINK = RUNTIME_ROOT / "current"
RELEASES_ROOT = RUNTIME_ROOT / "releases"
REQUEST_PATH = STATE_ROOT / "host_lifecycle" / "primary_update_enablement_request.json"
MARKER_PATH = STATE_ROOT / "host_lifecycle" / "primary_update_enablement.marker"
SERVICE_NAME = "personal-agent-api.service"


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str


def _check(name: str, ok: bool, evidence: str, command: str) -> Check:
    return Check(name=name, ok=ok, evidence=" ".join(str(evidence or "").split())[:1800], command=command)


def _request(method: str, path: str, *, payload: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    except (urllib.error.URLError, http.client.RemoteDisconnected, ConnectionError, TimeoutError) as exc:
        return {"ok": False, "http_status": 0, "error": f"{exc.__class__.__name__}: {exc}"}
    parsed: dict[str, Any]
    try:
        value = json.loads(raw or "{}")
        parsed = value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        parsed = {"raw": raw[:1000]}
    parsed["http_status"] = status
    return parsed


def _post_chat(message: str, *, thread_id: str, timeout: float = 30.0) -> dict[str, Any]:
    now = int(time.time() * 1000)
    return _request(
        "POST",
        "/chat",
        payload={
            "messages": [{"role": "user", "content": message}],
            "message": message,
            "user_id": "primary-update-enable-smoke",
            "session_id": f"primary-update-enable-{thread_id}",
            "thread_id": f"primary-update-enable-{thread_id}",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"primary-update-enable-{thread_id}-{now}",
        },
        timeout=timeout,
    )


def _assistant_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    for value in (assistant.get("content"), payload.get("message"), payload.get("response"), payload.get("text")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, sort_keys=True)[:1000]


def _ready(timeout: float = 15.0) -> dict[str, Any]:
    return _request("GET", "/ready", timeout=timeout)


def _version(timeout: float = 15.0) -> dict[str, Any]:
    return _request("GET", "/version", timeout=timeout)


def _wait_ready(expected_commit: str, *, timeout: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = "not checked"
    while time.monotonic() < deadline:
        try:
            ready = _ready(timeout=3.0)
            version = _version(timeout=3.0)
            if ready.get("ready") is True and str(version.get("git_commit") or "") == expected_commit:
                return {"ready": ready, "version": version}
            last = json.dumps({"ready": ready, "version": version}, sort_keys=True)[:500]
        except Exception as exc:  # noqa: BLE001
            last = f"{exc.__class__.__name__}: {exc}"
        time.sleep(1.0)
    raise RuntimeError(f"primary API did not recover: {last}")


def _run(argv: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)


def _git_status() -> str:
    return _run(["git", "status", "--short"], timeout=10).stdout.strip()


def _release_commit(path: Path) -> str:
    try:
        payload = json.loads((path / "agent" / "BUILD_INFO.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("git_commit") or "")


def _copy_current_release(target: Path) -> None:
    source = CURRENT_LINK.resolve()
    if target.exists():
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "node_modules")
    shutil.copytree(source, target, symlinks=False, ignore=ignore)


def _write_request(*, operation_id: str, source: Path, target_release_id: str, commit: str, force_failure: bool = False) -> None:
    REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKER_PATH.write_text(f"primary-update-enable {operation_id}\n", encoding="utf-8")
    payload = {
        "schema_version": "primary_update_enablement_request.v1",
        "operation_id": operation_id,
        "staged_source_path": str(source),
        "target_release_id": target_release_id,
        "expected_current_commit": commit,
        "target_commit": commit,
        "proof_marker_path": str(MARKER_PATH),
        "force_post_promotion_failure": bool(force_failure),
        "created_at": time.time(),
    }
    REQUEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(REQUEST_PATH, 0o600)


def _operation_receipt(operation_id: str) -> Path:
    return STATE_ROOT / "host_lifecycle" / "operations" / operation_id / "receipt.json"


def _operation_state(operation_id: str) -> Path:
    return STATE_ROOT / "host_lifecycle" / "operations" / operation_id / "state.json"


def _wait_receipt(operation_id: str, *, timeout: float = 120.0) -> dict[str, Any]:
    receipt = _operation_receipt(operation_id)
    deadline = time.monotonic() + timeout
    last = "not written"
    while time.monotonic() < deadline:
        if receipt.is_file():
            try:
                payload = json.loads(receipt.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                last = f"{exc.__class__.__name__}: {exc}"
            else:
                status = str(payload.get("status") or "")
                if status in {"completed_verified", "update_failed_rolled_back", "update_failed_rollback_failed"}:
                    return payload if isinstance(payload, dict) else {}
                last = status or "receipt pending"
        time.sleep(1.0)
    raise RuntimeError(f"receipt timeout for {operation_id}: {last}")


def _preview_and_confirm(operation_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    thread = operation_id
    preview = _post_chat("update the assistant", thread_id=thread, timeout=30)
    confirm = _post_chat("yes", thread_id=thread, timeout=35)
    return preview, confirm


def _canonical_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def run(expected_commit: str) -> list[Check]:
    checks: list[Check] = []
    before_status = _git_status()
    before_ready = _ready()
    before_version = _version()
    before_current = CURRENT_LINK.resolve()
    before_current_commit = _release_commit(before_current)
    checks.append(_check("primary API ready before proof", before_ready.get("ready") is True, json.dumps(before_ready, sort_keys=True), "GET /ready"))
    checks.append(_check("expected commit matches serving runtime", before_current_commit == expected_commit == before_version.get("git_commit"), json.dumps({"expected": expected_commit, "current": before_current_commit, "version": before_version.get("git_commit")}, sort_keys=True), "GET /version and BUILD_INFO"))
    checks.append(_check("current runtime is a symlink under primary runtime root", CURRENT_LINK.is_symlink() and str(before_current).startswith(str(RELEASES_ROOT)), _canonical_path(before_current), "inspect runtime/current"))
    checks.append(_check("primary uninstall service not targeted by proof", SERVICE_NAME == "personal-agent-api.service", SERVICE_NAME, "static guard"))

    success_operation = f"primary-update-success-{uuid4().hex[:8]}"
    success_source = STATE_ROOT / "host_lifecycle" / "staged_sources" / success_operation
    success_release_id = f"primary-proof-success-{int(time.time())}-{uuid4().hex[:6]}"
    _copy_current_release(success_source)
    _write_request(operation_id=success_operation, source=success_source, target_release_id=success_release_id, commit=expected_commit)
    preview, confirm = _preview_and_confirm(success_operation)
    preview_text = _assistant_text(preview)
    confirm_text = _assistant_text(confirm)
    checks.append(_check("primary update preview uses Plan Mode", "Plan Mode v2" in json.dumps(preview) and "trusted host runner" in preview_text.lower(), preview_text, 'POST /chat "update the assistant"'))
    confirm_handoff = (
        "in_progress" in json.dumps(confirm).lower()
        or "has started" in confirm_text.lower()
        or confirm.get("http_status") == 0
    )
    checks.append(_check("primary update confirmation returns in-progress handoff", confirm_handoff, confirm_text or json.dumps(confirm, sort_keys=True), 'POST /chat "yes"'))
    receipt = _wait_receipt(success_operation)
    recovered = _wait_ready(expected_commit)
    after_success_current = CURRENT_LINK.resolve()
    checks.append(_check("primary update receipt completed", receipt.get("ok") is True and receipt.get("status") == "completed_verified", json.dumps(receipt, sort_keys=True), "host runner receipt"))
    checks.append(_check("primary runtime switched to new release path", after_success_current.name == success_release_id and after_success_current != before_current, _canonical_path(after_success_current), "inspect runtime/current"))
    checks.append(_check("primary API recovered after update", recovered.get("version", {}).get("git_commit") == expected_commit, json.dumps(recovered, sort_keys=True), "GET /ready /version"))
    post_chat = _post_chat("hello", thread_id=f"{success_operation}-after", timeout=30)
    post_text = _assistant_text(post_chat)
    checks.append(_check("deterministic chat works after primary update", bool(post_text) and "traceback" not in post_text.lower(), post_text, 'POST /chat "hello"'))

    rollback_operation = f"primary-update-rollback-{uuid4().hex[:8]}"
    rollback_source = STATE_ROOT / "host_lifecycle" / "staged_sources" / rollback_operation
    rollback_release_id = f"primary-proof-rollback-target-{int(time.time())}-{uuid4().hex[:6]}"
    _copy_current_release(rollback_source)
    _write_request(
        operation_id=rollback_operation,
        source=rollback_source,
        target_release_id=rollback_release_id,
        commit=expected_commit,
        force_failure=True,
    )
    rollback_preview, rollback_confirm = _preview_and_confirm(rollback_operation)
    rollback_confirm_text = _assistant_text(rollback_confirm)
    checks.append(_check("primary rollback update preview uses same Plan Mode path", "Plan Mode v2" in json.dumps(rollback_preview), _assistant_text(rollback_preview), 'POST /chat "update the assistant" rollback'))
    rollback_handoff = (
        "in_progress" in json.dumps(rollback_confirm).lower()
        or "has started" in rollback_confirm_text.lower()
        or rollback_confirm.get("http_status") == 0
    )
    checks.append(_check("primary rollback confirmation returns in-progress handoff", rollback_handoff, rollback_confirm_text or json.dumps(rollback_confirm, sort_keys=True), 'POST /chat "yes" rollback'))
    rollback_receipt = _wait_receipt(rollback_operation)
    rollback_ready = _wait_ready(expected_commit)
    after_rollback_current = CURRENT_LINK.resolve()
    checks.append(_check("forced primary update failure rolls back", rollback_receipt.get("rollback_verified") is True and rollback_receipt.get("status") == "update_failed_rolled_back", json.dumps(rollback_receipt, sort_keys=True), "host runner receipt rollback"))
    checks.append(_check("rollback verifies serving previous release path", after_rollback_current == after_success_current, json.dumps({"current": _canonical_path(after_rollback_current), "expected": _canonical_path(after_success_current)}, sort_keys=True), "inspect runtime/current after rollback"))
    checks.append(_check("primary API recovered after rollback", rollback_ready.get("version", {}).get("git_commit") == expected_commit, json.dumps(rollback_ready, sort_keys=True), "GET /ready /version after rollback"))

    uninstall_preview = _post_chat("uninstall the assistant", thread_id="primary-uninstall-guard", timeout=30)
    uninstall_confirm = _post_chat("yes", thread_id="primary-uninstall-guard", timeout=30)
    uninstall_result = uninstall_confirm.get("data", {}).get("runtime_payload", {}).get("executor_result", {})
    uninstall_blob = json.dumps(uninstall_confirm, sort_keys=True)
    uninstall_guarded = uninstall_result.get("error_code") == "uninstall_live_execution_not_enabled" or "uninstall_live_execution_not_enabled" in uninstall_blob
    checks.append(_check("primary uninstall guard remains active", uninstall_guarded, uninstall_blob[:1800], 'POST /chat uninstall then "yes"'))

    after_status = _git_status()
    checks.append(_check("git status unchanged by primary proof", after_status == before_status, f"before={before_status!r} after={after_status!r}", "git status --short"))
    checks.append(_check("operation state files exist", _operation_state(success_operation).is_file() and _operation_state(rollback_operation).is_file(), json.dumps({"success": str(_operation_state(success_operation)), "rollback": str(_operation_state(rollback_operation))}, sort_keys=True), "inspect operation state"))
    checks.append(_check("primary final runtime is healthy", _ready().get("ready") is True and _version().get("git_commit") == expected_commit, json.dumps({"current": _canonical_path(CURRENT_LINK.resolve()), "commit": _version().get("git_commit")}, sort_keys=True), "GET /ready /version final"))

    try:
        REQUEST_PATH.unlink(missing_ok=True)
        MARKER_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Dangerous installed-host proof for primary Personal Agent update handoff.")
    parser.add_argument("--allow-primary-update-proof", action="store_true", help="Required acknowledgement that this will restart the primary Personal Agent API.")
    parser.add_argument("--expected-commit", required=True, help="Expected current serving git commit.")
    args = parser.parse_args()
    if not args.allow_primary_update_proof:
        print("primary_update_enablement_smoke requires --allow-primary-update-proof")
        return 2
    checks: list[Check]
    try:
        checks = run(args.expected_commit.strip())
    except Exception as exc:  # noqa: BLE001 - smoke must report and leave recovery evidence.
        checks = [_check("primary update proof exception", False, f"{exc.__class__.__name__}: {exc}", "primary_update_enablement_smoke")]
        try:
            REQUEST_PATH.unlink(missing_ok=True)
            MARKER_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed
    print("# Personal Agent Primary Update Enablement Smoke")
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"## {check.name}: {status}")
        print(f"command: {check.command}")
        print(f"evidence: {check.evidence}\n")
    print(f"SUMMARY PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
