#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.executor_registry import ExecutorRegistry, ExecutorSpec, execute_cleanup


DEFAULT_BASE_URL = "http://127.0.0.1:8765"


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str


def _pass(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=True, evidence=evidence.strip()[:1200], command=command)


def _fail(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=False, evidence=evidence.strip()[:1600], command=command)


def _request_json(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _post_chat(base_url: str, message: str, *, thread_id: str, timeout: float) -> dict[str, Any]:
    return _request_json(
        "POST",
        f"{base_url.rstrip('/')}/chat",
        payload={"message": message, "user_id": "cleanup-execution-smoke", "thread_id": thread_id},
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


def _git_status_short() -> str:
    result = subprocess.run(["git", "status", "--short"], check=False, text=True, capture_output=True, timeout=10)
    return result.stdout.strip()


def _fixture_size(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for child in path.rglob("*"):
        total += child.lstat().st_size
        count += 1
    return total, count


def _run_isolated_registry_fixture() -> list[Check]:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        tmp_root = root / "tmp"
        tmp_root.mkdir()
        artifact = tmp_root / "personal-agent-support-old"
        artifact.mkdir()
        (artifact / "summary.json").write_text("{}", encoding="utf-8")
        size, count = _fixture_size(artifact)
        preview = {
            "candidates": [
                {
                    "path": str(artifact),
                    "canonical_path": str(artifact.resolve()),
                    "classification": "old support bundle artifact",
                    "safe_to_delete_later": True,
                    "size_bytes": size,
                    "file_count": count,
                    "reason": "isolated cleanup execution fixture",
                }
            ],
            "protected": [],
        }
        registry = ExecutorRegistry(root / "executor_registry_journal.jsonl")
        registry.register(
            ExecutorSpec(
                executor_id="operator.cleanup.v1",
                action_type="operator.cleanup",
                status="enabled",
                run=execute_cleanup,
                rollback_available=False,
                rollback_hint="Cleanup deletion is not automatically reversible.",
            )
        )
        plan = {
            "plan_id": "confirm-cleanup-fixture",
            "action_type": "operator.cleanup",
            "target": "isolated cleanup fixture",
            "risk_level": "high",
            "executor_status": "enabled",
        }
        action = {"pending_id": "confirm-cleanup-fixture", "cleanup_preview": preview}
        with patch("tempfile.gettempdir", return_value=str(tmp_root)):
            result = registry.execute_confirmed_plan(plan=plan, action=action)
        checks.append(
            _pass("isolated cleanup deletes approved fixture", result.user_message, "ExecutorRegistry.execute_confirmed_plan")
            if result.ok and result.mutated and not artifact.exists() and result.executor_id == "operator.cleanup.v1"
            else _fail("isolated cleanup deletes approved fixture", json.dumps(result.to_dict(), sort_keys=True), "ExecutorRegistry.execute_confirmed_plan")
        )
        recent = registry.journal.recent(limit=5)
        checks.append(
            _pass("cleanup execution is journaled", json.dumps(recent[-1], sort_keys=True)[:1000], "read isolated executor journal")
            if recent and recent[-1].get("result", {}).get("action_type") == "operator.cleanup"
            else _fail("cleanup execution is journaled", json.dumps(recent, sort_keys=True), "read isolated executor journal")
        )
    return checks


def run(base_url: str, timeout: float) -> list[Check]:
    checks = _run_isolated_registry_fixture()
    before = _git_status_short()
    preview = _post_chat(base_url, "clean old runtime files", thread_id="cleanup-exec-installed", timeout=timeout)
    text = _assistant_text(preview)
    runtime = _runtime_payload(preview)
    plan = runtime.get("canonical_plan") if isinstance(runtime.get("canonical_plan"), dict) else {}
    checks.append(
        _pass("installed cleanup preview exposes enabled executor", f"plan_id={plan.get('plan_id')} action={plan.get('action_type')}", 'POST /chat {"message": "clean old runtime files"}')
        if "Plan Mode v2" in text and plan.get("action_type") == "operator.cleanup" and plan.get("executor_status") == "enabled"
        else _fail("installed cleanup preview exposes enabled executor", json.dumps(preview, sort_keys=True)[:1400], 'POST /chat {"message": "clean old runtime files"}')
    )
    cancel = _post_chat(base_url, "no", thread_id="cleanup-exec-installed", timeout=timeout)
    cancel_text = _assistant_text(cancel).lower()
    checks.append(
        _pass("installed cleanup plan cancels without deleting live artifacts", _assistant_text(cancel), 'POST /chat {"message": "no"}')
        if "cancel" in cancel_text
        else _fail("installed cleanup plan cancels without deleting live artifacts", json.dumps(cancel, sort_keys=True)[:1400], 'POST /chat {"message": "no"}')
    )
    after = _git_status_short()
    checks.append(
        _pass("git status unchanged", "working tree status unchanged", "git status --short")
        if before == after
        else _fail("git status unchanged", f"before={before!r} after={after!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup execution smoke using isolated destructive fixture plus installed preview.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed
    print("# Personal Agent Cleanup Execution Smoke")
    for check in checks:
        print()
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
    print()
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"CLEANUP_EXECUTION_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
