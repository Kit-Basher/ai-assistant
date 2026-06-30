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


EXPECTED_FILES = {
    "manifest.json",
    "doctor_summary.json",
    "version.json",
    "ready.json",
    "state_summary.json",
    "search_status.json",
    "telegram_status.json",
    "packs_state_summary.json",
    "executor_registry_journal_summary.json",
    "readiness_proof_summary.json",
    "git_runtime_freshness.json",
    "support_summary.json",
}


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


def _post_chat(base_url: str, message: str, *, thread_id: str = "support-bundle-v2", timeout: float = 30.0) -> dict[str, Any]:
    return _request_json(
        "POST",
        f"{base_url.rstrip('/')}/chat",
        payload={"message": message, "user_id": "support-bundle-v2-smoke", "thread_id": thread_id},
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
    return {}


def _pass(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=True, evidence=evidence.strip()[:1000], command=command)


def _fail(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=False, evidence=evidence.strip()[:1600], command=command)


def _git_status_short() -> str:
    result = subprocess.run(["git", "status", "--short"], check=False, text=True, capture_output=True, timeout=10)
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    baseline_git = _git_status_short()

    preview = _post_chat(base_url, "make a support bundle", timeout=timeout)
    preview_text = _assistant_text(preview)
    preview_payload = _runtime_payload(preview)
    plan = preview_payload.get("canonical_plan") if isinstance(preview_payload.get("canonical_plan"), dict) else {}
    checks.append(
        _pass("support bundle preview uses Plan Mode v2", f"plan_id={plan.get('plan_id')} executor_status={plan.get('executor_status')}", 'POST /chat {"message": "make a support bundle"}')
        if "Plan Mode v2" in preview_text and plan.get("action_type") == "operator.support_bundle" and plan.get("executor_status") == "enabled"
        else _fail("support bundle preview uses Plan Mode v2", json.dumps(preview, sort_keys=True)[:1400], 'POST /chat {"message": "make a support bundle"}')
    )

    confirmed = _post_chat(base_url, "yes", timeout=timeout)
    confirmed_text = _assistant_text(confirmed)
    runtime = _runtime_payload(confirmed)
    executor_result = runtime.get("executor_result") if isinstance(runtime.get("executor_result"), dict) else {}
    details = executor_result.get("details") if isinstance(executor_result.get("details"), dict) else {}
    artifact_path = Path(str(details.get("artifact_path") or ""))
    checks.append(
        _pass("confirmation executes through executor registry", json.dumps(executor_result, sort_keys=True)[:1000], 'POST /chat {"message": "yes"}')
        if bool(executor_result.get("ok")) and bool(executor_result.get("mutated")) and str(executor_result.get("executor_id")) == "operator.support_bundle.v1" and bool(executor_result.get("journal_id"))
        else _fail("confirmation executes through executor registry", json.dumps(confirmed, sort_keys=True)[:1600], 'POST /chat {"message": "yes"}')
    )
    checks.append(
        _pass("chat response gives local bundle path", confirmed_text, 'POST /chat {"message": "yes"}')
        if str(artifact_path) and str(artifact_path) in confirmed_text
        else _fail("chat response gives local bundle path", confirmed_text, 'POST /chat {"message": "yes"}')
    )
    checks.append(
        _pass("bundle artifact exists", str(artifact_path), "inspect support bundle path")
        if artifact_path.exists() and artifact_path.is_dir()
        else _fail("bundle artifact exists", str(artifact_path), "inspect support bundle path")
    )

    if artifact_path.exists():
        manifest_path = artifact_path / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        included = set(str(item) for item in manifest.get("included_files", []) if str(item).strip())
        missing = sorted(EXPECTED_FILES - included)
        checks.append(
            _pass("manifest exists and lists expected files", json.dumps(manifest, sort_keys=True)[:1000], "inspect manifest.json")
            if manifest.get("bundle_schema_version") == "support_bundle.v2" and not missing
            else _fail("manifest exists and lists expected files", f"missing={missing}; manifest={json.dumps(manifest, sort_keys=True)[:1200]}", "inspect manifest.json")
        )
        missing_files = sorted(name for name in EXPECTED_FILES if not (artifact_path / name).is_file())
        checks.append(
            _pass("expected summary files exist", ", ".join(sorted(EXPECTED_FILES)), "inspect support bundle files")
            if not missing_files
            else _fail("expected summary files exist", f"missing_files={missing_files}", "inspect support bundle files")
        )
        combined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in artifact_path.glob("*.json"))
        forbidden_values = (
            "123456:telegram",
            "bot token",
            "bearer abc",
            "sk-test",
            "password=letmein",
            "ultrasecretkey",
            "correct horse battery staple",
        )
        checks.append(
            _pass("bundle files do not contain obvious raw secrets", "no forbidden raw secret samples found", "scan support bundle files")
            if not any(value.lower() in combined.lower() for value in forbidden_values)
            else _fail("bundle files do not contain obvious raw secrets", combined[:1600], "scan support bundle files")
        )
        rollback_hint = str(executor_result.get("rollback_hint") or "")
        checks.append(
            _pass("rollback hint scoped to new bundle path only", rollback_hint, "inspect executor result")
            if str(artifact_path) in rollback_hint and "rm -rf /" not in rollback_hint.lower()
            else _fail("rollback hint scoped to new bundle path only", rollback_hint, "inspect executor result")
        )
    else:
        checks.extend(
            [
                _fail("manifest exists and lists expected files", "artifact missing", "inspect manifest.json"),
                _fail("expected summary files exist", "artifact missing", "inspect support bundle files"),
                _fail("bundle files do not contain obvious raw secrets", "artifact missing", "scan support bundle files"),
                _fail("rollback hint scoped to new bundle path only", "artifact missing", "inspect executor result"),
            ]
        )

    resources = executor_result.get("resources_touched") if isinstance(executor_result.get("resources_touched"), list) else []
    checks.append(
        _pass("executor result includes resources_touched", f"resources={len(resources)} journal={executor_result.get('journal_id')}", "inspect executor result")
        if resources and all(str(item).startswith(str(artifact_path)) for item in resources)
        else _fail("executor result includes resources_touched", json.dumps(executor_result, sort_keys=True)[:1200], "inspect executor result")
    )

    final_git = _git_status_short()
    checks.append(
        _pass("git status unchanged", "working tree status unchanged", "git status --short")
        if final_git == baseline_git
        else _fail("git status unchanged", f"before={baseline_git!r} after={final_git!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Installed-product Support Bundle v2 smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed

    print("# Personal Agent Support Bundle v2 Smoke")
    for check in checks:
        print()
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
    print()
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"SUPPORT_BUNDLE_V2_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

