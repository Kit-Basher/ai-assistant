#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str
    next_action: str | None = None


def _request_json(method: str, base_url: str, path: str, *, payload: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    parsed: Any
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        parsed = {"raw": raw[:1000]}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    parsed.setdefault("http_status", status)
    return parsed


def _request_text(method: str, base_url: str, path: str, *, timeout: float = 15.0) -> tuple[int, str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "text/html,application/json"}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def _post_chat(base_url: str, message: str, *, thread_id: str, timeout: float = 25.0) -> dict[str, Any]:
    return _request_json(
        "POST",
        base_url,
        "/chat",
        payload={
            "message": message,
            "user_id": "vm-proof-smoke",
            "thread_id": thread_id,
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
        },
        timeout=timeout,
    )


def _assistant_text(payload: dict[str, Any]) -> str:
    for key in ("response", "message", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for key in ("response", "message", "text", "summary"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, sort_keys=True)[:1000]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _git_status_short() -> str:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=False)
    return proc.stdout.strip()


def _pass(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=True, evidence=evidence.strip()[:1000], command=command)


def _fail(name: str, evidence: str, command: str, next_action: str | None = None) -> Check:
    return Check(name=name, ok=False, evidence=evidence.strip()[:1600], command=command, next_action=next_action)


def run(*, base_url: str, expected_commit: str | None, allow_existing_config: bool) -> list[Check]:
    checks: list[Check] = []
    before = _git_status_short()

    ready = _request_json("GET", base_url, "/ready")
    checks.append(
        _pass("GET /ready coherent", json.dumps({k: ready.get(k) for k in ("ready", "runtime_mode", "state_label", "chat_usable")}, sort_keys=True), "GET /ready")
        if int(ready.get("http_status") or 0) == 200 and ("ready" in ready or "runtime_mode" in ready)
        else _fail("GET /ready coherent", json.dumps(ready, sort_keys=True), "GET /ready", "Check personal-agent-api.service on the VM.")
    )

    state = _request_json("GET", base_url, "/state")
    checks.append(
        _pass("GET /state coherent", json.dumps({k: state.get(k) for k in ("ok", "ready", "runtime_mode", "state_label")}, sort_keys=True), "GET /state")
        if int(state.get("http_status") or 0) == 200 and isinstance(state, dict)
        else _fail("GET /state coherent", json.dumps(state, sort_keys=True), "GET /state")
    )

    version = _request_json("GET", base_url, "/version")
    version_ok = int(version.get("http_status") or 0) == 200 and bool(version.get("runtime_instance")) and bool(version.get("git_commit"))
    if expected_commit:
        version_ok = version_ok and str(version.get("git_commit")) == expected_commit
    checks.append(
        _pass("GET /version has runtime metadata", f"runtime_instance={version.get('runtime_instance')} git_commit={version.get('git_commit')}", "GET /version")
        if version_ok
        else _fail("GET /version has runtime metadata", json.dumps(version, sort_keys=True), "GET /version", "Confirm the VM is running the installed runtime from the intended commit.")
    )

    root_status, root_body = _request_text("GET", base_url, "/")
    checks.append(
        _pass("web UI root responds", f"http_status={root_status} body_len={len(root_body)}", "GET /")
        if root_status == 200 and root_body.strip()
        else _fail("web UI root responds", f"http_status={root_status} body={root_body[:300]}", "GET /")
    )

    telegram = _request_json("GET", base_url, "/telegram/status")
    telegram_text = json.dumps(telegram, sort_keys=True)
    telegram_ok = int(telegram.get("http_status") or 0) == 200 and (
        telegram.get("configured") is False or (allow_existing_config and "token" not in telegram_text.lower())
    )
    checks.append(
        _pass("Telegram missing/unconfigured is optional", telegram_text[:1000], "GET /telegram/status")
        if telegram_ok and _contains_any(telegram_text, ("optional", "disabled_optional", "not_configured", "token_source"))
        else _fail("Telegram missing/unconfigured is optional", telegram_text[:1200], "GET /telegram/status", "Clean VM proof should not require a Telegram token.")
    )

    search = _request_json("GET", base_url, "/search/status")
    search_state = str(search.get("search_state") or "")
    search_ok = int(search.get("http_status") or 0) == 200 and (
        search_state == "never_configured" or (allow_existing_config and search_state in {"configured_running", "configured_stopped"})
    )
    checks.append(
        _pass("search state is honest", json.dumps({k: search.get(k) for k in ("enabled", "available", "search_state", "reason", "next_action")}, sort_keys=True), "GET /search/status")
        if search_ok
        else _fail("search state is honest", json.dumps(search, sort_keys=True)[:1200], "GET /search/status", "Clean VM proof should start with search never_configured unless explicitly configured.")
    )

    search_chat = _post_chat(base_url, "what is dots.tts?", thread_id="vm-search-guidance")
    search_text = _assistant_text(search_chat)
    if search_state == "never_configured":
        search_guidance_ok = _contains_any(search_text, ("search is not configured", "web search is not set up", "set up local searxng", "reply yes", "say yes"))
        search_guidance_ok = search_guidance_ok and "missing podman" not in search_text.lower()
    else:
        search_guidance_ok = allow_existing_config and _contains_any(search_text, ("metadata-only", "search is not currently working", "assistant web search"))
    checks.append(
        _pass("search missing/configured guidance is safe", search_text[:1000], 'POST /chat {"message": "what is dots.tts?"}')
        if search_guidance_ok
        else _fail("search missing/configured guidance is safe", search_text[:1200], 'POST /chat {"message": "what is dots.tts?"}')
    )

    install = _post_chat(base_url, "install htop", thread_id="vm-install-preview")
    install_text = _assistant_text(install)
    checks.append(
        _pass("Plan Mode gates mutating package action", install_text[:1000], 'POST /chat {"message": "install htop"}')
        if _contains_any(install_text, ("Plan Mode v2", "Action type: package.install", "mutates the local system", "Say yes"))
        else _fail("Plan Mode gates mutating package action", install_text[:1200], 'POST /chat {"message": "install htop"}')
    )

    support = _post_chat(base_url, "make a support bundle", thread_id="vm-support-preview")
    support_text = _assistant_text(support)
    checks.append(
        _pass("support bundle preview works", support_text[:1000], 'POST /chat {"message": "make a support bundle"}')
        if _contains_any(support_text, ("Support bundle preview", "redacted support bundle", "raw tokens", "Plan Mode v2"))
        else _fail("support bundle preview works", support_text[:1200], 'POST /chat {"message": "make a support bundle"}')
    )

    backup = _post_chat(base_url, "back up the assistant", thread_id="vm-backup-preview")
    backup_text = _assistant_text(backup)
    checks.append(
        _pass("backup remains confirmation-gated", backup_text[:1000], 'POST /chat {"message": "back up the assistant"}')
        if _contains_any(backup_text, ("Backup assistant preview", "explicit confirmation", "secrets must remain", "Plan Mode v2"))
        else _fail("backup remains confirmation-gated", backup_text[:1200], 'POST /chat {"message": "back up the assistant"}')
    )

    restore = _post_chat(base_url, "restore from backup", thread_id="vm-restore-preview")
    restore_text = _assistant_text(restore)
    checks.append(
        _pass("restore remains confirmation-gated", restore_text[:1000], 'POST /chat {"message": "restore from backup"}')
        if _contains_any(restore_text, ("Restore from backup preview", "safety snapshot", "could not find a valid Backup v1 artifact"))
        else _fail("restore remains confirmation-gated", restore_text[:1200], 'POST /chat {"message": "restore from backup"}')
    )

    after = _git_status_short()
    checks.append(
        _pass("repo status unchanged by VM smoke", "working tree status unchanged", "git status --short")
        if after == before
        else _fail("repo status unchanged by VM smoke", f"before={before!r} after={after!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh Debian VM post-install smoke for Personal Agent.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument(
        "--allow-existing-config",
        action="store_true",
        help="Allow already configured Telegram/search for diagnostic runs on non-clean machines. Do not use for clean VM proof.",
    )
    args = parser.parse_args()

    checks = run(base_url=args.base_url, expected_commit=args.expected_commit, allow_existing_config=bool(args.allow_existing_config))
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed

    print("# Personal Agent VM Proof Smoke")
    print(f"Base URL: {args.base_url}")
    if args.expected_commit:
        print(f"Expected commit: {args.expected_commit}")
    print("")
    for check in checks:
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
        if check.next_action:
            print(f"- next action: {check.next_action}")
        print("")
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"VM_PROOF_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
