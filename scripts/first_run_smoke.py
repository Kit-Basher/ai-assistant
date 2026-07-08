#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str
    next_action: str | None = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(method: str, base_url: str, path: str, *, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
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


def _request_text(method: str, base_url: str, path: str, *, timeout: float = 10.0) -> tuple[int, str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "text/html,application/json"}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def _post_chat(base_url: str, message: str, *, thread_id: str, timeout: float = 20.0) -> dict[str, Any]:
    return _request_json(
        "POST",
        base_url,
        "/chat",
        payload={
            "message": message,
            "user_id": "first-run-smoke",
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
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    content = assistant.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for key in ("response", "message", "text", "summary"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, sort_keys=True)[:1000]


def _runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    runtime_payload = data.get("runtime_payload") if isinstance(data.get("runtime_payload"), dict) else {}
    if runtime_payload:
        return runtime_payload
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    runtime_payload = meta.get("runtime_payload") if isinstance(meta.get("runtime_payload"), dict) else {}
    return runtime_payload if isinstance(runtime_payload, dict) else {}


def _git_status_short() -> str:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=False)
    return proc.stdout.strip()


def _pass(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=True, evidence=evidence.strip()[:1000], command=command)


def _fail(name: str, evidence: str, command: str, next_action: str | None = None) -> Check:
    return Check(name=name, ok=False, evidence=evidence.strip()[:1600], command=command, next_action=next_action)


def _terminate(proc: subprocess.Popen[str], *, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def _tail(path: Path, *, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@contextlib.contextmanager
def _isolated_api(timeout: float) -> Iterator[tuple[str, Path, Path]]:
    port = _find_free_port()
    with tempfile.TemporaryDirectory(prefix="personal-agent-first-run-") as tmp:
        root = Path(tmp)
        home = root / "home"
        state = home / ".local/share/personal-agent"
        config = home / ".config/personal-agent"
        xdg_runtime = root / "xdg-runtime"
        xdg_cache = root / "xdg-cache"
        xdg_data = home / ".local/share"
        xdg_config = home / ".config"
        temp_root = root / "tmp"
        logs = root / "logs"
        for path in (home, state, config, xdg_runtime, xdg_cache, xdg_data, xdg_config, temp_root, logs):
            path.mkdir(parents=True, exist_ok=True)
        db_path = state / "agent.db"
        log_path = state / "agent.jsonl"
        registry_path = state / "llm_registry.json"
        api_log_path = logs / "api.log"
        registry_path.write_text("{}\n", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home),
                "XDG_DATA_HOME": str(xdg_data),
                "XDG_CONFIG_HOME": str(xdg_config),
                "XDG_CACHE_HOME": str(xdg_cache),
                "XDG_RUNTIME_DIR": str(xdg_runtime),
                "TMPDIR": str(temp_root),
                "TMP": str(temp_root),
                "TEMP": str(temp_root),
                "AGENT_API_HOST": "127.0.0.1",
                "AGENT_API_PORT": str(port),
                "AGENT_DB_PATH": str(db_path),
                "AGENT_LOG_PATH": str(log_path),
                "AGENT_SECRET_STORE_PATH": str(state / "secrets.enc.json"),
                "AGENT_AUDIT_LOG_PATH": str(state / "audit.jsonl"),
                "AGENT_PERMISSIONS_PATH": str(state / "permissions.json"),
                "LLM_REGISTRY_PATH": str(registry_path),
                "AGENT_MODEL_SCOUT_STATE_PATH": str(state / "model_scout_state.json"),
                "AGENT_MODEL_WATCH_STATE_PATH": str(state / "model_watch_state.json"),
                "AGENT_PROVIDER_CATALOG_STATE_PATH": str(state / "provider_catalog_state.json"),
                "LLM_HEALTH_STATE_PATH": str(state / "llm_health_state.json"),
                "AUTOPILOT_NOTIFY_STORE_PATH": str(state / "autopilot_notify_state.json"),
                "LLM_AUTOPILOT_STATE_PATH": str(state / "llm_autopilot_state.json"),
                "TELEGRAM_ENABLED": "0",
                "TELEGRAM_REQUIRED": "0",
                "SEARCH_ENABLED": "0",
                "SEARXNG_BASE_URL": "",
                "LLM_PROVIDER": "none",
                "ALLOW_CLOUD": "false",
                "PREFER_LOCAL": "true",
                "MODEL_SCOUT_ENABLED": "0",
                "AGENT_MODEL_WATCH_ENABLED": "0",
                "LLM_AUTOMATION_ENABLED": "0",
                "PERCEPTION_ENABLED": "0",
                "AGENT_MEMORY_V2_ENABLED": "0",
                "AGENT_SEMANTIC_MEMORY_ENABLED": "0",
                "AGENT_INTENT_LLM_RERANK_ENABLED": "0",
                "AGENT_LAUNCHER_OPEN_BROWSER": "0",
                "PERSONAL_AGENT_INSTANCE": "dev",
                "PYTHONUNBUFFERED": "1",
            }
        )
        log_handle = api_log_path.open("a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent.api_server", "--host", "127.0.0.1", "--port", str(port)],
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            log_handle.close()
        base_url = f"http://127.0.0.1:{port}"
        started = time.monotonic()
        last_error = ""
        try:
            while time.monotonic() - started < timeout:
                if proc.poll() is not None:
                    raise RuntimeError(f"first-run API exited early code={proc.returncode}\n{_tail(api_log_path)}")
                try:
                    ready = _request_json("GET", base_url, "/ready", timeout=2.0)
                    if int(ready.get("http_status") or 0) == 200:
                        yield base_url, home, api_log_path
                        return
                except Exception as exc:  # noqa: BLE001 - startup polling records last failure.
                    last_error = f"{exc.__class__.__name__}: {exc}"
                time.sleep(0.5)
            raise RuntimeError(f"first-run API did not become ready: {last_error}\n{_tail(api_log_path)}")
        finally:
            _terminate(proc)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def run(timeout: float) -> list[Check]:
    checks: list[Check] = []
    before = _git_status_short()
    with _isolated_api(timeout=timeout) as (base_url, home, api_log_path):
        state = home / ".local/share/personal-agent"
        config = home / ".config/personal-agent"
        checks.append(
            _pass("isolated required dirs exist", f"state={state} config={config}", "start isolated API")
            if state.is_dir() and config.is_dir()
            else _fail("isolated required dirs exist", f"state_exists={state.is_dir()} config_exists={config.is_dir()}", "start isolated API")
        )
        checks.append(
            _pass("real user state not targeted", f"isolated_home={home}", "inspect isolated HOME")
            if str(home).startswith("/tmp/")
            else _fail("real user state not targeted", f"home={home}", "inspect isolated HOME")
        )

        ready = _request_json("GET", base_url, "/ready", timeout=timeout)
        checks.append(
            _pass("GET /ready coherent", json.dumps({k: ready.get(k) for k in ("ready", "runtime_mode", "state_label", "chat_usable")}, sort_keys=True), "GET /ready")
            if int(ready.get("http_status") or 0) == 200 and isinstance(ready, dict) and ("runtime_mode" in ready or "ready" in ready)
            else _fail("GET /ready coherent", json.dumps(ready, sort_keys=True)[:1000], "GET /ready", "Inspect isolated API log.")
        )

        state_payload = _request_json("GET", base_url, "/state", timeout=timeout)
        checks.append(
            _pass("GET /state coherent", json.dumps({k: state_payload.get(k) for k in ("ok", "ready", "runtime_mode", "state_label")}, sort_keys=True), "GET /state")
            if int(state_payload.get("http_status") or 0) == 200 and isinstance(state_payload, dict)
            else _fail("GET /state coherent", json.dumps(state_payload, sort_keys=True)[:1000], "GET /state")
        )

        version = _request_json("GET", base_url, "/version", timeout=timeout)
        checks.append(
            _pass("GET /version has runtime metadata", f"runtime_instance={version.get('runtime_instance')} git_commit={version.get('git_commit')}", "GET /version")
            if int(version.get("http_status") or 0) == 200 and version.get("runtime_instance") and version.get("git_commit")
            else _fail("GET /version has runtime metadata", json.dumps(version, sort_keys=True)[:1000], "GET /version")
        )

        root_status, root_body = _request_text("GET", base_url, "/", timeout=timeout)
        checks.append(
            _pass("web UI entrypoint responds", f"http_status={root_status} body_len={len(root_body)}", "GET /")
            if root_status == 200 and root_body.strip()
            else _fail("web UI entrypoint responds", f"http_status={root_status} body={root_body[:300]}", "GET /")
        )

        telegram = _request_json("GET", base_url, "/telegram/status", timeout=timeout)
        telegram_text = json.dumps(telegram, sort_keys=True)
        checks.append(
            _pass("Telegram missing is optional", telegram_text[:1000], "GET /telegram/status")
            if int(telegram.get("http_status") or 0) == 200
            and telegram.get("configured") is False
            and _contains_any(telegram_text, ("not_configured", "telegram is optional", "optional", "token"))
            else _fail("Telegram missing is optional", telegram_text[:1200], "GET /telegram/status")
        )

        search_status = _request_json("GET", base_url, "/search/status", timeout=timeout)
        checks.append(
            _pass("search unconfigured is honest", json.dumps(search_status, sort_keys=True)[:1000], "GET /search/status")
            if int(search_status.get("http_status") or 0) == 200
            and search_status.get("enabled") is False
            and str(search_status.get("search_state") or search_status.get("reason") or "").strip()
            else _fail("search unconfigured is honest", json.dumps(search_status, sort_keys=True)[:1200], "GET /search/status")
        )

        search_chat = _post_chat(base_url, "what is dots.tts?", thread_id="search-missing", timeout=timeout)
        search_text = _assistant_text(search_chat)
        search_lower = search_text.lower()
        search_guidance_ok = (
            _contains_any(
                search_text,
                ("search is not configured", "web search is not set up", "set up local searxng", "plan mode", "reply yes", "say yes"),
            )
            and "missing podman" not in search_lower
            and not any(bad in search_lower for bad in ("i searched metadata-only", "search results are untrusted metadata"))
        )
        checks.append(
            _pass("search setup guidance is safe", search_text[:1000], 'POST /chat {"message": "what is dots.tts?"}')
            if search_guidance_ok
            else _fail("search setup guidance is safe", search_text[:1200], 'POST /chat {"message": "what is dots.tts?"}')
        )

        memory_chat = _post_chat(base_url, "what do you remember about me?", thread_id="memory-empty", timeout=timeout)
        memory_text = _assistant_text(memory_chat)
        checks.append(
            _pass("memory starts empty or explains no saved memory", memory_text[:1000], 'POST /chat {"message": "what do you remember about me?"}')
            if _contains_any(memory_text, ("no saved memory", "not have saved", "nothing useful", "do not have", "no useful memory"))
            else _fail("memory starts empty or explains no saved memory", memory_text[:1200], 'POST /chat {"message": "what do you remember about me?"}')
        )

        install_chat = _post_chat(base_url, "install htop", thread_id="install-preview", timeout=timeout)
        install_text = _assistant_text(install_chat)
        checks.append(
            _pass("Plan Mode gates package install", install_text[:1000], 'POST /chat {"message": "install htop"}')
            if _contains_any(install_text, ("Plan Mode v2", "Action type: package.install", "Say yes", "mutates the local system"))
            else _fail("Plan Mode gates package install", json.dumps(install_chat, sort_keys=True)[:1200], 'POST /chat {"message": "install htop"}')
        )

        support_chat = _post_chat(base_url, "make a support bundle", thread_id="support-preview", timeout=timeout)
        support_text = _assistant_text(support_chat)
        checks.append(
            _pass("support bundle remains safe preview", support_text[:1000], 'POST /chat {"message": "make a support bundle"}')
            if _contains_any(support_text, ("Support bundle preview", "redacted support bundle", "raw tokens"))
            else _fail("support bundle remains safe preview", support_text[:1200], 'POST /chat {"message": "make a support bundle"}')
        )

        backup_chat = _post_chat(base_url, "back up the assistant", thread_id="backup-preview", timeout=timeout)
        backup_text = _assistant_text(backup_chat)
        checks.append(
            _pass("backup remains confirmation-gated", backup_text[:1000], 'POST /chat {"message": "back up the assistant"}')
            if _contains_any(backup_text, ("Backup assistant preview", "secrets must remain", "explicit confirmation"))
            else _fail("backup remains confirmation-gated", backup_text[:1200], 'POST /chat {"message": "back up the assistant"}')
        )

        restore_chat = _post_chat(base_url, "restore from backup", thread_id="restore-preview", timeout=timeout)
        restore_text = _assistant_text(restore_chat)
        checks.append(
            _pass("restore is safe in fresh state", restore_text[:1000], 'POST /chat {"message": "restore from backup"}')
            if _contains_any(restore_text, ("Restore from backup preview", "safety snapshot", "could not find a valid Backup v1 artifact"))
            else _fail("restore is safe in fresh state", restore_text[:1200], 'POST /chat {"message": "restore from backup"}')
        )

        cleanup_chat = _post_chat(base_url, "clean old backup files", thread_id="cleanup-preview", timeout=timeout)
        cleanup_text = _assistant_text(cleanup_chat)
        checks.append(
            _pass("cleanup remains confirmation-gated and read-only before approval", cleanup_text[:1000], 'POST /chat {"message": "clean old backup files"}')
            if _contains_any(cleanup_text, ("Cleanup old Personal Agent files preview", "I did not delete anything"))
            else _fail("cleanup remains confirmation-gated and read-only before approval", cleanup_text[:1200], 'POST /chat {"message": "clean old backup files"}')
        )

        created_outside = [
            str(path)
            for path in (ROOT / "publishability_smoke_note.txt", ROOT / "first_run_smoke_note.txt")
            if path.exists()
        ]
        checks.append(
            _pass("no root-level smoke artifacts", "no known first-run temp artifacts in repo root", "inspect repo root")
            if not created_outside
            else _fail("no root-level smoke artifacts", ", ".join(created_outside), "inspect repo root")
        )
        if any(not check.ok for check in checks):
            checks.append(Check("isolated api log tail", True, _tail(api_log_path), "tail isolated api log"))

    after = _git_status_short()
    checks.append(
        _pass("git status unchanged by first-run smoke", "working tree status unchanged", "git status --short")
        if after == before
        else _fail("git status unchanged by first-run smoke", f"before={before!r} after={after!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="First-run/fresh-state Personal Agent smoke.")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(timeout=args.timeout)
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed

    print("# Personal Agent First-Run Smoke")
    for check in checks:
        print()
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
        if check.next_action:
            print(f"- next action: {check.next_action}")
    print()
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"FIRST_RUN_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
