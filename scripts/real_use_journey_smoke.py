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
    re.compile(r"bot\d{8,12}:[A-Za-z0-9_-]{30,}", re.IGNORECASE),
    re.compile(r"\b(?:sk|sk-proj|xoxb|ghp)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
)
TELEGRAM_BOT_URL_RE = re.compile(r"bot\d{8,12}:[A-Za-z0-9_-]{30,}", re.IGNORECASE)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    command: str
    next_action: str = "No action."


def _run(argv: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)


def _request_json(method: str, base_url: str, path: str, *, payload: dict[str, Any] | None = None, timeout: float) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    except urllib.error.URLError as exc:
        return {"ok": False, "http_status": 0, "error": f"URLError: {exc}"}
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        parsed = {"raw": raw[:1000]}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    parsed.setdefault("http_status", status)
    return parsed


def _post_chat(base_url: str, message: str, *, thread_id: str, timeout: float) -> dict[str, Any]:
    now = int(time.time() * 1000)
    return _request_json(
        "POST",
        base_url,
        "/chat",
        payload={
            "messages": [{"role": "user", "content": message}],
            "session_id": f"real-use-journey-{thread_id}",
            "thread_id": f"real-use-journey-{thread_id}",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"real-use-journey-{thread_id}-{now}",
        },
        timeout=timeout,
    )


def _assistant_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    for value in (assistant.get("content"), payload.get("message"), payload.get("response"), payload.get("text")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, sort_keys=True)[:1000]


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    if not meta and isinstance(data, dict):
        meta = data
    return meta if isinstance(meta, dict) else {}


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _redact_output(text: str) -> str:
    return TELEGRAM_BOT_URL_RE.sub("bot[REDACTED]", str(text or ""))


def _check(name: str, status: str, detail: str, command: str, next_action: str = "No action.") -> Check:
    if _contains_secret(detail):
        return Check(name, "FAIL", "Output looked like it exposed a secret/token.", command, "Redact this output.")
    return Check(name, status, " ".join(str(detail or "").split())[:1200], command, next_action)


def _pass(name: str, detail: str, command: str) -> Check:
    return _check(name, "PASS", detail, command)


def _warn(name: str, detail: str, command: str, next_action: str) -> Check:
    return _check(name, "WARN", detail, command, next_action)


def _fail(name: str, detail: str, command: str, next_action: str) -> Check:
    return _check(name, "FAIL", detail, command, next_action)


def _git_status_short() -> str:
    return str(_run(["git", "status", "--short"]).stdout or "").strip()


def _telegram_service_detail(timeout: float) -> str:
    status = _run(["systemctl", "--user", "status", "personal-agent-telegram.service", "--no-pager"], timeout=timeout)
    logs = _run(["journalctl", "--user", "-u", "personal-agent-telegram.service", "-n", "40", "--no-pager"], timeout=timeout)
    status_text = (status.stdout or status.stderr or "").strip()
    logs_text = (logs.stdout or logs.stderr or "").strip()
    if not logs_text:
        logs_text = "No recent service log entries."
    status_text = _redact_output(status_text)
    logs_text = _redact_output(logs_text)
    return f"systemctl rc={status.returncode}: {status_text[:700]}\njournalctl rc={logs.returncode}: {logs_text[:700]}"


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    before = _git_status_short()

    version = _request_json("GET", base_url, "/version", timeout=timeout)
    checks.append(
        _pass("installed API version", json.dumps({k: version.get(k) for k in ("git_commit", "checkout_git_commit", "runtime_instance")}, sort_keys=True), "GET /version")
        if int(version.get("http_status") or 0) == 200 and version.get("git_commit")
        else _fail("installed API version", json.dumps(version, sort_keys=True)[:1000], "GET /version", "Promote/start the installed API before running real-use journeys.")
    )

    hello = _post_chat(base_url, "hello", thread_id="hello", timeout=timeout)
    hello_text = _assistant_text(hello)
    hello_ok = int(hello.get("http_status") or 200) == 200 and bool(hello_text) and "traceback" not in hello_text.lower() and "runtime is ready" not in hello_text.lower()
    checks.append(
        _pass("web chat hello", hello_text, 'POST /chat {"message": "hello"}')
        if hello_ok
        else _fail("web chat hello", hello_text, 'POST /chat {"message": "hello"}', "Make a normal greeting produce a normal short reply.")
    )

    model_prompt = (
        "no there should be a few things running, i have my game website being served locally here "
        "and it uses the ollama llm and you are running here through ollama correct?"
    )
    model = _post_chat(base_url, model_prompt, thread_id="model-status", timeout=timeout)
    model_text = _assistant_text(model)
    model_meta = _meta(model)
    model_lower = model_text.lower()
    switched = "switched chat" in model_lower or "i switched" in model_lower
    model_ok = (
        int(model.get("http_status") or 200) == 200
        and not switched
        and str(model_meta.get("route") or "") == "model_status"
        and "configured model" in model_lower
    )
    checks.append(
        _pass("casual Ollama question does not switch", model_text, f'POST /chat {{"message": "{model_prompt}"}}')
        if model_ok
        else _fail("casual Ollama question does not switch", model_text, f'POST /chat {{"message": "{model_prompt}"}}', "Route casual model/provider questions to status, not model switching.")
    )

    why = _post_chat(base_url, "why", thread_id="model-status", timeout=timeout)
    why_text = _assistant_text(why)
    why_lower = why_text.lower()
    why_ok = (
        int(why.get("http_status") or 200) == 200
        and "ollama" in why_lower
        and "did not switch models" in why_lower
        and "doctor:" not in why_lower
        and "runtime is ready" not in why_lower
        and "alibaba cloud" not in why_lower
        and "previous conversation" not in why_lower
    )
    checks.append(
        _pass("why follows model status context", why_text, 'POST /chat {"message": "why"} after model status')
        if why_ok
        else _fail("why follows model status context", why_text, 'POST /chat {"message": "why"} after model status', "Keep immediate model-status context ahead of stale diagnostic context.")
    )

    ram_prompt = "can you do a quick system check and see if anything is eating ram?"
    ram = _post_chat(base_url, ram_prompt, thread_id="ram-check", timeout=timeout)
    ram_text = _assistant_text(ram)
    ram_lower = ram_text.lower()
    ram_lines = [line for line in ram_text.splitlines() if line.strip()]
    ram_ok = (
        int(ram.get("http_status") or 200) == 200
        and len(ram_text) <= 900
        and len(ram_lines) <= 10
        and "ram" in ram_lower
        and ("not under pressure" in ram_lower or "under pressure" in ram_lower)
        and ("baseline" in ram_lower or "usual" in ram_lower)
        and "top memory processes" not in ram_lower
        and "top cpu processes" not in ram_lower
        and "pid=" not in ram_lower
        and "likely cause:" not in ram_lower
    )
    checks.append(
        _pass("concise RAM check with baseline", ram_text, f'POST /chat {{"message": "{ram_prompt}"}}')
        if ram_ok
        else _fail("concise RAM check with baseline", ram_text, f'POST /chat {{"message": "{ram_prompt}"}}', "Keep quick system checks concise and baseline-aware by default.")
    )

    telegram = _request_json("GET", base_url, "/telegram/status", timeout=timeout)
    service_detail = _telegram_service_detail(timeout)
    configured = telegram.get("configured") is True
    active = telegram.get("service_active") is True
    poller_count = int(telegram.get("poller_count") or 0)
    embedded_running = telegram.get("embedded_running") is True
    next_action = str(telegram.get("next_action") or "").strip()
    telegram_summary = json.dumps(
        {
            "configured": telegram.get("configured"),
            "state": telegram.get("state"),
            "effective_state": telegram.get("effective_state"),
            "service_active": telegram.get("service_active"),
            "poller_count": telegram.get("poller_count"),
            "duplicate_pollers": telegram.get("duplicate_pollers"),
            "token_source": telegram.get("token_source"),
            "next_action": next_action,
        },
        sort_keys=True,
    )
    if int(telegram.get("http_status") or 0) != 200:
        checks.append(
            _fail("Telegram status reachable", json.dumps(telegram, sort_keys=True)[:1000], "GET /telegram/status", "Start the installed API before checking Telegram state.")
        )
    elif not configured:
        checks.append(_pass("Telegram optional not configured", telegram_summary, "GET /telegram/status"))
    elif active or embedded_running:
        polling_ok = poller_count > 0 or embedded_running
        checks.append(
            _pass("Telegram active polling evidence", telegram_summary, "GET /telegram/status")
            if polling_ok
            else _fail("Telegram active polling evidence", f"{telegram_summary}\n{service_detail}", "GET /telegram/status", "Telegram is active but no poller evidence was visible.")
        )
    else:
        stopped_ok = str(telegram.get("state") or "") == "stopped" and poller_count == 0 and (
            "telegram_enable" in next_action or "systemctl --user" in next_action
        )
        checks.append(
            _pass("Telegram stopped diagnostic is explicit", f"{telegram_summary}\n{service_detail}", "GET /telegram/status; systemctl --user status personal-agent-telegram.service")
            if stopped_ok
            else _fail("Telegram stopped diagnostic is explicit", f"{telegram_summary}\n{service_detail}", "GET /telegram/status; systemctl --user status personal-agent-telegram.service", "Report stopped Telegram with an exact bounded start action.")
        )

    telegram_chat = _post_chat(base_url, "is Telegram working?", thread_id="telegram-status", timeout=timeout)
    telegram_chat_text = _assistant_text(telegram_chat)
    telegram_chat_lower = telegram_chat_text.lower()
    telegram_chat_ok = int(telegram_chat.get("http_status") or 200) == 200 and (
        (not configured and "not configured" in telegram_chat_lower)
        or (active and "running" in telegram_chat_lower)
        or (configured and not active and "configured" in telegram_chat_lower and ("not currently running" in telegram_chat_lower or "stopped" in telegram_chat_lower) and "optional" in telegram_chat_lower)
    )
    checks.append(
        _pass("Telegram chat status matches runtime", telegram_chat_text, 'POST /chat {"message": "is Telegram working?"}')
        if telegram_chat_ok
        else _fail("Telegram chat status matches runtime", telegram_chat_text, 'POST /chat {"message": "is Telegram working?"}', "Make Telegram chat status match /telegram/status.")
    )

    after = _git_status_short()
    checks.append(
        _pass("git status unchanged", "clean" if not after else after, "git status --short")
        if before == after
        else _fail("git status unchanged", f"before={before!r} after={after!r}", "git status --short", "Real-use smoke must not change repo files.")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run non-destructive real-use journey checks against the installed Personal Agent API."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    checks = run(args.base_url, args.timeout)
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
        print(f"## {check.name}: {check.status}")
        print(f"command: {check.command}")
        print(f"evidence: {check.detail}")
        if check.status != "PASS":
            print(f"next action: {check.next_action}")
        print()
    print(f"SUMMARY: PASS={counts.get('PASS', 0)} WARN={counts.get('WARN', 0)} FAIL={counts.get('FAIL', 0)}")
    return 1 if counts.get("FAIL", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
