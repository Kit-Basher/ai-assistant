#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
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
STATE_DIR = Path.home() / ".local" / "share" / "personal-agent"
DB_PATH = STATE_DIR / "agent.db"
BASELINE_KEY = "system_resource_baseline_v1"
BASELINE_CONTEXT_KEY = "system_resource_baseline_context_v1"
SECRET_PATTERNS = (
    re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:sk|sk-proj|xoxb|ghp)[_-][A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
    re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
)
DIAGNOSTIC_LABELS = (
    "Likely cause:",
    "Normality:",
    "Evidence:",
    "Safe next action:",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    command: str
    next_action: str = "No action."


def _run(argv: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)


def _git_status_short() -> str:
    return str(_run(["git", "status", "--short"]).stdout or "").strip()


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


def _post_chat(
    base_url: str,
    message: str,
    *,
    thread_id: str,
    timeout: float,
    source_surface: str = "webui",
) -> dict[str, Any]:
    now = int(time.time() * 1000)
    return _request_json(
        "POST",
        base_url,
        "/chat",
        payload={
            "messages": [{"role": "user", "content": message}],
            "session_id": f"normal-user-acceptance-{thread_id}",
            "thread_id": f"normal-user-acceptance-{thread_id}",
            "source_surface": source_surface,
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"normal-user-acceptance-{thread_id}-{now}",
        },
        timeout=timeout,
    )


def _assistant_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    for value in (assistant.get("content"), payload.get("message"), payload.get("response"), payload.get("text")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, sort_keys=True)[:1000]


def _http_ok(payload: dict[str, Any]) -> bool:
    return int(payload.get("http_status") or 200) == 200 and "URLError:" not in str(payload.get("error") or "")


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _check(name: str, status: str, detail: str, command: str, next_action: str = "No action.") -> Check:
    compact = " ".join(str(detail or "").split())[:1400]
    if _contains_secret(compact):
        return Check(name, "FAIL", "Output looked like it exposed a secret/token.", command, "Redact this output.")
    return Check(name, status, compact, command, next_action)


def _pass(name: str, detail: str, command: str) -> Check:
    return _check(name, "PASS", detail, command)


def _fail(name: str, detail: str, command: str, next_action: str) -> Check:
    return _check(name, "FAIL", detail, command, next_action)


def _memory_preferences(keys: tuple[str, ...] = (BASELINE_KEY, BASELINE_CONTEXT_KEY)) -> dict[str, str]:
    if not DB_PATH.exists():
        return {}
    placeholders = ",".join("?" for _ in keys)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(f"SELECT key, value FROM preferences WHERE key IN ({placeholders})", keys).fetchall()
    return {str(key): str(value or "") for key, value in rows}


def _all_baseline_memory_text() -> str:
    prefs = _memory_preferences()
    return "\n".join(f"{key}: {value}" for key, value in sorted(prefs.items()))


def _looks_concise_ram_answer(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    problems: list[str] = []
    if len(text) > 900:
        problems.append(f"too long for Telegram-style default ({len(text)} chars)")
    if not ("ram" in lowered and ("not under pressure" in lowered or "under pressure" in lowered or "okay" in lowered or "fine" in lowered)):
        problems.append("does not clearly say whether RAM is okay/under pressure")
    if not re.search(r"\bused:\s*\d", lowered):
        problems.append("missing used RAM line")
    if "available:" not in lowered:
        problems.append("missing available RAM line")
    if not ("baseline" in lowered or "usual" in lowered):
        problems.append("missing baseline create/compare language")
    for label in DIAGNOSTIC_LABELS:
        if label.lower() in lowered:
            problems.append(f"contains verbose diagnostic label {label!r}")
    if "top memory processes" in lowered or "top cpu processes" in lowered or "pid=" in lowered:
        problems.append("included process table in default answer")
    return (not problems, "; ".join(problems))


def check_telegram_style_ram(base_url: str, timeout: float) -> Check:
    prompt = "can you do a quick system check and see if anything is eating ram?"
    command = f'POST /chat source_surface=telegram {{"message": "{prompt}"}}'
    payload = _post_chat(base_url, prompt, thread_id="telegram-ram", timeout=timeout, source_surface="telegram")
    text = _assistant_text(payload)
    if not _http_ok(payload):
        return _fail("Telegram-style RAM check is concise by default", text, command, "Start the installed API and allow localhost access.")
    ok, reason = _looks_concise_ram_answer(text)
    if not ok:
        return _fail("Telegram-style RAM check is concise by default", f"{reason}\n{text}", command, "Keep normal-user diagnostics short by default.")
    return _pass("Telegram-style RAM check is concise by default", text, command)


def check_detailed_ram(base_url: str, timeout: float) -> Check:
    prompt = "full detailed system check, show top memory and CPU processes"
    command = f'POST /chat {{"message": "{prompt}"}}'
    payload = _post_chat(base_url, prompt, thread_id="detailed-ram", timeout=timeout)
    text = _assistant_text(payload)
    if not _http_ok(payload):
        return _fail("Detailed RAM check remains available", text, command, "Start the installed API and allow localhost access.")
    lowered = text.lower()
    has_detail = "top memory processes" in lowered or "top cpu processes" in lowered or "pid=" in lowered
    has_summary = "summary" in lowered or "likely cause" in lowered or "memory" in lowered
    if not (has_detail and has_summary):
        return _fail("Detailed RAM check remains available", text, command, "Detailed prompts should show process detail.")
    return _pass("Detailed RAM check remains available", text[:1000], command)


def check_baseline_memory(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    prompt = "can you do a quick system check and see if anything is eating ram?"
    first_command = f'POST /chat {{"message": "{prompt}"}}'
    first = _post_chat(base_url, prompt, thread_id="baseline-first", timeout=timeout, source_surface="telegram")
    first_text = _assistant_text(first)
    if not _http_ok(first):
        checks.append(_fail("First RAM check stores safe baseline", first_text, first_command, "Start the installed API and allow localhost access."))
        return checks
    prefs_after_first = _memory_preferences()
    baseline_raw = prefs_after_first.get(BASELINE_KEY, "")
    if not baseline_raw:
        checks.append(_fail("First RAM check stores safe baseline", first_text, first_command, "Store a bounded baseline preference after useful system checks."))
    else:
        try:
            baseline = json.loads(baseline_raw)
        except json.JSONDecodeError:
            baseline = {}
        safe_shape = (
            isinstance(baseline, dict)
            and baseline.get("schema") == "system_resource_baseline.v1"
            and isinstance(baseline.get("memory"), dict)
            and len(baseline_raw) <= 4000
        )
        checks.append(
            _pass("First RAM check stores safe baseline", baseline_raw[:600], first_command)
            if safe_shape
            else _fail("First RAM check stores safe baseline", baseline_raw[:1000], first_command, "Baseline must be bounded JSON with memory summary fields.")
        )
    second = _post_chat(base_url, prompt, thread_id="baseline-second", timeout=timeout, source_surface="telegram")
    second_text = _assistant_text(second)
    if not _http_ok(second):
        checks.append(_fail("Second RAM check compares against baseline", second_text, f'POST /chat {{"message": "{prompt}"}} again', "Start the installed API and allow localhost access."))
        return checks
    second_lower = second_text.lower()
    compared = "baseline" in second_lower and any(token in second_lower for token in ("usual", "higher", "lower", "similar", "normal"))
    checks.append(
        _pass("Second RAM check compares against baseline", second_text, f'POST /chat {{"message": "{prompt}"}} again')
        if compared
        else _fail("Second RAM check compares against baseline", second_text, f'POST /chat {{"message": "{prompt}"}} again', "Use saved baseline comparison language on later checks.")
    )
    return checks


def check_user_context_memory(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    context_prompt = "my game website uses Ollama and should normally be running"
    context_command = f'POST /chat {{"message": "{context_prompt}"}}'
    context_response = _post_chat(base_url, context_prompt, thread_id="baseline-context", timeout=timeout)
    context_text = _assistant_text(context_response)
    if not _http_ok(context_response):
        checks.append(_fail("User-normal context is stored safely", context_text, context_command, "Start the installed API and allow localhost access."))
        return checks
    prefs = _memory_preferences()
    context_raw = prefs.get(BASELINE_CONTEXT_KEY, "")
    if "game website" in context_raw and "Ollama" in context_raw and "/" not in context_raw:
        checks.append(_pass("User-normal context is stored safely", context_raw[:600], context_command))
    else:
        checks.append(_fail("User-normal context is stored safely", context_raw[:1000], context_command, "Store bounded expected-app context without private paths."))
    ram_prompt = "can you do a quick system check and see if anything is eating ram?"
    ram = _post_chat(base_url, ram_prompt, thread_id="baseline-context-ram", timeout=timeout, source_surface="telegram")
    ram_text = _assistant_text(ram)
    if not _http_ok(ram):
        checks.append(_fail("RAM check treats expected Ollama context normally", ram_text, f'POST /chat {{"message": "{ram_prompt}"}} after context', "Start the installed API and allow localhost access."))
        return checks
    ram_lower = ram_text.lower()
    suspicious_ollama = bool(
        re.search(r"ollama.{0,80}\b(suspicious|unexpected|problem|runaway)\b", ram_lower)
        or re.search(r"\b(suspicious|unexpected|problem|runaway)\b.{0,80}ollama", ram_lower)
    )
    expected_ollama = "ollama" not in ram_lower or any(token in ram_lower for token in ("normal", "expected", "baseline", "usual"))
    checks.append(
        _pass("RAM check treats expected Ollama context normally", ram_text, f'POST /chat {{"message": "{ram_prompt}"}} after context')
        if not suspicious_ollama and expected_ollama
        else _fail("RAM check treats expected Ollama context normally", ram_text, f'POST /chat {{"message": "{ram_prompt}"}} after context', "Do not flag expected Ollama usage as suspicious solely for being present.")
    )
    return checks


def check_no_unsafe_memory_storage() -> Check:
    command = f"read bounded baseline preferences from {DB_PATH}"
    text = _all_baseline_memory_text()
    if not text:
        return _fail("System baseline memory avoids unsafe storage", "No baseline preferences found.", command, "Run the baseline checks first.")
    lowered = text.lower()
    forbidden = [
        "top memory processes",
        "top cpu processes",
        "likely cause:",
        "normality:",
        "evidence:",
        "safe next action:",
        "journalctl",
        "traceback",
        "pid=",
    ]
    found = [item for item in forbidden if item in lowered]
    if found:
        return _fail("System baseline memory avoids unsafe storage", f"found={found}; {text[:1000]}", command, "Store summaries, not raw diagnostics or logs.")
    if _contains_secret(text):
        return _fail("System baseline memory avoids unsafe storage", text[:1000], command, "Redact token-like values before memory writes.")
    if len(text) > 9000:
        return _fail("System baseline memory avoids unsafe storage", f"baseline preference text too large: {len(text)} chars", command, "Keep baseline memory bounded.")
    return _pass("System baseline memory avoids unsafe storage", text[:1000], command)


def check_channel_profile(base_url: str, timeout: float) -> Check:
    concise_prompt = "can you do a quick system check and see if anything is eating ram?"
    detailed_prompt = "full detailed system check, show top memory and CPU processes"
    concise_payload = _post_chat(base_url, concise_prompt, thread_id="channel-telegram", timeout=timeout, source_surface="telegram")
    detailed_payload = _post_chat(base_url, detailed_prompt, thread_id="channel-web", timeout=timeout, source_surface="webui")
    concise = _assistant_text(concise_payload)
    detailed = _assistant_text(detailed_payload)
    if not _http_ok(concise_payload) or not _http_ok(detailed_payload):
        return _fail(
            "Telegram/default profile is shorter than detailed output",
            f"telegram={concise[:700]} detailed={detailed[:700]}",
            "POST /chat source_surface=telegram and detailed web prompt",
            "Start the installed API and allow localhost access.",
        )
    if len(concise) <= 900 and len(concise) < len(detailed) and ("top memory processes" not in concise.lower()):
        return _pass("Telegram/default profile is shorter than detailed output", f"telegram_chars={len(concise)} detailed_chars={len(detailed)}", "POST /chat source_surface=telegram and detailed web prompt")
    return _fail(
        "Telegram/default profile is shorter than detailed output",
        f"telegram_chars={len(concise)} detailed_chars={len(detailed)} telegram={concise[:700]} detailed={detailed[:700]}",
        "POST /chat source_surface=telegram and detailed web prompt",
        "Keep Telegram/default diagnostics concise while preserving explicit detail mode.",
    )


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    before = _git_status_short()
    version = _request_json("GET", base_url, "/version", timeout=timeout)
    checks.append(
        _pass("installed API reachable", json.dumps({k: version.get(k) for k in ("git_commit", "runtime_instance")}, sort_keys=True), "GET /version")
        if int(version.get("http_status") or 0) == 200 and version.get("git_commit")
        else _fail("installed API reachable", json.dumps(version, sort_keys=True)[:1000], "GET /version", "Promote/start the installed API.")
    )
    checks.append(check_telegram_style_ram(base_url, timeout))
    checks.append(check_detailed_ram(base_url, timeout))
    checks.extend(check_baseline_memory(base_url, timeout))
    checks.extend(check_user_context_memory(base_url, timeout))
    checks.append(check_no_unsafe_memory_storage())
    checks.append(check_channel_profile(base_url, timeout))
    after = _git_status_short()
    checks.append(
        _pass("git status unchanged", "clean" if not after else after, "git status --short")
        if before == after
        else _fail("git status unchanged", f"before={before!r} after={after!r}", "git status --short", "Normal-user acceptance smoke must not change repo files.")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run normal-user response quality acceptance checks against the installed Personal Agent API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    counts = {"PASS": 0, "FAIL": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
        print(f"## {check.name}: {check.status}")
        print(f"command: {check.command}")
        print(f"evidence: {check.detail}")
        if check.status != "PASS":
            print(f"next action: {check.next_action}")
        print()
    print(f"SUMMARY: PASS={counts.get('PASS', 0)} FAIL={counts.get('FAIL', 0)}")
    print("NORMAL_USER_ACCEPTANCE_SMOKE: " + ("pass" if counts.get("FAIL", 0) == 0 else "fail"))
    return 1 if counts.get("FAIL", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
