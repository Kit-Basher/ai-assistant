#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8765"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "end_user_closure"
FIXTURE_FILE = FIXTURE_DIR / "closure-note.txt"
FIXTURE_MARKER = "PERSONAL_AGENT_CLOSURE_FILE_OK"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    evidence: str
    next_action: str = "No action."


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any], str]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json,text/html"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    return status, parsed if isinstance(parsed, dict) else {}, raw


def _chat(base_url: str, text: str, *, check_id: str, timeout: float) -> dict[str, Any]:
    stamp = time.time_ns()
    _status, payload, _raw = _request(
        base_url,
        "POST",
        "/chat",
        payload={
            "messages": [{"role": "user", "content": text}],
            "source_surface": "webui",
            "user_id": f"end-user-closure-{check_id}",
            "thread_id": f"end-user-closure-{check_id}-{stamp}",
            "session_id": f"end-user-closure-{stamp}",
        },
        timeout=timeout,
    )
    return payload


def _message(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    return str(assistant.get("content") or payload.get("message") or "").strip()


def _filesystem_ok(payload: dict[str, Any]) -> bool:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    tools = meta.get("used_tools") if isinstance(meta.get("used_tools"), list) else []
    return str(meta.get("route") or "") == "action_tool" and "filesystem" in tools and meta.get("used_llm") is False


def _pass(name: str, evidence: str) -> Check:
    return Check(name, "PASS", evidence)


def _fail(name: str, evidence: str, next_action: str) -> Check:
    return Check(name, "FAIL", evidence, next_action)


def run(base_url: str, *, timeout: float) -> list[Check]:
    checks: list[Check] = []

    root_status, _root_payload, root_html = _request(base_url, "GET", "/", timeout=timeout)
    asset_path = ""
    for token in root_html.split('src="')[1:]:
        candidate = token.split('"', 1)[0]
        if candidate.startswith("/assets/index-") and candidate.endswith(".js"):
            asset_path = candidate
            break
    asset_status, _asset_payload, asset_body = _request(base_url, "GET", asset_path, timeout=timeout) if asset_path else (0, {}, "")
    chat_markers = ("chat-product-shell", "Ask for help naturally.", "What can I help you with?")
    if root_status == 200 and asset_status == 200 and all(marker in asset_body for marker in chat_markers):
        checks.append(_pass("chat-first web UI", f"root=200 asset=200 markers={len(chat_markers)}"))
    else:
        checks.append(_fail("chat-first web UI", f"root={root_status} asset={asset_status} path={asset_path or 'missing'}", "Run bash scripts/build_webui.sh and verify AGENT_WEBUI_DIST_PATH."))

    state_status, state, _state_raw = _request(base_url, "GET", "/state", timeout=timeout)
    llm_status_code, llm, _llm_raw = _request(base_url, "GET", "/llm/status", timeout=timeout)
    runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
    model = str(llm.get("default_model") or llm.get("resolved_default_model") or "").strip()
    provider = str(llm.get("default_provider") or "").strip()
    if state_status == 200 and llm_status_code == 200 and state.get("ok") is True and model and provider:
        checks.append(_pass("backend runtime/model truth", f"runtime={runtime.get('runtime_mode') or runtime.get('state') or 'reported'} provider={provider} model={model}"))
    else:
        checks.append(_fail("backend runtime/model truth", f"state={state_status} llm={llm_status_code} provider={provider or 'missing'} model={model or 'missing'}", "Run python -m agent doctor and inspect /state plus /llm/status."))

    list_result = _chat(base_url, f"list files in {FIXTURE_DIR}", check_id="file-list", timeout=timeout)
    list_ok = _filesystem_ok(list_result) and "closure-note.txt" in _message(list_result)
    checks.append(_pass("bounded file list", "filesystem tool listed closure-note.txt") if list_ok else _fail("bounded file list", _message(list_result)[:300], "Inspect deterministic filesystem list routing."))

    search_result = _chat(base_url, f"find files named closure-note in {FIXTURE_DIR}", check_id="file-search", timeout=timeout)
    search_ok = _filesystem_ok(search_result) and "closure-note.txt" in _message(search_result)
    checks.append(_pass("bounded file search", "filesystem tool found closure-note.txt") if search_ok else _fail("bounded file search", _message(search_result)[:300], "Inspect filename-query/root extraction and bounded search."))

    read_result = _chat(base_url, f"read {FIXTURE_FILE}", check_id="file-read", timeout=timeout)
    read_ok = _filesystem_ok(read_result) and FIXTURE_MARKER in _message(read_result)
    checks.append(_pass("bounded file read", f"filesystem tool returned marker {FIXTURE_MARKER}") if read_ok else _fail("bounded file read", _message(read_result)[:300], "Inspect deterministic filesystem read routing."))

    sensitive_result = _chat(base_url, "read ~/.ssh/config", check_id="file-sensitive", timeout=timeout)
    sensitive_message = _message(sensitive_result).lower()
    sensitive_ok = (
        _filesystem_ok(sensitive_result)
        and sensitive_result.get("ok") is False
        and str(sensitive_result.get("error_kind") or "") == "sensitive_path_blocked"
        and any(token in sensitive_message for token in ("protected", "blocked", "privacy"))
    )
    checks.append(_pass("sensitive file denial", "sensitive_path_blocked with clear user wording") if sensitive_ok else _fail("sensitive file denial", _message(sensitive_result)[:300], "Inspect filesystem sensitive-root policy and response serialization."))

    sources_status, sources, _sources_raw = _request(base_url, "GET", "/pack_sources", timeout=timeout)
    packs_status, packs, _packs_raw = _request(base_url, "GET", "/packs/state", timeout=timeout)
    if sources_status == 200 and packs_status == 200 and isinstance(sources.get("sources"), list) and packs.get("ok") is True:
        checks.append(_pass("external pack discovery/state", f"sources={len(sources.get('sources') or [])} pack_state=ok"))
    else:
        checks.append(_fail("external pack discovery/state", f"sources={sources_status} packs={packs_status}", "Inspect /pack_sources and /packs/state."))

    pack_proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pack_route_smoke.py"), "--base-url", base_url],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=max(60.0, timeout * 8),
        check=False,
    )
    pack_summary = next((line for line in reversed(pack_proc.stdout.splitlines()) if line.strip()), "no output")
    if pack_proc.returncode == 0 and pack_summary == "dead_end_warnings: none":
        checks.append(_pass("external pack round-trip/policy", "temporary discovery source confirmed, queried, previewed, and removed; ingestion succeeded or returned the exact Safe Mode policy block"))
    else:
        checks.append(_fail("external pack round-trip/policy", f"exit={pack_proc.returncode} summary={pack_summary}", "Run python scripts/pack_route_smoke.py for per-route evidence."))

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove the live Personal Agent end-user closure contract.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    checks = run(str(args.base_url), timeout=max(1.0, float(args.timeout)))
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
        print(f"{check.status}: {check.name} — {check.evidence}")
        if check.status != "PASS":
            print(f"  next: {check.next_action}")
    print(f"SUMMARY: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
