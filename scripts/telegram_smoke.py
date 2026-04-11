#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.telegram_bridge import handle_telegram_text


DEFAULT_BASE_URL = os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8765"
CANONICAL_PROMPT = "what is the runtime status?"
WARMUP_PROMPT = "what do i have for ram and vram right now?"
API_CHAT_TIMEOUT_SECONDS = 45.0


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _json_from_response(body: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _post_chat(base_url: str, payload: dict[str, Any], *, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            ok = int(getattr(response, "status", 200)) < 400
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        ok = False
        status = int(getattr(exc, "code", 500))
    except urllib.error.URLError as exc:
        return {"ok": False, "status": 0, "error": f"transport error: {exc.reason}", "raw": "", "payload": {}}
    except Exception as exc:  # pragma: no cover - defensive live smoke guard
        return {"ok": False, "status": 0, "error": f"transport error: {exc}", "raw": "", "payload": {}}
    payload = _json_from_response(raw)
    return {"ok": bool(ok), "status": status, "raw": raw, "payload": payload}


def _build_api_chat_payload(*, text: str, trace_id: str, user_id: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": str(text or "")}],
        "purpose": "chat",
        "task_type": "chat",
        "source_surface": "operator_smoke",
        "user_id": user_id,
        "thread_id": f"{user_id}:thread",
        "trace_id": trace_id,
    }


def _response_summary(payload: dict[str, Any]) -> dict[str, Any]:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    text = str(assistant.get("content") or payload.get("message") or payload.get("error") or "").strip()
    return {
        "ok": bool(payload.get("ok", True)),
        "text": text,
        "first_line": _first_line(text),
        "route": str(meta.get("route") or "").strip().lower() or "unknown",
    }


def _status_text(result: dict[str, Any]) -> str:
    if bool(result.get("ok")):
        return "ok"
    status = result.get("status")
    if isinstance(status, int) and status:
        return str(status)
    if str(result.get("route") or "").strip():
        return str(result.get("route"))
    return str(result.get("error") or "error").strip() or "error"


def _dead_end_warnings(*, first_line: str, text: str, ok: bool) -> list[str]:
    lowered = f"{first_line}\n{text}".lower()
    warnings: list[str] = []
    if not ok:
        warnings.append("transport not ok")
    if not first_line:
        warnings.append("empty first line")
    if first_line.startswith("{") or first_line.startswith("["):
        warnings.append("raw dump")
    if any(
        token in lowered
        for token in (
            "need more context",
            "i need more context",
            "couldn't read",
            "could not read",
            "need to know",
            "can't help",
            "cannot help",
        )
    ):
        warnings.append("dead-end wording")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live Telegram-vs-API chat parity smoke against the canonical runtime."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    parser.add_argument("--prompt", default=CANONICAL_PROMPT, help="Prompt to send through chat and Telegram bridge.")
    args = parser.parse_args(argv)

    trace_id = f"operator-live-smoke-runtime-{int(time.time())}"
    chat_id = "telegram-smoke"
    api_user_id = "operator-smoke-runtime"
    warmup_payload = _build_api_chat_payload(text=WARMUP_PROMPT, trace_id=f"{trace_id}-warmup", user_id="operator-smoke-hardware")
    _post_chat(str(args.base_url), warmup_payload, timeout=20.0)
    payload = _build_api_chat_payload(text=str(args.prompt), trace_id=trace_id, user_id=api_user_id)

    api_result = _post_chat(str(args.base_url), payload, timeout=API_CHAT_TIMEOUT_SECONDS)
    api_payload = api_result.get("payload") if isinstance(api_result.get("payload"), dict) else {}
    api_summary = _response_summary(api_payload)

    def _proxy(chat_payload: dict[str, Any]) -> dict[str, Any]:
        prompt_text = str((chat_payload.get("messages") or [{}])[0].get("content") or "")
        proxy_payload = _build_api_chat_payload(text=prompt_text, trace_id=trace_id, user_id=api_user_id)
        result = _post_chat(str(args.base_url), proxy_payload, timeout=API_CHAT_TIMEOUT_SECONDS)
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        return payload if isinstance(payload, dict) and payload else {"_proxy_error": {"kind": "invalid_response"}}

    telegram_result = handle_telegram_text(
        text=str(args.prompt),
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=None,
        orchestrator=None,
        fetch_local_api_chat_json=_proxy,
    )
    telegram_text = str(telegram_result.get("text") or "").strip()
    telegram_summary = {
        "ok": bool(telegram_result.get("ok", True)),
        "text": telegram_text,
        "first_line": _first_line(telegram_text),
        "route": str(telegram_result.get("route") or "").strip().lower() or "unknown",
    }

    api_warnings = _dead_end_warnings(
        first_line=str(api_summary.get("first_line") or ""),
        text=str(api_summary.get("text") or ""),
        ok=bool(api_summary.get("ok")),
    )
    telegram_warnings = _dead_end_warnings(
        first_line=str(telegram_summary.get("first_line") or ""),
        text=str(telegram_summary.get("text") or ""),
        ok=bool(telegram_summary.get("ok")),
    )
    route_match = str(api_summary.get("route") or "") == str(telegram_summary.get("route") or "")
    first_line_match = str(api_summary.get("first_line") or "") == str(telegram_summary.get("first_line") or "")

    print(f"[api_chat] route: {api_summary['route']}")
    print(f"[api_chat] status: {_status_text(api_result)}")
    print(f"[api_chat] first_line: {api_summary['first_line']}")
    if str(api_result.get("error") or "").strip():
        print(f"[api_chat] error: {api_result.get('error')}")
    if not api_summary["first_line"] and str(api_result.get("raw") or "").strip():
        print(f"[api_chat] raw_start: {str(api_result.get('raw') or '')[:240]}")
    print(f"[api_chat] dead_end_warnings: {', '.join(api_warnings) if api_warnings else 'none'}")
    print(f"[telegram] route: {telegram_summary['route']}")
    print(f"[telegram] status: {_status_text(telegram_result)}")
    print(f"[telegram] first_line: {telegram_summary['first_line']}")
    if not telegram_summary["first_line"] and str(telegram_result.get("text") or "").strip():
        print(f"[telegram] raw_text_start: {str(telegram_result.get('text') or '')[:240]}")
    print(f"[telegram] dead_end_warnings: {', '.join(telegram_warnings) if telegram_warnings else 'none'}")
    print(f"[parity] route_match: {'yes' if route_match else 'no'}")
    print(f"[parity] first_line_match: {'yes' if first_line_match else 'no'}")

    if api_warnings or telegram_warnings or not route_match or not first_line_match:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
