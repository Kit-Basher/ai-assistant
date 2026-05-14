#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.telegram_bridge import handle_telegram_text


BAD_TEXT_MARKERS = (
    "read-only guard",
    "nl path refused",
    "source_surface",
    "thread_id",
    "user_id",
    "runtime_payload",
    "runtime_state_failure_reason",
    "default model updated",
)


def _post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        decoded = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(decoded)
    if not isinstance(parsed, dict):
        raise AssertionError(f"{path} did not return a JSON object")
    return parsed


def _get_json(base_url: str, path: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        decoded = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(decoded)
    if not isinstance(parsed, dict):
        raise AssertionError(f"{path} did not return a JSON object")
    return parsed


def _assert_user_facing(label: str, text: Any) -> None:
    rendered = str(text or "").strip()
    if not rendered:
        raise AssertionError(f"{label}: empty user-facing text")
    if rendered in {"OK", "Ok", "Done.", "I’m here and ready"}:
        raise AssertionError(f"{label}: stale placeholder text: {rendered!r}")
    lowered = rendered.lower()
    for marker in BAD_TEXT_MARKERS:
        if marker in lowered:
            raise AssertionError(f"{label}: leaked internal marker {marker!r}: {rendered!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test intentional pre-chat and Telegram bypass behavior.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    failures: list[str] = []

    for path in ("/ready", "/llm/status"):
        try:
            payload = _get_json(args.base_url, path, float(args.timeout))
            print(f"[{path}] ok keys={','.join(sorted(payload.keys())[:8])}", flush=True)
        except Exception as exc:
            failures.append(f"{path} failed: {exc}")

    chat_cases = [
        (
            "empty-chat",
            {
                "messages": [],
                "source_surface": "smoke",
                "user_id": "smoke:bypass",
                "thread_id": "smoke:bypass:thread",
                "trace_id": f"bypass-smoke-empty-{int(time.time())}",
            },
        ),
    ]
    for label, payload in chat_cases:
        try:
            response = _post_json(args.base_url, "/chat", payload, float(args.timeout))
            text = response.get("message") or (response.get("assistant") or {}).get("content")
            _assert_user_facing(label, text)
            print(f"[chat:{label}] route={(response.get('meta') or {}).get('route') or response.get('intent') or 'unknown'}", flush=True)
        except Exception as exc:
            failures.append(f"chat {label} failed: {exc}")

    telegram_cases = ["/start", "/help", "/status", "/setup", "hello are you working?"]
    for index, text in enumerate(telegram_cases, start=1):
        try:
            result = handle_telegram_text(
                text=text,
                chat_id="bypass-smoke",
                trace_id=f"tg-bypass-smoke-{index}",
                runtime=None,
                orchestrator=None,
                fetch_local_api_json=lambda path: _get_json(args.base_url, path, float(args.timeout)),
                fetch_local_api_chat_json=lambda payload: _post_json(args.base_url, "/chat", payload, float(args.timeout)),
            )
            _assert_user_facing(f"telegram {text}", result.get("text"))
            print(f"[telegram:{text}] route={result.get('route')} handler={result.get('handler_name')}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError, AssertionError, Exception) as exc:
            failures.append(f"telegram {text!r} failed: {exc}")

    try:
        numeric = handle_telegram_text(
            text="1",
            chat_id="bypass-smoke",
            trace_id="tg-bypass-smoke-number",
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=lambda _payload: {
                "ok": True,
                "assistant": {"content": "I need a little more context. Tell me what option 1 refers to."},
                "message": "I need a little more context. Tell me what option 1 refers to.",
                "meta": {"route": "generic_chat"},
            },
        )
        _assert_user_facing("telegram numeric no-wizard", numeric.get("text"))
        if numeric.get("handler_name") != "api_chat_proxy":
            raise AssertionError(f"numeric no-wizard used {numeric.get('handler_name')!r}")
        print(f"[telegram:1] route={numeric.get('route')} handler={numeric.get('handler_name')}", flush=True)
    except Exception as exc:
        failures.append(f"telegram numeric no-wizard failed: {exc}")

    if failures:
        print("[bypass_behavior_smoke] FAIL", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        return 1
    print("[bypass_behavior_smoke] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
