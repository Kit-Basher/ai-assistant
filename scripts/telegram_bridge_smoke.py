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
TARGET_MODEL = "ollama:qwen3.6:35b-a3b"
PROMPTS = [
    "what model am i using",
    "why arent you working",
    "is memory on",
    "do you remember what we were doing",
    f"use {TARGET_MODEL} for this chat session only",
    "yes",
    "what model am i using",
]
FORBIDDEN_INTERNAL_TEXT = (
    "source_surface",
    "thread_id",
    "selection_policy",
    "runtime_payload",
    "traceback",
    "guardrail",
    "read-only guard",
)
OLD_SOCIAL_TEXT = (
    "i'm here",
    "i’m here",
    "what should i do next",
    "got it.",
)


def _load_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_json(base_url: str, path: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        method="GET",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return _load_json(response.read())


def _post_chat(base_url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = _load_json(response.read())
            if isinstance(result, dict) and result:
                return result
            return {"_proxy_error": {"kind": "invalid_response", "status_code": int(response.status)}}
    except urllib.error.HTTPError as exc:
        body = _load_json(exc.read())
        if body:
            return body
        return {"_proxy_error": {"kind": "http_error", "status_code": int(exc.code)}}
    except Exception as exc:
        return {"_proxy_error": {"kind": "unreachable", "detail": str(exc)}}


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _model_entry(status: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = status.get("models") if isinstance(status.get("models"), list) else []
    for item in models:
        if isinstance(item, dict) and str(item.get("id") or "") == model_id:
            return item
    return {}


def _text(result: dict[str, Any]) -> str:
    return str(result.get("text") or "").strip()


def _run_prompt(
    *,
    base_url: str,
    timeout: float,
    chat_id: str,
    prompt: str,
    index: int,
) -> dict[str, Any]:
    trace_id = f"telegram-bridge-smoke-{int(time.time())}-{index}"

    def _proxy(payload: dict[str, Any]) -> dict[str, Any]:
        return _post_chat(base_url, payload, timeout=timeout)

    return handle_telegram_text(
        text=prompt,
        chat_id=chat_id,
        trace_id=trace_id,
        runtime=None,
        orchestrator=None,
        fetch_local_api_chat_json=_proxy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exercise Telegram's local message-to-/chat bridge without contacting Telegram servers."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--chat-id", default=f"telegram-bridge-smoke-{os.getpid()}")
    args = parser.parse_args(argv)

    failures: list[str] = []
    try:
        status = _get_json(str(args.base_url), "/llm/status", timeout=float(args.timeout))
    except Exception as exc:
        print(f"[llm/status] failed: {exc}")
        return 1

    current_model = str(status.get("effective_chat_model") or status.get("chat_model") or "").strip()
    target = _model_entry(status, TARGET_MODEL)
    print(f"[llm/status] current_model={current_model or 'unknown'}")
    print(
        "[llm/status] target "
        f"available={bool(target)} selectable={bool(target.get('selectable_now'))} "
        f"chat_usable={bool(target.get('chat_usable'))}"
    )
    _assert(bool(target), f"{TARGET_MODEL} missing from /llm/status", failures)
    _assert(bool(target.get("selectable_now")), f"{TARGET_MODEL} is not selectable_now", failures)
    _assert(bool(target.get("chat_usable")), f"{TARGET_MODEL} is not chat_usable", failures)

    rows: list[tuple[str, dict[str, Any]]] = []
    for index, prompt in enumerate(PROMPTS, start=1):
        result = _run_prompt(
            base_url=str(args.base_url),
            timeout=float(args.timeout),
            chat_id=str(args.chat_id),
            prompt=prompt,
            index=index,
        )
        rows.append((prompt, result))
        line = _text(result).splitlines()[0] if _text(result) else ""
        print(
            f"[telegram] prompt={prompt!r} ok={bool(result.get('ok', True))} "
            f"route={result.get('route') or 'unknown'} handler={result.get('handler_name') or 'unknown'} "
            f"first_line={line[:160]}"
        )

        text_lower = _text(result).lower()
        _assert(bool(result.get("ok", True)), f"{prompt!r} returned ok=false", failures)
        _assert(
            str(result.get("handler_name") or "") == "api_chat_proxy",
            f"{prompt!r} did not use api_chat_proxy",
            failures,
        )
        _assert(
            bool(result.get("legacy_compatibility")) is False,
            f"{prompt!r} used legacy compatibility path",
            failures,
        )
        for forbidden in FORBIDDEN_INTERNAL_TEXT:
            _assert(forbidden not in text_lower, f"{prompt!r} leaked internal text {forbidden!r}", failures)

    why_text = _text(rows[1][1]).lower()
    for old in OLD_SOCIAL_TEXT:
        _assert(old not in why_text, "'why arent you working' returned old social-presence wording", failures)

    switch_text = _text(rows[4][1]).lower()
    confirm_text = _text(rows[5][1]).lower()
    combined_switch = f"{switch_text}\n{confirm_text}"
    _assert("temporary" in combined_switch, "temporary switch response did not say temporary", failures)
    _assert("does not change your default model" in combined_switch, "temporary switch response missed default warning", failures)
    _assert("default model updated" not in combined_switch, "temporary switch said default model updated", failures)
    _assert("default chat model updated" not in combined_switch, "temporary switch said default chat model updated", failures)

    final_model_text = _text(rows[-1][1]).lower()
    _assert("qwen3.6" in final_model_text or TARGET_MODEL.lower() in final_model_text, "final model answer did not report target model", failures)

    if failures:
        print("[result] failed")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("[result] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
