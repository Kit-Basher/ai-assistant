#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


TARGET_MODEL = "ollama:qwen3.6:35b-a3b"


def _request(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    if not raw.strip():
        return {}
    return json.loads(raw)


def _find_model(status: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    models = status.get("models")
    if not isinstance(models, list):
        return None
    for row in models:
        if isinstance(row, dict) and str(row.get("id") or "") == model_id:
            return row
    return None


def _chat(base_url: str, message: str, *, thread_id: str, timeout: float) -> dict[str, Any]:
    return _request(
        "POST",
        base_url,
        "/chat",
        {
            "message": message,
            "thread_id": thread_id,
            "user_id": "live-model-switch-smoke",
            "source_surface": "web",
        },
        timeout=timeout,
    )


def _response_text(payload: dict[str, Any]) -> str:
    for key in ("response", "message", "text", "assistant", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    nested = payload.get("reply") or payload.get("assistant_message")
    if isinstance(nested, dict):
        for key in ("content", "text", "message"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return json.dumps(payload, sort_keys=True)


def _needs_confirmation(payload: dict[str, Any]) -> bool:
    text = _response_text(payload).lower()
    if payload.get("requires_confirmation") is True or payload.get("needs_confirmation") is True:
        return True
    return any(phrase in text for phrase in ("confirm", "are you sure", "say yes", "do you want me to"))


def _temporary_switch_prompt(model_id: str) -> str:
    return f"Use {model_id} for this chat session only. Do not change my default model."


def _switch_chat_model(base_url: str, model_id: str, *, thread_id: str, timeout: float, label: str) -> dict[str, Any]:
    payload = _chat(base_url, _temporary_switch_prompt(model_id), thread_id=thread_id, timeout=timeout)
    print(f"{label}_response={_response_text(payload)}")
    if _needs_confirmation(payload):
        payload = _chat(base_url, "yes", thread_id=thread_id, timeout=timeout)
        print(f"{label}_confirm_response={_response_text(payload)}")
    return payload


def run_smoke(base_url: str, model_id: str, timeout: float) -> int:
    status = _request("GET", base_url, "/llm/status", timeout=timeout)
    current_model = str(status.get("effective_chat_model") or status.get("current_model") or "")
    target = _find_model(status, model_id)
    print(f"current_model={current_model or 'unknown'}")
    if target is None:
        print(f"target_model_missing={model_id}", file=sys.stderr)
        return 2

    selectable = bool(target.get("selectable_now", False))
    chat_usable = bool(target.get("chat_usable", False))
    print(
        "target_model="
        + json.dumps(
            {
                "id": target.get("id"),
                "health": target.get("health"),
                "selectable_now": selectable,
                "chat_usable": chat_usable,
                "requires_probe": target.get("requires_probe"),
            },
            sort_keys=True,
        )
    )
    if not selectable or not chat_usable:
        print(f"target_model_not_selectable_or_chat_usable={model_id}", file=sys.stderr)
        return 3

    thread_id = f"live-model-switch-smoke-{int(time.time())}"
    _switch_chat_model(base_url, model_id, thread_id=thread_id, timeout=timeout, label="switch")
    verify_payload = _chat(base_url, "what model am i using", thread_id=thread_id, timeout=timeout)
    verify_text = _response_text(verify_payload)
    print(f"verify_response={verify_text}")
    if model_id.lower() not in verify_text.lower():
        print(f"model_switch_not_reported={model_id}", file=sys.stderr)
        return 4
    if current_model and current_model != model_id:
        _switch_chat_model(base_url, current_model, thread_id=thread_id, timeout=timeout, label="restore")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live smoke for temporary assistant model switching.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    try:
        return run_smoke(args.base_url, args.model, args.timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"live_model_switch_smoke_failed={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
