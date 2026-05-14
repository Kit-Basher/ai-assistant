#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.telegram_bridge import handle_telegram_text


DEFAULT_BASE_URL = os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8765"
DEFAULT_USER_ID = "live-user-barrage"
DEFAULT_THREAD_ID = "live-user-barrage:thread"

INTERNAL_LEAK_MARKERS = (
    "source_surface",
    "thread_id",
    "runtime_payload",
    "read-only guard",
    "nl path refused",
    "traceback",
)
DIAGNOSTIC_READY_MARKERS = (
    "i'm here and ready to help",
    "i’m here and ready to help",
)
WHOLE_ANSWER_FORBIDDEN = {"ok", "done."}
MODEL_ACQUISITION_MARKERS = (
    "which model do you want me to acquire",
    "model acquisition",
    "acquire a model",
    "pull a model",
)


@dataclass(frozen=True)
class PromptCase:
    category: str
    prompt: str


PROMPT_CASES: tuple[PromptCase, ...] = (
    PromptCase("runtime_status", "what model am i using"),
    PromptCase("runtime_status", "what is your runtime status"),
    PromptCase("runtime_status", "are you actually connected to a model right now"),
    PromptCase("runtime_status", "which provider are you using"),
    PromptCase("runtime_status", "is the local API healthy"),
    PromptCase("memory", "is memory on"),
    PromptCase("memory", "do you remember what we were doing"),
    PromptCase("memory", "what do you remember about my preferences"),
    PromptCase("memory", "where were we before"),
    PromptCase("frustration", "why arent you working"),
    PromptCase("frustration", "you keep giving useless answers, diagnose yourself"),
    PromptCase("frustration", "this feels broken, what is wrong"),
    PromptCase("frustration", "fix yourself"),
    PromptCase("app_setup", "open the app"),
    PromptCase("app_setup", "how do i open the web UI"),
    PromptCase("app_setup", "is setup complete"),
    PromptCase("app_setup", "help me set this up"),
    PromptCase("model_switch", "use ollama:qwen3.6:35b-a3b for this chat session only"),
    PromptCase("model_switch", "yes"),
    PromptCase("model_switch", "what model am i using now"),
    PromptCase("model_switch", "make qwen3.6 the default model"),
    PromptCase("skill_install", "install a skill that lets you browse"),
    PromptCase("skill_install", "can you add browser capabilities"),
    PromptCase("skill_install", "what skills can you install for web research"),
    PromptCase("skill_install", "add a capability for reading webpages"),
    PromptCase("vague", "fix it"),
    PromptCase("vague", "do it"),
    PromptCase("vague", "1"),
    PromptCase("vague", "yes"),
    PromptCase("vague", "continue"),
    PromptCase("system_slow", "my computer is slow"),
    PromptCase("system_slow", "why is my system lagging"),
    PromptCase("system_slow", "check ram and cpu pressure"),
    PromptCase("system_slow", "is something eating resources"),
    PromptCase("open_chat", "help me plan the next hour"),
    PromptCase("open_chat", "explain this project in plain english"),
    PromptCase("open_chat", "what can you help me with right now"),
    PromptCase("open_chat", "give me a concise checklist for testing this app"),
    PromptCase("open_chat", "write a short note saying the assistant is working"),
    PromptCase("open_chat", "what should I ask you next"),
)


def first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def load_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", 200)), load_json(response.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), load_json(exc.read())


def require_ready(base_url: str, *, timeout: float) -> dict[str, Any]:
    status, payload = request_json("GET", base_url, "/ready", timeout=timeout)
    if status >= 400 or not payload:
        raise RuntimeError(f"/ready unavailable: status={status}")
    if payload.get("ok") is not True or payload.get("ready") is not True:
        runtime_mode = str(payload.get("runtime_mode") or payload.get("phase") or "unknown")
        next_action = str(payload.get("next_action") or "")
        raise RuntimeError(
            f"/ready is not ready: ok={payload.get('ok')!r} ready={payload.get('ready')!r} "
            f"runtime_mode={runtime_mode} next_action={next_action}"
        )
    return payload


def extract_chat_result(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    text = str(
        assistant.get("content")
        or payload.get("message")
        or payload.get("response")
        or payload.get("text")
        or payload.get("error")
        or ""
    ).strip()
    return {
        "status": status,
        "ok": bool(payload.get("ok", status < 400)),
        "payload": payload,
        "text": text,
        "first_line": first_line(text),
        "route": str(meta.get("route") or payload.get("intent") or "unknown").strip().lower() or "unknown",
        "used_llm": bool(meta.get("used_llm", False)),
        "used_runtime_state": bool(meta.get("used_runtime_state", False)),
    }


def classify_barrage_response(case: PromptCase, response: dict[str, Any]) -> list[str]:
    text = str(response.get("text") or "").strip()
    line = str(response.get("first_line") or first_line(text)).strip()
    lowered = f"{line}\n{text}".lower()
    failures: list[str] = []

    if not text:
        failures.append("empty response")
    if text.lower() in WHOLE_ANSWER_FORBIDDEN:
        failures.append("stale whole-answer placeholder")
    for marker in INTERNAL_LEAK_MARKERS:
        if marker in lowered:
            failures.append(f"internal leak: {marker}")
    if case.category == "frustration" and any(marker in lowered for marker in DIAGNOSTIC_READY_MARKERS):
        failures.append("diagnostic/frustration prompt got stale ready-to-help wording")
    if case.category == "model_switch" and "session only" in case.prompt.lower():
        if "default model updated" in lowered or "default chat model updated" in lowered:
            failures.append("temporary switch claimed default was updated")
    if case.category == "skill_install" and ("browse" in case.prompt.lower() or "browser" in case.prompt.lower()):
        if any(marker in lowered for marker in MODEL_ACQUISITION_MARKERS):
            failures.append("browse skill request was treated as model acquisition")
    return failures


def post_chat_case(
    base_url: str,
    case: PromptCase,
    *,
    user_id: str,
    thread_id: str,
    trace_id: str,
    timeout: float,
) -> dict[str, Any]:
    status, payload = request_json(
        "POST",
        base_url,
        "/chat",
        payload={
            "messages": [{"role": "user", "content": case.prompt}],
            "purpose": "chat",
            "task_type": "chat",
            "source_surface": "operator_smoke",
            "user_id": user_id,
            "thread_id": thread_id,
            "trace_id": trace_id,
        },
        timeout=timeout,
    )
    return extract_chat_result(status, payload)


def print_result(prefix: str, case: PromptCase, response: dict[str, Any], failures: list[str]) -> None:
    print(f"[{prefix}] prompt={case.prompt!r}")
    print(
        f"[{prefix}] route={response.get('route') or 'unknown'} "
        f"used_llm={bool(response.get('used_llm'))} "
        f"used_runtime_state={bool(response.get('used_runtime_state'))}"
    )
    print(f"[{prefix}] first_line={str(response.get('first_line') or '')[:220]}")
    print(f"[{prefix}] result={'FAIL: ' + '; '.join(failures) if failures else 'PASS'}")


def telegram_proxy_factory(base_url: str, timeout: float) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _proxy(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            status, result = request_json("POST", base_url, "/chat", payload=payload, timeout=timeout)
        except Exception as exc:
            return {"_proxy_error": {"kind": "unreachable", "detail": str(exc)}}
        if result:
            return result
        return {"_proxy_error": {"kind": "http_error", "status_code": status}}

    return _proxy


def run_api_barrage(base_url: str, *, timeout: float, limit: int | None = None) -> list[str]:
    failures: list[str] = []
    cases = PROMPT_CASES[: max(0, int(limit))] if limit is not None else PROMPT_CASES
    for index, case in enumerate(cases, start=1):
        trace_id = f"live-user-barrage-{index}-{int(time.time())}"
        try:
            response = post_chat_case(
                base_url,
                case,
                user_id=DEFAULT_USER_ID,
                thread_id=DEFAULT_THREAD_ID,
                trace_id=trace_id,
                timeout=timeout,
            )
            case_failures = classify_barrage_response(case, response)
        except Exception as exc:
            response = {
                "route": "transport_error",
                "used_llm": False,
                "used_runtime_state": False,
                "first_line": "",
                "text": "",
            }
            case_failures = [f"transport error: {exc}"]
        print_result(f"api:{index:02d}:{case.category}", case, response, case_failures)
        failures.extend(f"api {index:02d} {case.prompt!r}: {failure}" for failure in case_failures)
    return failures


def run_telegram_barrage(base_url: str, *, timeout: float, limit: int | None = None) -> list[str]:
    failures: list[str] = []
    cases = PROMPT_CASES[: max(0, int(limit))] if limit is not None else PROMPT_CASES
    proxy = telegram_proxy_factory(base_url, timeout)
    chat_id = "live-user-barrage-telegram"
    for index, case in enumerate(cases, start=1):
        try:
            result = handle_telegram_text(
                text=case.prompt,
                chat_id=chat_id,
                trace_id=f"live-user-barrage-tg-{index}-{int(time.time())}",
                runtime=None,
                orchestrator=None,
                fetch_local_api_chat_json=proxy,
            )
            response = {
                "route": str(result.get("route") or "unknown"),
                "used_llm": bool(result.get("used_llm", False)),
                "used_runtime_state": bool(result.get("used_runtime_state", False)),
                "text": str(result.get("text") or ""),
                "first_line": first_line(str(result.get("text") or "")),
            }
            case_failures = classify_barrage_response(case, response)
            if str(result.get("handler_name") or "") != "api_chat_proxy" and case.prompt not in {
                "/start",
                "/help",
                "/status",
                "/setup",
            }:
                if case.category not in {"frustration"} or "are you" not in case.prompt.lower():
                    pass
        except Exception as exc:
            response = {
                "route": "telegram_error",
                "used_llm": False,
                "used_runtime_state": False,
                "text": "",
                "first_line": "",
            }
            case_failures = [f"telegram bridge error: {exc}"]
        print_result(f"tg:{index:02d}:{case.category}", case, response, case_failures)
        failures.extend(f"telegram {index:02d} {case.prompt!r}: {failure}" for failure in case_failures)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run messy real-user prompts against the live /chat path.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--telegram-bridge", action="store_true", help="Also run prompts through Telegram bridge api_chat_proxy.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N prompt cases.")
    args = parser.parse_args(argv)

    try:
        ready = require_ready(str(args.base_url), timeout=float(args.timeout))
    except Exception as exc:
        print(f"[ready] FAIL: {exc}", file=sys.stderr)
        return 2
    print(
        f"[ready] ok=True ready=True runtime_mode={ready.get('runtime_mode') or ready.get('phase') or 'unknown'} "
        f"chat_usable={bool(ready.get('chat_usable', False))}"
    )

    failures = run_api_barrage(str(args.base_url), timeout=float(args.timeout), limit=args.limit)
    if args.telegram_bridge:
        failures.extend(run_telegram_barrage(str(args.base_url), timeout=float(args.timeout), limit=args.limit))

    if failures:
        print("[live_user_barrage] FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("[live_user_barrage] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
