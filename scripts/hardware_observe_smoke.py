#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.api_server import AgentRuntime
from agent.config import load_config


HARDWARE_PROMPT = "what do i have for ram and vram right now?"


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _build_runtime() -> AgentRuntime:
    config = load_config(require_telegram_token=False)
    return AgentRuntime(config, defer_bootstrap_warmup=True)


def _run_hardware_prompt(runtime: AgentRuntime, prompt: str) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "source_surface": "operator_smoke",
        "purpose": "chat",
        "task_type": "chat",
        "user_id": "operator-smoke",
        "thread_id": "operator-smoke:thread",
        "trace_id": f"operator-hardware-smoke-{int(time.time())}",
    }
    ok, body = runtime.chat(payload)
    response = body if isinstance(body, dict) else {}
    assistant = response.get("assistant") if isinstance(response.get("assistant"), dict) else {}
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    text = str(assistant.get("content") or response.get("message") or "").strip()
    first_line = _first_line(text)
    route = str(meta.get("route") or "").strip().lower() or "unknown"
    return {
        "ok": bool(ok),
        "text": text,
        "first_line": first_line,
        "route": route,
        "meta": meta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live, non-blocking RAM/VRAM answer-shape smoke check against the canonical runtime."
    )
    parser.add_argument("--prompt", default=HARDWARE_PROMPT, help="Prompt to send to the runtime.")
    args = parser.parse_args(argv)

    runtime = _build_runtime()
    try:
        result = _run_hardware_prompt(runtime, str(args.prompt))
    finally:
        try:
            runtime.close()
        except Exception:
            pass

    first_line = str(result.get("first_line") or "")
    route = str(result.get("route") or "unknown")
    print(f"route: {route}")
    print(f"first_line: {first_line}")

    if not first_line:
        print("FAIL: empty first line")
        return 1
    lowered = first_line.lower()
    if first_line.startswith("{") or first_line.startswith("["):
        print("FAIL: response looks like a raw dump")
        return 1
    if "ram" not in lowered:
        print("FAIL: first line does not directly address RAM")
        return 1
    if "vram" not in lowered and "unavailable" not in lowered:
        print("FAIL: first line does not mention VRAM availability")
        return 1
    if any(token in lowered for token in ("pid", "process", "system health")):
        print("FAIL: diagnostic detail appeared before the direct answer")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
