#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.api_server import AgentRuntime
from agent.config import load_config


SMOKES: tuple[tuple[str, str], ...] = (
    ("hardware", "what do i have for ram and vram right now?"),
    ("runtime", "what is the runtime status?"),
    ("discovery", "there is a brand new tiny Gemma 4 model, can you look into it?"),
)


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _build_runtime() -> AgentRuntime:
    config = load_config(require_telegram_token=False)
    return AgentRuntime(config, defer_bootstrap_warmup=True)


def _run_prompt(runtime: AgentRuntime, label: str, prompt: str) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "source_surface": "operator_smoke",
        "purpose": "chat",
        "task_type": "chat",
        "user_id": f"operator-smoke-{label}",
        "thread_id": f"operator-smoke-{label}:thread",
        "trace_id": f"operator-live-smoke-{label}-{int(time.time())}",
    }
    ok, body = runtime.chat(payload)
    response = body if isinstance(body, dict) else {}
    assistant = response.get("assistant") if isinstance(response.get("assistant"), dict) else {}
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    text = str(assistant.get("content") or response.get("message") or "").strip()
    return {
        "ok": bool(ok),
        "text": text,
        "first_line": _first_line(text),
        "route": str(meta.get("route") or "").strip().lower() or "unknown",
        "used_tools": list(meta.get("used_tools") or []) if isinstance(meta.get("used_tools"), list) else [],
    }


def _failure_warnings(label: str, first_line: str, text: str, ok: bool) -> list[str]:
    lowered = f"{first_line}\n{text}".lower()
    warnings: list[str] = []
    if not ok:
        warnings.append("transport not ok")
    if not first_line:
        warnings.append("empty first line")
    if first_line.startswith("{") or first_line.startswith("["):
        warnings.append("raw dump")
    if any(token in lowered for token in ("need more context", "does not exist", "i can't", "i cannot", "couldn't complete", "no discovery sources are enabled")):
        warnings.append("dead-end wording")
    if label == "hardware" and not any(token in lowered for token in ("ram", "vram")):
        warnings.append("missing ram/vram answer")
    if label == "runtime":
        if not any(token in lowered for token in ("runtime", "ready", "health")):
            warnings.append("missing runtime/readiness answer")
        if any(token in lowered for token in ("degraded", "not ready", "failed")):
            warnings.append("runtime not ready")
    if label == "discovery" and not any(token in lowered for token in ("model", "source", "discover")):
        warnings.append("missing discovery answer")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live, non-blocking smoke family against the canonical runtime answer-shape paths."
    )
    parser.add_argument("--prompt", choices=[name for name, _ in SMOKES], help="Run only one smoke family member.")
    args = parser.parse_args(argv)

    runtime = _build_runtime()
    try:
        runs = [(name, prompt) for name, prompt in SMOKES if not args.prompt or args.prompt == name]
        exit_code = 0
        for label, prompt in runs:
            result = _run_prompt(runtime, label, prompt)
            first_line = str(result.get("first_line") or "")
            route = str(result.get("route") or "unknown")
            text = str(result.get("text") or "")
            warnings = _failure_warnings(label, first_line, text, bool(result.get("ok")))

            print(f"[{label}] route: {route}")
            print(f"[{label}] first_line: {first_line}")
            print(f"[{label}] dead_end_warnings: {', '.join(warnings) if warnings else 'none'}")

            if warnings:
                exit_code = 1
        return exit_code
    finally:
        try:
            runtime.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
