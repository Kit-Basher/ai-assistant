#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BASE_URL = os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8765"
PROMPTS: tuple[tuple[str, str, float], ...] = (
    ("gemma", "there is a brand new tiny Gemma 4 model, can you look into it?", 75.0),
    ("local_coding", "what's a small local coding model?", 60.0),
    ("local_vision", "is there a lightweight vision model I could run locally?", 60.0),
    ("qwen_chat", "what's newer than qwen2.5 3b for chat?", 75.0),
)

_DEAD_END_MARKERS = (
    "need more context",
    "i need more context",
    "couldn't read",
    "could not read",
    "need to know",
    "can't help",
    "cannot help",
    "no likely models matched",
    "no models matched",
    "no discovery sources are enabled",
)
_CANDIDATE_PATTERNS = (
    re.compile(r"\blikely family match:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"\bpractical local fit:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"\brelated alternative:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"\blikely fits:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"\btop matches:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"\bnearby candidates:\s*([^\n]+)", re.IGNORECASE),
)


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _json_from_response(body: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _post_chat(base_url: str, prompt: str, *, user_id: str, thread_id: str, trace_id: str, timeout: float) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": str(prompt or "")}],
        "purpose": "chat",
        "task_type": "chat",
        "source_surface": "operator_smoke",
        "user_id": user_id,
        "thread_id": thread_id,
        "trace_id": trace_id,
    }
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
    except TimeoutError:
        raw = json.dumps({"error": "transport timeout"})
        ok = False
        status = 0
    except urllib.error.URLError as exc:
        raw = json.dumps({"error": f"transport error: {exc.reason}"})
        ok = False
        status = 0
    except Exception as exc:  # pragma: no cover - defensive live-smoke guard
        raw = json.dumps({"error": f"transport error: {exc}"})
        ok = False
        status = 0

    payload = _json_from_response(raw)
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    text = str(assistant.get("content") or payload.get("message") or payload.get("error") or "").strip()
    return {
        "ok": bool(ok),
        "status": status,
        "payload": payload,
        "text": text,
        "first_line": _first_line(text),
        "route": str(meta.get("route") or "").strip().lower() or "unknown",
        "used_tools": [str(item).strip() for item in (meta.get("used_tools") if isinstance(meta.get("used_tools"), list) else []) if str(item).strip()],
        "raw": raw,
    }


def _status_text(result: dict[str, Any]) -> str:
    if bool(result.get("ok")):
        return "ok"
    status = result.get("status")
    if isinstance(status, int) and status:
        return str(status)
    return str(result.get("error") or "error").strip() or "error"


def _broadening_used(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ("broadened the search", "broadened to:", "i broadened the search", "broadened search:"))


def _candidate_names(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in _CANDIDATE_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        raw = match.group(1)
        for part in re.split(r"[,;]", raw):
            cleaned = part.strip().strip(".")
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
    return candidates[:5]


def _warnings(first_line: str, text: str, ok: bool) -> list[str]:
    lowered = f"{first_line}\n{text}".lower()
    warnings: list[str] = []
    if not ok:
        warnings.append("transport not ok")
    if not first_line:
        warnings.append("empty first line")
    if first_line.startswith("{") or first_line.startswith("["):
        warnings.append("raw dump")
    if "generic_chat" in lowered or "generic chat" in lowered:
        warnings.append("generic fallback route")
    if any(marker in lowered for marker in _DEAD_END_MARKERS):
        warnings.append("dead-end wording")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live discovery-quality smoke against the canonical runtime."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    parser.add_argument("--prompt", choices=[label for label, _, _ in PROMPTS], help="Run only one prompt.")
    args = parser.parse_args(argv)

    exit_code = 0
    for label, prompt, timeout_seconds in PROMPTS:
        if args.prompt and args.prompt != label:
            continue
        result = _post_chat(
            str(args.base_url),
            prompt,
            user_id=f"operator-smoke-{label}",
            thread_id=f"operator-smoke-{label}:thread",
            trace_id=f"operator-discovery-quality-{label}-{int(time.time())}",
            timeout=timeout_seconds,
        )
        first_line = str(result.get("first_line") or "")
        text = str(result.get("text") or "")
        route = str(result.get("route") or "unknown")
        warnings = _warnings(first_line, text, bool(result.get("ok")))
        candidates = _candidate_names(text)
        broadening = "yes" if _broadening_used(text) else "no"

        print(f"[{label}] route: {route}")
        print(f"[{label}] status: {_status_text(result)}")
        print(f"[{label}] first_line: {first_line}")
        print(f"[{label}] broadening: {broadening}")
        print(f"[{label}] top_candidates: {', '.join(candidates) if candidates else 'none'}")
        print(f"[{label}] dead_end_warnings: {', '.join(warnings) if warnings else 'none'}")

        if warnings:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
