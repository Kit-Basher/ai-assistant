#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("ASSISTANT_REAL_WORLD_SMOKE_TIMEOUT_SECONDS", "60"))

PROMPTS: tuple[tuple[str, str], ...] = (
    ("files", "list the files in this repo"),
    ("memory", "what do you remember about my preferences?"),
    ("memory_simple", "how much RAM am I using right now?"),
    ("hardware", "what do i have for ram and vram right now?"),
    ("runtime", "what is the runtime status?"),
    ("discovery", "there is a brand new tiny Gemma 4 model, can you look into it?"),
    ("skill_packs", "what skill packs can you use for extra abilities?"),
)


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


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

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
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
        "used_tools": [
            str(item).strip()
            for item in (meta.get("used_tools") if isinstance(meta.get("used_tools"), list) else [])
            if str(item).strip()
        ],
    }


def _warnings(label: str, first_line: str, text: str, ok: bool) -> list[str]:
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
            "i can't",
            "i cannot",
        )
        ):
        warnings.append("dead-end wording")
    if "(confidence " in lowered:
        warnings.append("confidence leaked")

    if label == "files" and not any(token in lowered for token in ("entries", "contains", "directory", "repo", "file", "/home")):
        warnings.append("missing file answer")
    if label == "memory" and not any(token in lowered for token in ("remember", "memory", "preferences")):
        warnings.append("missing memory answer")
    if label == "memory_simple":
        if not any(token in lowered for token in ("ram", "memory", "using", "available")):
            warnings.append("missing ram answer")
        if any(token in lowered for token in ("likely cause:", "top cpu processes", "pid=")):
            warnings.append("diagnostic detail leaked")
    if label == "hardware" and not any(token in lowered for token in ("ram", "vram")):
        warnings.append("missing ram/vram answer")
    if label == "runtime" and not any(token in lowered for token in ("runtime", "ready", "health")):
        warnings.append("missing runtime/readiness answer")
    if label == "discovery" and not any(token in lowered for token in ("model", "source", "discover")):
        warnings.append("missing discovery answer")
    if label == "skill_packs" and not any(token in lowered for token in ("skill pack", "pack", "abilities", "skills")):
        warnings.append("missing skill-pack answer")
    return warnings


def _filesystem_canary(base_url: str, *, timeout: float) -> int:
    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="fs-canary-", dir=str(Path(__file__).resolve().parents[1])) as tmpdir:
        canary_root = Path(tmpdir)
        nested_dir = canary_root / "nested" / "deeper"
        nested_dir.mkdir(parents=True, exist_ok=True)
        token = f"assistant-files-canary-{int(time.time())}"
        file_path = nested_dir / "note.txt"
        file_path.write_text(
            "\n".join(
                [
                    f"CANARY_TOKEN={token}",
                    "This file exists so the assistant has to read real local content.",
                    f"Path={file_path}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        symlink_path = canary_root / "alias-note.txt"
        try:
            os.symlink(file_path, symlink_path)
        except FileExistsError:
            pass
        prompts = (
            (
                "filesystem_list",
                f"list the files in {nested_dir}",
                "note.txt",
            ),
            (
                "filesystem_read",
                f"read the file {file_path}",
                token,
            ),
            (
                "filesystem_symlink",
                f"read the file {symlink_path}",
                token,
            ),
        )
        for label, prompt, expected in prompts:
            trace_id = f"assistant-real-world-{label}-{int(time.time())}"
            result = _post_chat(
                base_url,
                prompt,
                user_id=f"assistant-real-world-{label}",
                thread_id=f"assistant-real-world-{label}:thread",
                trace_id=trace_id,
                timeout=timeout,
            )
            first_line = str(result.get("first_line") or "")
            text = str(result.get("text") or "")
            route = str(result.get("route") or "unknown")
            used_tools = [str(item).strip().lower() for item in (result.get("used_tools") or []) if str(item).strip()]
            lowered = f"{first_line}\n{text}".lower()
            warnings = _warnings(label, first_line, text, bool(result.get("ok")))
            if expected.lower() not in lowered:
                warnings.append(f"missing expected canary text: {expected}")
            if "filesystem" not in used_tools:
                warnings.append("filesystem tool not used")

            print(f"[{label}] route: {route}")
            print(f"[{label}] status: {'ok' if bool(result.get('ok')) else result.get('status') or 'error'}")
            print(f"[{label}] first_line: {first_line}")
            print(f"[{label}] dead_end_warnings: {', '.join(warnings) if warnings else 'none'}")

            if warnings:
                exit_code = 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real-world assistant smoke against the live API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    parser.add_argument("--prompt", choices=[name for name, _ in PROMPTS], help="Run only one prompt.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout in seconds.")
    args = parser.parse_args(argv)

    exit_code = 0
    runs = [(name, prompt) for name, prompt in PROMPTS if not args.prompt or args.prompt == name]
    for label, prompt in runs:
        trace_id = f"assistant-real-world-{label}-{int(time.time())}"
        result = _post_chat(
            str(args.base_url),
            prompt,
            user_id=f"assistant-real-world-{label}",
            thread_id=f"assistant-real-world-{label}:thread",
            trace_id=trace_id,
            timeout=float(args.timeout),
        )
        first_line = str(result.get("first_line") or "")
        text = str(result.get("text") or "")
        route = str(result.get("route") or "unknown")
        warnings = _warnings(label, first_line, text, bool(result.get("ok")))

        print(f"[{label}] route: {route}")
        print(f"[{label}] status: {'ok' if bool(result.get('ok')) else result.get('status') or 'error'}")
        print(f"[{label}] first_line: {first_line}")
        print(f"[{label}] dead_end_warnings: {', '.join(warnings) if warnings else 'none'}")

        if warnings:
            exit_code = 1

    canary_exit_code = _filesystem_canary(str(args.base_url), timeout=float(args.timeout))
    if canary_exit_code != 0:
        exit_code = canary_exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
