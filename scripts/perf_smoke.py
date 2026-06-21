#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8765"


@dataclass
class Measurement:
    name: str
    command: str
    status: str
    elapsed_ms: int
    budget_ms: int
    detail: str


def _json_request(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError("non_object_json")
    return parsed


def _time_call(name: str, command: str, budget_ms: int, func) -> Measurement:  # type: ignore[no-untyped-def]
    started = time.monotonic()
    try:
        detail = func()
        status = "PASS"
    except Exception as exc:  # noqa: BLE001 - smoke reports class only.
        detail = f"{exc.__class__.__name__}: {exc}"
        status = "FAIL"
    elapsed_ms = int(max(0.0, time.monotonic() - started) * 1000)
    if status == "PASS" and elapsed_ms > budget_ms:
        status = "WARN"
    return Measurement(
        name=name,
        command=command,
        status=status,
        elapsed_ms=elapsed_ms,
        budget_ms=budget_ms,
        detail=str(detail or "ok")[:500],
    )


def _chat_payload(prompt: str, *, suffix: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "user_id": f"perf-smoke-{suffix}",
        "thread_id": f"perf-smoke-thread-{suffix}",
        "source_surface": "api",
        "purpose": "chat",
        "task_type": "chat",
        "trace_id": f"perf-smoke-{suffix}-{int(time.time())}",
    }


def _assistant_meta(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return dict(meta)


def _chat_detail(payload: dict[str, Any]) -> str:
    meta = _assistant_meta(payload)
    timings = meta.get("chat_timing_ms") if isinstance(meta.get("chat_timing_ms"), dict) else {}
    orchestrator_timings = (
        meta.get("orchestrator_timing_ms")
        if isinstance(meta.get("orchestrator_timing_ms"), dict)
        else {}
    )
    return (
        f"route={meta.get('route')} used_llm={meta.get('used_llm')} "
        f"used_tools={meta.get('used_tools')} timing_ms={timings} "
        f"orchestrator_timing_ms={orchestrator_timings}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Personal Agent latency smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    base_url = str(args.base_url).rstrip("/")

    checks: list[Measurement] = []
    for path, budget in (
        ("/ready", 750),
        ("/state", 1000),
        ("/search/status", 1200),
        ("/packs/state", 1500),
    ):
        checks.append(
            _time_call(
                f"GET {path}",
                f"GET {path}",
                budget,
                lambda path=path: f"ok={_json_request('GET', base_url + path, timeout=args.timeout).get('ok', True)}",
            )
        )

    chat_cases = (
        ("runtime status chat", "give me a runtime check", 1500),
        ("telegram status chat", "is telegram working?", 1500),
        ("search status chat", "is search working?", 1800),
        ("search-disabled/setup chat", "what is dots.tts?", 2500),
        ("install preview chat", "Can you install htop on this machine?", 2000),
    )
    for index, (name, prompt, budget) in enumerate(chat_cases, start=1):
        command = f"POST /chat {json.dumps({'message': prompt}, ensure_ascii=True)}"
        checks.append(
            _time_call(
                name,
                command,
                budget,
                lambda prompt=prompt, index=index: _chat_detail(
                    _json_request(
                        "POST",
                        base_url + "/chat",
                        payload=_chat_payload(prompt, suffix=str(index)),
                        timeout=args.timeout,
                    )
                ),
            )
        )

    def _chat_eval() -> str:
        started = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "scripts/chat_eval.py"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        elapsed = int(max(0.0, time.monotonic() - started) * 1000)
        if proc.returncode != 0:
            raise RuntimeError((proc.stdout or "").splitlines()[-1:] or ["chat_eval failed"])
        return f"chat_eval_ms={elapsed}"

    checks.append(_time_call("chat_eval runtime", "python scripts/chat_eval.py", 10000, _chat_eval))

    pass_count = sum(1 for row in checks if row.status == "PASS")
    warn_count = sum(1 for row in checks if row.status == "WARN")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    print("# Personal Agent Performance Smoke")
    for row in checks:
        print(f"## {row.name}: {row.status}")
        print(f"- command/API path: {row.command}")
        print(f"- elapsed_ms: {row.elapsed_ms} budget_ms: {row.budget_ms}")
        print(f"- evidence: {row.detail}")
    print("## Summary")
    print(f"PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
