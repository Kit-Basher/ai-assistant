#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
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
    samples_ms: list[int]
    p95_ms: int


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


def _dist(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min_ms": 0, "median_ms": 0, "p90_ms": 0, "p95_ms": 0, "max_ms": 0}
    ordered = sorted(values)
    p90_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.9) - 1))
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return {
        "min_ms": int(ordered[0]),
        "median_ms": int(statistics.median(ordered)),
        "p90_ms": int(ordered[p90_index]),
        "p95_ms": int(ordered[p95_index]),
        "max_ms": int(ordered[-1]),
    }


def _time_call(name: str, command: str, budget_ms: int, func, *, samples: int = 3) -> Measurement:  # type: ignore[no-untyped-def]
    samples_ms: list[int] = []
    details: list[str] = []
    started = time.monotonic()
    status = "PASS"
    for _ in range(max(1, int(samples))):
        sample_started = time.monotonic()
        try:
            detail = func()
            details.append(str(detail or "ok")[:500])
        except Exception as exc:  # noqa: BLE001 - smoke reports class only.
            details.append(f"{exc.__class__.__name__}: {exc}")
            status = "FAIL"
        samples_ms.append(int(max(0.0, time.monotonic() - sample_started) * 1000))
        if status == "FAIL":
            break
        time.sleep(0.05)
    distribution = _dist(samples_ms)
    elapsed_ms = distribution["median_ms"] if samples_ms else int(max(0.0, time.monotonic() - started) * 1000)
    p95_ms = distribution["p95_ms"]
    if status == "PASS" and (elapsed_ms > budget_ms or p95_ms > budget_ms * 2):
        status = "WARN"
    detail_payload = {
        "distribution": distribution,
        "samples_ms": samples_ms,
        "last": details[-1] if details else "ok",
        "mode": "warm_distribution" if len(samples_ms) > 1 else "single_sample",
    }
    return Measurement(
        name=name,
        command=command,
        status=status,
        elapsed_ms=elapsed_ms,
        budget_ms=budget_ms,
        detail=json.dumps(detail_payload, sort_keys=True)[:900],
        samples_ms=samples_ms,
        p95_ms=p95_ms,
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
        ("/ready", 1000),
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
                samples=5,
            )
        )

    chat_cases = (
        ("runtime status chat", "give me a runtime check", 1500),
        ("telegram status chat", "is telegram working?", 1500),
        ("search status chat", "is search working?", 1800),
        ("search-disabled/setup chat", "what is dots.tts?", 2500),
        ("install preview chat", "Can you install htop on this machine?", 2500),
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
                samples=3,
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

    checks.append(_time_call("chat_eval runtime", "python scripts/chat_eval.py", 10000, _chat_eval, samples=1))

    pass_count = sum(1 for row in checks if row.status == "PASS")
    warn_count = sum(1 for row in checks if row.status == "WARN")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    print("# Personal Agent Performance Smoke")
    for row in checks:
        print(f"## {row.name}: {row.status}")
        print(f"- command/API path: {row.command}")
        print(f"- median_ms: {row.elapsed_ms} p95_ms: {row.p95_ms} budget_ms: {row.budget_ms}")
        print(f"- evidence: {row.detail}")
    print("## Summary")
    print(f"PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
