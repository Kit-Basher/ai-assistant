#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8765"


@dataclass
class Check:
    name: str
    status: str
    evidence: str


def _request_json(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    elapsed_ms = int(max(0.0, time.monotonic() - started) * 1000)
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    return elapsed_ms, parsed


def _dist(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min_ms": 0, "median_ms": 0, "p90_ms": 0, "max_ms": 0}
    ordered = sorted(values)
    p90_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.9) - 1))
    return {
        "min_ms": int(min(ordered)),
        "median_ms": int(statistics.median(ordered)),
        "p90_ms": int(ordered[p90_index]),
        "max_ms": int(max(ordered)),
    }


def _chat_payload(message: str, *, suffix: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": message}],
        "message": message,
        "user_id": f"rc1-latency-{suffix}",
        "session_id": f"rc1-latency-{suffix}",
        "thread_id": f"rc1-latency-{suffix}",
        "source_surface": "webui",
        "purpose": "chat",
        "task_type": "chat",
        "trace_id": f"rc1-latency-{suffix}-{int(time.time() * 1000)}",
    }


def _package_state(package: str) -> tuple[int, dict[str, Any]]:
    code = (
        "from agent.shell_skill import ShellSkill; "
        "s=ShellSkill(allowed_roots=['.'], base_dir='.'); "
        f"import json; print(json.dumps(s.debian_package_state({package!r}), sort_keys=True))"
    )
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=5,
        check=False,
    )
    elapsed_ms = int(max(0.0, time.monotonic() - started) * 1000)
    if proc.returncode != 0:
        return elapsed_ms, {"ok": False, "error": (proc.stdout or "").strip()[:500]}
    return elapsed_ms, json.loads((proc.stdout or "{}").strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="RC1 latency closure smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    base = str(args.base_url).rstrip("/")
    checks: list[Check] = []

    ready_cold_ms, ready_cold = _request_json("GET", f"{base}/ready", timeout=args.timeout)
    ready_warm: list[int] = []
    ready_timings: list[dict[str, Any]] = []
    for _ in range(10):
        elapsed, payload = _request_json("GET", f"{base}/ready", timeout=args.timeout)
        ready_warm.append(elapsed)
        ready_timings.append(payload.get("timing_ms") if isinstance(payload.get("timing_ms"), dict) else {})
        time.sleep(0.1)
    ready_stats = _dist(ready_warm)
    checks.append(
        Check(
            "/ready warm budget",
            "PASS" if ready_stats["median_ms"] < 250 and ready_stats["p90_ms"] < 500 else "WARN",
            json.dumps({"cold_ms": ready_cold_ms, "warm": ready_stats, "cold_timing": ready_cold.get("timing_ms")}, sort_keys=True),
        )
    )

    state_samples = [_request_json("GET", f"{base}/state", timeout=args.timeout)[0] for _ in range(10)]
    checks.append(Check("/state distribution", "PASS", json.dumps(_dist(state_samples), sort_keys=True)))

    search_miss_ms, search_miss = _request_json("GET", f"{base}/search/status", timeout=args.timeout)
    search_hit_ms, search_hit = _request_json("GET", f"{base}/search/status", timeout=args.timeout)
    checks.append(
        Check(
            "/search/status cache",
            "PASS" if bool(search_hit.get("cached")) and search_hit_ms <= search_miss_ms + 20 else "WARN",
            json.dumps({"miss_ms": search_miss_ms, "hit_ms": search_hit_ms, "hit_cached": search_hit.get("cached"), "miss_timing": search_miss.get("timing_ms")}, sort_keys=True),
        )
    )

    package_samples: list[int] = []
    package_payloads: list[dict[str, Any]] = []
    for _ in range(10):
        elapsed, payload = _package_state("htop")
        package_samples.append(elapsed)
        package_payloads.append(payload)
    package_stats = _dist(package_samples)
    checks.append(
        Check(
            "htop package-state lookup",
            "PASS" if package_stats["median_ms"] < 250 and str(package_payloads[-1].get("binary")) == "/usr/bin/dpkg-query" else "WARN",
            json.dumps({"distribution": package_stats, "last": package_payloads[-1]}, sort_keys=True)[:1200],
        )
    )

    htop_plan_samples: list[int] = []
    htop_plan_payloads: list[dict[str, Any]] = []
    for i in range(10):
        elapsed, payload = _request_json(
            "POST",
            f"{base}/chat",
            payload=_chat_payload("install htop", suffix=f"plan-{i}"),
            timeout=args.timeout,
        )
        htop_plan_samples.append(elapsed)
        htop_plan_payloads.append(payload)
    htop_plan_stats = _dist(htop_plan_samples)
    last_plan = htop_plan_payloads[-1]
    text = json.dumps(last_plan, sort_keys=True).lower()
    no_mutation = "installed htop" not in text and "installing htop" not in text
    plan_server_timings = [
        payload.get("orchestrator_timing_ms")
        for payload in htop_plan_payloads
        if isinstance(payload.get("orchestrator_timing_ms"), dict)
    ]
    checks.append(
        Check(
            "htop plan rendering",
            "PASS" if htop_plan_stats["median_ms"] < 1500 and htop_plan_stats["p90_ms"] < 2500 and no_mutation else "WARN",
            json.dumps(
                {
                    "distribution": htop_plan_stats,
                    "samples_ms": htop_plan_samples,
                    "server_timing_ms": plan_server_timings[-3:],
                    "no_mutation": no_mutation,
                },
                sort_keys=True,
            )[:2000],
        )
    )

    confirm_samples: list[int] = []
    confirm_payload: dict[str, Any] = {}
    for _ in range(5):
        confirm_elapsed, confirm_payload = _request_json(
            "POST",
            f"{base}/chat",
            payload=_chat_payload("show the pending action", suffix="plan-9"),
            timeout=args.timeout,
        )
        confirm_samples.append(confirm_elapsed)
    confirm_stats = _dist(confirm_samples)
    checks.append(
        Check(
            "deterministic confirmation lookup",
            "PASS" if confirm_stats["median_ms"] < 2500 and confirm_stats["p90_ms"] < 4000 and "package.install" in json.dumps(confirm_payload).lower() else "WARN",
            json.dumps({"distribution": confirm_stats, "has_package_install": "package.install" in json.dumps(confirm_payload).lower()}, sort_keys=True),
        )
    )

    pass_count = sum(1 for row in checks if row.status == "PASS")
    warn_count = sum(1 for row in checks if row.status == "WARN")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    for row in checks:
        print(f"## {row.name}: {row.status}")
        print(f"- evidence: {row.evidence}")
    print(
        f"PASS={pass_count} WARN={warn_count} FAIL={fail_count} "
        f"READY_MEDIAN_MS={ready_stats['median_ms']} READY_P90_MS={ready_stats['p90_ms']} "
        f"READY_MAX_MS={ready_stats['max_ms']} HTOP_PLAN_MEDIAN_MS={htop_plan_stats['median_ms']} "
        f"HTOP_PLAN_P90_MS={htop_plan_stats['p90_ms']}"
    )
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
