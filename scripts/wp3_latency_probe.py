from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _post(base_url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, (time.perf_counter() - started) * 1000.0


def _get(base_url: str, path: str, timeout: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, (time.perf_counter() - started) * 1000.0


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95)))
    return {
        "samples": len(values),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[index], 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def run(base_url: str, *, samples: int, label: str, include_task: bool, fetch_task_state: bool, timeout: float) -> dict[str, Any]:
    routes = {
        "presence_fast_path": "are you here?",
        "system_fast_path": "show current system status",
    }
    if include_task:
        routes["multi_capability_task"] = "check system health; then show installed local models"
    results: dict[str, Any] = {}
    session = f"wp3-latency-{label}-{int(time.time())}"
    for route_name, text in routes.items():
        client_values: list[float] = []
        server_values: list[float] = []
        understanding_values: list[float] = []
        planning_generations: list[int] = []
        observed_routes: list[str] = []
        for index in range(samples):
            body, elapsed = _post(base_url, {
                "messages": [{"role": "user", "content": text}],
                "session_id": session,
                "thread_id": f"{session}-{route_name}-{index}",
                "source_surface": "api",
            }, timeout)
            meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
            timing = meta.get("chat_timing_ms") if isinstance(meta.get("chat_timing_ms"), dict) else {}
            setup = body.get("setup") if isinstance(body.get("setup"), dict) else {}
            client_values.append(elapsed)
            server_values.append(float(timing.get("total_ms") or elapsed))
            understanding_values.append(float(timing.get("request_understanding_ms") or 0))
            planning_generations.append(int(setup.get("planning_generations") or 0))
            observed_routes.append(str(meta.get("route") or ""))
        results[route_name] = {
            "client_end_to_end": _summary(client_values),
            "server_end_to_end": _summary(server_values),
            "request_understanding": _summary(understanding_values),
            "planning_generations_max": max(planning_generations),
            "observed_routes": sorted(set(observed_routes)),
        }
    if fetch_task_state:
        fetch_values = []
        query = urllib.parse.urlencode({"session_id": session, "source_surface": "api", "limit": 20})
        for _ in range(samples):
            _body, elapsed = _get(base_url, f"/tasks?{query}", timeout)
            fetch_values.append(elapsed)
        results["task_state_fetch"] = {"client_end_to_end": _summary(fetch_values)}
    return {"schema_version": "wp3-latency.v1", "label": label, "base_url": base_url, "samples_per_route": samples, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeatable WP3 routing/task latency probe.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--include-task", action="store_true")
    parser.add_argument("--skip-task-fetch", action="store_true", help="Support pre-WP3 baselines without /tasks.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.base_url,
        samples=max(3, min(100, args.samples)),
        label=args.label,
        include_task=args.include_task,
        fetch_task_state=not args.skip_task_fetch,
        timeout=args.timeout,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
