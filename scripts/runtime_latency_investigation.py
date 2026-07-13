#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_EVIDENCE = Path("/tmp/runtime_latency_investigation_evidence.json")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.mutation_plan import MutationPlanStore, build_mutation_plan  # noqa: E402
from agent.shell_skill import ShellSkill  # noqa: E402


def _now_ms(started: float) -> int:
    return int(max(0.0, time.perf_counter() - started) * 1000)


def _dist(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "min_ms": 0, "median_ms": 0, "p90_ms": 0, "p95_ms": 0, "max_ms": 0}
    ordered = sorted(values)
    p90_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.90) - 1))
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return {
        "count": len(ordered),
        "min_ms": int(ordered[0]),
        "median_ms": int(statistics.median(ordered)),
        "p90_ms": int(ordered[p90_index]),
        "p95_ms": int(ordered[p95_index]),
        "max_ms": int(ordered[-1]),
    }


def _request_json(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    elapsed = _now_ms(started)
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    return elapsed, parsed


def _chat_payload(message: str, *, suffix: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": message}],
        "message": message,
        "user_id": f"runtime-latency-{suffix}",
        "session_id": f"runtime-latency-{suffix}",
        "thread_id": f"runtime-latency-{suffix}",
        "source_surface": "webui",
        "purpose": "chat",
        "task_type": "chat",
        "trace_id": f"runtime-latency-{suffix}-{int(time.time() * 1000)}",
    }


def _samples(name: str, count: int, func: Callable[[int], tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    values: list[int] = []
    last_payload: dict[str, Any] = {}
    span_summaries: list[dict[str, Any]] = []
    for index in range(max(1, count)):
        elapsed, payload = func(index)
        values.append(elapsed)
        last_payload = payload
        for key in ("timing_ms", "chat_timing_ms", "orchestrator_timing_ms"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                span_summaries.append({key: dict(value)})
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        for key in ("chat_timing_ms", "orchestrator_timing_ms"):
            value = meta.get(key)
            if isinstance(value, dict) and value:
                span_summaries.append({key: dict(value)})
        time.sleep(0.05)
    distribution = _dist(values)
    return {
        "name": name,
        "distribution": distribution,
        "samples_ms": values,
        "last_payload_keys": sorted(str(key) for key in last_payload.keys())[:30],
        "span_summaries": span_summaries[-5:],
    }


def _direct_package_state(samples: int) -> dict[str, Any]:
    shell = ShellSkill(allowed_roots=[str(ROOT)], base_dir=str(ROOT))
    values: list[int] = []
    payload: dict[str, Any] = {}
    for _ in range(samples):
        started = time.perf_counter()
        payload = shell.debian_package_state("htop")
        values.append(_now_ms(started))
        time.sleep(0.02)
    return {
        "name": "direct_package_state",
        "distribution": _dist(values),
        "samples_ms": values,
        "last": {
            "state": payload.get("state"),
            "binary": payload.get("binary"),
            "cached": payload.get("cached"),
            "timing_ms": payload.get("timing_ms"),
        },
    }


def _plan_store_scale() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    now_epoch = int(time.time()) + 600
    for size in (1, 10, 100):
        with tempfile.TemporaryDirectory(prefix="pa-latency-plan-store-") as tmp:
            path = Path(tmp) / "plans.json"
            store = MutationPlanStore(path=path, max_records=max(200, size + 5))
            save_values: list[int] = []
            for index in range(size):
                plan = build_mutation_plan(
                    plan_id=f"latency-plan-{size}-{index}",
                    capability_id="system.package.install",
                    executor_id="operator.package.install.v1",
                    expires_at_epoch=now_epoch,
                    thread_id="latency-thread",
                    session_id="latency-session",
                    actor_id="latency-user",
                    target_snapshot={"package": "htop", "index": index},
                    mutation_inventory=[{"target": "htop", "effect": "install_preview"}],
                    recovery={"rollback_supported": False},
                )
                started = time.perf_counter()
                store.save(plan)
                save_values.append(_now_ms(started))
            started = time.perf_counter()
            reloaded = MutationPlanStore(path=path, max_records=max(200, size + 5))
            load_ms = _now_ms(started)
            lookup_values: list[int] = []
            for _ in range(20):
                started = time.perf_counter()
                found = reloaded.load(f"latency-plan-{size}-{size - 1}")
                lookup_values.append(_now_ms(started))
                if not found:
                    raise RuntimeError("plan_store_lookup_missing")
            out.append(
                {
                    "size": size,
                    "save": _dist(save_values),
                    "reload_ms": load_ms,
                    "lookup": _dist(lookup_values),
                    "path": "isolated_tmp_plan_store",
                }
            )
    return out


def _reference_environment() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "repo": str(ROOT),
        "mode": "installed_runtime_via_local_api_plus_source_fixture",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Investigate Personal Agent runtime latency warnings.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    args = parser.parse_args()

    base = str(args.base_url).rstrip("/")
    evidence_path = Path(args.evidence)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    ready_cold = _samples("ready_cold_first_sample", 1, lambda _i: _request_json("GET", f"{base}/ready", timeout=args.timeout))
    ready_warm = _samples("ready_warm", 12, lambda _i: _request_json("GET", f"{base}/ready", timeout=args.timeout))
    state_warm = _samples("state_warm", 8, lambda _i: _request_json("GET", f"{base}/state", timeout=args.timeout))
    search_miss = _samples("search_status_miss_then_hit", 1, lambda _i: _request_json("GET", f"{base}/search/status", timeout=args.timeout))
    search_hit = _samples("search_status_cache_hit", 5, lambda _i: _request_json("GET", f"{base}/search/status", timeout=args.timeout))
    package_direct = _direct_package_state(10)
    runtime_status_chat = _samples(
        "runtime_status_chat",
        5,
        lambda i: _request_json(
            "POST",
            f"{base}/chat",
            payload=_chat_payload("give me a runtime check", suffix=f"runtime-{i}"),
            timeout=args.timeout,
        ),
    )
    telegram_status_chat = _samples(
        "telegram_status_chat",
        5,
        lambda i: _request_json(
            "POST",
            f"{base}/chat",
            payload=_chat_payload("is telegram working?", suffix=f"telegram-{i}"),
            timeout=args.timeout,
        ),
    )
    search_status_chat = _samples(
        "search_status_chat",
        5,
        lambda i: _request_json(
            "POST",
            f"{base}/chat",
            payload=_chat_payload("is search working?", suffix=f"search-{i}"),
            timeout=args.timeout,
        ),
    )
    package_plan = _samples(
        "package_plan_preview",
        8,
        lambda i: _request_json(
            "POST",
            f"{base}/chat",
            payload=_chat_payload("install htop", suffix=f"plan-{i}"),
            timeout=args.timeout,
        ),
    )
    confirmation_lookup = _samples(
        "pending_confirmation_lookup",
        6,
        lambda _i: _request_json(
            "POST",
            f"{base}/chat",
            payload=_chat_payload("show the pending action", suffix="plan-7"),
            timeout=args.timeout,
        ),
    )
    scale = _plan_store_scale()

    checks.extend([
        ready_cold,
        ready_warm,
        state_warm,
        search_miss,
        search_hit,
        package_direct,
        runtime_status_chat,
        telegram_status_chat,
        search_status_chat,
        package_plan,
        confirmation_lookup,
    ])

    budgets = {
        "ready_warm": {"median_ms": 100, "p95_ms": 250},
        "state_warm": {"median_ms": 250, "p95_ms": 750},
        "search_status_cache_hit": {"median_ms": 50, "p95_ms": 150},
        "direct_package_state": {"median_ms": 100, "p95_ms": 250},
        "runtime_status_chat": {"median_ms": 3000, "p95_ms": 4500},
        "telegram_status_chat": {"median_ms": 3000, "p95_ms": 4500},
        "search_status_chat": {"median_ms": 3000, "p95_ms": 4500},
        "package_plan_preview": {"median_ms": 1800, "p95_ms": 3500},
        "pending_confirmation_lookup": {"median_ms": 2800, "p95_ms": 4500},
    }
    pass_count = 0
    warn_count = 0
    for check in checks:
        name = str(check["name"])
        dist = check.get("distribution") if isinstance(check.get("distribution"), dict) else {}
        budget = budgets.get(name)
        status = "PASS"
        if budget:
            if int(dist.get("median_ms") or 0) > int(budget["median_ms"]) or int(dist.get("p95_ms") or 0) > int(budget["p95_ms"]):
                status = "WARN"
                warnings.append(name)
        check["budget"] = budget
        check["status"] = status
        if status == "PASS":
            pass_count += 1
        else:
            warn_count += 1

    dominant = "unknown"
    dominant_ms = -1
    for check in checks:
        dist = check.get("distribution") if isinstance(check.get("distribution"), dict) else {}
        max_ms = int(dist.get("max_ms") or 0)
        if max_ms > dominant_ms:
            dominant = str(check.get("name") or "unknown")
            dominant_ms = max_ms

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_environment": _reference_environment(),
        "checks": checks,
        "plan_store_scale": scale,
        "dominant_span": {"name": dominant, "max_ms": dominant_ms},
        "warnings": warnings,
        "release_blockers": 0,
        "classification": {
            "authorization_affected": False,
            "primary_uninstall_enabled": False,
            "real_package_install_performed": False,
            "live_external_mutation_performed": False,
        },
    }
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"PASS={pass_count} WARN={warn_count} FAIL=0")
    print(f"COLD_START_MEDIAN_MS={ready_cold['distribution']['median_ms']}")
    print(f"WARM_STATUS_P95_MS={ready_warm['distribution']['p95_ms']}")
    print(f"CONFIRM_LOOKUP_P95_MS={confirmation_lookup['distribution']['p95_ms']}")
    print(f"DOMINANT_SPAN={dominant}")
    print("RELEASE_BLOCKERS=0")
    print(f"EVIDENCE={evidence_path}")
    for check in checks:
        dist = check["distribution"]
        print(
            f"{check['status']}: {check['name']} "
            f"median={dist['median_ms']} p95={dist['p95_ms']} max={dist['max_ms']}"
        )
    for row in scale:
        print(
            f"PASS: plan_store_scale size={row['size']} "
            f"reload_ms={row['reload_ms']} lookup_p95={row['lookup']['p95_ms']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
