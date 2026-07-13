#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_PATH = ROOT / "docs" / "operator" / "RUNTIME_LATENCY_ACCEPTANCE_V1.json"
EVIDENCE_PATH = Path("/tmp/runtime_latency_investigation_evidence.json")


@dataclass
class Check:
    name: str
    status: str
    evidence: str


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"non_object_json:{path}")
    return parsed


def _check(name: str, condition: bool, evidence: str, *, warn: bool = False) -> Check:
    if condition:
        return Check(name, "PASS", evidence)
    return Check(name, "WARN" if warn else "FAIL", evidence)


def main() -> int:
    checks: list[Check] = []
    acceptance: dict[str, Any] = {}
    evidence: dict[str, Any] = {}

    checks.append(_check("latency acceptance record exists", ACCEPTANCE_PATH.exists(), str(ACCEPTANCE_PATH)))
    if ACCEPTANCE_PATH.exists():
        acceptance = _load_json(ACCEPTANCE_PATH)
    checks.append(_check("latency evidence exists", EVIDENCE_PATH.exists(), str(EVIDENCE_PATH)))
    if EVIDENCE_PATH.exists():
        evidence = _load_json(EVIDENCE_PATH)

    accepted = acceptance.get("accepted_warnings") if isinstance(acceptance.get("accepted_warnings"), list) else []
    warning_ids = {str(row.get("warning_id") or "") for row in accepted if isinstance(row, dict)}
    checks.append(_check("accepted warnings have formal ids", {"cold_ready_observability", "search_status_miss", "package_plan_chat_roundtrip_noise"}.issubset(warning_ids), json.dumps(sorted(warning_ids))))
    checks.append(_check("release decision is explicit", str(acceptance.get("release_decision") or "").startswith("accepted_for_v0.2.2"), str(acceptance.get("release_decision"))))

    env = evidence.get("reference_environment") if isinstance(evidence.get("reference_environment"), dict) else {}
    checks.append(_check("reference environment recorded", bool(env.get("os")) and bool(env.get("python")), json.dumps(env, sort_keys=True)[:500]))

    checks_payload = evidence.get("checks") if isinstance(evidence.get("checks"), list) else []
    by_name = {str(row.get("name") or ""): row for row in checks_payload if isinstance(row, dict)}
    required = {
        "ready_cold_first_sample",
        "ready_warm",
        "state_warm",
        "search_status_miss_then_hit",
        "search_status_cache_hit",
        "direct_package_state",
        "runtime_status_chat",
        "telegram_status_chat",
        "search_status_chat",
        "package_plan_preview",
        "pending_confirmation_lookup",
    }
    checks.append(_check("cold and warm paths separated", required.issubset(set(by_name)), json.dumps(sorted(by_name.keys()))[:1000]))

    def dist(name: str) -> dict[str, Any]:
        row = by_name.get(name) if isinstance(by_name.get(name), dict) else {}
        return row.get("distribution") if isinstance(row.get("distribution"), dict) else {}

    def metric_ms(payload: dict[str, Any], key: str, *, missing: int = 999999) -> int:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return int(value)
        return missing

    checks.append(_check("multiple samples used", int(dist("ready_warm").get("count") or 0) >= 10 and int(dist("package_plan_preview").get("count") or 0) >= 6, json.dumps({"ready": dist("ready_warm"), "package_plan": dist("package_plan_preview")}, sort_keys=True)[:1000]))
    checks.append(_check("warm readiness within budget", metric_ms(dist("ready_warm"), "p95_ms") <= 250, json.dumps(dist("ready_warm"), sort_keys=True)))
    checks.append(_check("direct package state within budget", metric_ms(dist("direct_package_state"), "p95_ms") <= 250, json.dumps(dist("direct_package_state"), sort_keys=True)))
    checks.append(_check("runtime status chat within accepted ceiling", metric_ms(dist("runtime_status_chat"), "p95_ms") <= 4500, json.dumps(dist("runtime_status_chat"), sort_keys=True), warn=True))
    checks.append(_check("telegram status chat within accepted ceiling", metric_ms(dist("telegram_status_chat"), "p95_ms") <= 4500, json.dumps(dist("telegram_status_chat"), sort_keys=True), warn=True))
    checks.append(_check("search status chat within accepted ceiling", metric_ms(dist("search_status_chat"), "p95_ms") <= 4500, json.dumps(dist("search_status_chat"), sort_keys=True), warn=True))
    checks.append(_check("package Plan preview within accepted ceiling", metric_ms(dist("package_plan_preview"), "p95_ms") <= 3500, json.dumps(dist("package_plan_preview"), sort_keys=True), warn=True))
    checks.append(_check("confirmation lookup within accepted ceiling", metric_ms(dist("pending_confirmation_lookup"), "p95_ms") <= 4500, json.dumps(dist("pending_confirmation_lookup"), sort_keys=True), warn=True))

    scale = evidence.get("plan_store_scale") if isinstance(evidence.get("plan_store_scale"), list) else []
    scale_sizes = {int(row.get("size") or 0) for row in scale if isinstance(row, dict)}
    scale_ok = {1, 10, 100}.issubset(scale_sizes)
    lookup_ok = all(metric_ms(row.get("lookup") if isinstance(row.get("lookup"), dict) else {}, "p95_ms") <= 5 for row in scale if isinstance(row, dict))
    checks.append(_check("confirmation lookup scale tested", scale_ok and lookup_ok, json.dumps(scale, sort_keys=True)[:1200]))

    dominant = evidence.get("dominant_span") if isinstance(evidence.get("dominant_span"), dict) else {}
    checks.append(_check("dominant span identified", bool(dominant.get("name")), json.dumps(dominant, sort_keys=True)))
    checks.append(_check("authorization invariants unaffected", evidence.get("classification", {}).get("authorization_affected") is False, json.dumps(evidence.get("classification", {}), sort_keys=True)))
    checks.append(_check("no live mutation performed", not bool(evidence.get("classification", {}).get("real_package_install_performed")) and not bool(evidence.get("classification", {}).get("live_external_mutation_performed")), json.dumps(evidence.get("classification", {}), sort_keys=True)))

    pass_count = sum(1 for row in checks if row.status == "PASS")
    warn_count = sum(1 for row in checks if row.status == "WARN")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    for row in checks:
        print(f"{row.status}: {row.name}: {row.evidence}")
    print(f"PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
