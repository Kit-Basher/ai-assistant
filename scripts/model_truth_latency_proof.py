#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "agent" / "data" / "model_runtime_evidence.json"
REPORT_JSON = ROOT / "build" / "reports" / "model-truth-latency-proof.json"
REPORT_TEXT = ROOT / "build" / "reports" / "model-truth-latency-proof.txt"
REQUIRED_FILES = (
    "agent/llm/model_runtime_truth.py",
    "agent/data/model_runtime_evidence.json",
    "scripts/model_runtime_evaluation.py",
    "tests/test_model_truth_latency_wp45.py",
    "docs/design/MODEL_TRUTH_AND_LATENCY_WP4_5.md",
    "desktop/src/components/ModelTruthTab.jsx",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _fingerprint() -> dict[str, str]:
    head = _git("rev-parse", "HEAD")
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", ":(exclude)build/reports"],
        cwd=ROOT,
    )
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    digest = hashlib.sha256(diff)
    for relative in sorted(item for item in untracked if not item.startswith("build/reports/")):
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return {"head": head, "tracked_tree": _git("write-tree"), "candidate_diff_sha256": digest.hexdigest()}


def _live_tags() -> list[dict[str, Any]]:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3.0) as response:
        payload = json.loads(response.read(2 * 1024 * 1024).decode("utf-8"))
    return [dict(row) for row in payload.get("models", []) if isinstance(row, dict)]


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail[:500]}


def evaluate(*, sensitivity: str | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        evidence = {}
        checks.append(_check("evidence readable", False, exc.__class__.__name__))
    checks.append(_check("evaluation contract", evidence.get("contract") == "personal-agent.model-evaluation.v1", str(evidence.get("contract"))))
    checks.append(_check("required implementation files", all((ROOT / item).is_file() for item in REQUIRED_FILES), ",".join(REQUIRED_FILES)))

    try:
        tags = _live_tags()
        live_error = None
    except Exception as exc:
        tags = []
        live_error = exc.__class__.__name__
    observed = evidence.get("installed_observation") if isinstance(evidence.get("installed_observation"), list) else []
    live_by_digest = {str(row.get("digest") or "") for row in tags if str(row.get("digest") or "")}
    evidence_by_digest = {str(row.get("digest") or "") for row in observed if isinstance(row, dict) and str(row.get("digest") or "")}
    if sensitivity == "omit_installed" and evidence_by_digest:
        evidence_by_digest.pop()
    checks.append(_check("physical inventory complete", bool(tags) and live_by_digest == evidence_by_digest, f"live={len(live_by_digest)} evidence={len(evidence_by_digest)} error={live_error}"))
    checks.append(_check("canonical digest identities unique", len(live_by_digest) == len(tags), f"tags={len(tags)} unique={len(live_by_digest)}"))

    evaluated = evidence.get("evaluated_models") if isinstance(evidence.get("evaluated_models"), list) else []
    excluded = evidence.get("excluded_models") if isinstance(evidence.get("excluded_models"), list) else []
    accounted = {str(row.get("model") or "").lower() for row in [*evaluated, *excluded] if isinstance(row, dict)}
    live_names = {str(row.get("name") or row.get("model") or "").lower() for row in tags}
    checks.append(_check("every installed model dispositioned", live_names == accounted, f"live={len(live_names)} accounted={len(accounted)}"))
    checks.append(_check("exclusions objective", all(str(row.get("reason") or "") in {"embedding_only", "minimum_contract_timeout", "non_chat_format"} for row in excluded if isinstance(row, dict)), json.dumps(excluded, sort_keys=True)))
    checks.append(_check("evaluation cases complete", bool(evaluated) and all(int((row.get("score") or {}).get("total") or 0) == 9 for row in evaluated if isinstance(row, dict)), f"evaluated={len(evaluated)}"))

    recommendation = evidence.get("recommendation") if isinstance(evidence.get("recommendation"), dict) else {}
    recommended = str(recommendation.get("default_general_assistant") or "")
    evaluated_names = {str(row.get("model") or "") for row in evaluated if isinstance(row, dict)}
    if sensitivity == "recommendation_without_evidence":
        recommended = "not-evaluated:latest"
    checks.append(_check("recommendation has current comparable evidence", bool(recommended) and recommended in evaluated_names, recommended))
    limits = evidence.get("limits") if isinstance(evidence.get("limits"), dict) else {}
    checks.append(_check("benchmark mutation boundary", limits.get("default_mutated") is False and limits.get("downloads_or_deletions") is False, json.dumps(limits, sort_keys=True)))
    source = (ROOT / "agent" / "api_server.py").read_text(encoding="utf-8")
    frontdoor = source[source.index("    def assistant_frontdoor_active"):source.index("    def should_use_assistant_frontdoor")]
    if sensitivity == "deterministic_probe":
        frontdoor += "ready_status()"
    checks.append(_check("deterministic frontdoor has no readiness probe", "ready_status(" not in frontdoor and "model_inventory_status(" not in frontdoor, "static ownership check"))
    checks.append(_check("canonical API/UI mapping", 'path == "/llm/models/truth"' in source and '"/llm/models/truth"' in (ROOT / "desktop/src/App.jsx").read_text(encoding="utf-8"), "API and UI consume canonical truth"))
    latency_source = (ROOT / "scripts" / "runtime_latency_investigation.py").read_text(encoding="utf-8")
    checks.append(_check("latency blockers enforced", 'presence_chat_body_complete' in latency_source and 'return 1 if release_blockers else 0' in latency_source, "presence/status budgets release-blocking"))
    failures = [row for row in checks if row["status"] == "fail"]
    return {
        "contract": "personal-agent.model-truth-latency-proof.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": _fingerprint(),
        "checks": checks,
        "summary": {"passed": len(checks) - len(failures), "failed": len(failures), "total": len(checks)},
        "status": "pass" if not failures else "fail",
        "sensitivity": sensitivity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-tests", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    if args.execute_tests:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_model_truth_latency_wp45.py"], cwd=ROOT, check=False)
        if result.returncode:
            return int(result.returncode)
    if args.sensitivity:
        for mutation in ("omit_installed", "recommendation_without_evidence", "deterministic_probe"):
            mutated = evaluate(sensitivity=mutation)
            if mutated["status"] != "fail":
                print(f"sensitivity failed to detect {mutation}")
                return 1
            print(f"SENSITIVITY PASS: {mutation}")
    report = evaluate()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"{row['status'].upper()}: {row['name']} — {row['detail']}" for row in report["checks"]]
    lines.append(f"TOTAL: {report['summary']['passed']}/{report['summary']['total']} passed")
    REPORT_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
