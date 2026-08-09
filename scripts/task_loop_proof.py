from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_capability_proof import _build_registry


MANIFEST_PATH = ROOT / "config" / "task_loop_proof.json"
DEFAULT_JSON = ROOT / "build" / "reports" / "task-loop-proof.json"
DEFAULT_TEXT = ROOT / "build" / "reports" / "task-loop-proof.txt"
SECRET_MARKERS = ("bearer ", "api_key", "secret-token", "private.invalid", "chain-of-thought", "hidden reasoning")


def _candidate_fingerprint() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    staged = subprocess.run(["git", "diff", "--binary", "--cached"], cwd=ROOT, capture_output=True, check=True).stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    digest = hashlib.sha256(head.encode("utf-8") + diff + staged + "\n".join(sorted(untracked)).encode("utf-8")).hexdigest()
    return {"commit": head, "diff_fingerprint": digest, "clean": not bool(diff or staged or untracked)}


def _source_scenarios() -> set[str]:
    source = (ROOT / "tests" / "test_wp3_scenarios.py").read_text(encoding="utf-8")
    return set(__import__("re").findall(r'^\s+"([a-z_]+)",?$', source, flags=__import__("re").M))


def run_proof() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    registry, cleanup = _build_registry()
    failures: list[dict[str, str]] = []
    try:
        definitions = {item.capability_id: item for item in registry.definitions()}
        expected = set(manifest["expected_task_composable_capabilities"])
        actual = {item.capability_id for item in definitions.values() if item.task_composable}
        for capability_id in sorted(expected - actual):
            failures.append({"category": "registry", "item": capability_id, "reason": "task_composable_capability_missing"})
        for capability_id in sorted(actual - expected):
            failures.append({"category": "registry", "item": capability_id, "reason": "unproved_task_composable_capability"})
        capability_rows = []
        for capability_id in sorted(expected & actual):
            definition = definitions[capability_id]
            required = {"task_schema", "task_dispatch", "task_verification", "task_restart"}
            if definition.mode.value == "mutating":
                required |= {"task_approval", "task_cancellation", "task_indeterminate"}
            missing = sorted(required - set(definition.proof_nodes))
            if missing:
                failures.append({"category": "capability", "item": capability_id, "reason": "missing_orchestration_proof:" + ",".join(missing)})
            if definition.mode.value == "mutating" and (definition.retry_safety != "reconcile_first" or definition.resume_policy != "reconcile_first"):
                failures.append({"category": "policy", "item": capability_id, "reason": "mutation_reconcile_policy_downgraded"})
            if definition.mode.value == "mutating" and definition.approval_policy.value != "required":
                failures.append({"category": "policy", "item": capability_id, "reason": "mutation_approval_downgraded"})
            if definition.mode.value == "mutating" and definition.independent_verification_hook is None:
                failures.append({"category": "verification", "item": capability_id, "reason": "mutation_independent_verifier_missing"})
            capability_rows.append({
                "id": capability_id, "mode": definition.mode.value, "approval": definition.approval_policy.value,
                "retry_safety": definition.retry_safety, "resume_policy": definition.resume_policy,
                "proof_categories": sorted(required), "status": "fail" if missing else "pass",
            })
        declared_categories = set(manifest.get("proof_nodes", {}))
        required_categories = set(manifest["required_proof_categories"])
        for category in sorted(required_categories - declared_categories):
            failures.append({"category": "proof", "item": category, "reason": "required_proof_category_missing"})
        for category, nodes in manifest.get("proof_nodes", {}).items():
            for node in nodes:
                if not (ROOT / str(node)).exists():
                    failures.append({"category": "proof", "item": str(node), "reason": f"proof_node_missing:{category}"})
        scenarios = _source_scenarios()
        for category in sorted(set(manifest["required_scenarios"]) - scenarios):
            failures.append({"category": "scenario", "item": category, "reason": "required_scenario_missing"})
        candidate = _candidate_fingerprint()
        return {
            "schema_version": "task-loop-proof-report.v1",
            "candidate": candidate,
            "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            "ok": not failures,
            "totals": {
                "expected_task_composable": len(expected), "registered_task_composable": len(actual),
                "capabilities_passed": sum(1 for row in capability_rows if row["status"] == "pass"),
                "proof_categories": len(required_categories), "scenario_categories": len(manifest["required_scenarios"]),
                "failures": len(failures),
            },
            "capabilities": capability_rows,
            "proof_categories": {key: {"nodes": value, "status": "declared"} for key, value in sorted(manifest["proof_nodes"].items())},
            "scenarios": {key: "declared" for key in manifest["required_scenarios"]},
            "ceilings": manifest["ceilings"],
            "redaction": {"passed": True, "markers_checked": len(SECRET_MARKERS)},
            "failures": failures,
        }
    finally:
        _db, temp = cleanup
        try:
            _db.close()
        finally:
            temp.cleanup()


def execute_tests(report: dict[str, Any]) -> None:
    nodes = sorted({str(node) for item in report["proof_categories"].values() for node in item["nodes"] if str(node).startswith(("tests/", "desktop/tests/"))})
    python_nodes = [node for node in nodes if node.startswith("tests/")]
    js_nodes = [node for node in nodes if node.startswith("desktop/tests/")]
    executions = []
    if python_nodes:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *python_nodes], cwd=ROOT, text=True, capture_output=True, check=False)
        executions.append({"kind": "python", "nodes": python_nodes, "returncode": proc.returncode, "summary": (proc.stdout + proc.stderr)[-2000:]})
    if js_nodes:
        proc = subprocess.run(["node", "--test", *js_nodes], cwd=ROOT, text=True, capture_output=True, check=False)
        executions.append({"kind": "javascript", "nodes": js_nodes, "returncode": proc.returncode, "summary": (proc.stdout + proc.stderr)[-2000:]})
    failed = [row for row in executions if int(row["returncode"]) != 0]
    report["proof_execution"] = {"status": "fail" if failed else "pass", "executions": executions}
    if failed:
        report["ok"] = False
        report["failures"].append({"category": "proof_execution", "item": failed[0]["kind"], "reason": "declared_proof_failed"})
    else:
        for item in report["proof_categories"].values():
            item["status"] = "pass"
        for key in report["scenarios"]:
            report["scenarios"][key] = "pass"


def _write(report: dict[str, Any], json_path: Path, text_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    lowered = serialized.lower()
    leaked = [marker for marker in SECRET_MARKERS if marker in lowered]
    if leaked:
        report["ok"] = False
        report["redaction"] = {"passed": False, "leaked_markers": leaked}
        serialized = json.dumps(report, indent=2, sort_keys=True)
    json_path.write_text(serialized + "\n", encoding="utf-8")
    totals = report["totals"]
    text_path.write_text(
        "Task loop proof\n"
        f"status: {'PASS' if report['ok'] else 'FAIL'}\n"
        f"candidate: {report['candidate']['commit']} clean={report['candidate']['clean']}\n"
        f"task-composable: {totals['capabilities_passed']}/{totals['expected_task_composable']}\n"
        f"proof categories: {totals['proof_categories']}\n"
        f"scenario categories: {totals['scenario_categories']}\n"
        f"failures: {len(report['failures'])}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and execute the protected WP3 task-loop proof.")
    parser.add_argument("--execute-tests", action="store_true")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    args = parser.parse_args()
    report = run_proof()
    if args.execute_tests and report["ok"]:
        execute_tests(report)
    _write(report, args.json, args.text)
    print(json.dumps({"ok": report["ok"], "totals": report["totals"], "candidate": report["candidate"], "report": str(args.json)}, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
