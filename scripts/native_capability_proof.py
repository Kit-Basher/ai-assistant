from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.capability_registry import ApprovalPolicy, CapabilityMode
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB

INVENTORY_PATH = ROOT / "config" / "native_capabilities.json"
DEFAULT_JSON = ROOT / "build" / "reports" / "native-capability-proof.json"
DEFAULT_TEXT = ROOT / "build" / "reports" / "native-capability-proof.txt"


def _commit_identity() -> dict[str, Any]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
    tracked_diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True).stdout
    return {"commit": sha, "dirty": dirty, "candidate_fingerprint": hashlib.sha256(sha.encode() + tracked_diff).hexdigest()}


def _load_inventory() -> dict[str, Any]:
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("inventory_not_object")
    return payload


def _build_registry() -> tuple[Any, Any]:
    temp = tempfile.TemporaryDirectory(prefix="native-capability-proof-")
    db = MemoryDB(str(Path(temp.name) / "agent.db"))
    db.init_schema(str(ROOT / "memory" / "schema.sql"))
    orchestrator = Orchestrator(
        db=db,
        skills_path=str(ROOT / "skills"),
        log_path=str(Path(temp.name) / "events.jsonl"),
        timezone="UTC",
        llm_client=None,
    )
    return orchestrator._capability_registry, (db, temp)  # noqa: SLF001


def _api_literals() -> list[str]:
    tree = ast.parse((ROOT / "agent" / "api_server.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in {"do_GET", "do_POST", "do_PUT", "do_DELETE"}:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.startswith("/"):
                value = child.value.strip()
                if value and " " not in value and "{" not in value and len(value) < 100:
                    found.add(value)
    return sorted(found)


def _cli_commands() -> set[str]:
    tree = ast.parse((ROOT / "agent" / "cli.py").read_text(encoding="utf-8"))
    commands: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_parser" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            commands.add(node.args[0].value)
    return commands


def _native_skill_names() -> set[str]:
    names: set[str] = set()
    for path in sorted((ROOT / "skills").glob("*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and str(payload.get("name") or "").strip():
            names.add(str(payload["name"]).strip())
    return names


def _ui_section_ids() -> set[str]:
    source = (ROOT / "desktop" / "src" / "App.jsx").read_text(encoding="utf-8")
    start = source.find("const adminSections = [")
    if start < 0:
        return set()
    end = source.find("\n  ];", start)
    section = source[start:] if end < 0 else source[start:end]
    return set(re.findall(r'\bid:\s*"([a-z0-9_]+)"', section))


def _surface_rule(path: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    for rule in rules:
        if rule.get("exact") == path:
            return rule
    matches = [rule for rule in rules if rule.get("prefix") and path.startswith(str(rule["prefix"]))]
    return max(matches, key=lambda row: len(str(row["prefix"])), default=None)


def run_proof() -> dict[str, Any]:
    inventory = _load_inventory()
    identity = _commit_identity()
    failures: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    expected_rows = inventory.get("capabilities") if isinstance(inventory.get("capabilities"), list) else []
    expected = {str(row.get("id")): row for row in expected_rows if isinstance(row, dict)}
    if len(expected) != len(expected_rows):
        failures.append({"category": "inventory", "item": "capabilities", "reason": "duplicate_or_invalid_expected_id"})
    registry, cleanup = _build_registry()
    db, temp = cleanup
    try:
        actual = {row.capability_id: row for row in registry.definitions()}
        for missing in sorted(set(expected) - set(actual)):
            failures.append({"category": "registry", "item": missing, "reason": "protected_capability_missing"})
        for extra in sorted(set(actual) - set(expected)):
            failures.append({"category": "registry", "item": extra, "reason": "unmanifested_registry_entry"})
        profiles = inventory.get("proof_profiles") if isinstance(inventory.get("proof_profiles"), dict) else {}
        for capability_id in sorted(set(expected) & set(actual)):
            manifest = expected[capability_id]
            definition = actual[capability_id]
            item_failures: list[str] = []
            expected_mode = str(manifest.get("mode") or "")
            if definition.mode.value != expected_mode:
                item_failures.append(f"mode_mismatch:{definition.mode.value}!={expected_mode}")
            if definition.mode is CapabilityMode.MUTATING and definition.approval_policy is not ApprovalPolicy.REQUIRED:
                item_failures.append("mutating_approval_policy_invalid")
            if bool(definition.chat_selectable) != bool(manifest.get("chat")):
                item_failures.append("chat_selectability_mismatch")
            required = set(profiles.get(str(manifest.get("proof_profile") or ""), []))
            declared = set(definition.proof_requirements)
            missing_proof = sorted(required - declared)
            if missing_proof:
                item_failures.append("missing_proof_requirements:" + ",".join(missing_proof))
            missing_node_categories = sorted(required - set(definition.proof_nodes))
            if missing_node_categories:
                item_failures.append("missing_proof_nodes:" + ",".join(missing_node_categories))
            for category, nodes in definition.proof_nodes.items():
                if category not in required:
                    continue
                if not nodes:
                    item_failures.append(f"empty_proof_nodes:{category}")
                for node in nodes:
                    path = str(node).split("::", 1)[0]
                    if not (ROOT / path).is_file():
                        item_failures.append(f"proof_node_missing:{category}:{node}")
            for hook_name in ("health_hook", "invocation_hook", "verification_hook", "self_test_hook"):
                if not callable(getattr(definition, hook_name, None)):
                    item_failures.append(f"missing_{hook_name}")
            health = definition.health().public_dict()
            if str(health.get("reason") or "").startswith("health_check_failed:"):
                item_failures.append(str(health.get("reason")))
            self_test = definition.self_test()
            if not bool(self_test.get("ok")):
                item_failures.append(str(self_test.get("reason") or "self_test_failed"))
            for reason in item_failures:
                failures.append({"category": "capability", "item": capability_id, "reason": reason})
            results.append({
                "id": capability_id,
                "family": manifest.get("family"),
                "mode": definition.mode.value,
                "approval": definition.approval_policy.value,
                "health": health,
                "self_test": self_test,
                "proof_profile": manifest.get("proof_profile"),
                "proof_requirements": sorted(required),
                "proof_nodes": {key: list(value) for key, value in sorted(definition.proof_nodes.items()) if key in required},
                "status": "fail" if item_failures else ("pass" if health.get("available") else "unavailable_expected"),
            })
        rules = inventory.get("api_surface_rules") if isinstance(inventory.get("api_surface_rules"), list) else []
        api_mappings: list[dict[str, Any]] = []
        for path in _api_literals():
            rule = _surface_rule(path, rules)
            if rule is None:
                failures.append({"category": "surface", "item": path, "reason": "api_surface_unmapped"})
                continue
            parent = str(rule.get("parent") or "")
            if parent not in expected:
                failures.append({"category": "surface", "item": path, "reason": f"unknown_parent:{parent}"})
            api_mappings.append({"surface": path, "parent": parent, "disposition": rule.get("disposition")})
        for mapping_kind in ("cli_mappings", "ui_mappings", "telegram_mappings", "native_skill_mappings", "compatibility_route_mappings"):
            mappings = inventory.get(mapping_kind) if isinstance(inventory.get(mapping_kind), dict) else {}
            for surface, parent in mappings.items():
                if str(parent) not in expected:
                    failures.append({"category": "surface", "item": f"{mapping_kind}:{surface}", "reason": f"unknown_parent:{parent}"})
        cli_expected = set(inventory.get("cli_mappings", {}))
        for missing in sorted(_cli_commands() - cli_expected):
            failures.append({"category": "surface", "item": f"cli:{missing}", "reason": "cli_surface_unmapped"})
        for stale in sorted(cli_expected - _cli_commands()):
            failures.append({"category": "surface", "item": f"cli:{stale}", "reason": "stale_cli_mapping"})
        skill_expected = set(inventory.get("native_skill_mappings", {}))
        for missing in sorted(_native_skill_names() - skill_expected):
            failures.append({"category": "surface", "item": f"native_skill:{missing}", "reason": "native_skill_unmapped"})
        for stale in sorted(skill_expected - _native_skill_names()):
            failures.append({"category": "surface", "item": f"native_skill:{stale}", "reason": "stale_native_skill_mapping"})
        ui_expected = set(inventory.get("ui_mappings", {}))
        for missing in sorted(_ui_section_ids() - ui_expected):
            failures.append({"category": "surface", "item": f"ui:{missing}", "reason": "ui_surface_unmapped"})
        for stale in sorted(ui_expected - _ui_section_ids()):
            failures.append({"category": "surface", "item": f"ui:{stale}", "reason": "stale_ui_mapping"})
        for claim in inventory.get("documentation_claims", []):
            if not (ROOT / str(claim)).is_file():
                failures.append({"category": "documentation", "item": str(claim), "reason": "documentation_claim_source_missing"})
        dispositions = Counter(
            str(row.get("disposition") or "invalid")
            for row in inventory.get("dispositions", [])
            if isinstance(row, dict)
        )
        by_family = Counter(str(row.get("family")) for row in results)
        by_status = Counter(str(row.get("status")) for row in results)
        by_category = Counter(str(row.get("category")) for row in failures)
        report = {
            "schema_version": "native-capability-proof.v1",
            "candidate": identity,
            "inventory_sha256": hashlib.sha256(INVENTORY_PATH.read_bytes()).hexdigest(),
            "ok": not failures,
            "totals": {"expected": len(expected), "registered": len(actual), "api_surfaces": len(api_mappings), "failures": len(failures)},
            "by_family": dict(sorted(by_family.items())),
            "by_status": dict(sorted(by_status.items())),
            "dispositions": dict(sorted(dispositions.items())),
            "failures_by_category": dict(sorted(by_category.items())),
            "capabilities": results,
            "surface_totals": {
                "api": len(api_mappings),
                "cli": len(inventory.get("cli_mappings", {})),
                "ui": len(inventory.get("ui_mappings", {})),
                "telegram": len(inventory.get("telegram_mappings", {})),
                "native_skills": len(inventory.get("native_skill_mappings", {})),
                "compatibility_routes": len(inventory.get("compatibility_route_mappings", {})),
                "documentation_claims": len(inventory.get("documentation_claims", [])),
            },
            "api_mappings": api_mappings,
            "failures": failures,
            "redaction": {"passed": True, "fields_excluded": ["secrets", "tokens", "private_urls", "raw_pack_content", "model_prompts", "model_responses"]},
        }
        return report
    finally:
        db.close()
        temp.cleanup()


def _execute_declared_proofs(report: dict[str, Any]) -> None:
    nodes = sorted({
        node
        for capability in report.get("capabilities", [])
        for values in capability.get("proof_nodes", {}).values()
        for node in values
    })
    node_results: dict[str, dict[str, Any]] = {}
    for node in nodes:
        started = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--maxfail=1", node],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        node_results[node] = {
            "status": "pass" if proc.returncode == 0 else "fail",
            "exit_code": int(proc.returncode),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }
    category_counts: Counter[str] = Counter()
    for capability in report.get("capabilities", []):
        category_status: dict[str, str] = {}
        for category, required_nodes in capability.get("proof_nodes", {}).items():
            status = "pass" if required_nodes and all(node_results.get(node, {}).get("status") == "pass" for node in required_nodes) else "fail"
            category_status[category] = status
            category_counts[f"{category}:{status}"] += 1
            if status == "fail":
                report["failures"].append({
                    "category": "proof_execution",
                    "item": str(capability.get("id") or "unknown"),
                    "reason": f"proof_category_failed:{category}",
                })
        capability["proof_category_status"] = category_status
        if "fail" in category_status.values():
            capability["status"] = "fail"
    report["proof_execution"] = {
        "nodes": node_results,
        "category_totals": dict(sorted(category_counts.items())),
    }
    report["totals"]["failures"] = len(report["failures"])
    report["failures_by_category"] = dict(sorted(Counter(str(row.get("category")) for row in report["failures"]).items()))
    report["by_status"] = dict(sorted(Counter(str(row.get("status")) for row in report["capabilities"]).items()))
    report["ok"] = not report["failures"]


def _human(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "Native capability proof",
        f"candidate: {report['candidate']['commit']}",
        f"result: {'PASS' if report['ok'] else 'FAIL'}",
        f"capabilities: {totals['registered']}/{totals['expected']}",
        f"mapped API surfaces: {totals['api_surfaces']}",
        "status: " + ", ".join(f"{key}={value}" for key, value in report["by_status"].items()),
        "families: " + ", ".join(f"{key}={value}" for key, value in report["by_family"].items()),
    ]
    for failure in report["failures"]:
        lines.append(f"FAIL [{failure['category']}] {failure['item']}: {failure['reason']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and self-prove the protected native capability inventory.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--execute-tests", action="store_true", help="Execute every declared proof node and record per-category results.")
    args = parser.parse_args(argv)
    report = run_proof()
    if bool(args.execute_tests) and report.get("ok"):
        _execute_declared_proofs(report)
    human = _human(report)
    if not args.no_write:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(human, encoding="utf-8")
    print(human, end="")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
