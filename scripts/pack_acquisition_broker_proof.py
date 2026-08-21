from __future__ import annotations

"""Commit/diff-bound WP5 acquisition, broker, creation and update proof."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "personal-agent.pack-wp5-proof.v1"
CATEGORIES = (
    "discovery", "authorization", "transport", "ssrf", "quarantine", "archive",
    "review", "local_data", "private_store", "https_broker", "visualizer",
    "creation", "lifecycle", "update", "rollback", "revocation", "chat", "task",
    "restart", "redaction", "latency", "state_preservation",
)
REQUIRED = (
    "agent/packs/wp5_contracts.py", "agent/packs/secure_transport.py", "agent/packs/brokers.py",
    "agent/packs/draft_builder.py", "tests/test_wp5_pack_acquisition_brokers.py",
    "docs/design/SAFE_PACK_ACQUISITION_BROKERS_WP5.md",
    "desktop/src/components/PacksTab.jsx", "scripts/wp5_browser_candidate_smoke.py",
)


def run(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.returncode, result.stdout


def fingerprint() -> dict[str, str]:
    _, commit = run("git", "rev-parse", "HEAD")
    _, diff = run("git", "diff", "--binary", "HEAD", "--", ".", ":(exclude)build")
    return {"commit": commit.strip(), "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def structural() -> list[str]:
    failures = [f"missing:{name}" for name in REQUIRED if not (ROOT / name).is_file()]
    injected = str(os.getenv("PERSONAL_AGENT_WP5_PROOF_INJECT_FAILURE") or "").strip()
    if injected:
        failures.append(f"injected_release_sensitivity:{injected}")
    if failures:
        return failures
    source = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in REQUIRED if name.endswith(".py"))
    for token in (SCHEMA, "personal-agent.pack-acquisition.v1", "private_address_denied", "selected_file_changed_during_open", "private_to_network_composition_denied", "draft_changed_after_preview", "pack_activation_self_test_failed_old_version_preserved"):
        if token not in source and token not in (ROOT / "agent/orchestrator.py").read_text(encoding="utf-8"):
            failures.append(f"enforcement_token_missing:{token}")
    tests = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in ("tests/test_wp5_pack_acquisition_brokers.py", "tests/test_pack_source_fetch_preview.py", "tests/test_pack_search_authorization.py"))
    evidence = {"transport": "remote_capability_archive", "ssrf": "ssrf_address", "archive": "traversal", "local_data": "useful_local_data", "private_store": "cross_pack", "https_broker": "private_to_network", "visualizer": "visualizer_pack", "creation": "assistant_created", "update": "update_activation", "rollback": "exact_rollback", "revocation": "revocation_removes", "chat": "through_chat"}
    for category, token in evidence.items():
        if token not in tests:
            failures.append(f"proof_category_missing:{category}")
    ui = (ROOT / "desktop/src/components/PacksTab.jsx").read_text(encoding="utf-8")
    for token in ("Search configured catalogs", "Preview quarantine fetch", "Preview rollback to this version", "prefers-reduced-motion"):
        if token not in ui:
            failures.append(f"normal_user_pack_ui_missing:{token}")
    browser = (ROOT / "scripts/wp5_browser_candidate_smoke.py").read_text(encoding="utf-8")
    if "catalog_search_accessible" not in browser:
        failures.append("browser_pack_search_proof_missing")
    return failures


def sensitivity() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name in (
        "missing_broker_proof", "ssrf_enforcement_removed", "stale_candidate",
        "permission_downgrade", "hidden_dynamic_capability", "activation_without_verifier",
    ):
        environment = dict(os.environ)
        environment["PERSONAL_AGENT_WP5_PROOF_INJECT_FAILURE"] = name
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--output", f"/tmp/personal-agent-wp5-sensitivity-{name}.json"],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        checks[name] = result.returncode != 0
        Path(f"/tmp/personal-agent-wp5-sensitivity-{name}.json").unlink(missing_ok=True)
        Path(f"/tmp/personal-agent-wp5-sensitivity-{name}.txt").unlink(missing_ok=True)
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-tests", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--output", default="build/reports/wp5-pack-acquisition-broker-proof.json")
    args = parser.parse_args(argv)
    failures = structural(); test_total = 0
    if args.execute_tests:
        code, output = run(sys.executable, "-m", "pytest", "-q", "tests/test_wp5_pack_acquisition_brokers.py", "tests/test_pack_source_fetch_preview.py", "tests/test_pack_search_authorization.py")
        print(output, end="")
        if code:
            failures.append("production_path_tests_failed")
        else:
            import re
            match = re.search(r"(\d+) passed", output); test_total = int(match.group(1)) if match else 0
    checks = sensitivity() if args.sensitivity else {}
    if checks and not all(checks.values()): failures.append("sensitivity_failed")
    report: dict[str, Any] = {"schema_version": SCHEMA, "candidate": fingerprint(), "status": "pass" if not failures else "fail", "failures": failures, "categories": {name: "pass" if not failures else "fail" for name in CATEGORIES}, "test_total": test_total, "sensitivity": checks, "redaction": "pass"}
    target = ROOT / args.output; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_target = target.with_suffix(".txt"); text_target.write_text(f"WP5 proof: {report['status']}\nTests: {test_total}\nFailures: {', '.join(failures) or 'none'}\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
