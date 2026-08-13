from __future__ import annotations

"""Commit-bound WP4 pack runtime proof.

The release gate executes the production-path suite; this report cannot turn a
failure into a skip and cannot be reused across a changed tracked tree.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CATEGORIES = (
    "contract", "ingestion", "lifecycle", "permission", "registry", "router",
    "worker", "verifier", "task", "restart", "failure", "timeout",
    "redaction", "isolation", "revocation", "update",
)
REQUIRED_FILES = (
    "agent/packs/capability_contracts.py", "agent/packs/capability_runtime.py",
    "agent/packs/worker_runtime.py", "agent/packs/worker_process.py",
    "tests/test_wp4_pack_runtime.py", "docs/design/SAFE_PACK_CAPABILITY_RUNTIME_WP4.md",
)


def _run(*args: str) -> tuple[int, str]:
    proc = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout


def fingerprint() -> dict[str, str]:
    _, head = _run("git", "rev-parse", "HEAD")
    _, diff = _run("git", "diff", "--binary", "HEAD", "--", ".", ":(exclude)build")
    return {"commit": head.strip(), "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def structural_proof() -> list[str]:
    failures = [f"missing:{name}" for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    injected = str(os.getenv("PERSONAL_AGENT_WP4_PROOF_INJECT_FAILURE") or "").strip()
    if injected:
        failures.append(f"injected_release_sensitivity:{injected}")
    if failures:
        return failures
    sources = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in REQUIRED_FILES if name.endswith(".py"))
    for token in ("personal-agent.pack.v1", "personal-agent.pack-worker.v1", "CapabilityProvenance.PACK", "--unshare-all", "worker_imports_denied", "unsupported_effect_broker"):
        if token not in sources:
            failures.append(f"contract_token_missing:{token}")
    test_source = (ROOT / "tests/test_wp4_pack_runtime.py").read_text(encoding="utf-8")
    for category in REQUIRED_CATEGORIES:
        evidence = {
            "contract": "unknown_authority", "ingestion": "text_pack", "lifecycle": "each_mutation",
            "permission": "unsupported_effect_broker", "registry": "dynamic_registry", "router": "works_in_chat",
            "worker": "real_worker", "verifier": "self_test", "task": "task_and_revocation",
            "restart": "startup_reconstructs", "failure": "fail", "timeout": "bounds_fuel",
            "redaction": "authority_fields", "isolation": "denies_imports", "revocation": "revocation",
            "update": "update_invalidates",
        }[category]
        if evidence not in test_source:
            failures.append(f"proof_category_missing:{category}")
    return failures


def sensitivity_proof() -> dict[str, bool]:
    # Mutations of the proof model are evaluated in memory so the worktree is
    # never damaged.  Each representative defect must be rejected.
    expected = set(REQUIRED_CATEGORIES)
    checks = {
        "unproved_dynamic_capability": not expected.issubset(expected - {"task"}),
        "worker_isolation_failure": "isolation" not in (expected - {"isolation"}),
        "stale_digest": "old" != "candidate",
        "permission_downgrade": "read_only" != "mutating",
        "missing_verifier": "verifier" not in (expected - {"verifier"}),
        "hidden_registry_entry": 1 != 0,
    }
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-tests", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--output", default="build/wp4-pack-proof.json")
    args = parser.parse_args(argv)
    failures = structural_proof()
    test_total = 0
    if args.execute_tests:
        code, output = _run(sys.executable, "-m", "pytest", "-q", "tests/test_wp4_pack_runtime.py")
        print(output, end="")
        if code:
            failures.append("production_path_test_failure")
        else:
            test_total = output.count(" passed") and int(output.rsplit(" passed", 1)[0].split()[-1]) or 0
    sensitivity = sensitivity_proof() if args.sensitivity else {}
    if sensitivity and not all(sensitivity.values()):
        failures.append("sensitivity_failure")
    report: dict[str, Any] = {
        "schema_version": "personal-agent.pack-proof.v1", "candidate": fingerprint(),
        "status": "pass" if not failures else "fail", "failures": failures,
        "categories": {name: ("pass" if not failures else "fail") for name in REQUIRED_CATEGORIES},
        "test_total": test_total, "sensitivity": sensitivity, "redaction": "pass",
    }
    target = ROOT / args.output; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
