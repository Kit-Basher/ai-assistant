#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PREFIX = "b0a7fe5"


def _login_command(*argv: str) -> tuple[str, ...]:
    return ("bash", "-lc", " ".join(shlex.quote(part) for part in argv))


@dataclass(frozen=True)
class Gate:
    phase: str
    name: str
    command: tuple[str, ...]
    timeout: int
    warn_only: bool = False


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    status: str
    seconds: float
    output: str


def build_gates(*, expected_commit: str = EXPECTED_PREFIX, include_primary_update: bool = True) -> list[Gate]:
    py = sys.executable
    gates = [
        Gate("checkpoint", "checkpoint status", _login_command("git", "status", "--short"), 30),
        Gate("repository_hygiene", "git diff whitespace", _login_command("git", "diff", "--check"), 30),
        Gate("host_policy_status", "primary uninstall policy status", _login_command(py, "scripts/primary_uninstall_policy.py", "status"), 30),
        Gate("promotion", "promote local stable", _login_command("bash", "scripts/promote_local_stable.sh"), 420),
        Gate("latency_closure", "rc1 latency closure", _login_command(py, "scripts/rc1_latency_closure_smoke.py"), 180),
        Gate("core_unit_tests", "release smoke", _login_command(py, "scripts/release_smoke.py"), 240),
        Gate("lifecycle_proofs", "primary uninstall activation policy", _login_command(py, "scripts/primary_uninstall_policy_smoke.py"), 120),
        Gate(
            "lifecycle_proofs",
            "primary uninstall production-shaped proof",
            _login_command(
                py,
                "scripts/primary_uninstall_enablement_smoke.py",
                "--allow-primary-uninstall-shaped-proof",
                "--expected-commit",
                expected_commit,
            ),
            180,
        ),
    ]
    if include_primary_update:
        gates.append(
            Gate(
                "lifecycle_proofs",
                "primary update proof",
                _login_command(
                    py,
                    "scripts/primary_update_enablement_smoke.py",
                    "--allow-primary-update-proof",
                    "--expected-commit",
                    expected_commit,
                ),
                420,
            )
        )
    gates.extend(
        [
            Gate("lifecycle_proofs", "active host enablement", _login_command(py, "scripts/active_host_enablement_smoke.py"), 420),
            Gate("lifecycle_proofs", "host lifecycle runner", _login_command(py, "scripts/host_lifecycle_runner_smoke.py"), 120),
            Gate("lifecycle_proofs", "host lifecycle systemd", _login_command(py, "scripts/host_lifecycle_systemd_smoke.py"), 120),
            Gate("lifecycle_proofs", "update execution", _login_command(py, "scripts/update_execution_smoke.py"), 120),
            Gate("lifecycle_proofs", "uninstall execution", _login_command(py, "scripts/uninstall_execution_smoke.py"), 120),
            Gate("installed_product_proofs", "operator lifecycle", _login_command(py, "scripts/operator_lifecycle_smoke.py"), 180),
            Gate("backup_restore", "restore execution", _login_command(py, "scripts/restore_execution_smoke.py"), 180),
            Gate("installed_product_proofs", "cleanup execution", _login_command(py, "scripts/cleanup_execution_smoke.py"), 180),
            Gate("installed_product_proofs", "executor registry", _login_command(py, "scripts/executor_registry_smoke.py"), 180),
            Gate("installed_product_proofs", "plan mode", _login_command(py, "scripts/plan_mode_v2_smoke.py"), 180),
            Gate("browser_proofs", "browser ui survival", _login_command(py, "scripts/browser_ui_survival_smoke.py"), 300),
            Gate("normal_user_proofs", "normal user acceptance", _login_command(py, "scripts/normal_user_acceptance_smoke.py"), 180),
            Gate("normal_user_proofs", "real use journey", _login_command(py, "scripts/real_use_journey_smoke.py"), 180),
            Gate("installed_product_proofs", "daily driver maturity", _login_command(py, "scripts/daily_driver_maturity_audit.py"), 240),
            Gate("installed_product_proofs", "installed product abuse", _login_command(py, "scripts/installed_product_abuse.py"), 240),
            Gate("release_readiness", "prove ready", _login_command(py, "scripts/prove_ready.py"), 420, warn_only=True),
            Gate("docs_truth", "docs truth", _login_command(py, "scripts/docs_truth_smoke.py"), 120),
            Gate("release_matrix", "release gate matrix", _login_command(py, "scripts/release_gate_matrix_smoke.py"), 120),
            Gate("final_runtime_state", "primary uninstall policy final status", _login_command(py, "scripts/primary_uninstall_policy.py", "status"), 30),
            Gate("final_runtime_state", "git status final", _login_command("git", "status", "--short"), 30),
        ]
    )
    return gates


def run_gate(gate: Gate) -> GateResult:
    started = time.monotonic()
    proc = subprocess.run(
        list(gate.command),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=gate.timeout,
        check=False,
    )
    seconds = time.monotonic() - started
    status = "PASS" if proc.returncode == 0 else ("WARN" if gate.warn_only else "FAIL")
    return GateResult(gate=gate, status=status, seconds=seconds, output=(proc.stdout or "")[-4000:])


def summarize(results: Iterable[GateResult]) -> dict[str, int]:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential v0.2.1 release closure proof runner.")
    parser.add_argument("--expected-commit", default=EXPECTED_PREFIX)
    parser.add_argument("--skip-primary-update", action="store_true", help="Skip the disruptive primary update proof.")
    parser.add_argument("--list", action="store_true", help="List gates without running them.")
    args = parser.parse_args()

    gates = build_gates(expected_commit=args.expected_commit, include_primary_update=not args.skip_primary_update)
    if args.list:
        for index, gate in enumerate(gates, start=1):
            print(f"{index:02d} {gate.phase}: {gate.name}: {' '.join(gate.command)}")
        print("NOTE: this runner never enables the primary uninstall marker and never runs active primary uninstall.")
        return 0

    started = time.monotonic()
    results: list[GateResult] = []
    print("# v0.2.1 Release Closure Sequential Proof")
    for gate in gates:
        print(f"## {gate.phase}: {gate.name}")
        print(f"command: {' '.join(gate.command)}")
        try:
            result = run_gate(gate)
        except subprocess.TimeoutExpired as exc:
            result = GateResult(gate=gate, status="WARN" if gate.warn_only else "FAIL", seconds=gate.timeout, output=f"timeout after {gate.timeout}s: {exc}")
        results.append(result)
        print(f"status: {result.status} seconds={result.seconds:.1f}")
        if result.output.strip():
            print(result.output.strip())
        if result.status == "FAIL":
            break
    counts = summarize(results)
    total_seconds = int(time.monotonic() - started)
    print(
        f"PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']} SKIP={counts['SKIP']} "
        f"TOTAL_SECONDS={total_seconds}"
    )
    recommendation = "v0.2.1" if counts["FAIL"] == 0 and counts["WARN"] == 0 else "v0.2.1-rc2"
    print(f"RELEASE_RECOMMENDATION={recommendation}")
    print("UNINSTALL_MARKER_ENABLED_BY_RUNNER=false")
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
