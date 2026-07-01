#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Gate:
    name: str
    command: tuple[str, ...]
    timeout_seconds: int = 300
    release_blocking: bool = True


@dataclass
class GateResult:
    gate: Gate
    returncode: int
    elapsed_seconds: float
    output: str
    status: str
    category: str
    next_action: str


CORE_PY_COMPILE: tuple[str, ...] = (
    "agent/api_server.py",
    "agent/policy.py",
    "agent/bot.py",
    "agent/executor_registry.py",
    "agent/orchestrator.py",
    "agent/setup_chat_flow.py",
    "agent/packs/store.py",
    "agent/packs/state_truth.py",
    "agent/runtime_truth_service.py",
    "agent/doctor.py",
    "scripts/chat_eval.py",
    "scripts/llm_behavior_eval.py",
    "scripts/perf_smoke.py",
    "scripts/executor_registry_smoke.py",
    "scripts/support_bundle_v2_smoke.py",
    "scripts/backup_v1_smoke.py",
    "scripts/cleanup_preview_smoke.py",
    "scripts/restore_validator_smoke.py",
    "scripts/first_run_smoke.py",
    "scripts/vm_proof_smoke.py",
    "scripts/daily_driver_maturity_audit.py",
    "scripts/prove_ready.py",
)


def _has_frontend() -> bool:
    return (ROOT / "desktop" / "package.json").exists() and (ROOT / "desktop" / "node_modules").is_dir()


def _gates() -> list[Gate]:
    gates = [
        Gate("py_compile core files", (sys.executable, "-m", "py_compile", *CORE_PY_COMPILE), 120),
        Gate("chat_eval deterministic adversarial routing", (sys.executable, "scripts/chat_eval.py"), 180),
        Gate("llm_behavior_eval deterministic e2e behavior", (sys.executable, "scripts/llm_behavior_eval.py"), 180),
        Gate("perf_smoke read-only latency smoke", (sys.executable, "scripts/perf_smoke.py"), 180),
        Gate("release_smoke canonical smoke suite", (sys.executable, "scripts/release_smoke.py"), 420),
        Gate(
            "direct behavior release gates",
            (
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_chat_behavior_audit.py",
                "tests/test_live_user_barrage.py",
                "tests/test_assistant_behavior_release_gate.py",
            ),
            180,
        ),
        Gate("daily_driver_smoke live user-facing path", (sys.executable, "scripts/daily_driver_smoke.py", "--timeout", "90"), 240),
        Gate("external_pack_safety_smoke safety gates", (sys.executable, "scripts/external_pack_safety_smoke.py"), 180),
        Gate("prove_core_workflows product proof", (sys.executable, "scripts/prove_core_workflows.py"), 240),
        Gate("git diff whitespace check", ("git", "diff", "--check"), 60),
    ]
    if _has_frontend():
        gates.insert(3, Gate("frontend build", ("npm", "run", "build"), 180))
    return gates


def _summarize_output(output: str, *, max_lines: int = 16) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines // 2]
    tail = lines[-(max_lines // 2) :]
    return "\n".join([*head, "...", *tail])


def _extract_count(label: str, output: str) -> int:
    match = re.search(rf"\b{re.escape(label)}=(\d+)\b", output)
    if match:
        return int(match.group(1))
    match = re.search(rf"^{re.escape(label)}:\s*(\d+)\b", output, re.MULTILINE | re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def _classify(gate: Gate, returncode: int, output: str) -> tuple[str, str, str]:
    if returncode != 0:
        return "FAIL", "release-blocking", "Fix the failed command above, then rerun python scripts/prove_ready.py."
    if gate.name.startswith("daily_driver_smoke"):
        blocked = _extract_count("BLOCKED", output)
        failed = _extract_count("FAIL", output)
        if failed:
            return "FAIL", "release-blocking", "Fix the failing daily-driver workflow before release proof."
        if blocked:
            lowered = output.lower()
            if "search unavailable" in lowered or "search_disabled" in lowered or "/search/status" in lowered:
                return "WARN", "runtime-state", "Search is disabled or unconfigured in this runtime; configure trusted SearXNG for live search proof."
            return "WARN", "runtime-state", "A daily-driver dependency is blocked; inspect the blocked section before release."
    if gate.name.startswith("prove_core_workflows"):
        lowered = output.lower()
        if "fail workflows: none" not in lowered:
            return "FAIL", "release-blocking", "Fix the failing core workflow proof before release."
        not_proven_match = re.search(r"^not_proven workflows:\s*(.+)$", output, re.MULTILINE | re.IGNORECASE)
        if not_proven_match and not_proven_match.group(1).strip().lower() != "none":
            return "WARN", "expected-isolated-proof", "prove_core_workflows.py marks nested release gates NOT_PROVEN by design; prove_ready.py runs the direct behavior gate separately above."
        if "blocked workflows: internet/search status" in lowered:
            return "WARN", "expected-isolated-proof", "Isolated proof search BLOCKED is expected unless trusted SearXNG is configured."
    if gate.name.startswith("perf_smoke"):
        failed = _extract_count("FAIL", output)
        warned = _extract_count("WARN", output)
        if failed:
            return "FAIL", "release-blocking", "Fix the failing read-only performance/status probe before VM proof."
        if warned:
            return "WARN", "runtime-state", "One or more read-only latency probes exceeded the generous warning budget; inspect perf_smoke output."
    return "PASS", "none", "No action."


def _run_gate(gate: Gate) -> GateResult:
    started = time.monotonic()
    cwd = ROOT / "desktop" if gate.name == "frontend build" else ROOT
    try:
        proc = subprocess.run(
            gate.command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=gate.timeout_seconds,
            check=False,
        )
        output = proc.stdout or ""
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        output = f"{output}\nTIMEOUT after {gate.timeout_seconds}s"
        returncode = 124
    elapsed = time.monotonic() - started
    status, category, next_action = _classify(gate, returncode, output)
    return GateResult(gate=gate, returncode=returncode, elapsed_seconds=elapsed, output=output, status=status, category=category, next_action=next_action)


def main() -> int:
    os.chdir(ROOT)
    results: list[GateResult] = []
    print("# Personal Agent Prove Ready")
    print(f"Repo: {ROOT}")
    print("")
    for gate in _gates():
        command_text = " ".join(gate.command)
        print(f"## {gate.name}")
        print(f"command: {command_text}")
        result = _run_gate(gate)
        results.append(result)
        print(f"status: {result.status} category={result.category} exit={result.returncode} elapsed={result.elapsed_seconds:.1f}s")
        if result.status != "PASS":
            print(f"next_action: {result.next_action}")
            print("output:")
            print(_summarize_output(result.output))
        print("")

    blocking_failures = [row for row in results if row.status == "FAIL" and row.gate.release_blocking]
    warnings = [row for row in results if row.status == "WARN"]
    passed = [row for row in results if row.status == "PASS"]
    print("## Summary")
    print(f"PASS={len(passed)} WARN={len(warnings)} FAIL={len(blocking_failures)}")
    print(f"READY_FOR_VM_PROOF: {'yes' if not blocking_failures else 'no'}")
    print(f"RELEASE_BLOCKERS: {len(blocking_failures)}")
    print(f"WARNINGS: {len(warnings)}")
    print("NEXT_ACTIONS:")
    if warnings:
        for row in warnings:
            print(f"- [{row.category}] {row.gate.name}: {row.next_action}")
    elif not blocking_failures:
        print("- Run the fresh Debian VM install proof when ready.")
    if blocking_failures:
        first = blocking_failures[0]
        print("Release-blocking failure:")
        print(f"- {first.gate.name}: {' '.join(first.gate.command)}")
        print(f"- next_action: {first.next_action}")
        return 1
    print("No release-blocking readiness gate failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
