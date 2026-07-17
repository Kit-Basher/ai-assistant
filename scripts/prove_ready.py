#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
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
    "agent/capability_policy.py",
    "agent/executor_registry.py",
    "agent/mutation_boundary.py",
    "agent/mutation_plan.py",
    "agent/skill_pack_permissions.py",
    "agent/orchestrator.py",
    "agent/setup_chat_flow.py",
    "agent/packs/store.py",
    "agent/packs/state_truth.py",
    "agent/runtime_truth_service.py",
    "agent/doctor.py",
    "agent/host_lifecycle.py",
    "scripts/chat_eval.py",
    "scripts/llm_behavior_eval.py",
    "scripts/perf_smoke.py",
    "scripts/rc1_latency_closure_smoke.py",
    "scripts/runtime_latency_investigation.py",
    "scripts/runtime_latency_closure_smoke.py",
    "scripts/version_consistency_smoke.py",
    "scripts/upgrade_compatibility_smoke.py",
    "scripts/release_artifact_smoke.py",
    "scripts/clean_checkout_reproducibility_smoke.py",
    "scripts/clean_checkout_debian_package_smoke.py",
    "scripts/final_release_audit.py",
    "scripts/full_pytest_failure_triage.py",
    "scripts/full_pytest_closure_smoke.py",
    "scripts/skipped_test_debt_inventory.py",
    "scripts/skipped_test_debt_closure_smoke.py",
    "scripts/telegram_transport_diagnostic.py",
    "scripts/telegram_transport_smoke.py",
    "scripts/telegram_token_redaction_smoke.py",
    "scripts/telegram_first_reply_latency_smoke.py",
    "scripts/assistant_personality_memory_smoke.py",
    "scripts/local_intent_routing_smoke.py",
    "scripts/local_system_inspection_smoke.py",
    "scripts/capability_policy_smoke.py",
    "scripts/capability_policy_audit.py",
    "scripts/universal_plan_mode_smoke.py",
    "scripts/universal_plan_mode_audit.py",
    "scripts/skill_pack_permission_boundary_smoke.py",
    "scripts/generic_mutation_bypass_audit.py",
    "scripts/generic_mutation_bypass_smoke.py",
    "scripts/full_adversarial_authorization_proof.py",
    "scripts/executor_registry_smoke.py",
    "scripts/support_bundle_v2_smoke.py",
    "scripts/backup_v1_smoke.py",
    "scripts/cleanup_preview_smoke.py",
    "scripts/restore_validator_smoke.py",
    "scripts/restore_execution_smoke.py",
    "scripts/update_execution_smoke.py",
    "scripts/uninstall_execution_smoke.py",
    "scripts/host_lifecycle_runner.py",
    "scripts/host_lifecycle_runner_smoke.py",
    "scripts/host_lifecycle_systemd_smoke.py",
    "scripts/active_host_enablement_smoke.py",
    "scripts/primary_update_enablement_smoke.py",
    "scripts/primary_uninstall_enablement_smoke.py",
    "scripts/uninstall_helper.py",
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
        Gate("capability_policy_smoke central authorization proof", (sys.executable, "scripts/capability_policy_smoke.py"), 120),
        Gate("capability_policy_audit executor binding audit", (sys.executable, "scripts/capability_policy_audit.py"), 120),
        Gate("universal_plan_mode_smoke shared mutation plan proof", (sys.executable, "scripts/universal_plan_mode_smoke.py"), 120),
        Gate("universal_plan_mode_audit migration audit", (sys.executable, "scripts/universal_plan_mode_audit.py"), 120),
        Gate("skill_pack_permission_boundary_smoke platform API boundary proof", (sys.executable, "scripts/skill_pack_permission_boundary_smoke.py"), 120),
        Gate("generic_mutation_bypass_audit static mutation-surface audit", (sys.executable, "scripts/generic_mutation_bypass_audit.py"), 120),
        Gate("generic_mutation_bypass_smoke dynamic bypass denial proof", (sys.executable, "scripts/generic_mutation_bypass_smoke.py"), 120),
        Gate("full_adversarial_authorization_proof end-to-end authorization attack matrix", (sys.executable, "scripts/full_adversarial_authorization_proof.py"), 120),
        Gate("runtime_latency_investigation measured latency evidence", (sys.executable, "scripts/runtime_latency_investigation.py"), 180),
        Gate("runtime_latency_closure_smoke accepted latency record", (sys.executable, "scripts/runtime_latency_closure_smoke.py"), 120),
        Gate("version_consistency_smoke product version truth", (sys.executable, "scripts/version_consistency_smoke.py"), 120),
        Gate("upgrade_compatibility_smoke isolated state compatibility", (sys.executable, "scripts/upgrade_compatibility_smoke.py"), 120),
        Gate("release_artifact_smoke source and bundle artifact audit", (sys.executable, "scripts/release_artifact_smoke.py"), 180),
        Gate("clean_checkout_reproducibility_smoke clean source proof", (sys.executable, "scripts/clean_checkout_reproducibility_smoke.py"), 360),
        Gate("clean_checkout_debian_package_smoke package reproducibility", (sys.executable, "scripts/clean_checkout_debian_package_smoke.py"), 240),
        Gate("full_pytest_closure_smoke default source-tree pytest closure", (sys.executable, "scripts/full_pytest_closure_smoke.py"), 1800),
        Gate("full_pytest_failure_triage classified baseline inventory", (sys.executable, "scripts/full_pytest_failure_triage.py"), 1800),
        Gate("skipped_test_debt_inventory historical skip-debt accounting", (sys.executable, "scripts/skipped_test_debt_inventory.py"), 120),
        Gate("skipped_test_debt_closure_smoke non-environmental skip debt closed", (sys.executable, "scripts/skipped_test_debt_closure_smoke.py"), 1800),
        Gate("telegram_transport_diagnostic no-send live Telegram status", (sys.executable, "scripts/telegram_transport_diagnostic.py"), 120),
        Gate("telegram_transport_smoke fixture Telegram inbound-to-reply proof", (sys.executable, "scripts/telegram_transport_smoke.py"), 120),
        Gate("telegram_token_redaction_smoke Telegram secret redaction proof", (sys.executable, "scripts/telegram_token_redaction_smoke.py"), 120),
        Gate("telegram_first_reply_latency_smoke first reply and greeting proof", (sys.executable, "scripts/telegram_first_reply_latency_smoke.py"), 120),
        Gate("assistant_personality_memory_smoke assistant UX and memory policy proof", (sys.executable, "scripts/assistant_personality_memory_smoke.py"), 120),
        Gate("local_intent_routing_smoke current-device routing proof", (sys.executable, "scripts/local_intent_routing_smoke.py"), 120),
        Gate("local_system_inspection_smoke process-level local inspection proof", (sys.executable, "scripts/local_system_inspection_smoke.py"), 120),
        Gate("final_release_audit version decision and release truth", (sys.executable, "scripts/final_release_audit.py"), 2400),
        Gate("rc1_latency_closure_smoke latency distributions", (sys.executable, "scripts/rc1_latency_closure_smoke.py"), 180),
        Gate("perf_smoke read-only latency smoke", (sys.executable, "scripts/perf_smoke.py"), 180),
        Gate("release_smoke canonical smoke suite", (sys.executable, "scripts/release_smoke.py"), 420),
        Gate("host_lifecycle_runner_smoke shared host runner", (sys.executable, "scripts/host_lifecycle_runner_smoke.py"), 120),
        Gate("update_execution_smoke isolated update executor", (sys.executable, "scripts/update_execution_smoke.py"), 120),
        Gate("uninstall_execution_smoke isolated uninstall executor", (sys.executable, "scripts/uninstall_execution_smoke.py"), 120),
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


def _latency_acceptance_covers(gate_name: str) -> bool:
    path = ROOT / "docs" / "operator" / "RUNTIME_LATENCY_ACCEPTANCE_V1.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False
    if not str(parsed.get("release_decision") or "").startswith("accepted_for_v0.2.2"):
        return False
    accepted = parsed.get("accepted_warnings")
    if not isinstance(accepted, list):
        return False
    for row in accepted:
        if not isinstance(row, dict):
            continue
        gates = row.get("gate_names")
        if isinstance(gates, list) and gate_name in {str(item) for item in gates}:
            return True
    return False


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
            return "NOTE", "expected-isolated-proof", "prove_core_workflows.py marks nested release gates NOT_PROVEN by design; prove_ready.py runs the direct behavior gate separately above."
        if "blocked workflows: internet/search status" in lowered:
            return "NOTE", "expected-isolated-proof", "Isolated proof search BLOCKED is expected unless trusted SearXNG is configured."
    if gate.name.startswith("rc1_latency_closure_smoke"):
        failed = _extract_count("FAIL", output)
        warned = _extract_count("WARN", output)
        if failed:
            return "FAIL", "release-blocking", "Fix the failing RC1 latency closure probe before final release."
        if warned:
            if _latency_acceptance_covers(gate.name):
                return "WARN_ACCEPTED", "runtime-latency-accepted", "Latency warning is covered by docs/operator/RUNTIME_LATENCY_ACCEPTANCE_V1.json; revisit if trigger thresholds are exceeded."
            return "WARN", "runtime-state", "One or more RC1 latency closure distributions exceeded the measured budget."
    if gate.name.startswith("perf_smoke"):
        failed = _extract_count("FAIL", output)
        warned = _extract_count("WARN", output)
        if failed:
            return "FAIL", "release-blocking", "Fix the failing read-only performance/status probe before VM proof."
        if warned:
            if _latency_acceptance_covers(gate.name):
                return "WARN_ACCEPTED", "runtime-latency-accepted", "Latency warning is covered by docs/operator/RUNTIME_LATENCY_ACCEPTANCE_V1.json; revisit if trigger thresholds are exceeded."
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
    accepted_warnings = [row for row in results if row.status == "WARN_ACCEPTED"]
    notes = [row for row in results if row.status == "NOTE"]
    passed = [row for row in results if row.status == "PASS"]
    print("## Summary")
    print(f"PASS={len(passed)} WARN={len(warnings)} WARN_ACCEPTED={len(accepted_warnings)} FAIL={len(blocking_failures)} NOTES={len(notes)}")
    print(f"READY_FOR_VM_PROOF: {'yes' if not blocking_failures else 'no'}")
    print(f"RELEASE_BLOCKERS: {len(blocking_failures)}")
    print(f"WARNINGS_UNRESOLVED: {len(warnings)}")
    print(f"WARNINGS_ACCEPTED: {len(accepted_warnings)}")
    print(f"NOTES: {len(notes)}")
    print(f"READY_TO_RELEASE: {'true' if not blocking_failures and not warnings else 'false'}")
    print("NEXT_ACTIONS:")
    if warnings:
        for row in warnings:
            print(f"- [{row.category}] {row.gate.name}: {row.next_action}")
    if accepted_warnings:
        for row in accepted_warnings:
            print(f"- accepted [{row.category}] {row.gate.name}: {row.next_action}")
    if notes:
        for row in notes:
            print(f"- note [{row.category}] {row.gate.name}: {row.next_action}")
    if not warnings and not blocking_failures:
        print("- Ready for manual release commit/tag review; notes and accepted warnings are listed above when present.")
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
