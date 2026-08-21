from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release_smoke import EXTENDED_TEST_NODES, MAIN_TEST_NODES

WP1_UNIFIED_ROUTING_TEST_NODES: tuple[str, ...] = (
    "tests/test_unified_conversation_routing.py",
)
WP2_NATIVE_CAPABILITY_TEST_NODES: tuple[str, ...] = (
    "tests/test_native_capability_proof.py",
    "tests/test_confirmation_transactions.py",
    "tests/test_adversarial_authorization.py",
    "tests/test_filesystem_api_contract.py",
    "tests/test_safe_web_search.py",
    "tests/test_api_packs_endpoints.py",
    "tests/test_memory_runtime.py",
    "tests/test_telegram_runtime_state.py",
    "tests/test_model_switch_semantics.py",
)
WP3_TASK_LOOP_TEST_NODES: tuple[str, ...] = (
    "tests/test_general_task_loop.py",
    "tests/test_task_loop_properties.py",
    "tests/test_wp3_scenarios.py",
    "tests/test_task_loop_proof.py",
)
WP4_PACK_RUNTIME_TEST_NODES: tuple[str, ...] = (
    "tests/test_wp4_pack_runtime.py",
)
WP45_MODEL_TRUTH_TEST_NODES: tuple[str, ...] = (
    "tests/test_model_truth_latency_wp45.py",
)
WP5_PACK_ACQUISITION_TEST_NODES: tuple[str, ...] = (
    "tests/test_wp5_pack_acquisition_brokers.py",
    "tests/test_pack_source_fetch_preview.py",
    "tests/test_pack_search_authorization.py",
)

PY_COMPILE_TARGETS: tuple[str, ...] = (
    "agent/api_server.py",
    "memory/db.py",
    "agent/orchestrator.py",
    "agent/capability_registry.py",
    "agent/request_understanding.py",
    "agent/task_loop.py",
    "agent/setup_chat_flow.py",
    "agent/filesystem_skill.py",
    "agent/executor_registry.py",
    "agent/runtime_truth_service.py",
    "agent/packs/store.py",
    "agent/packs/state_truth.py",
    "agent/packs/capability_recommendation.py",
    "agent/failure_ux.py",
    "agent/recovery_contract.py",
    "agent/state_transitions.py",
    "agent/ux/llm_fixit_wizard.py",
    "agent/persona.py",
    "scripts/release_smoke.py",
    "scripts/release_validation_extended.py",
    "scripts/assistant_viability_smoke.py",
    "scripts/restart_memory_smoke.py",
    "scripts/provider_matrix_smoke.py",
    "scripts/assistant_real_world_smoke.py",
    "scripts/assistant_interaction_barrage.py",
    "scripts/split_smoke.py",
    "scripts/reference_pack_workflow_smoke.py",
    "scripts/webui_smoke.py",
    "scripts/browser_ui_survival_smoke.py",
    "scripts/executor_registry_smoke.py",
    "scripts/support_bundle_v2_smoke.py",
    "scripts/backup_v1_smoke.py",
    "scripts/cleanup_preview_smoke.py",
    "scripts/restore_validator_smoke.py",
    "scripts/first_run_smoke.py",
    "scripts/vm_proof_smoke.py",
    "scripts/daily_driver_maturity_audit.py",
    "scripts/chat_frontdoor_smoke.py",
    "scripts/wp1_latency_probe.py",
    "scripts/native_capability_proof.py",
    "scripts/task_loop_proof.py",
    "scripts/wp3_latency_probe.py",
    "agent/packs/capability_contracts.py",
    "agent/packs/capability_runtime.py",
    "agent/packs/worker_runtime.py",
    "agent/packs/worker_process.py",
    "scripts/pack_capability_proof.py",
    "scripts/wp4_latency_probe.py",
    "agent/llm/model_runtime_truth.py",
    "scripts/model_runtime_evaluation.py",
    "scripts/model_truth_latency_proof.py",
    "scripts/wp45_latency_probe.py",
    "scripts/wp45_isolated_candidate.py",
    "agent/packs/wp5_contracts.py",
    "agent/packs/secure_transport.py",
    "agent/packs/brokers.py",
    "agent/packs/draft_builder.py",
    "scripts/pack_acquisition_broker_proof.py",
    "scripts/wp5_browser_candidate_smoke.py",
)

def _pytest_command(test_nodes: tuple[str, ...]) -> tuple[str, ...]:
    return (sys.executable, "-m", "pytest", "-q", "--maxfail=1", *test_nodes)


RELEASE_GATE_COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "-m", "py_compile", *PY_COMPILE_TARGETS),
    (sys.executable, "scripts/native_capability_proof.py", "--execute-tests"),
    (sys.executable, "scripts/task_loop_proof.py", "--execute-tests"),
    (sys.executable, "scripts/pack_capability_proof.py", "--execute-tests", "--sensitivity"),
    (sys.executable, "scripts/model_truth_latency_proof.py", "--execute-tests", "--sensitivity"),
    (sys.executable, "scripts/pack_acquisition_broker_proof.py", "--execute-tests", "--sensitivity"),
    ("bash", "scripts/build_webui.sh"),
    (
        "node",
        "--test",
        "desktop/tests/chatExperienceRobustness.test.js",
        "desktop/tests/chatUiHelpers.test.js",
        "desktop/tests/packStateUiHelpers.test.js",
        "desktop/tests/stateUiHelpers.test.js",
        "desktop/tests/taskUiHelpers.test.js",
    ),
    _pytest_command((*MAIN_TEST_NODES, *WP1_UNIFIED_ROUTING_TEST_NODES, *WP2_NATIVE_CAPABILITY_TEST_NODES, *WP3_TASK_LOOP_TEST_NODES, *WP4_PACK_RUNTIME_TEST_NODES, *WP45_MODEL_TRUTH_TEST_NODES, *WP5_PACK_ACQUISITION_TEST_NODES)),
    _pytest_command(EXTENDED_TEST_NODES),
    ("git", "diff", "--check"),
)

PY_COMPILE_COMMAND: tuple[str, ...] = RELEASE_GATE_COMMANDS[0]


def _print_commands() -> None:
    print("release gate commands:", flush=True)
    for command in RELEASE_GATE_COMMANDS:
        print(" ".join(command), flush=True)


def _run_command(command: tuple[str, ...]) -> int:
    proc = subprocess.run(command, cwd=ROOT, check=False)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the canonical Personal Agent release gate.")
    parser.add_argument("--list", action="store_true", help="Print the exact commands without running them.")
    parser.add_argument(
        "--py-compile-only",
        action="store_true",
        help="Run only the canonical release-gate py_compile target list.",
    )
    args = parser.parse_args(argv)
    if bool(args.list):
        _print_commands()
        return 0
    if bool(args.py_compile_only):
        print(f"Running: {' '.join(PY_COMPILE_COMMAND)}", flush=True)
        return _run_command(PY_COMPILE_COMMAND)
    for command in RELEASE_GATE_COMMANDS:
        print(f"Running: {' '.join(command)}", flush=True)
        exit_code = _run_command(command)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
