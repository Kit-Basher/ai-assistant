from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release_smoke import EXTENDED_TEST_NODES, MAIN_TEST_NODES

PY_COMPILE_TARGETS: tuple[str, ...] = (
    "agent/api_server.py",
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
    "scripts/reference_pack_workflow_smoke.py",
    "scripts/webui_smoke.py",
)

def _pytest_command(test_nodes: tuple[str, ...]) -> tuple[str, ...]:
    return (sys.executable, "-m", "pytest", "-q", "--maxfail=1", *test_nodes)


RELEASE_GATE_COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "-m", "py_compile", *PY_COMPILE_TARGETS),
    _pytest_command(MAIN_TEST_NODES),
    _pytest_command(EXTENDED_TEST_NODES),
    ("git", "diff", "--check"),
)


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
    args = parser.parse_args(argv)
    if bool(args.list):
        _print_commands()
        return 0
    for command in RELEASE_GATE_COMMANDS:
        print(f"Running: {' '.join(command)}", flush=True)
        exit_code = _run_command(command)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
