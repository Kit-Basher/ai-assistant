from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

MAIN_TEST_NODES: tuple[str, ...] = (
    "tests/test_publishability_smoke.py",
    "tests/test_golden_path_smoke.py",
    "tests/test_install_first_run_hardening.py::TestInstallFirstRunHardening::test_resolved_default_paths_use_canonical_location_for_fresh_install",
    "tests/test_install_first_run_hardening.py::TestInstallFirstRunHardening::test_doctor_required_dirs_warn_when_install_dirs_missing",
    "tests/test_install_first_run_hardening.py::TestInstallFirstRunHardening::test_startup_checks_fail_when_registry_json_is_invalid",
    "tests/test_memory_hardening.py::TestMemoryHardening::test_memory_status_reports_disabled_components_and_fresh_state",
    "tests/test_memory_hardening.py::TestMemoryHardening::test_memory_status_detects_corrupt_continuity_state",
    "tests/test_memory_hardening.py::TestMemoryHardening::test_memory_v2_selection_failure_degrades_clearly_without_breaking_chat",
    "tests/test_api_server.py::TestAPIServerRuntime::test_health_is_explicit_before_and_after_process_restart_with_deferred_warmup",
    "tests/test_doctor_cli.py::TestDoctorCLI::test_collect_diagnostics_writes_redacted_bundle_with_recovery_manifest",
    "tests/test_api_pack_sources_endpoints.py::TestAPIPackSourceEndpoints::test_pack_source_list_search_preview_and_install_handoff",
    "tests/test_api_pack_sources_endpoints.py::TestAPIPackSourceEndpoints::test_pack_source_policy_blocks_list_search_and_preview_consistently",
    "tests/test_api_packs_endpoints.py::TestAPIPacksEndpoints::test_install_native_code_pack_is_blocked_but_recorded",
)

EXTENDED_TEST_NODES: tuple[str, ...] = MAIN_TEST_NODES + (
    "tests/test_packaging_build.py::TestPackagingBuild::test_fresh_wheel_install_entry_points_work_without_repo_path",
    "tests/test_doctor_cli.py",
    "tests/test_runtime_concurrency.py::TestRuntimeConcurrency::test_first_chat_after_deferred_startup_ready_succeeds",
    "tests/test_runtime_concurrency.py::TestRuntimeConcurrency::test_first_chat_and_ready_probe_do_not_race_after_deferred_startup",
)


def _build_pytest_args(test_nodes: tuple[str, ...], *, quiet: bool = True) -> list[str]:
    args: list[str] = ["--maxfail=1"]
    if quiet:
        args.insert(0, "-q")
    args.extend(test_nodes)
    return args


def _print_nodes(name: str, test_nodes: tuple[str, ...]) -> None:
    print(f"{name} ({len(test_nodes)} pytest node(s))", flush=True)
    for node in test_nodes:
        print(node, flush=True)


def _run_suite(name: str, test_nodes: tuple[str, ...], *, list_only: bool = False, quiet: bool = True) -> int:
    os.chdir(ROOT)
    if list_only:
        _print_nodes(name, test_nodes)
        return 0
    print(f"Running {name} from {ROOT}", flush=True)
    return int(pytest.main(_build_pytest_args(test_nodes, quiet=quiet)))


def run_main_suite(*, list_only: bool = False, quiet: bool = True) -> int:
    return _run_suite("release smoke suite", MAIN_TEST_NODES, list_only=list_only, quiet=quiet)


def run_extended_suite(*, list_only: bool = False, quiet: bool = True) -> int:
    return _run_suite("extended release validation suite", EXTENDED_TEST_NODES, list_only=list_only, quiet=quiet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the canonical Personal Agent release-readiness smoke suite.")
    parser.add_argument("--list", action="store_true", help="Print the exact pytest nodes without running them.")
    parser.add_argument("--no-quiet", action="store_true", help="Run pytest without -q.")
    args = parser.parse_args(argv)
    return run_main_suite(list_only=bool(args.list), quiet=not bool(args.no_quiet))


if __name__ == "__main__":
    raise SystemExit(main())
