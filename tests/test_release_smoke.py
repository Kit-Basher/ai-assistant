from __future__ import annotations

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReleaseSmokeScripts(unittest.TestCase):
    def test_release_smoke_lists_expected_core_nodes(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "release_smoke.py", "release_smoke_script")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = module.main(["--list"])
        self.assertEqual(0, exit_code)
        rendered = stdout.getvalue()
        self.assertIn("tests/test_publishability_smoke.py", rendered)
        self.assertIn("tests/test_golden_path_smoke.py", rendered)
        self.assertIn("tests/test_install_local_flow.py", rendered)
        self.assertIn("tests/test_debian_package.py", rendered)
        self.assertIn("tests/test_release_bundle.py", rendered)
        self.assertIn("tests/test_desktop_launcher.py", rendered)
        self.assertIn("tests/test_first_run_release_smoke.py", rendered)
        self.assertIn("tests/test_webui_conversation_smoke.py", rendered)
        self.assertIn(
            "tests/test_memory_hardening.py::TestMemoryHardening::test_memory_status_reports_disabled_components_and_fresh_state",
            rendered,
        )
        self.assertIn("tests/test_assistant_behavior_release_gate.py", rendered)
        self.assertIn(
            "tests/test_api_server.py::TestAPIServerRuntime::test_health_is_explicit_before_and_after_process_restart_with_deferred_warmup",
            rendered,
        )
        self.assertIn(
            "tests/test_doctor_cli.py::TestDoctorCLI::test_collect_diagnostics_writes_redacted_bundle_with_recovery_manifest",
            rendered,
        )
        self.assertIn(
            "tests/test_api_packs_endpoints.py::TestAPIPacksEndpoints::test_install_native_code_pack_is_blocked_but_recorded",
            rendered,
        )

    def test_release_smoke_invokes_pytest_with_fixed_args(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "release_smoke.py", "release_smoke_script_run")
        with patch.object(module.subprocess, "run", return_value=type("Proc", (), {"returncode": 0})()) as run_mock:
            exit_code = module.main([])
        self.assertEqual(0, exit_code)
        run_mock.assert_called()
        args = list(run_mock.call_args.args[0])
        self.assertEqual(module.sys.executable, args[0])
        self.assertEqual("-m", args[1])
        self.assertEqual("pytest", args[2])
        self.assertEqual("-q", args[3])
        self.assertEqual("--maxfail=1", args[4])
        self.assertEqual(list(module.MAIN_TEST_NODES), args[5:])

    def test_extended_release_validation_wraps_main_plus_heavier_nodes(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "release_smoke.py", "release_smoke_script_extended")
        self.assertGreater(len(module.EXTENDED_TEST_NODES), len(module.MAIN_TEST_NODES))
        self.assertIn(
            "tests/test_packaging_build.py::TestPackagingBuild::test_fresh_wheel_install_entry_points_work_without_repo_path",
            module.EXTENDED_TEST_NODES,
        )
        self.assertIn(
            "tests/test_runtime_concurrency.py::TestRuntimeConcurrency::test_first_chat_after_deferred_startup_ready_succeeds",
            module.EXTENDED_TEST_NODES,
        )
        self.assertIn(
            "tests/test_runtime_concurrency.py::TestRuntimeConcurrency::test_first_chat_and_ready_probe_do_not_race_after_deferred_startup",
            module.EXTENDED_TEST_NODES,
        )
        self.assertIn("tests/test_clean_context_validation.py", module.EXTENDED_TEST_NODES)
        self.assertIn("tests/test_extended_soak.py", module.EXTENDED_TEST_NODES)
        self.assertIn("tests/test_concurrency_stress.py", module.EXTENDED_TEST_NODES)
