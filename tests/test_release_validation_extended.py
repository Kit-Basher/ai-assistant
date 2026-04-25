from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReleaseValidationExtended(unittest.TestCase):
    def test_release_validation_extended_invokes_restart_churn_smoke_when_enabled(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "release_validation_extended.py", "release_validation_extended_script")
        calls: list[list[str]] = []

        def _fake_live_product_smoke(args):  # noqa: ANN001
            calls.append(["live_product_smoke", *list(args)])
            return 0

        @contextmanager
        def _fake_temp_live_api_server():  # noqa: ANN001
            yield "http://127.0.0.1:54321"

        def _fake_split_smoke(args):  # noqa: ANN001
            calls.append(["split_smoke", *list(args)])
            return 0

        def _fake_provider_matrix_smoke(args):  # noqa: ANN001
            calls.append(["provider_matrix_smoke", *list(args)])
            return 0

        def _fake_restart_memory_smoke(args):  # noqa: ANN001
            calls.append(["restart_memory_smoke", *list(args)])
            return 0

        def _fake_assistant_real_world_smoke(args):  # noqa: ANN001
            calls.append(["assistant_real_world_smoke", *list(args)])
            return 0

        def _fake_assistant_interaction_barrage(args):  # noqa: ANN001
            calls.append(["assistant_interaction_barrage", *list(args)])
            return 0

        def _fake_assistant_viability_smoke(args):  # noqa: ANN001
            calls.append(["assistant_viability_smoke", *list(args)])
            return 0

        @contextmanager
        def _fake_temp_live_api_server():  # noqa: ANN001
            yield "http://127.0.0.1:54321"

        with patch.object(module, "run_extended_suite", return_value=0), patch.object(
            module,
            "run_live_product_smoke",
            side_effect=_fake_live_product_smoke,
        ), patch.object(
            module,
            "run_provider_matrix_smoke",
            side_effect=_fake_provider_matrix_smoke,
        ), patch.object(
            module,
            "run_restart_memory_smoke",
            side_effect=_fake_restart_memory_smoke,
        ), patch.object(
            module,
            "run_assistant_real_world_smoke",
            side_effect=_fake_assistant_real_world_smoke,
        ), patch.object(
            module,
            "run_assistant_interaction_barrage",
            side_effect=_fake_assistant_interaction_barrage,
        ), patch.object(
            module,
            "run_assistant_viability_smoke",
            side_effect=_fake_assistant_viability_smoke,
        ), patch.object(
            module,
            "run_split_smoke",
            side_effect=_fake_split_smoke,
        ), patch.object(
            module,
            "_temp_live_api_server",
            side_effect=_fake_temp_live_api_server,
        ), patch.object(
            module,
            "_stable_split_smoke_available",
            return_value=True,
        ):
            exit_code = module.main(["--with-live-smokes"])

        self.assertEqual(0, exit_code)
        self.assertEqual(["restart_memory_smoke", "--cycles", "3"], calls[0])
        self.assertEqual(["provider_matrix_smoke", "--base-url", "http://127.0.0.1:54321"], calls[1])
        self.assertEqual(["assistant_real_world_smoke", "--base-url", "http://127.0.0.1:54321"], calls[2])
        self.assertEqual(["assistant_interaction_barrage", "--base-url", "http://127.0.0.1:54321"], calls[3])
        self.assertEqual(
            [
                "assistant_viability_smoke",
                "--base-url",
                "http://127.0.0.1:54321",
                "--timeout",
                "180",
                "--retry-attempts",
                "2",
                "--surface",
                "webui",
                "--scenario",
                "long_human_like_session_webui",
            ],
            calls[4],
        )
        self.assertEqual(["live_product_smoke", "--base-url", "http://127.0.0.1:54321"], calls[5])
        self.assertEqual(["split_smoke"], calls[6])

    def test_release_validation_extended_skips_live_smokes_when_not_requested(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "release_validation_extended.py", "release_validation_extended_script_skip")
        with patch.object(module, "run_extended_suite", return_value=0), patch.object(
            module,
            "run_restart_memory_smoke",
            side_effect=AssertionError("restart smoke should not run without --with-live-smokes"),
        ), patch.object(
            module,
            "run_provider_matrix_smoke",
            side_effect=AssertionError("provider matrix smoke should not run without --with-live-smokes"),
        ), patch.object(
            module,
            "run_assistant_real_world_smoke",
            side_effect=AssertionError("assistant real-world smoke should not run without --with-live-smokes"),
        ), patch.object(
            module,
            "run_assistant_interaction_barrage",
            side_effect=AssertionError("assistant barrage should not run without --with-live-smokes"),
        ), patch.object(
            module,
            "run_assistant_viability_smoke",
            side_effect=AssertionError("assistant viability smoke should not run without --with-live-smokes"),
        ), patch.object(
            module,
            "run_live_product_smoke",
            side_effect=AssertionError("live smoke should not run without --with-live-smokes"),
        ):
            exit_code = module.main([])

        self.assertEqual(0, exit_code)

    def test_release_validation_extended_skips_split_smoke_when_stable_service_is_inactive(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "release_validation_extended.py", "release_validation_extended_script_split_skip")
        calls: list[list[str]] = []

        def _fake_restart_memory_smoke(args):  # noqa: ANN001
            calls.append(["restart_memory_smoke", *list(args)])
            return 0

        def _fake_provider_matrix_smoke(args):  # noqa: ANN001
            calls.append(["provider_matrix_smoke", *list(args)])
            return 0

        def _fake_assistant_real_world_smoke(args):  # noqa: ANN001
            calls.append(["assistant_real_world_smoke", *list(args)])
            return 0

        def _fake_assistant_interaction_barrage(args):  # noqa: ANN001
            calls.append(["assistant_interaction_barrage", *list(args)])
            return 0

        def _fake_assistant_viability_smoke(args):  # noqa: ANN001
            calls.append(["assistant_viability_smoke", *list(args)])
            return 0

        def _fake_live_product_smoke(args):  # noqa: ANN001
            calls.append(["live_product_smoke", *list(args)])
            return 0

        @contextmanager
        def _fake_temp_live_api_server():  # noqa: ANN001
            yield "http://127.0.0.1:54321"

        with patch.object(module, "run_extended_suite", return_value=0), patch.object(
            module,
            "run_restart_memory_smoke",
            side_effect=_fake_restart_memory_smoke,
        ), patch.object(
            module,
            "run_provider_matrix_smoke",
            side_effect=_fake_provider_matrix_smoke,
        ), patch.object(
            module,
            "run_assistant_real_world_smoke",
            side_effect=_fake_assistant_real_world_smoke,
        ), patch.object(
            module,
            "run_assistant_interaction_barrage",
            side_effect=_fake_assistant_interaction_barrage,
        ), patch.object(
            module,
            "run_assistant_viability_smoke",
            side_effect=_fake_assistant_viability_smoke,
        ), patch.object(
            module,
            "run_live_product_smoke",
            side_effect=_fake_live_product_smoke,
        ), patch.object(
            module,
            "run_split_smoke",
            side_effect=AssertionError("split smoke should not run without an active stable service"),
        ), patch.object(
            module,
            "_stable_split_smoke_available",
            return_value=False,
        ), patch.object(
            module,
            "_temp_live_api_server",
            side_effect=_fake_temp_live_api_server,
        ):
            exit_code = module.main(["--with-live-smokes"])

        self.assertEqual(0, exit_code)
        self.assertEqual(["restart_memory_smoke", "--cycles", "3"], calls[0])
        self.assertEqual(["provider_matrix_smoke", "--base-url", "http://127.0.0.1:54321"], calls[1])
        self.assertEqual(["assistant_real_world_smoke", "--base-url", "http://127.0.0.1:54321"], calls[2])
        self.assertEqual(["assistant_interaction_barrage", "--base-url", "http://127.0.0.1:54321"], calls[3])
        self.assertEqual(
            [
                "assistant_viability_smoke",
                "--base-url",
                "http://127.0.0.1:54321",
                "--timeout",
                "180",
                "--retry-attempts",
                "2",
                "--surface",
                "webui",
                "--scenario",
                "long_human_like_session_webui",
            ],
            calls[4],
        )
        self.assertEqual(["live_product_smoke", "--base-url", "http://127.0.0.1:54321"], calls[5])


if __name__ == "__main__":
    unittest.main()
