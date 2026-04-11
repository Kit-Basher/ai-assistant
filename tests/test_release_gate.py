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


class TestReleaseGate(unittest.TestCase):
    def test_release_gate_lists_expected_commands(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "release_gate.py", "release_gate_script")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = module.main(["--list"])
        self.assertEqual(0, exit_code)
        rendered = stdout.getvalue()
        self.assertIn("python -m py_compile", rendered)
        self.assertIn("python -m pytest -q --maxfail=1 tests/test_publishability_smoke.py", rendered)
        self.assertIn("tests/test_clean_context_validation.py", rendered)
        self.assertIn("git diff --check", rendered)

    def test_release_gate_runs_commands_in_order(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "release_gate.py", "release_gate_script_run")
        calls: list[list[str]] = []

        def _fake_run(command, cwd=None, check=False):  # noqa: ANN001
            _ = cwd
            _ = check
            calls.append(list(command))

            class _Result:
                returncode = 0

            return _Result()

        with patch.object(module.subprocess, "run", side_effect=_fake_run):
            exit_code = module.main([])

        self.assertEqual(0, exit_code)
        self.assertEqual(len(module.RELEASE_GATE_COMMANDS), len(calls))
        self.assertTrue(calls[0][0].endswith("python") or "python" in calls[0][0])
        self.assertEqual(["git", "diff", "--check"], calls[-1])


if __name__ == "__main__":
    unittest.main()
