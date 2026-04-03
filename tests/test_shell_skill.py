import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agent.shell_skill import ShellSkill


class TestShellSkill(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.safe_root = os.path.join(self.tmpdir.name, "safe")
        self.private_root = os.path.join(self.safe_root, "private")
        self.outside_root = os.path.join(self.tmpdir.name, "outside")
        os.makedirs(self.safe_root, exist_ok=True)
        os.makedirs(self.private_root, exist_ok=True)
        os.makedirs(self.outside_root, exist_ok=True)
        self.skill = ShellSkill(
            allowed_roots=[self.safe_root],
            base_dir=self.safe_root,
            sensitive_roots=[self.private_root],
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_safe_read_only_command_executes_through_allowlist(self) -> None:
        with patch(
            "agent.shell_skill.subprocess.run",
            return_value=subprocess.CompletedProcess(["python", "--version"], 0, "Python 3.12.1\n", ""),
        ) as run_mock:
            result = self.skill.execute_safe_command("python_version")

        self.assertTrue(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("python_version", result["command_name"])
        self.assertEqual(["python", "--version"], result["argv"])
        self.assertIn("Python 3.12.1", result["stdout"])
        run_mock.assert_called_once()

    def test_unsupported_commands_are_rejected_cleanly(self) -> None:
        result = self.skill.execute_safe_command("cat")

        self.assertFalse(result["ok"])
        self.assertEqual("unsupported_command", result["blocked_reason"])

    def test_no_shell_interpolation_arguments_are_allowed(self) -> None:
        result = self.skill.execute_safe_command("which", subject="pip;whoami")

        self.assertFalse(result["ok"])
        self.assertEqual("invalid_argument", result["blocked_reason"])

    def test_command_timeout_is_reported_cleanly(self) -> None:
        with patch(
            "agent.shell_skill.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["python", "--version"], 2.0, output="partial", stderr=""),
        ):
            result = self.skill.execute_safe_command("python_version")

        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual("timeout", result["error_kind"])

    def test_command_output_is_capped_and_marked_truncated(self) -> None:
        oversized = "x" * 5000
        with patch(
            "agent.shell_skill.subprocess.run",
            return_value=subprocess.CompletedProcess(["uname", "-a"], 0, oversized, ""),
        ):
            result = self.skill.execute_safe_command("uname", max_output_chars=128)

        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["stdout"]), 128)

    def test_install_requests_use_structured_install_path(self) -> None:
        with patch(
            "agent.shell_skill.subprocess.run",
            return_value=subprocess.CompletedProcess(["apt-get", "-s", "install", "-y", "ripgrep"], 0, "ok", ""),
        ) as run_mock:
            result = self.skill.install_package(manager="apt", package="ripgrep", dry_run=True)

        self.assertTrue(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual(["apt-get", "-s", "install", "-y", "ripgrep"], result["argv"])
        run_mock.assert_called_once()

    def test_install_requests_reject_invalid_values(self) -> None:
        invalid_manager = self.skill.install_package(manager="brew", package="ripgrep")
        invalid_package = self.skill.install_package(manager="apt", package="ripgrep;rm")

        self.assertFalse(invalid_manager["ok"])
        self.assertEqual("unsupported_manager", invalid_manager["blocked_reason"])
        self.assertFalse(invalid_package["ok"])
        self.assertEqual("invalid_package_name", invalid_package["blocked_reason"])

    def test_create_directory_works_only_inside_allowed_roots(self) -> None:
        inside = self.skill.create_directory("logs")
        outside = self.skill.create_directory("../outside/logs")

        self.assertTrue(inside["ok"])
        self.assertTrue(inside["mutated"])
        self.assertTrue(os.path.isdir(os.path.join(self.safe_root, "logs")))
        self.assertFalse(outside["ok"])
        self.assertEqual("outside_allowed_roots", outside["blocked_reason"])

    def test_create_directory_blocks_sensitive_paths(self) -> None:
        result = self.skill.create_directory(os.path.join(self.private_root, "logs"))

        self.assertFalse(result["ok"])
        self.assertEqual("sensitive_path_blocked", result["blocked_reason"])

    def test_destructive_commands_are_blocked(self) -> None:
        result = self.skill.execute_safe_command("rm")

        self.assertFalse(result["ok"])
        self.assertEqual("destructive_operation_blocked", result["blocked_reason"])
