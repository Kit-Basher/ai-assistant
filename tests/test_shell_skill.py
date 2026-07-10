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

    def test_debian_package_state_uses_exact_dpkg_query_without_shell(self) -> None:
        with patch(
            "agent.shell_skill.subprocess.run",
            return_value=subprocess.CompletedProcess(["/usr/bin/dpkg-query"], 0, "installed\n", ""),
        ) as run_mock:
            result = self.skill.debian_package_state("htop")

        self.assertTrue(result["ok"])
        self.assertTrue(result["installed"])
        argv = run_mock.call_args.args[0]
        self.assertEqual("/usr/bin/dpkg-query", argv[0])
        self.assertFalse(run_mock.call_args.kwargs.get("shell", False))
        self.assertLessEqual(run_mock.call_args.kwargs["timeout"], 2.0)

    def test_package_preview_adds_state_without_mutation_or_apt_update(self) -> None:
        with patch(
            "agent.shell_skill.subprocess.run",
            return_value=subprocess.CompletedProcess(["/usr/bin/dpkg-query"], 1, "", "no packages found matching htop\n"),
        ) as run_mock:
            result = self.skill.preview_install_package(manager="apt", package="htop")

        self.assertTrue(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("unknown_package", result["package_state"]["state"])
        calls = [" ".join(call.args[0]) for call in run_mock.call_args_list]
        self.assertFalse(any("apt update" in call or "apt-get update" in call for call in calls))

    def test_package_state_cache_reuses_and_invalidates(self) -> None:
        with patch(
            "agent.shell_skill.subprocess.run",
            return_value=subprocess.CompletedProcess(["/usr/bin/dpkg-query"], 0, "installed\n", ""),
        ) as run_mock:
            first = self.skill.debian_package_state("htop")
            second = self.skill.debian_package_state("htop")
            self.skill.invalidate_package_state_cache()
            third = self.skill.debian_package_state("htop")

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertFalse(third["cached"])
        self.assertEqual(2, run_mock.call_count)

    def test_install_requests_reject_invalid_values(self) -> None:
        invalid_manager = self.skill.install_package(manager="brew", package="ripgrep")
        invalid_package = self.skill.install_package(manager="apt", package="ripgrep;rm")

        self.assertFalse(invalid_manager["ok"])
        self.assertEqual("unsupported_manager", invalid_manager["blocked_reason"])
        self.assertFalse(invalid_package["ok"])
        self.assertEqual("invalid_package_name", invalid_package["blocked_reason"])

    def test_install_mutation_requires_trusted_invocation_context(self) -> None:
        with patch("agent.shell_skill.subprocess.run") as run_mock:
            result = self.skill.install_package(manager="apt", package="ripgrep")

        self.assertFalse(result["ok"])
        self.assertFalse(result["mutated"])
        self.assertEqual("generic_bypass_blocked", result["blocked_reason"])
        run_mock.assert_not_called()

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
