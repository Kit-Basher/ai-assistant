from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent import cli
from agent.setup_wizard import SetupWizardResult


class TestAgentCLI(unittest.TestCase):
    def test_doctor_subcommand_forwards_args(self) -> None:
        with patch("agent.cli.doctor_main", return_value=0) as doctor_main:
            code = cli.main(["doctor", "--json"])
        self.assertEqual(0, code)
        doctor_main.assert_called_once_with(["--json"])

    def test_status_subcommand_uses_ready_payload(self) -> None:
        payload = {
            "ok": True,
            "ready": True,
            "message": "Ready.",
            "telegram": {"state": "running"},
        }
        output = io.StringIO()
        with patch("agent.cli._http_json", return_value=(True, payload)), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("Agent is ready.", text)
        self.assertIn("telegram: running", text)
        self.assertIn("message: Ready.", text)

    def test_health_subcommand_failure_returns_structured_error(self) -> None:
        output = io.StringIO()
        with patch("agent.cli._http_json", return_value=(False, "boom")), redirect_stdout(output):
            code = cli.main(["health"])
        self.assertEqual(1, code)
        text = output.getvalue()
        self.assertIn("trace_id:", text)
        self.assertIn("component: agent.cli.health", text)
        self.assertIn("next_action: run `agent doctor`", text)

    def test_health_system_subcommand_uses_local_skill_summary(self) -> None:
        output = io.StringIO()
        with patch("agent.cli.collect_system_health", return_value={"warnings": []}), patch(
            "agent.cli.render_system_health_summary",
            return_value="System health\nCPU: ok",
        ), redirect_stdout(output):
            code = cli.main(["health_system"])
        self.assertEqual(0, code)
        self.assertEqual("System health\nCPU: ok\n", output.getvalue())

    def test_logs_subcommand_tails_last_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.log"
            path.write_text("\n".join(f"line-{idx}" for idx in range(1, 11)) + "\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = cli.main(["logs", "--path", str(path), "--lines", "3"])
        self.assertEqual(0, code)
        text = output.getvalue().strip()
        self.assertIn("Showing last 3 lines from", text)
        self.assertTrue(text.endswith("line-8\nline-9\nline-10"))

    def test_version_subcommand_prints_version_and_commit(self) -> None:
        output = io.StringIO()
        with patch("agent.cli._resolve_git_commit", return_value="abc1234"), redirect_stdout(output):
            code = cli.main(["version"])
        self.assertEqual(0, code)
        text = output.getvalue().strip()
        self.assertIn("version=", text)
        self.assertIn("commit=abc1234", text)

    def test_memory_subcommand_prints_summary(self) -> None:
        payload = {"ok": True, "message": "Memory summary (thread user:1):\nPending items: 0"}
        output = io.StringIO()
        with patch("agent.cli._http_json", return_value=(True, payload)), redirect_stdout(output):
            code = cli.main(["memory"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("Memory summary", text)
        self.assertIn("Pending items: 0", text)

    def test_main_uses_sys_argv_for_doctor_when_argv_none(self) -> None:
        with patch.object(sys, "argv", ["python -m agent", "doctor", "--json"]):
            with patch("agent.cli.doctor_main", return_value=0) as doctor_main:
                code = cli.main(None)
        self.assertEqual(0, code)
        doctor_main.assert_called_once_with(["--json"])

    def test_main_uses_sys_argv_for_status_when_argv_none(self) -> None:
        payload = {
            "ok": True,
            "ready": True,
            "message": "Ready.",
            "telegram": {"state": "running"},
        }
        output = io.StringIO()
        with patch.object(sys, "argv", ["python -m agent", "status"]):
            with patch("agent.cli._http_json", return_value=(True, payload)), redirect_stdout(output):
                code = cli.main(None)
        self.assertEqual(0, code)
        self.assertIn("Agent is ready.", output.getvalue())

    def test_main_uses_sys_argv_for_version_when_argv_none(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["python -m agent", "version"]):
            with patch("agent.cli._resolve_git_commit", return_value="abc1234"), redirect_stdout(output):
                code = cli.main(None)
        self.assertEqual(0, code)
        text = output.getvalue().strip()
        self.assertIn("version=", text)
        self.assertIn("commit=abc1234", text)

    def test_setup_subcommand_prints_readable_output(self) -> None:
        report = SetupWizardResult(
            trace_id="setup-1",
            generated_at="2026-03-06T00:00:00+00:00",
            onboarding_state="TOKEN_MISSING",
            recovery_mode="TOKEN_INVALID",
            summary="Telegram bot token is missing.",
            why="Telegram token is missing from secret store or environment.",
            next_action="Run: python -m agent.secrets set telegram:bot_token",
            steps=[
                "Run: python -m agent.secrets set telegram:bot_token",
                "Run: systemctl --user restart personal-agent-telegram.service",
                "Run: python -m agent status",
            ],
            suggestions=[
                "python -m agent.secrets set telegram:bot_token",
                "systemctl --user restart personal-agent-telegram.service",
            ],
            dry_run=False,
            api_reachable=True,
        )
        output = io.StringIO()
        with patch("agent.cli.run_setup_wizard", return_value=report), redirect_stdout(output):
            code = cli.main(["setup"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("1) State: TOKEN_MISSING", text)
        self.assertIn("3) Next action:", text)
        self.assertIn("trace_id: setup-1", text)

    def test_setup_subcommand_json_output(self) -> None:
        report = SetupWizardResult(
            trace_id="setup-2",
            generated_at="2026-03-06T00:00:00+00:00",
            onboarding_state="READY",
            recovery_mode="UNKNOWN_FAILURE",
            summary="Setup complete. The agent is ready.",
            why="All required services and chat model checks are healthy.",
            next_action="No action needed.",
            steps=["Use Telegram naturally."],
            suggestions=[],
            dry_run=False,
            api_reachable=True,
        )
        output = io.StringIO()
        with patch("agent.cli.run_setup_wizard", return_value=report), redirect_stdout(output):
            code = cli.main(["setup", "--json"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn('"onboarding_state": "READY"', text)
        self.assertIn('"trace_id": "setup-2"', text)

    def test_setup_subcommand_dry_run_flag_is_passed(self) -> None:
        report = SetupWizardResult(
            trace_id="setup-3",
            generated_at="2026-03-06T00:00:00+00:00",
            onboarding_state="NOT_STARTED",
            recovery_mode="UNKNOWN_FAILURE",
            summary="Setup has not started.",
            why="Setup has not been completed yet.",
            next_action="Run: python -m agent setup",
            steps=["Run: python -m agent setup"],
            suggestions=["python -m agent setup --dry-run"],
            dry_run=True,
            api_reachable=False,
        )
        output = io.StringIO()
        with patch("agent.cli.run_setup_wizard", return_value=report) as run_setup_wizard:
            with redirect_stdout(output):
                code = cli.main(["setup", "--dry-run"])
        self.assertEqual(0, code)
        run_setup_wizard.assert_called_once()
        kwargs = run_setup_wizard.call_args.kwargs
        self.assertTrue(bool(kwargs.get("dry_run")))
        self.assertIn("Dry-run: no changes were applied.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
