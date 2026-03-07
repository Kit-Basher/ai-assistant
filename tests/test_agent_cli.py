from __future__ import annotations

import io
import json
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
            "agent.cli.build_system_health_report",
            return_value={"observed": {"warnings": []}, "analysis": {"status": "ok", "warnings": [], "suggestions": []}},
        ), patch(
            "agent.cli.render_system_health_summary",
            return_value="System health\nCPU: ok\nOverall: OK",
        ), redirect_stdout(output):
            code = cli.main(["health_system"])
        self.assertEqual(0, code)
        self.assertEqual("System health\nCPU: ok\nOverall: OK\n", output.getvalue())

    def test_health_system_subcommand_json_outputs_observed_and_analysis(self) -> None:
        output = io.StringIO()
        report = {
            "observed": {"warnings": [], "cpu": {"usage_pct": 12.5}},
            "analysis": {"status": "warn", "warnings": [{"id": "cpu_usage_warn"}], "suggestions": []},
        }
        with patch("agent.cli.collect_system_health", return_value={"warnings": []}), patch(
            "agent.cli.build_system_health_report",
            return_value=report,
        ), redirect_stdout(output):
            code = cli.main(["health_system", "--json"])
        self.assertEqual(0, code)
        payload = json.loads(output.getvalue())
        self.assertEqual(report, payload)

    def test_llm_inventory_subcommand_json(self) -> None:
        output = io.StringIO()
        with patch(
            "agent.cli._load_llm_control_state",
            return_value={
                "inventory": [
                    {
                        "id": "ollama:qwen2.5:3b-instruct",
                        "provider": "ollama",
                        "local": True,
                        "healthy": True,
                    }
                ]
            },
        ), redirect_stdout(output):
            code = cli.main(["llm_inventory", "--json"])
        self.assertEqual(0, code)
        payload = json.loads(output.getvalue())
        self.assertEqual("ollama:qwen2.5:3b-instruct", payload["inventory"][0]["id"])

    def test_llm_inventory_plain_text_prioritizes_useful_rows_and_hides_noise_by_default(self) -> None:
        output = io.StringIO()
        with patch(
            "agent.cli._load_llm_control_state",
            return_value={
                "inventory": [
                    {
                        "id": "openrouter:catalog-noise",
                        "provider": "openrouter",
                        "local": False,
                        "installed": False,
                        "available": True,
                        "healthy": True,
                        "approved": False,
                        "configured": False,
                        "capabilities": ["chat"],
                    },
                    {
                        "id": "ollama:qwen2.5:3b-instruct",
                        "provider": "ollama",
                        "local": True,
                        "installed": True,
                        "available": True,
                        "healthy": True,
                        "approved": True,
                        "configured": True,
                        "capabilities": ["chat"],
                    },
                ]
            },
        ), redirect_stdout(output):
            code = cli.main(["llm_inventory"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("ollama:qwen2.5:3b-instruct", text)
        self.assertIn("configured=true", text)
        self.assertNotIn("openrouter:catalog-noise", text)
        self.assertIn("additional rows hidden; use --all", text)

    def test_llm_select_subcommand_prints_reason_and_selection(self) -> None:
        output = io.StringIO()
        with patch(
            "agent.cli._load_llm_control_state",
            return_value={
                "task_request": {"task_type": "coding", "requirements": ["chat", "json"], "preferred_local": True},
                "selection": {
                    "selected_model": "ollama:qwen2.5:7b-instruct",
                    "provider": "ollama",
                    "reason": "healthy+approved+local_first+task=coding",
                    "fallbacks": ["ollama:qwen2.5:3b-instruct"],
                },
            },
        ), redirect_stdout(output):
            code = cli.main(["llm_select", "--task", "debug this code"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("task_type: coding", text)
        self.assertIn("selected_model: ollama:qwen2.5:7b-instruct", text)
        self.assertIn("reason: healthy+approved+local_first+task=coding", text)

    def test_llm_plan_subcommand_json(self) -> None:
        output = io.StringIO()
        with patch(
            "agent.cli._load_llm_control_state",
            return_value={
                "task_request": {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
                "plan": {
                    "needed": True,
                    "approved": True,
                    "next_action": "Run: python -m agent llm_install --model ollama:qwen2.5:3b-instruct --approve",
                    "plan": [{"id": "01_pull_model", "action": "ollama.pull_model", "model": "qwen2.5:3b-instruct"}],
                },
            },
        ), redirect_stdout(output):
            code = cli.main(["llm_plan", "--task", "hello", "--json"])
        self.assertEqual(0, code)
        payload = json.loads(output.getvalue())
        self.assertTrue(bool(payload["plan"]["needed"]))
        self.assertEqual("ollama.pull_model", payload["plan"]["plan"][0]["action"])

    def test_llm_plan_subcommand_plain_text_shows_candidates_and_install_command(self) -> None:
        output = io.StringIO()
        with patch(
            "agent.cli._load_llm_control_state",
            return_value={
                "task_request": {"task_type": "vision", "requirements": ["chat", "vision"], "preferred_local": True},
                "plan": {
                    "needed": True,
                    "approved": True,
                    "reason": "no_local_model_with_required_capabilities",
                    "install_command": "ollama pull llava:7b",
                    "next_action": "Run: python -m agent llm_install --model ollama:llava:7b --approve",
                    "candidates": [
                        {
                            "model_id": "ollama:llava:7b",
                            "install_name": "llava:7b",
                            "size_hint": "7B",
                            "preferred": True,
                            "reason": "approved local vision baseline",
                        }
                    ],
                    "plan": [{"id": "01_pull_model", "action": "ollama.pull_model", "model": "llava:7b"}],
                },
            },
        ), redirect_stdout(output):
            code = cli.main(["llm_plan", "--task", "analyze this image"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("reason: no_local_model_with_required_capabilities", text)
        self.assertIn("install_command: ollama pull llava:7b", text)
        self.assertIn("candidates:", text)
        self.assertIn("ollama:llava:7b", text)

    def test_llm_install_subcommand_without_approve_shows_preview_only(self) -> None:
        output = io.StringIO()
        plan = {
            "needed": True,
            "approved": True,
            "approval_required": True,
            "reason": "no_local_model_with_required_capabilities",
            "install_command": "ollama pull llava:7b",
            "next_action": "Run: python -m agent llm_install --model ollama:llava:7b --approve",
            "candidates": [{"model_id": "ollama:llava:7b", "install_name": "llava:7b"}],
            "plan": [{"id": "01_pull_model", "action": "ollama.pull_model", "model": "llava:7b"}],
        }
        with patch("agent.cli.load_config"), patch("agent.cli.load_registry"), patch(
            "agent.cli.build_model_inventory", return_value=[]
        ), patch(
            "agent.cli.build_install_plan_for_model", return_value=plan
        ), patch(
            "agent.cli.execute_install_plan"
        ) as execute_mock, redirect_stdout(output):
            code = cli.main(["llm_install", "--model", "ollama:llava:7b"])
        self.assertEqual(0, code)
        execute_mock.assert_not_called()
        text = output.getvalue()
        self.assertIn("approval_required: true", text)
        self.assertIn("model_id: ollama:llava:7b", text)

    def test_llm_install_subcommand_with_approve_executes(self) -> None:
        output = io.StringIO()
        plan = {
            "needed": True,
            "approved": True,
            "approval_required": True,
            "reason": "no_local_model_with_required_capabilities",
            "install_command": "ollama pull llava:7b",
            "next_action": "Run: python -m agent llm_install --model ollama:llava:7b --approve",
            "candidates": [{"model_id": "ollama:llava:7b", "install_name": "llava:7b"}],
            "plan": [{"id": "01_pull_model", "action": "ollama.pull_model", "model": "llava:7b"}],
        }
        result = {
            "ok": True,
            "executed": True,
            "model_id": "ollama:llava:7b",
            "install_name": "llava:7b",
            "trace_id": "cli-install-1",
            "error_kind": None,
            "message": "Installed and verified ollama:llava:7b.",
            "verification": {"found": True, "installed": True, "available": True, "healthy": True},
        }
        with patch("agent.cli.load_config"), patch("agent.cli.load_registry"), patch(
            "agent.cli.build_model_inventory", return_value=[]
        ), patch(
            "agent.cli.build_install_plan_for_model", return_value=plan
        ), patch(
            "agent.cli.execute_install_plan", return_value=result
        ) as execute_mock, redirect_stdout(output):
            code = cli.main(["llm_install", "--model", "ollama:llava:7b", "--approve"])
        self.assertEqual(0, code)
        execute_mock.assert_called_once()
        text = output.getvalue()
        self.assertIn("LLM install result", text)
        self.assertIn("executed: true", text)
        self.assertIn("ollama:llava:7b", text)

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
