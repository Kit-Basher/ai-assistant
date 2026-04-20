from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent import cli
from agent.setup_wizard import SetupWizardResult
from agent.version import BuildInfo


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
        with patch("agent.cli._load_ready_status_payload", return_value=(True, payload)), patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(False, "runtime_unavailable"),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(False, "runtime_history_unavailable"),
        ), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("Ready.", text)
        self.assertIn("telegram: enabled_running", text)
        self.assertIn("message: Ready.", text)

    def test_status_subcommand_ready_local_provider_payload_stays_consistent(self) -> None:
        payload = {
            "ok": True,
            "ready": True,
            "message": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            "runtime_mode": "READY",
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            },
            "telegram": {"state": "running"},
            "llm": {"provider": "ollama", "model": "ollama:qwen2.5:7b-instruct"},
        }
        output = io.StringIO()
        with patch("agent.cli._load_ready_status_payload", return_value=(True, payload)), patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(False, "runtime_unavailable"),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(False, "runtime_history_unavailable"),
        ), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("runtime_mode: READY", text)
        self.assertIn("ollama:qwen2.5:7b-instruct", text)
        self.assertNotIn("provider unavailable", text.lower())

    def test_status_subcommand_ready_remote_provider_payload_stays_consistent(self) -> None:
        payload = {
            "ok": True,
            "ready": True,
            "message": "Ready. Using openrouter / openrouter:ai21/jamba-large-1.7.",
            "runtime_mode": "READY",
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using openrouter / openrouter:ai21/jamba-large-1.7.",
            },
            "telegram": {"state": "running"},
            "llm": {"provider": "openrouter", "model": "openrouter:ai21/jamba-large-1.7"},
        }
        output = io.StringIO()
        with patch("agent.cli._load_ready_status_payload", return_value=(True, payload)), patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(False, "runtime_unavailable"),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(False, "runtime_history_unavailable"),
        ), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("runtime_mode: READY", text)
        self.assertIn("openrouter:ai21/jamba-large-1.7", text)
        self.assertNotIn("provider unavailable", text.lower())

    def test_status_subcommand_retries_ready_timeout_before_failing(self) -> None:
        payload = {
            "ok": True,
            "ready": True,
            "message": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            "runtime_mode": "READY",
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            },
            "telegram": {"state": "running"},
            "llm": {"provider": "ollama", "model": "ollama:qwen2.5:7b-instruct"},
        }
        output = io.StringIO()
        with patch(
            "agent.cli._load_ready_status_payload",
            return_value=(True, payload),
        ) as ready_mock, patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(False, "runtime_unavailable"),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(False, "runtime_history_unavailable"),
        ), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        ready_mock.assert_called_once()
        text = output.getvalue()
        self.assertIn("runtime_mode: READY", text)
        self.assertNotIn("Agent status unavailable", text)

    def test_status_subcommand_failure_returns_agent_status_unavailable(self) -> None:
        output = io.StringIO()
        with patch("agent.cli._load_ready_status_payload", return_value=(False, "boom")), patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(False, "runtime_unavailable"),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(False, "runtime_history_unavailable"),
        ), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(1, code)
        text = output.getvalue()
        self.assertIn("Agent status unavailable", text)
        self.assertIn("component: agent.cli.status", text)

    def test_status_subcommand_uses_runtime_endpoint_when_available(self) -> None:
        ready_payload = {
            "ok": True,
            "ready": True,
            "message": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            "runtime_mode": "READY",
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            },
            "telegram": {"state": "running"},
            "llm": {"provider": "ollama", "model": "ollama:qwen2.5:7b-instruct"},
        }
        runtime_payload = {
            "ok": True,
            "phase": "ready",
            "default_chat_model": "ollama:qwen2.5:7b-instruct",
            "health_summary": {"ok": 1, "degraded": 0, "down": 0},
        }
        output = io.StringIO()
        with patch("agent.cli._load_ready_status_payload", return_value=(True, ready_payload)), patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(True, runtime_payload),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(False, "runtime_history_unavailable"),
        ), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("phase: ready", text)
        self.assertIn("default_model: ollama:qwen2.5:7b-instruct", text)
        self.assertIn("provider_health: ok=1 degraded=0 down=0", text)

    def test_status_subcommand_shows_recent_runtime_events_when_available(self) -> None:
        ready_payload = {
            "ok": True,
            "ready": True,
            "message": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            "runtime_mode": "READY",
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            },
            "telegram": {"state": "running"},
            "llm": {"provider": "ollama", "model": "ollama:qwen2.5:7b-instruct"},
        }
        runtime_history_payload = {
            "ok": True,
            "events": [
                {"event": "runtime_phase_change", "phase_from": "warmup", "phase_to": "ready"},
                {"event": "default_model_change", "new_model": "ollama:qwen2.5:7b-instruct"},
                {
                    "event": "provider_health_transition",
                    "provider": "ollama",
                    "old_status": "degraded",
                    "new_status": "ok",
                },
            ],
        }
        output = io.StringIO()
        with patch("agent.cli._load_ready_status_payload", return_value=(True, ready_payload)), patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(False, "runtime_unavailable"),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(True, runtime_history_payload),
        ), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("Runtime events:", text)
        self.assertIn("- warmup -> ready", text)
        self.assertIn("- default model set: ollama:qwen2.5:7b-instruct", text)
        self.assertIn("- ollama health: degraded -> ok", text)

    def test_load_ready_status_payload_uses_direct_http_for_loopback(self) -> None:
        payload = {"ok": True, "ready": True, "message": "Ready."}
        response = unittest.mock.Mock()
        response.status = 200
        response.reason = "OK"
        response.read.return_value = json.dumps(payload).encode("utf-8")
        connection = unittest.mock.Mock()
        connection.getresponse.return_value = response
        with patch("agent.cli.http.client.HTTPConnection", return_value=connection) as connection_cls:
            ok, payload_or_error = cli._load_ready_status_payload(base_url="http://127.0.0.1:8765")
        self.assertTrue(ok)
        self.assertEqual(payload, payload_or_error)
        connection_cls.assert_called_once_with("127.0.0.1", port=8765, timeout=6.0)
        connection.request.assert_called_once_with("GET", "/ready", body=None, headers={})

    def test_load_ready_status_payload_normalizes_localhost_to_127(self) -> None:
        payload = {"ok": True, "ready": True, "message": "Ready."}
        response = unittest.mock.Mock()
        response.status = 200
        response.reason = "OK"
        response.read.return_value = json.dumps(payload).encode("utf-8")
        connection = unittest.mock.Mock()
        connection.getresponse.return_value = response
        with patch("agent.cli.http.client.HTTPConnection", return_value=connection) as connection_cls:
            ok, payload_or_error = cli._load_ready_status_payload(base_url="http://localhost:8765")
        self.assertTrue(ok)
        self.assertEqual(payload, payload_or_error)
        connection_cls.assert_called_once_with("127.0.0.1", port=8765, timeout=6.0)

    def test_load_ready_status_payload_retries_timeout_then_succeeds_via_direct_http(self) -> None:
        payload = {"ok": True, "ready": True, "message": "Ready."}
        success_response = unittest.mock.Mock()
        success_response.status = 200
        success_response.reason = "OK"
        success_response.read.return_value = json.dumps(payload).encode("utf-8")
        timeout_connection = unittest.mock.Mock()
        timeout_connection.request.side_effect = TimeoutError("timed out")
        success_connection = unittest.mock.Mock()
        success_connection.getresponse.return_value = success_response
        with patch(
            "agent.cli.http.client.HTTPConnection",
            side_effect=[
                timeout_connection,
                success_connection,
            ],
        ) as connection_cls:
            ok, payload_or_error = cli._load_ready_status_payload(base_url="http://127.0.0.1:8765")
        self.assertTrue(ok)
        self.assertEqual(payload, payload_or_error)
        self.assertEqual(2, connection_cls.call_count)

    def test_load_ready_status_payload_returns_json_error_for_invalid_json(self) -> None:
        response = unittest.mock.Mock()
        response.status = 200
        response.reason = "OK"
        response.read.return_value = b"{not-json"
        connection = unittest.mock.Mock()
        connection.getresponse.return_value = response
        with patch("agent.cli.http.client.HTTPConnection", return_value=connection):
            ok, payload_or_error = cli._load_ready_status_payload(base_url="http://127.0.0.1:8765")
        self.assertFalse(ok)
        self.assertEqual("json_error:JSONDecodeError", payload_or_error)

    def test_load_ready_status_payload_logs_concrete_transport_error(self) -> None:
        with patch(
            "agent.cli.http.client.HTTPConnection",
            side_effect=ConnectionRefusedError("connection refused"),
        ), self.assertLogs("agent.cli", level="DEBUG") as logs:
            ok, payload_or_error = cli._load_ready_status_payload(base_url="http://127.0.0.1:8765")
        self.assertFalse(ok)
        self.assertEqual("ConnectionRefusedError:connection refused", payload_or_error)
        self.assertTrue(any("transport=direct_http_loopback" in line for line in logs.output))
        self.assertTrue(any("ConnectionRefusedError:connection refused" in line for line in logs.output))

    def test_status_subcommand_end_to_end_uses_direct_http_ready_path(self) -> None:
        payload = {
            "ok": True,
            "ready": True,
            "message": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            "runtime_mode": "READY",
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using ollama / ollama:qwen2.5:7b-instruct.",
            },
            "llm": {"provider": "ollama", "model": "ollama:qwen2.5:7b-instruct"},
        }
        response = unittest.mock.Mock()
        response.status = 200
        response.reason = "OK"
        response.read.return_value = json.dumps(payload).encode("utf-8")
        connection = unittest.mock.Mock()
        connection.getresponse.return_value = response
        output = io.StringIO()
        with patch(
            "agent.cli.http.client.HTTPConnection",
            return_value=connection,
        ) as connection_cls, patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(False, "runtime_unavailable"),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(False, "runtime_history_unavailable"),
        ), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ), redirect_stdout(output):
            code = cli.main(["status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("runtime_mode: READY", text)
        self.assertIn("ollama:qwen2.5:7b-instruct", text)
        connection_cls.assert_called_once_with("127.0.0.1", port=8765, timeout=6.0)

    def test_telegram_status_command_reports_disabled_optional(self) -> None:
        output = io.StringIO()
        with patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={
                "enabled": False,
                "config_source": "default",
                "service_installed": True,
                "service_active": False,
                "token_configured": False,
                "lock_present": False,
                "effective_state": "disabled_optional",
                "next_action": "Run: python -m agent telegram_enable",
            },
        ), redirect_stdout(output):
            code = cli.main(["telegram_status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("enabled: false", text)
        self.assertIn("effective_state: disabled_optional", text)

    def test_telegram_enable_command_starts_service(self) -> None:
        output = io.StringIO()
        states = [
            {
                "enabled": True,
                "service_installed": True,
                "service_active": False,
                "token_configured": True,
                "effective_state": "enabled_stopped",
                "next_action": "Run: python -m agent telegram_enable",
            },
            {
                "enabled": True,
                "config_source": "config",
                "service_installed": True,
                "service_active": True,
                "token_configured": True,
                "lock_present": False,
                "effective_state": "enabled_running",
                "next_action": "No action needed.",
            },
        ]
        with patch("agent.cli.write_telegram_enablement"), patch(
            "agent.cli.get_telegram_runtime_state",
            side_effect=states,
        ), patch(
            "agent.cli.resolve_telegram_token_with_source",
            return_value=("123:token", "secret_store"),
        ), patch(
            "agent.cli.clear_stale_telegram_locks",
            return_value=[],
        ), patch(
            "agent.cli._run_systemctl_user",
        ) as systemctl_mock, redirect_stdout(output):
            code = cli.main(["telegram_enable"])
        self.assertEqual(0, code)
        commands = [call.args[0] for call in systemctl_mock.call_args_list]
        self.assertIn(["daemon-reload"], commands)
        self.assertIn(["restart", "personal-agent-telegram.service"], commands)
        self.assertIn("effective_state: enabled_running", output.getvalue())

    def test_run_systemctl_user_retries_once_after_timeout(self) -> None:
        calls: list[float | None] = []

        def _runner(args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs.get("timeout"))
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with patch("agent.cli.subprocess.run", side_effect=_runner):
            result = cli._run_systemctl_user(["restart", "personal-agent-telegram.service"], timeout_seconds=0.1)

        self.assertEqual(0, result.returncode)
        self.assertGreaterEqual(len(calls), 2)

    def test_telegram_disable_command_stops_service(self) -> None:
        output = io.StringIO()
        states = [
            {
                "enabled": False,
                "service_installed": True,
                "service_active": True,
                "token_configured": True,
                "effective_state": "enabled_running",
                "next_action": "No action needed.",
            },
            {
                "enabled": False,
                "config_source": "config",
                "service_installed": True,
                "service_active": False,
                "token_configured": True,
                "lock_present": False,
                "effective_state": "disabled_optional",
                "next_action": "Run: python -m agent telegram_enable",
            },
        ]
        with patch("agent.cli.write_telegram_enablement"), patch(
            "agent.cli.get_telegram_runtime_state",
            side_effect=states,
        ), patch(
            "agent.cli._run_systemctl_user",
        ) as systemctl_mock, redirect_stdout(output):
            code = cli.main(["telegram_disable"])
        self.assertEqual(0, code)
        commands = [call.args[0] for call in systemctl_mock.call_args_list]
        self.assertIn(["daemon-reload"], commands)
        self.assertIn(["stop", "personal-agent-telegram.service"], commands)
        self.assertIn("effective_state: disabled_optional", output.getvalue())

    def test_health_subcommand_failure_returns_structured_error(self) -> None:
        output = io.StringIO()
        with patch("agent.cli._http_json", return_value=(False, "boom")), redirect_stdout(output):
            code = cli.main(["health"])
        self.assertEqual(1, code)
        text = output.getvalue()
        self.assertIn("trace_id:", text)
        self.assertIn("component: agent.cli.health", text)
        self.assertIn("next_action: run `agent doctor`", text)

    def test_health_subcommand_prefers_ready_embedded_llm_payload(self) -> None:
        output = io.StringIO()

        def _fetch(*, base_url: str, path: str, timeout_seconds: float) -> tuple[bool, dict[str, object] | str]:
            _ = (base_url, timeout_seconds)
            if path == "/ready":
                return True, {
                    "ok": True,
                    "llm": {
                        "default_provider": "ollama",
                        "default_model": "ollama:qwen2.5:3b-instruct",
                        "resolved_default_model": "ollama:qwen2.5:3b-instruct",
                        "active_provider_health": {"status": "ok"},
                        "active_model_health": {"status": "ok"},
                    },
                }
            if path == "/llm/status":
                raise AssertionError("cli health should use /ready llm before /llm/status")
            raise AssertionError(path)

        with patch("agent.cli._http_json", side_effect=_fetch), redirect_stdout(output):
            code = cli.main(["health"])

        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("provider=ollama", text)
        self.assertIn("model=ollama:qwen2.5:3b-instruct", text)

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
            "agent.cli._execute_llm_install_via_model_manager"
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
            "agent.cli._execute_llm_install_via_model_manager", return_value=result
        ) as execute_mock, redirect_stdout(output):
            code = cli.main(["llm_install", "--model", "ollama:llava:7b", "--approve"])
        self.assertEqual(0, code)
        execute_mock.assert_called_once()
        text = output.getvalue()
        self.assertIn("LLM install result", text)
        self.assertIn("executed: true", text)
        self.assertIn("ollama:llava:7b", text)

    def test_execute_llm_install_via_model_manager_routes_approved_plan_through_runtime_manager(self) -> None:
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
        expected = {
            "ok": True,
            "executed": True,
            "model_id": "ollama:llava:7b",
            "install_name": "llava:7b",
            "trace_id": "cli-install-1",
            "error_kind": None,
            "message": "Installed and verified ollama:llava:7b.",
            "verification": {"found": True, "installed": True, "available": True, "healthy": True},
        }

        class _FakeManager:
            def __init__(self) -> None:
                self.called_with: tuple[tuple[object, ...], dict[str, object]] | None = None

            def execute_request(self, *args: object, **kwargs: object) -> dict[str, object]:
                self.called_with = (args, kwargs)
                return dict(expected)

        class _FakeRuntime:
            last_instance: "_FakeRuntime | None" = None

            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.manager = _FakeManager()
                _FakeRuntime.last_instance = self

            def _model_manager(self) -> _FakeManager:
                return self.manager

        with patch("agent.api_server.AgentRuntime", _FakeRuntime):
            result = cli._execute_llm_install_via_model_manager(
                config=object(),
                plan=plan,
                trace_id="cli-install-1",
            )

        self.assertEqual(expected, result)
        runtime = _FakeRuntime.last_instance
        self.assertIsNotNone(runtime)
        manager = runtime._model_manager() if runtime is not None else None
        self.assertIsNotNone(manager)
        args, kwargs = manager.called_with if manager is not None and manager.called_with is not None else ((), {})
        self.assertEqual(({"kind": "approved_ollama_pull", "model_ref": "ollama:llava:7b"},), args)
        self.assertEqual(
            {
                "approve": True,
                "trace_id": "cli-install-1",
                "timeout_seconds": 1800.0,
                "source": "cli.llm_install",
            },
            kwargs,
        )

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
        with patch(
            "agent.cli.read_build_info",
            return_value=BuildInfo(version="9.9.9", version_source="repo_file", git_commit="abc1234"),
        ), redirect_stdout(output):
            code = cli.main(["version"])
        self.assertEqual(0, code)
        text = output.getvalue().strip()
        self.assertIn("version=9.9.9", text)
        self.assertIn("commit=abc1234", text)

    def test_split_status_subcommand_prints_runtime_identity(self) -> None:
        output = io.StringIO()
        with patch("agent.cli.runtime_instance", return_value="stable"), patch(
            "agent.cli.runtime_root_path",
            return_value=Path("/opt/personal-agent/runtime/current"),
        ), patch(
            "agent.cli.runtime_service_name",
            return_value="personal-agent-api.service",
        ), patch(
            "agent.cli.runtime_launcher_name",
            return_value="personal-agent-webui",
        ), patch(
            "agent.cli.runtime_api_base_url",
            return_value="http://127.0.0.1:8765",
        ), patch("agent.cli.runtime_port", return_value=8765), patch(
            "agent.cli.Path.home",
            return_value=Path("/home/test"),
        ), redirect_stdout(output):
            code = cli.main(["split_status"])
        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("runtime_instance: stable", text)
        self.assertIn("runtime_root: /opt/personal-agent/runtime/current", text)
        self.assertIn("service_name: personal-agent-api.service", text)
        self.assertIn("launcher_target: /home/test/.local/share/personal-agent/bin/personal-agent-webui", text)
        self.assertIn("api_base_url: http://127.0.0.1:8765", text)
        self.assertIn("legacy_checkout_service: retired", text)

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
            with patch("agent.cli._load_ready_status_payload", return_value=(True, payload)), patch(
                "agent.cli._load_runtime_status_payload",
                return_value=(False, "runtime_unavailable"),
            ), patch(
                "agent.cli._load_runtime_history_payload",
                return_value=(False, "runtime_history_unavailable"),
            ), redirect_stdout(output):
                code = cli.main(None)
        self.assertEqual(0, code)
        self.assertIn("Ready.", output.getvalue())

    def test_main_uses_sys_argv_for_version_when_argv_none(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["python -m agent", "version"]):
            with patch(
                "agent.cli.read_build_info",
                return_value=BuildInfo(version="9.9.9", version_source="repo_file", git_commit="abc1234"),
            ), redirect_stdout(output):
                code = cli.main(None)
        self.assertEqual(0, code)
        text = output.getvalue().strip()
        self.assertIn("version=9.9.9", text)
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
