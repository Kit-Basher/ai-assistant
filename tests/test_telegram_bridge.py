from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from agent.telegram_bridge import (
    build_telegram_error,
    classify_telegram_text_command,
    handle_telegram_command,
    handle_telegram_text,
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeOrchestrator:
    def __init__(self, reply_text: str = "chat") -> None:
        self.reply_text = reply_text
        self.calls: list[tuple[str, str]] = []

    def handle_message(self, text: str, *, user_id: str) -> _FakeResponse:
        self.calls.append((text, user_id))
        return _FakeResponse(self.reply_text)


class _ReadyRuntime:
    version = "0.2.0"
    git_commit = "abc123def456"
    started_at = datetime.now(timezone.utc) - timedelta(seconds=55)

    def ready_status(self) -> dict[str, object]:
        return {
            "ok": True,
            "ready": True,
            "runtime_status": {
                "runtime_mode": "READY",
                "summary": "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
                "next_action": None,
            },
            "telegram": {"state": "running"},
            "onboarding": {
                "state": "READY",
                "summary": "Setup complete.",
                "next_action": "No action needed.",
            },
            "api": {"version": "0.2.0", "git_commit": "abc123def456", "uptime_seconds": 55},
        }

    def llm_status(self) -> dict[str, object]:
        return {
            "default_provider": "ollama",
            "default_model": "ollama:qwen2.5:3b-instruct",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "active_provider_health": {"status": "ok"},
            "active_model_health": {"status": "ok"},
        }


class _NotStartedRuntime:
    def ready_status(self) -> dict[str, object]:
        return {
            "ok": True,
            "ready": False,
            "runtime_status": {
                "runtime_mode": "BOOTSTRAP_REQUIRED",
                "summary": "System is still initializing. Startup is still in progress. Next: Wait for startup to finish, then try again.",
                "next_action": "Wait for startup to finish, then try again.",
            },
            "telegram": {"state": "running"},
            "onboarding": {
                "state": "NOT_STARTED",
                "summary": "Setup has not started.",
                "next_action": "Run: python -m agent setup",
            },
        }

    def llm_status(self) -> dict[str, object]:
        return {}


class _StaleOnboardingReadyRuntime(_ReadyRuntime):
    def ready_status(self) -> dict[str, object]:
        payload = super().ready_status()
        payload["onboarding"] = {
            "state": "SERVICES_DOWN",
            "summary": "Core services are down or still starting.",
            "next_action": "Run: systemctl --user restart personal-agent-api.service",
        }
        return payload


class TestTelegramBridge(unittest.TestCase):
    def test_setup_runtime_prefers_embedded_ready_llm_snapshot(self) -> None:
        class _EmbeddedReadyRuntime(_ReadyRuntime):
            def ready_status(self) -> dict[str, object]:
                payload = super().ready_status()
                payload["llm"] = {
                    "default_provider": "ollama",
                    "resolved_default_model": "ollama:qwen2.5:3b-instruct",
                    "active_provider_health": {"status": "ok"},
                    "active_model_health": {"status": "ok"},
                }
                return payload

            def llm_status(self) -> dict[str, object]:
                raise AssertionError("telegram setup bridge should use ready.llm before llm_status")

        result = handle_telegram_text(
            text="setup",
            chat_id="1",
            trace_id="tg-embedded-ready",
            runtime=_EmbeddedReadyRuntime(),
            orchestrator=_FakeOrchestrator(),
        )

        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("setup", result.get("route"))
        self.assertIn("Setup is complete.", str(result.get("text") or ""))

    def test_classify_command_aliases(self) -> None:
        self.assertEqual("/brief", classify_telegram_text_command("breif"))
        self.assertEqual("/memory", classify_telegram_text_command("what are we doing?"))
        self.assertIsNone(classify_telegram_text_command("what can you do"))
        self.assertIsNone(classify_telegram_text_command("fix setup"))
        self.assertIsNone(classify_telegram_text_command("repair openrouter"))
        self.assertIsNone(classify_telegram_text_command("openrouter unavailable"))
        self.assertIsNone(classify_telegram_text_command("configure ollama"))

    def test_help_ready_returns_command_list(self) -> None:
        result = handle_telegram_text(
            text="help",
            chat_id="1",
            trace_id="tg-1",
            runtime=_ReadyRuntime(),
            orchestrator=_FakeOrchestrator(),
        )
        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("help", result.get("route"))
        self.assertIn("Available commands:", str(result.get("text") or ""))

    def test_help_api_probe_failure_still_returns_command_list(self) -> None:
        result = handle_telegram_text(
            text="/help",
            chat_id="1",
            trace_id="tg-help-probe-failure",
            runtime=None,
            orchestrator=_FakeOrchestrator(),
            fetch_local_api_json=lambda _path: {},
        )

        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("help", result.get("route"))
        text = str(result.get("text") or "")
        self.assertIn("Available commands:", text)
        self.assertNotIn("Setup state: degraded", text)
        self.assertNotIn("API may be restarting", text)

    def test_presence_probe_uses_fast_path_without_chat_proxy(self) -> None:
        result = handle_telegram_text(
            text="hello are you working?",
            chat_id="1",
            trace_id="tg-presence-fast-path",
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=lambda _payload: (_ for _ in ()).throw(
                AssertionError("presence checks must not call chat proxy")
            ),
        )

        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("generic_chat", result.get("route"))
        self.assertFalse(bool(result.get("used_llm")))
        self.assertIn("I’m here", str(result.get("text") or ""))

    def test_setup_not_started_returns_setup_summary(self) -> None:
        result = handle_telegram_text(
            text="setup",
            chat_id="1",
            trace_id="tg-2",
            runtime=_NotStartedRuntime(),
            orchestrator=_FakeOrchestrator(),
        )
        text = str(result.get("text") or "")
        self.assertIn("Setup state:", text)
        self.assertIn("Next:", text)

    def test_setup_ignores_stale_onboarding_when_runtime_is_ready(self) -> None:
        result = handle_telegram_command(
            command="/setup",
            chat_id="1",
            trace_id="tg-stale-setup",
            runtime=_StaleOnboardingReadyRuntime(),
            orchestrator=_FakeOrchestrator(),
        )
        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("setup", result.get("route"))
        text = str(result.get("text") or "")
        self.assertNotIn("Setup state: services down", text)
        self.assertNotIn("API service is down.", text)
        self.assertIn("Setup is complete.", text)
        diagnosis = result.get("diagnosis") if isinstance(result.get("diagnosis"), dict) else {}
        self.assertEqual("READY", diagnosis.get("mapped_state"))
        self.assertEqual("confirmed", diagnosis.get("confidence"))

    def test_setup_transport_failure_with_active_service_is_uncertain(self) -> None:
        with patch(
            "agent.setup_wizard.probe_api_service_state",
            return_value={
                "checked": True,
                "available": True,
                "active_state": "active",
                "confidence": "confirmed",
            },
        ):
            result = handle_telegram_text(
                text="setup",
                chat_id="1",
                trace_id="tg-setup-uncertain",
                runtime=None,
                orchestrator=_FakeOrchestrator(),
                fetch_local_api_json=lambda _path: {},
            )

        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("setup", result.get("route"))
        text = str(result.get("text") or "")
        self.assertIn("Setup state: degraded", text)
        self.assertIn("The API may be restarting or temporarily unavailable.", text)
        self.assertNotIn("API service is down.", text)
        diagnosis = result.get("diagnosis") if isinstance(result.get("diagnosis"), dict) else {}
        self.assertEqual("service_check", diagnosis.get("source"))
        self.assertEqual("uncertain", diagnosis.get("confidence"))

    def test_status_returns_runtime_mode_text(self) -> None:
        result = handle_telegram_command(
            command="/status",
            chat_id="1",
            trace_id="tg-3",
            runtime=_ReadyRuntime(),
            orchestrator=_FakeOrchestrator(),
        )
        text = str(result.get("text") or "")
        self.assertIn("runtime_mode: READY", text)
        self.assertIn("telegram: running", text)

    def test_memory_like_telegram_text_uses_chat_proxy_when_runtime_is_unbound(self) -> None:
        result = handle_telegram_text(
            text="continue from here",
            chat_id="1",
            trace_id="tg-memory-proxy",
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=lambda _payload: {
                "ok": True,
                "assistant": {"role": "assistant", "content": "We were focused on today_plan. Ask me to continue there."},
                "message": "We were focused on today_plan. Ask me to continue there.",
                "meta": {"route": "agent_memory", "used_memory": True},
            },
        )

        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("agent_memory", result.get("route"))
        self.assertIn("today_plan", str(result.get("text") or ""))

    def test_health_uses_orchestrator_command(self) -> None:
        orchestrator = _FakeOrchestrator(reply_text="health ok")
        result = handle_telegram_command(
            command="/health",
            chat_id="42",
            trace_id="tg-4",
            runtime=None,
            orchestrator=orchestrator,
        )
        self.assertEqual("health", result.get("route"))
        self.assertEqual("health ok", result.get("text"))
        self.assertEqual([("/health", "42")], orchestrator.calls)

    def test_doctor_summary_path(self) -> None:
        result = handle_telegram_command(
            command="/doctor",
            chat_id="42",
            trace_id="tg-5",
            runtime=None,
            orchestrator=None,
        )
        text = str(result.get("text") or "")
        self.assertIn("Doctor:", text)
        self.assertIn("agent doctor --json", text)

    def test_error_shape_is_deterministic(self) -> None:
        result = build_telegram_error(
            title="❌ Runtime unavailable",
            trace_id="tg-e1",
            component="telegram.bridge",
            next_action="run `agent doctor`",
        )
        self.assertFalse(bool(result.get("ok")))
        self.assertEqual("tg-e1", result.get("trace_id"))
        self.assertEqual("run `agent doctor`", result.get("next_action"))
        self.assertIn("trace_id: tg-e1", str(result.get("text") or ""))

    def test_chat_proxy_success_preserves_route_and_first_line(self) -> None:
        def _proxy(_payload: dict[str, object]) -> dict[str, object]:
            return {
                "ok": True,
                "assistant": {"content": "Ready.\nUsing ollama / ollama:qwen2.5:3b-instruct."},
                "meta": {"route": "runtime_status", "used_tools": ["runtime_status"]},
            }

        result = handle_telegram_text(
            text="what is the runtime status?",
            chat_id="42",
            trace_id="tg-proxy-ok",
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=_proxy,
        )
        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("runtime_status", result.get("route"))
        self.assertEqual("Ready.", str(result.get("text") or "").splitlines()[0])
        self.assertEqual(["runtime_status"], result.get("used_tools"))

    def test_chat_proxy_unavailable_returns_truthful_backend_message(self) -> None:
        def _proxy(_payload: dict[str, object]) -> dict[str, object]:
            return {"_proxy_error": {"kind": "unreachable", "backend_reachable": False, "backend_ready": False}}

        result = handle_telegram_text(
            text="what is the runtime status?",
            chat_id="42",
            trace_id="tg-proxy-down",
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=_proxy,
        )
        self.assertTrue(bool(result.get("handled")))
        self.assertEqual("chat_proxy_error", result.get("route"))
        self.assertIn("couldn't reach the agent backend", str(result.get("text") or "").lower())


if __name__ == "__main__":
    unittest.main()
