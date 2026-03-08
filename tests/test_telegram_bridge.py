from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

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
                "summary": "Agent is ready. Using ollama / ollama:qwen2.5:3b-instruct.",
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
                "summary": "Setup needed. No chat model is ready yet.",
                "next_action": "Run: python -m agent setup",
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


class TestTelegramBridge(unittest.TestCase):
    def test_classify_command_aliases(self) -> None:
        self.assertEqual("/brief", classify_telegram_text_command("breif"))
        self.assertEqual("/memory", classify_telegram_text_command("what are we doing?"))
        self.assertEqual("/setup", classify_telegram_text_command("fix setup"))
        self.assertEqual("/setup", classify_telegram_text_command("repair openrouter"))
        self.assertEqual("/setup", classify_telegram_text_command("openrouter unavailable"))
        self.assertEqual("/setup", classify_telegram_text_command("configure ollama"))

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


if __name__ == "__main__":
    unittest.main()
