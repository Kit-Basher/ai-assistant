from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.telegram_bridge import handle_telegram_command, handle_telegram_text


_BAD_TEXT_MARKERS = (
    "read-only guard",
    "nl path refused",
    "source_surface",
    "thread_id",
    "user_id",
    "runtime_payload",
    "runtime_state_failure_reason",
    "default model updated",
)


def _config(registry_path: str, db_path: str) -> Config:
    return Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path="/tmp/skills",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=registry_path,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
        llm_health_state_path=os.path.join(os.path.dirname(db_path), "llm_health_state.json"),
        llm_automation_enabled=False,
        model_scout_state_path=os.path.join(os.path.dirname(db_path), "model_scout_state.json"),
        autopilot_notify_store_path=os.path.join(os.path.dirname(db_path), "llm_notifications.json"),
    )


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime: AgentRuntime, path: str, payload: dict[str, object]) -> None:
        self.runtime = runtime
        self.path = path
        self.headers = {"Content-Length": "0"}
        self.client_address = ("127.0.0.1", 12345)
        self._payload = dict(payload)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


class _Response:
    def __init__(self, text: str, data: dict[str, object] | None = None) -> None:
        self.text = text
        self.data = data or {}


class _Orchestrator:
    def __init__(self, text: str) -> None:
        self.text = text

    def handle_message(self, text: str, *, user_id: str) -> _Response:
        return _Response(self.text, {"route": "generic_chat", "used_llm": False})


class TestBypassBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self) -> AgentRuntime:
        return AgentRuntime(_config(self.registry_path, self.db_path))

    def assert_user_facing(self, text: object) -> None:
        rendered = str(text or "").strip()
        self.assertTrue(rendered)
        self.assertNotIn(rendered, {"OK", "Ok", "Done.", "I’m here and ready"})
        lowered = rendered.lower()
        for marker in _BAD_TEXT_MARKERS:
            self.assertNotIn(marker, lowered)

    def test_api_legacy_choice_empty_message_gets_actionable_fallback(self) -> None:
        runtime = self._runtime()
        payload = {"messages": [{"role": "user", "content": "status"}], "trace_id": "choice-empty"}
        with patch.object(
            runtime,
            "consume_clarify_recovery_choice",
            return_value=(True, {"ok": True, "intent": "chat", "did_work": True, "message": ""}),
        ):
            handler = _HandlerForTest(runtime, "/ask", payload)
            handler.do_POST()

        self.assertEqual(200, handler.status_code)
        message = str(handler.response_payload.get("message") or "")
        self.assert_user_facing(message)
        self.assertIn("next request", message.lower())

    def test_api_binary_choice_empty_message_gets_actionable_fallback(self) -> None:
        runtime = self._runtime()
        payload = {"messages": [{"role": "user", "content": "1"}], "trace_id": "binary-empty"}
        with patch.object(runtime, "consume_clarify_recovery_choice", return_value=(False, {})), patch.object(
            runtime,
            "consume_binary_clarification_choice",
            return_value=(True, {"ok": True, "intent": "chat", "did_work": True, "message": ""}),
        ):
            handler = _HandlerForTest(runtime, "/ask", payload)
            handler.do_POST()

        self.assertEqual(200, handler.status_code)
        message = str(handler.response_payload.get("message") or "")
        self.assert_user_facing(message)
        self.assertIn("next request", message.lower())

    def test_api_intent_choice_empty_message_gets_actionable_fallback(self) -> None:
        runtime = self._runtime()
        payload = {"messages": [{"role": "user", "content": "chat"}], "trace_id": "intent-empty"}
        with patch.object(runtime, "consume_clarify_recovery_choice", return_value=(False, {})), patch.object(
            runtime,
            "consume_binary_clarification_choice",
            return_value=(False, {}),
        ), patch.object(
            runtime,
            "consume_intent_choice",
            return_value=(True, {"ok": True, "intent": "chat", "did_work": True, "message": ""}),
        ):
            handler = _HandlerForTest(runtime, "/ask", payload)
            handler.do_POST()

        self.assertEqual(200, handler.status_code)
        message = str(handler.response_payload.get("message") or "")
        self.assert_user_facing(message)
        self.assertIn("next request", message.lower())

    def test_api_thread_resume_reenters_runtime_chat_with_clean_payload(self) -> None:
        runtime = self._runtime()
        payload = {
            "messages": [{"role": "user", "content": "same thread"}],
            "user_id": "api:resume",
            "thread_id": "api:resume:thread",
            "trace_id": "original-trace",
            "request_id": "original-request",
        }
        runtime.set_thread_integrity_prompt(
            source="api",
            user_id="api:resume",
            pending_text="what model am I using?",
            payload_template=payload,
        )
        captured: list[dict[str, object]] = []

        def _chat(resume_payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
            captured.append(dict(resume_payload))
            return True, {
                "ok": True,
                "assistant": {"role": "assistant", "content": "You are using the active chat model for this thread."},
                "message": "You are using the active chat model for this thread.",
                "meta": {"route": "model_status"},
            }

        with patch.object(runtime, "consume_clarify_recovery_choice", return_value=(False, {})), patch.object(
            runtime,
            "consume_binary_clarification_choice",
            return_value=(False, {}),
        ), patch.object(runtime, "consume_intent_choice", return_value=(False, {})), patch.object(runtime, "chat", side_effect=_chat):
            handler = _HandlerForTest(runtime, "/chat", payload)
            handler.do_POST()

        self.assertEqual(200, handler.status_code)
        self.assertEqual(1, len(captured))
        self.assertEqual([{"role": "user", "content": "what model am I using?"}], captured[0].get("messages"))
        self.assertNotIn("trace_id", captured[0])
        self.assertNotIn("request_id", captured[0])
        self.assert_user_facing(handler.response_payload.get("message"))

    def test_telegram_start_and_status_are_explicit_command_bypasses(self) -> None:
        start = handle_telegram_text(text="/start", chat_id="7", trace_id="tg-start", runtime=None, orchestrator=None)
        status = handle_telegram_command(command="/status", chat_id="7", trace_id="tg-status", runtime=None, orchestrator=None)

        self.assertEqual("help", start.get("route"))
        self.assertEqual("command_bridge", start.get("handler_name"))
        self.assertIn("Available commands", str(start.get("text") or ""))
        self.assertEqual("status", status.get("route"))
        self.assertIn("runtime_mode:", str(status.get("text") or ""))
        self.assert_user_facing(start.get("text"))
        self.assert_user_facing(status.get("text"))

    def test_telegram_numeric_without_wizard_routes_to_chat_proxy(self) -> None:
        calls: list[dict[str, object]] = []

        def _proxy(payload: dict[str, object]) -> dict[str, object]:
            calls.append(payload)
            return {
                "ok": True,
                "assistant": {"content": "I need a little more context. Tell me what option 1 refers to."},
                "message": "I need a little more context. Tell me what option 1 refers to.",
                "meta": {"route": "generic_chat"},
            }

        result = handle_telegram_text(
            text="1",
            chat_id="7",
            trace_id="tg-number",
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=_proxy,
        )

        self.assertEqual(1, len(calls))
        self.assertEqual("api_chat_proxy", result.get("handler_name"))
        self.assertEqual("generic_chat", result.get("route"))
        self.assert_user_facing(result.get("text"))
        self.assertIn("context", str(result.get("text") or "").lower())

    def test_telegram_proxy_and_direct_fallback_sanitize_internal_text(self) -> None:
        proxy_result = handle_telegram_text(
            text="what happened",
            chat_id="7",
            trace_id="tg-proxy-internal",
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=lambda _payload: {
                "ok": True,
                "assistant": {"content": '{"source_surface":"telegram","thread_id":"secret"}'},
                "message": '{"source_surface":"telegram","thread_id":"secret"}',
                "meta": {"route": "generic_chat"},
            },
        )
        direct_result = handle_telegram_text(
            text="what happened",
            chat_id="7",
            trace_id="tg-direct-internal",
            runtime=None,
            orchestrator=_Orchestrator("Read-only guard / NL path refused"),
        )

        self.assert_user_facing(proxy_result.get("text"))
        self.assert_user_facing(direct_result.get("text"))
        self.assertIn("status", str(proxy_result.get("text") or "").lower())
        self.assertIn("status", str(direct_result.get("text") or "").lower())


if __name__ == "__main__":
    unittest.main()
