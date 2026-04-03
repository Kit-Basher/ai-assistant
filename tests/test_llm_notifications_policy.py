from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config


def _config(registry_path: str, db_path: str, **overrides: object) -> Config:
    base = Config(
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
    return base.__class__(**{**base.__dict__, **overrides})


class TestLLMNotificationsPolicy(unittest.TestCase):
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

    def test_policy_loopback_unset_knob_allows(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=None))
        runtime.set_listening("127.0.0.1", 8765)
        policy = runtime.llm_notifications_policy()["policy"]
        self.assertEqual("127.0.0.1", policy["bind_host"])
        self.assertTrue(policy["is_loopback"])
        self.assertTrue(policy["allow_test_effective"])
        self.assertEqual("loopback_auto", policy["allow_reason"])

    def test_policy_non_loopback_unset_knob_requires_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=None))
        runtime.set_listening("0.0.0.0", 8765)
        policy = runtime.llm_notifications_policy()["policy"]
        self.assertEqual("0.0.0.0", policy["bind_host"])
        self.assertFalse(policy["is_loopback"])
        self.assertFalse(policy["allow_test_effective"])
        self.assertEqual("permission_required", policy["allow_reason"])

    def test_policy_knob_true_always_allows(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=True))
        runtime.set_listening("0.0.0.0", 8765)
        policy = runtime.llm_notifications_policy()["policy"]
        self.assertEqual("explicit_true", policy["allow_reason"])
        self.assertTrue(policy["allow_test_effective"])

    def test_policy_knob_false_disables_auto_allow(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=False))
        runtime.set_listening("127.0.0.1", 8765)
        policy = runtime.llm_notifications_policy()["policy"]
        self.assertEqual("explicit_false", policy["allow_reason"])
        self.assertFalse(policy["allow_test_effective"])

    def test_policy_endpoint_returns_current_policy(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=None))
        runtime.set_listening("127.0.0.1", 8765)

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
                self.runtime = runtime_obj
                self.path = path
                self.headers = {}
                self.status_code = 0
                self.content_type = ""
                self.body = b""

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                self.status_code = status
                self.content_type = "application/json"
                self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

            def _send_bytes(
                self,
                status: int,
                body: bytes,
                *,
                content_type: str,
                cache_control: str | None = None,
            ) -> None:
                _ = cache_control
                self.status_code = status
                self.content_type = content_type
                self.body = body

        handler = _HandlerForTest(runtime, "/llm/notifications/policy")
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual("loopback_auto", payload["policy"]["allow_reason"])
        self.assertTrue(payload["policy"]["allow_test_effective"])

    def test_notifications_test_behavior_unchanged(self) -> None:
        non_loopback = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=None))
        non_loopback.set_listening("0.0.0.0", 8765)
        denied_ok, denied_body = non_loopback.llm_notifications_test({"actor": "test", "confirm": True})
        self.assertFalse(denied_ok)
        self.assertEqual("action_not_permitted", denied_body["error"])

        loopback = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=None))
        loopback.set_listening("127.0.0.1", 8765)
        with patch.object(loopback, "_send_telegram_message", return_value=None), patch.object(
            loopback, "_resolve_telegram_target", return_value=("token", "chat-1")
        ):
            allowed_ok, allowed_body = loopback.llm_notifications_test({"actor": "test", "confirm": True})
        self.assertTrue(allowed_ok)
        self.assertTrue(allowed_body["ok"])
        self.assertEqual("sent", allowed_body["result"]["outcome"])


if __name__ == "__main__":
    unittest.main()

