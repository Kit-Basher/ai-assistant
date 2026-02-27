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


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload or {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
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

    def _read_json(self) -> dict[str, object]:
        return self._payload


class TestLLMFixitEndpointFlow(unittest.TestCase):
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

    def test_fixit_choice_confirm_flow(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime._health_monitor.state["providers"] = {
            "openrouter": {
                "status": "down",
                "last_error_kind": "provider_unavailable",
                "status_code": 502,
                "last_checked_at": 100,
                "cooldown_until": 0,
                "down_since": 100,
                "failure_streak": 12,
                "next_probe_at": 160,
            },
            "ollama": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
        }

        first = _HandlerForTest(runtime, "/llm/fixit", {})
        first.do_POST()
        self.assertEqual(200, first.status_code)
        first_payload = json.loads(first.body.decode("utf-8"))
        self.assertTrue(first_payload["ok"])
        self.assertEqual("needs_user_choice", first_payload["status"])
        self.assertEqual("openrouter_down", first_payload["issue_code"])
        self.assertLessEqual(len(first_payload["choices"]), 3)

        second = _HandlerForTest(runtime, "/llm/fixit", {"answer": "1"})
        second.do_POST()
        self.assertEqual(200, second.status_code)
        second_payload = json.loads(second.body.decode("utf-8"))
        self.assertEqual("needs_confirmation", second_payload["status"])
        self.assertTrue(second_payload["confirm_token"])

        with patch.object(
            runtime,
            "_execute_llm_fixit_plan",
            return_value=(True, {"ok": True, "executed_steps": [{"id": "01"}], "blocked_steps": [], "failed_steps": []}),
        ):
            third = _HandlerForTest(
                runtime,
                "/llm/fixit",
                {"confirm": True, "confirm_token": second_payload["confirm_token"]},
            )
            third.do_POST()

        self.assertEqual(200, third.status_code)
        third_payload = json.loads(third.body.decode("utf-8"))
        self.assertTrue(third_payload["ok"])
        self.assertTrue(third_payload["did_work"])
        self.assertEqual("llm_fixit", third_payload["intent"])
        self.assertFalse(bool(runtime._llm_fixit_store.state.get("active")))  # type: ignore[attr-defined]

    def test_llm_status_endpoint_returns_defaults_and_health(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        handler = _HandlerForTest(runtime, "/llm/status")
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertIn("default_provider", payload)
        self.assertIn("default_model", payload)
        self.assertIn("active_provider_health", payload)
        self.assertIn("active_model_health", payload)
        self.assertIn("safe_mode", payload)


if __name__ == "__main__":
    unittest.main()
