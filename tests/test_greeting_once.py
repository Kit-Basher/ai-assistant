from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.bootstrap.snapshot import BootstrapSnapshot
from agent.config import Config


_GREETING = "Hi — I’m here and ready to help. What can I do for you?"


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
        memory_v2_enabled=True,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _fake_snapshot() -> BootstrapSnapshot:
    return BootstrapSnapshot(
        created_at_ts=1_700_333_000,
        os={"name": "Ubuntu", "version": "24.04", "pretty_name": "Ubuntu", "kernel": "6.8", "arch": "x86_64", "hostname": "h", "os_release": {}},
        hardware={"cpu_count_logical": 8, "cpu_freq_mhz": 2300.0, "cpu_load_1m": 0.2, "mem_total_bytes": 1, "swap_total_bytes": 0, "gpu": {"available": False, "memory_total_mb": 0, "usage_pct": 0.0, "error": "none"}},
        interfaces={"api": {"listening": "http://127.0.0.1:8765"}, "memory_v2_enabled": True, "model_watch_enabled": True, "llm_automation_enabled": False, "telegram_configured": False, "webui_dev_proxy": False},
        providers={"enabled_ids": ["ollama"], "rows": [{"id": "ollama", "enabled": True, "local": True, "health": {"status": "ok", "last_error_kind": None, "status_code": None}}], "defaults": {"default_provider": "ollama", "default_model": "ollama:llama3", "routing_mode": "auto"}},
        capsules={"installed": ["llm"]},
        routes={"methods": {"GET": ["/health"], "POST": ["/chat"], "PUT": [], "DELETE": []}, "counts": {"GET": 1, "POST": 1, "PUT": 0, "DELETE": 0}, "total": 2},
        notes=["gpu_unavailable"],
    )


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object]) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {"Content-Type": "application/json"}
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload

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

    def _read_json(self) -> dict[str, object]:
        return self._payload


class TestGreetingOnce(unittest.TestCase):
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

    def test_chat_greeting_is_emitted_once(self) -> None:
        with patch("agent.api_server.collect_bootstrap_snapshot", return_value=_fake_snapshot()):
            runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        def _fake_chat(_payload: dict[str, object]):
            return True, {
                "ok": True,
                "assistant": {"role": "assistant", "content": "Working on it."},
                "meta": {"provider": "test", "model": "test"},
            }

        runtime.chat = _fake_chat  # type: ignore[assignment]

        first = _HandlerForTest(runtime, "/chat", {"messages": [{"role": "user", "content": "hello"}]})
        first.do_POST()
        first_payload = json.loads(first.body.decode("utf-8"))
        first_text = str(((first_payload.get("assistant") or {}).get("content") if isinstance(first_payload.get("assistant"), dict) else "") or "")
        self.assertIn(_GREETING, first_text)

        second = _HandlerForTest(runtime, "/chat", {"messages": [{"role": "user", "content": "hello again"}]})
        second.do_POST()
        second_payload = json.loads(second.body.decode("utf-8"))
        second_text = str(((second_payload.get("assistant") or {}).get("content") if isinstance(second_payload.get("assistant"), dict) else "") or "")
        self.assertNotIn(_GREETING, second_text)

    def test_empty_chat_prefers_greeting_once_then_normal_clarification(self) -> None:
        with patch("agent.api_server.collect_bootstrap_snapshot", return_value=_fake_snapshot()):
            runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        first = _HandlerForTest(runtime, "/chat", {})
        first.do_POST()
        first_payload = json.loads(first.body.decode("utf-8"))
        self.assertEqual(200, first.status_code)
        self.assertEqual("needs_clarification", first_payload.get("error_kind"))
        self.assertEqual(_GREETING, first_payload.get("message"))

        second = _HandlerForTest(runtime, "/chat", {})
        second.do_POST()
        second_payload = json.loads(second.body.decode("utf-8"))
        self.assertEqual(200, second.status_code)
        self.assertEqual("needs_clarification", second_payload.get("error_kind"))
        self.assertNotEqual(_GREETING, second_payload.get("message"))


if __name__ == "__main__":
    unittest.main()
