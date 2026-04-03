from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.llm.model_selector import select_model_for_task


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


def _inventory_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "ollama:qwen3.5:4b",
            "provider": "ollama",
            "available": True,
            "healthy": True,
            "approved": True,
            "local": True,
            "routable": True,
            "capabilities": ["chat"],
            "model_name": "qwen3.5:4b",
            "quality_rank": 6,
            "context_window": 32768,
        },
        {
            "id": "ollama:qwen2.5:3b-instruct",
            "provider": "ollama",
            "available": True,
            "healthy": True,
            "approved": True,
            "local": True,
            "routable": True,
            "capabilities": ["chat"],
            "model_name": "qwen2.5:3b-instruct",
            "quality_rank": 5,
            "context_window": 32768,
        },
        {
            "id": "ollama:qwen2.5:7b-instruct",
            "provider": "ollama",
            "available": True,
            "healthy": True,
            "approved": True,
            "local": True,
            "routable": True,
            "capabilities": ["chat"],
            "model_name": "qwen2.5:7b-instruct",
            "quality_rank": 9,
            "context_window": 32768,
        },
        {
            "id": "ollama:llava:7b",
            "provider": "ollama",
            "available": True,
            "healthy": True,
            "approved": True,
            "local": True,
            "routable": True,
            "capabilities": ["chat", "vision"],
            "model_name": "llava:7b",
            "quality_rank": 10,
            "context_window": 8192,
        },
    ]


class TestLatencyRouting(unittest.TestCase):
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

    def test_telegram_channel_prefers_small_local_model(self) -> None:
        selection = select_model_for_task(
            _inventory_rows(),
            {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            allow_remote_fallback=True,
            channel="telegram",
        )
        self.assertEqual("ollama:qwen3.5:4b", selection["selected_model"])

    def test_api_channel_still_allows_stronger_large_model(self) -> None:
        selection = select_model_for_task(
            [row for row in _inventory_rows() if str(row.get("id") or "") != "ollama:llava:7b"],
            {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            allow_remote_fallback=True,
            channel="api",
        )
        self.assertEqual("ollama:qwen2.5:7b-instruct", selection["selected_model"])

    def test_telegram_channel_hard_filters_vision_and_slow_models_for_plain_text(self) -> None:
        selection = select_model_for_task(
            _inventory_rows(),
            {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            allow_remote_fallback=True,
            channel="telegram",
        )
        self.assertEqual("ollama:qwen3.5:4b", selection["selected_model"])
        self.assertNotEqual("ollama:llava:7b", selection["selected_model"])
        self.assertNotEqual("ollama:qwen2.5:7b-instruct", selection["selected_model"])

    def test_telegram_latency_fallback_only_uses_fast_allowlist(self) -> None:
        selection = select_model_for_task(
            _inventory_rows(),
            {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            allow_remote_fallback=True,
            channel="telegram",
            latency_fallback=True,
        )
        self.assertIn(
            selection["selected_model"],
            {"ollama:qwen3.5:4b", "ollama:qwen2.5:3b-instruct"},
        )

    def test_telegram_chat_timeout_triggers_latency_fallback(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        route_calls: list[dict[str, object]] = []

        def _fake_route_inference(**kwargs: object) -> dict[str, object]:
            route_calls.append(dict(kwargs))
            metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else {}
            latency_fallback = bool(metadata.get("latency_fallback"))
            if not latency_fallback:
                self.assertEqual("telegram", metadata.get("channel"))
                self.assertEqual(5.0, kwargs.get("timeout_seconds"))
                return {
                    "ok": False,
                    "text": "",
                    "provider": "ollama",
                    "model": "ollama:qwen3.5:4b",
                    "task_type": "chat",
                    "selection_reason": "healthy+approved+local_first+task=chat",
                    "fallback_used": False,
                    "error_kind": "timeout",
                    "error_class": "timeout",
                    "next_action": None,
                    "trace_id": "latency-test",
                    "duration_ms": 5000,
                    "attempts": [{"provider": "ollama", "model": "ollama:qwen3.5:4b", "reason": "timeout"}],
                    "data": {
                        "task_request": {
                            "task_type": "chat",
                            "requirements": ["chat"],
                            "preferred_local": True,
                        },
                        "selection": {
                            "selected_model": "ollama:qwen3.5:4b",
                            "provider": "ollama",
                            "reason": "healthy+approved+local_first+task=chat",
                            "fallbacks": ["ollama:qwen2.5:3b-instruct"],
                        },
                    },
                }
            self.assertEqual("telegram", metadata.get("channel"))
            self.assertTrue(latency_fallback)
            self.assertEqual(4.0, kwargs.get("timeout_seconds"))
            return {
                "ok": True,
                "text": "Fast reply",
                "provider": "ollama",
                "model": "ollama:qwen2.5:3b-instruct",
                "task_type": "chat",
                "selection_reason": "healthy+approved+local_first+task=chat",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "latency-test",
                "duration_ms": 700,
                "attempts": [],
                "data": {
                    "task_request": {
                        "task_type": "chat",
                        "requirements": ["chat"],
                        "preferred_local": True,
                    },
                    "selection": {
                        "selected_model": "ollama:qwen2.5:3b-instruct",
                        "provider": "ollama",
                        "reason": "healthy+approved+local_first+task=chat",
                        "fallbacks": ["ollama:qwen3.5:4b"],
                    },
                },
            }

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            runtime._router,
            "enabled",
            return_value=True,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=_fake_route_inference,
        ):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "tell me a joke"}],
                    "source_surface": "telegram",
                }
            )

        self.assertTrue(ok)
        self.assertEqual("Fast reply", body["assistant"]["content"])
        self.assertEqual("generic_chat", body["meta"]["route"])
        self.assertEqual(2, len(route_calls))

        events = runtime.runtime_event_history(limit=20)["events"]
        guard = next((row for row in events if row.get("event") == "telegram_latency_guard"), None)
        fallback = next((row for row in events if row.get("event") == "telegram_latency_fallback"), None)
        self.assertIsNotNone(guard)
        self.assertIsNotNone(fallback)
        self.assertEqual("ollama:qwen3.5:4b", guard["model_selected"])
        self.assertEqual("ollama:qwen3.5:4b", fallback["slow_model"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", fallback["fallback_model"])

    def test_api_chat_propagates_api_channel_without_latency_fallback(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        def _fake_route_inference(**kwargs: object) -> dict[str, object]:
            metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else {}
            self.assertEqual("api", metadata.get("channel"))
            self.assertFalse(bool(metadata.get("latency_fallback")))
            self.assertIsNone(kwargs.get("timeout_seconds"))
            return {
                "ok": True,
                "text": "API reply",
                "provider": "ollama",
                "model": "ollama:qwen2.5:7b-instruct",
                "task_type": "chat",
                "selection_reason": "healthy+approved+local_first+task=chat",
                "fallback_used": False,
                "error_kind": None,
                "next_action": None,
                "trace_id": "api-latency-test",
                "duration_ms": 120,
                "attempts": [],
                "data": {
                    "task_request": {
                        "task_type": "chat",
                        "requirements": ["chat"],
                        "preferred_local": True,
                    },
                    "selection": {
                        "selected_model": "ollama:qwen2.5:7b-instruct",
                        "provider": "ollama",
                        "reason": "healthy+approved+local_first+task=chat",
                        "fallbacks": ["ollama:qwen3.5:4b"],
                    },
                },
            }

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            runtime._router,
            "enabled",
            return_value=True,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=_fake_route_inference,
        ):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "tell me a joke"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok)
        self.assertEqual("API reply", body["assistant"]["content"])


if __name__ == "__main__":
    unittest.main()
