from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
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
        llm_registry_prune_allow_apply=None,
    )
    return base.__class__(**{**base.__dict__, **overrides})


class TestLLMCleanupAPI(unittest.TestCase):
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

    def test_scheduler_non_loopback_denies_cleanup_without_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_registry_prune_allow_apply=None))
        runtime.set_listening("0.0.0.0", 8765)
        ok, body = runtime.llm_cleanup_apply({"actor": "scheduler"}, trigger="scheduler")
        self.assertFalse(ok)
        self.assertEqual("action_not_permitted", body["error"])

    def test_scheduler_loopback_auto_allows_cleanup_apply_without_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_registry_prune_allow_apply=None))
        runtime.set_listening("127.0.0.1", 8765)
        preview_plan = {
            "ok": True,
            "changes": [
                {
                    "kind": "model",
                    "id": "ollama:llama3",
                    "field": "available",
                    "before": True,
                    "after": False,
                    "reason": "missing_from_catalog",
                }
            ],
            "prune_candidates": [],
            "impact": {"changes_count": 1},
            "updated_document": runtime.registry_document,
        }
        apply_plan = {
            "ok": True,
            "changes": [
                {
                    "kind": "model",
                    "id": "ollama:llama3",
                    "field": "available",
                    "before": True,
                    "after": False,
                    "reason": "missing_from_catalog",
                }
            ],
            "prune_candidates": [],
            "impact": {"changes_count": 1},
            "updated_document": runtime.registry_document,
        }
        with patch("agent.api_server.build_registry_cleanup_plan", side_effect=[preview_plan, apply_plan]):
            ok, body = runtime.llm_cleanup_apply({"actor": "scheduler"}, trigger="scheduler")
        self.assertTrue(ok)
        self.assertTrue(body["applied"])


if __name__ == "__main__":
    unittest.main()
