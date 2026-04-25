from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.bootstrap.snapshot import BootstrapSnapshot
from agent.config import Config
from agent.memory_v2.types import MemoryLevel


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
        created_at_ts=1_700_222_000,
        os={"name": "Ubuntu", "version": "24.04", "pretty_name": "Ubuntu", "kernel": "6.8", "arch": "x86_64", "hostname": "h", "os_release": {}},
        hardware={"cpu_count_logical": 8, "cpu_freq_mhz": 2300.0, "cpu_load_1m": 0.2, "mem_total_bytes": 1, "swap_total_bytes": 0, "gpu": {"available": False, "memory_total_mb": 0, "usage_pct": 0.0, "error": "none"}},
        interfaces={"api": {"listening": "http://127.0.0.1:8765"}, "memory_v2_enabled": True, "model_watch_enabled": True, "llm_automation_enabled": False, "telegram_configured": False, "webui_dev_proxy": False},
        providers={"enabled_ids": ["ollama"], "rows": [{"id": "ollama", "enabled": True, "local": True, "health": {"status": "ok", "last_error_kind": None, "status_code": None}}], "defaults": {"default_provider": "ollama", "default_model": "ollama:llama3", "routing_mode": "auto"}},
        capsules={"installed": ["llm"]},
        routes={"methods": {"GET": ["/health"], "POST": ["/chat"], "PUT": [], "DELETE": []}, "counts": {"GET": 1, "POST": 1, "PUT": 0, "DELETE": 0}, "total": 2},
        notes=["gpu_unavailable"],
    )


class TestBootstrapIdempotent(unittest.TestCase):
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

    def test_bootstrap_runs_once_and_second_pass_is_noop(self) -> None:
        with patch("agent.api_server.collect_bootstrap_snapshot", return_value=_fake_snapshot()) as snapshot_mock:
            runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        self.assertIsNotNone(runtime._memory_v2_store)
        self.assertEqual(1, snapshot_mock.call_count)
        first_episodic = runtime._memory_v2_store.list_episodic_events(limit=100)
        first_semantic = runtime._memory_v2_store.list_memory_items(level=MemoryLevel.SEMANTIC)
        self.assertTrue(first_episodic)
        self.assertTrue(first_semantic)

        with patch("agent.api_server.collect_bootstrap_snapshot", side_effect=AssertionError("should not rerun")):
            runtime._initialize_memory_v2_bootstrap()

        second_episodic = runtime._memory_v2_store.list_episodic_events(limit=100)
        second_semantic = runtime._memory_v2_store.list_memory_items(level=MemoryLevel.SEMANTIC)
        self.assertEqual(len(first_episodic), len(second_episodic))
        self.assertEqual(len(first_semantic), len(second_semantic))
        self.assertTrue(runtime._memory_v2_bootstrap_completed())

    def test_auto_bootstrap_preserves_persisted_chat_model_choice(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_default_chat_model("ollama:llama3")

        with patch.object(
            runtime,
            "runtime_truth_service",
            side_effect=AssertionError("bootstrap chooser should not be queried for an explicit saved model"),
        ):
            result = runtime._auto_bootstrap_local_chat_model()

        self.assertIsNone(result)
        self.assertEqual("ollama:llama3", runtime.get_defaults()["chat_model"])


if __name__ == "__main__":
    unittest.main()
