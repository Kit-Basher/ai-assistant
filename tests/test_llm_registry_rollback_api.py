from __future__ import annotations

import copy
import os
import tempfile
import unittest

from agent.api_server import AgentRuntime
from agent.config import Config


def _config(registry_path: str, db_path: str, snapshots_dir: str, **overrides: object) -> Config:
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
        llm_registry_snapshots_dir=snapshots_dir,
        llm_registry_rollback_allow=None,
    )
    return base.__class__(**{**base.__dict__, **overrides})


class TestRegistryRollbackAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.snapshots_dir = os.path.join(self.tmpdir.name, "snapshots")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_non_loopback_denies_without_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.snapshots_dir, llm_registry_rollback_allow=None))
        runtime.set_listening("0.0.0.0", 8765)
        snap = runtime._registry_snapshot_store.create_snapshot(runtime.registry_store.path, runtime.registry_document)
        ok, body = runtime.llm_registry_rollback({"actor": "webui", "snapshot_id": snap["snapshot_id"]})
        self.assertFalse(ok)
        self.assertEqual("action_not_permitted", body["error"])

    def test_loopback_auto_allows_and_restores_snapshot(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.snapshots_dir, llm_registry_rollback_allow=None))
        runtime.set_listening("127.0.0.1", 8765)
        before = copy.deepcopy(runtime.registry_document)
        snap = runtime._registry_snapshot_store.create_snapshot(runtime.registry_store.path, before)

        changed = copy.deepcopy(runtime.registry_document)
        defaults = changed.get("defaults") if isinstance(changed.get("defaults"), dict) else {}
        defaults["allow_remote_fallback"] = not bool(defaults.get("allow_remote_fallback", True))
        changed["defaults"] = defaults
        saved, error = runtime._persist_registry_document(changed)
        self.assertTrue(saved)
        self.assertIsNone(error)

        ok, body = runtime.llm_registry_rollback({"actor": "webui", "snapshot_id": snap["snapshot_id"]})
        self.assertTrue(ok)
        self.assertEqual(snap["snapshot_id"], body["snapshot_id"])
        self.assertEqual(before.get("defaults"), runtime.registry_document.get("defaults"))

        snapshots = runtime.llm_registry_snapshots(limit=20)
        self.assertTrue(snapshots["ok"])
        self.assertTrue(any(row["snapshot_id"] == snap["snapshot_id"] for row in snapshots["snapshots"]))


if __name__ == "__main__":
    unittest.main()
