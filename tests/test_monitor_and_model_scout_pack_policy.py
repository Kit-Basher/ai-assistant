from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.orchestrator import Orchestrator
from agent.skills_loader import SkillLoader
from memory.db import MemoryDB


def _config(registry_path: str, db_path: str, skills_path: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=skills_path,
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


class TestMonitorAndModelScoutPackPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
        self.db = MemoryDB(self.db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        self.db.close()
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_system_monitor_registered(self) -> None:
        loader = SkillLoader(self.skills_path)
        skills = loader.load_all()
        self.assertIn("runtime_status", skills)
        self.assertIn("service_health_report", skills)

        runtime_status = skills["runtime_status"]
        self.assertEqual("native", runtime_status.pack_trust)
        self.assertEqual("runtime_status", runtime_status.pack_id)
        self.assertIn("runtime_status", (runtime_status.pack_permissions or {}).get("ifaces", []))

        service_health = skills["service_health_report"]
        self.assertEqual("native", service_health.pack_trust)
        self.assertEqual("service_health_report", service_health.pack_id)
        self.assertIn("service_health_report", (service_health.pack_permissions or {}).get("ifaces", []))

    def test_model_scout_registered(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.skills_path))
        packs = runtime.list_packs().get("packs") if isinstance(runtime.list_packs().get("packs"), list) else []
        row = next((item for item in packs if isinstance(item, dict) and item.get("pack_id") == "model_scout"), None)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual("native", row.get("trust"))
        self.assertTrue(bool(row.get("enabled")))

    def test_system_monitor_pack_disabled_or_mismatch_denies_skill_execution(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )
        orchestrator._pack_store.set_enabled("runtime_status", False)
        denied_disabled = orchestrator._call_skill(
            "user1",
            "runtime_status",
            "runtime_status",
            {},
            ["db:read"],
        )
        self.assertIn("not allowed to call runtime_status", denied_disabled.text)

        orchestrator._pack_store.set_enabled("runtime_status", True)
        orchestrator._pack_store.set_approval_hash("runtime_status", "mismatch")
        denied_mismatch = orchestrator._call_skill(
            "user1",
            "runtime_status",
            "runtime_status",
            {},
            ["db:read"],
        )
        self.assertIn("not allowed to call runtime_status", denied_mismatch.text)

    def test_model_scout_pack_disabled_or_mismatch_denies_runtime_actions(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.skills_path))
        runtime.pack_store.set_enabled("model_scout", False)
        ok_disabled, body_disabled = runtime.run_model_scout()
        self.assertFalse(ok_disabled)
        self.assertEqual("pack_permission_denied", body_disabled.get("error"))

        runtime.pack_store.set_enabled("model_scout", True)
        runtime.pack_store.set_approval_hash("model_scout", "mismatch")
        ok_mismatch, body_mismatch = runtime.run_model_scout()
        self.assertFalse(ok_mismatch)
        self.assertEqual("pack_permission_denied", body_mismatch.get("error"))

    def test_system_monitor_smoke_via_orchestrator(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )
        response = orchestrator.handle_message("/runtime_status", "user1")
        self.assertTrue(str(response.text).strip())

    def test_model_scout_endpoints_smoke_no_network(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.skills_path))
        with patch.object(runtime.model_scout, "status", return_value={"ok": True, "last_run": {"ok": True}}), patch.object(
            runtime.model_scout,
            "list_suggestions",
            return_value=[],
        ), patch.object(
            runtime.model_scout,
            "run",
            return_value={"ok": True, "suggestions": [], "new_suggestions": []},
        ):
            status = runtime.model_scout_status()
            self.assertTrue(status.get("ok"))

            suggestions = runtime.model_scout_suggestions()
            self.assertTrue(suggestions.get("ok"))
            self.assertEqual([], suggestions.get("suggestions"))

            run_ok, run_body = runtime.run_model_scout()
            self.assertTrue(run_ok)
            self.assertTrue(run_body.get("ok"))

    def test_model_scout_scheduler_exception_does_not_escape(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.skills_path, llm_automation_enabled=True))
        future = time.time() + 100000.0
        runtime._scheduler_next_run = {
            "refresh": future,
            "bootstrap": future,
            "catalog": future,
            "capabilities_reconcile": future,
            "health": future,
            "hygiene": future,
            "cleanup": future,
            "self_heal": future,
            "autoconfig": future,
            "model_scout": 0.0,
        }
        with patch.object(runtime, "run_model_scout", side_effect=RuntimeError("boom")):
            runtime._scheduler_loop(
                sleep_fn=lambda _seconds: None,
                stop_event=runtime._scheduler_stop,
                max_iters=2,
            )
        self.assertGreater(float(runtime._scheduler_next_run.get("model_scout") or 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
