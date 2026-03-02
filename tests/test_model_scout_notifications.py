from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config


def _config(registry_path: str, db_path: str) -> Config:
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
        autopilot_notify_enabled=True,
        autopilot_notify_rate_limit_seconds=0,
        autopilot_notify_dedupe_window_seconds=0,
        llm_notifications_allow_send=None,
    )
    return base.__class__(**dict(base.__dict__))


class TestModelScoutNotifications(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        os.environ["AGENT_MODEL_SCOUT_NOTIFY_STATE_PATH"] = os.path.join(
            self.tmpdir.name,
            "model_scout_notify_state.json",
        )

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_scout_noop_does_not_send_telegram(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch.object(
            runtime.model_scout,
            "run",
            return_value={"ok": True, "error": None, "suggestions": [], "new_suggestions": []},
        ), patch.object(runtime, "_send_telegram_message", return_value=None) as send_mock:
            ok, body = runtime.run_model_scout()

        self.assertTrue(ok)
        self.assertTrue(body.get("ok"))
        self.assertEqual([], body.get("model_scout_change_events"))
        self.assertFalse(bool(body.get("notification_emitted")))
        self.assertEqual("no_change", str(body.get("notification_reason") or ""))
        send_mock.assert_not_called()

    def test_scout_change_sends_telegram_and_audits(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document = {
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "openrouter",
                "default_model": "openrouter:old-model",
                "allow_remote_fallback": True,
            },
            "providers": {
                "openrouter": {"enabled": True, "available": True, "local": False},
                "ollama": {"enabled": True, "available": True, "local": True},
            },
            "models": {
                "openrouter:old-model": {
                    "provider": "openrouter",
                    "model": "old-model",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "capabilities": ["chat"],
                },
                "openrouter:new-model": {
                    "provider": "openrouter",
                    "model": "new-model",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "capabilities": ["chat"],
                },
            },
        }
        candidate = {
            "id": "remote:openrouter:openrouter:new-model",
            "kind": "remote",
            "repo_id": None,
            "provider_id": "openrouter",
            "model_id": "openrouter:new-model",
            "score": 91.25,
            "rationale": "better quality",
            "install_cmd": None,
        }
        with patch.object(
            runtime.model_scout,
            "run",
            return_value={"ok": True, "error": None, "suggestions": [candidate], "new_suggestions": [candidate]},
        ), patch.object(
            runtime.model_scout.store,
            "latest_baseline",
            return_value={
                "snapshot": {
                    "remote": {"openrouter": {"model_id": "openrouter:old-model", "score": 80.0}},
                    "local": {"provider_id": "ollama", "model_id": None, "score": 60.0},
                }
            },
        ), patch.object(runtime, "telegram_status", return_value={"state": "running"}), patch.object(
            runtime,
            "_resolve_telegram_target",
            return_value=("token", "123456789"),
        ), patch.object(runtime, "_send_telegram_message", return_value=None) as send_mock:
            ok, body = runtime.run_model_scout()

        self.assertTrue(ok)
        self.assertTrue(body.get("ok"))
        self.assertTrue(bool(body.get("notification_emitted")))
        self.assertEqual("sent", str(body.get("notification_reason") or ""))
        events = body.get("model_scout_change_events") if isinstance(body.get("model_scout_change_events"), list) else []
        self.assertTrue(events)
        self.assertEqual("better_default_candidate", str(events[0].get("reason") or ""))
        self.assertEqual("openrouter:old-model", str(events[0].get("from_model") or ""))
        self.assertEqual("openrouter:new-model", str(events[0].get("to_model") or ""))
        self.assertEqual("openrouter", str(events[0].get("provider") or ""))
        self.assertAlmostEqual(11.25, float(events[0].get("score_delta") or 0.0), places=2)

        send_mock.assert_called_once()
        sent_message = str(send_mock.call_args.args[2] if len(send_mock.call_args.args) >= 3 else "")
        self.assertIn("Current default: openrouter:old-model", sent_message)
        self.assertIn("Recommended default: openrouter:new-model", sent_message)
        self.assertIn("PUT http://127.0.0.1:8765/defaults", sent_message)

        audit_entries = runtime.get_audit(limit=30).get("entries", [])
        notify_rows = [row for row in audit_entries if row.get("action") == "llm.model_scout.notify"]
        self.assertTrue(notify_rows)
        self.assertEqual("sent", notify_rows[0].get("outcome"))
        changed_rows = [row for row in audit_entries if row.get("action") == "llm.model_scout.changed"]
        self.assertTrue(changed_rows)


if __name__ == "__main__":
    unittest.main()
