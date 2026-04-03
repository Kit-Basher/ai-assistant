from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config


def _config(registry_path: str, db_path: str, snapshots_dir: str, ledger_path: str, **overrides: object) -> Config:
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
        llm_autopilot_ledger_path=ledger_path,
        llm_autopilot_safe_mode=True,
    )
    return base.__class__(**{**base.__dict__, **overrides})


class TestLLMAutopilotSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.snapshots_dir = os.path.join(self.tmpdir.name, "snapshots")
        self.ledger_path = os.path.join(self.tmpdir.name, "ledger.json")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_safe_mode_blocks_remote_default_switch_for_autoconfig(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.snapshots_dir, self.ledger_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.autoconfig.apply": True}})
        remote_plan = {
            "ok": True,
            "changes": [
                {
                    "kind": "defaults",
                    "field": "default_provider",
                    "before": "ollama",
                    "after": "openrouter",
                    "reason": "selected_best_available_candidate",
                },
                {
                    "kind": "defaults",
                    "field": "default_model",
                    "before": "ollama:llama3",
                    "after": "openrouter:openai/gpt-4o-mini",
                    "reason": "selected_best_available_candidate",
                },
            ],
            "reasons": ["selected remote model openrouter:openai/gpt-4o-mini"],
            "proposed_defaults": {
                "routing_mode": "prefer_best",
                "default_provider": "openrouter",
                "default_model": "openrouter:openai/gpt-4o-mini",
                "allow_remote_fallback": True,
            },
            "impact": {"changes_count": 2},
        }
        with patch("agent.api_server.build_autoconfig_plan", return_value=remote_plan):
            ok, body = runtime.llm_autoconfig_apply({"actor": "webui", "confirm": True})
        self.assertTrue(ok)
        self.assertFalse(body["applied"])
        self.assertTrue(body["safe_mode_blocked"])
        self.assertIn("blocked switching default", str(body["safe_mode_blocked_reason"] or ""))
        health = runtime.llm_health_summary()["health"]
        self.assertTrue(bool((health.get("autopilot") or {}).get("safe_mode")))
        self.assertIn("blocked switching default", str((health.get("autopilot") or {}).get("last_blocked_reason") or ""))

    def test_safe_mode_blocks_remote_default_switch_for_self_heal(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.snapshots_dir, self.ledger_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.self_heal.apply": True}})
        remote_plan = {
            "ok": True,
            "drift": {"has_drift": True, "reasons": ["default_model_missing"], "details": {}},
            "changes": [
                {
                    "kind": "defaults",
                    "field": "default_provider",
                    "before": "ollama",
                    "after": "openrouter",
                    "reason": "self_heal_drift_repair",
                },
                {
                    "kind": "defaults",
                    "field": "default_model",
                    "before": "ollama:llama3",
                    "after": "openrouter:openai/gpt-4o-mini",
                    "reason": "self_heal_drift_repair",
                },
            ],
            "impact": {"changes_count": 2},
            "proposed_defaults": {
                "routing_mode": "prefer_best",
                "default_provider": "openrouter",
                "default_model": "openrouter:openai/gpt-4o-mini",
                "allow_remote_fallback": True,
            },
        }
        with patch("agent.api_server.build_self_heal_plan", return_value=remote_plan):
            ok, body = runtime.llm_self_heal_apply({"actor": "webui", "confirm": True}, trigger="manual")
        self.assertTrue(ok)
        self.assertFalse(body["applied"])
        self.assertTrue(body["safe_mode_blocked"])
        self.assertIn("blocked switching default", str(body["safe_mode_blocked_reason"] or ""))

    def test_safe_mode_blocks_remote_enable_changes_for_cleanup(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.snapshots_dir, self.ledger_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.registry.prune": True}})
        preview_plan = {
            "ok": True,
            "changes": [
                {
                    "kind": "provider",
                    "id": "openrouter",
                    "field": "enabled",
                    "before": False,
                    "after": True,
                    "reason": "test_enable_remote",
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
                    "kind": "provider",
                    "id": "openrouter",
                    "field": "enabled",
                    "before": False,
                    "after": True,
                    "reason": "test_enable_remote",
                }
            ],
            "prune_candidates": [],
            "impact": {"changes_count": 1},
            "updated_document": runtime.registry_document,
        }
        with patch("agent.api_server.build_registry_cleanup_plan", side_effect=[preview_plan, apply_plan]):
            ok, body = runtime.llm_cleanup_apply({"actor": "webui", "confirm": True}, trigger="manual")
        self.assertTrue(ok)
        self.assertFalse(body["applied"])
        self.assertTrue(body["safe_mode_blocked"])
        self.assertIn("blocked enabling remote provider", str(body["safe_mode_blocked_reason"] or ""))

    def test_ledger_records_snapshot_hash_and_sorted_changed_ids(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                self.snapshots_dir,
                self.ledger_path,
                llm_autopilot_safe_mode=False,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.autoconfig.apply": True}})
        plan = {
            "ok": True,
            "changes": [
                {
                    "kind": "defaults",
                    "field": "allow_remote_fallback",
                    "before": True,
                    "after": False,
                    "reason": "test_change",
                },
                {
                    "kind": "defaults",
                    "field": "routing_mode",
                    "before": "auto",
                    "after": "prefer_local_lowest_cost_capable",
                    "reason": "test_change",
                },
            ],
            "reasons": ["test_change"],
            "proposed_defaults": {
                "routing_mode": "prefer_local_lowest_cost_capable",
                "default_provider": "ollama",
                "default_model": "ollama:llama3",
                "allow_remote_fallback": False,
            },
            "impact": {"changes_count": 2},
        }
        with patch("agent.api_server.build_autoconfig_plan", return_value=plan):
            ok, body = runtime.llm_autoconfig_apply({"actor": "webui", "confirm": True})
        self.assertTrue(ok)
        self.assertTrue(body["applied"])
        self.assertTrue(str(body.get("snapshot_id") or ""))
        self.assertEqual(64, len(str(body.get("resulting_registry_hash") or "")))

        ledger_rows = runtime.llm_autopilot_ledger(limit=5)["entries"]
        self.assertTrue(ledger_rows)
        latest = ledger_rows[0]
        self.assertEqual("llm.autoconfig.apply", latest["action"])
        self.assertEqual(str(body["snapshot_id"]), latest["snapshot_id"])
        self.assertEqual(str(body["resulting_registry_hash"]), latest["resulting_registry_hash"])
        self.assertEqual(sorted(body["modified_ids"]), latest["changed_ids"])
        audit_entries = runtime.get_audit(limit=10)["entries"]
        apply_entry = next(
            (
                row
                for row in audit_entries
                if row.get("action") == "llm.autoconfig.apply" and str(row.get("outcome") or "") == "success"
            ),
            None,
        )
        self.assertIsNotNone(apply_entry)
        params = (
            (apply_entry or {}).get("params_redacted")
            if isinstance((apply_entry or {}).get("params_redacted"), dict)
            else {}
        )
        self.assertEqual(str(body["snapshot_id"]), str(params.get("snapshot_id") or ""))
        self.assertEqual(str(body["resulting_registry_hash"]), str(params.get("resulting_registry_hash") or ""))
        self.assertEqual(sorted(body["modified_ids"]), sorted(params.get("changed_ids") or []))


if __name__ == "__main__":
    unittest.main()
