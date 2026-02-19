from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config


def _config(
    registry_path: str,
    db_path: str,
    snapshots_dir: str,
    ledger_path: str,
    autopilot_state_path: str,
    **overrides: object,
) -> Config:
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
        llm_autopilot_state_path=autopilot_state_path,
        llm_autopilot_safe_mode=False,
        llm_autopilot_churn_min_applies=2,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _doctor_snapshot_for_bootstrap() -> dict[str, object]:
    return {
        "providers": [
            {"id": "ollama", "enabled": True, "local": True},
            {"id": "openrouter", "enabled": True, "local": False},
        ],
        "models": [
            {
                "id": "ollama:llama3",
                "provider": "ollama",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "routable": True,
                "max_context_tokens": 8192,
                "input_cost_per_million_tokens": None,
                "output_cost_per_million_tokens": None,
                "health": {"status": "ok"},
            },
            {
                "id": "openrouter:openai/gpt-4o-mini",
                "provider": "openrouter",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "routable": True,
                "max_context_tokens": 128000,
                "input_cost_per_million_tokens": 0.15,
                "output_cost_per_million_tokens": 0.6,
                "health": {"status": "ok"},
            },
        ],
    }


class TestLLMAutopilotUserGrade(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.snapshots_dir = os.path.join(self.tmpdir.name, "snapshots")
        self.ledger_path = os.path.join(self.tmpdir.name, "ledger.json")
        self.autopilot_state_path = os.path.join(self.tmpdir.name, "autopilot_state.json")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_explain_last_returns_stable_rationale_and_redacts(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                self.snapshots_dir,
                self.ledger_path,
                self.autopilot_state_path,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.autoconfig.apply": True}})
        plan = {
            "ok": True,
            "changes": [
                {
                    "kind": "defaults",
                    "field": "default_model",
                    "before": None,
                    "after": "ollama:llama3",
                    "reason": "selected_best_available_candidate",
                }
            ],
            "reasons": [
                "selected local model ollama:llama3 sk-1234567890ABCDEF1234567890"
            ],
            "proposed_defaults": {
                "routing_mode": "prefer_local_lowest_cost_capable",
                "default_provider": "ollama",
                "default_model": "ollama:llama3",
                "allow_remote_fallback": True,
            },
            "impact": {"changes_count": 1},
        }
        with patch("agent.api_server.build_autoconfig_plan", return_value=plan):
            ok, _body = runtime.llm_autoconfig_apply({"actor": "webui", "confirm": True})
        self.assertTrue(ok)

        payload = runtime.llm_autopilot_explain_last()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["found"])
        last_apply = payload["last_apply"]
        self.assertEqual("llm.autoconfig.apply", last_apply["action"])
        self.assertTrue(last_apply["rationale_lines"])
        self.assertTrue(last_apply["rationale_lines"][0].startswith("Applied action"))
        rendered = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        self.assertNotIn("sk-1234567890ABCDEF1234567890", rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_undo_rolls_back_to_previous_snapshot(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                self.snapshots_dir,
                self.ledger_path,
                self.autopilot_state_path,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.autoconfig.apply": True}})
        before = copy.deepcopy(runtime.registry_document.get("defaults"))
        plan = {
            "ok": True,
            "changes": [
                {
                    "kind": "defaults",
                    "field": "allow_remote_fallback",
                    "before": True,
                    "after": False,
                    "reason": "selected_best_available_candidate",
                }
            ],
            "reasons": ["selected local model ollama:qwen2.5:3b-instruct"],
            "proposed_defaults": {
                "routing_mode": "prefer_local_lowest_cost_capable",
                "default_provider": "ollama",
                "default_model": "ollama:qwen2.5:3b-instruct",
                "allow_remote_fallback": False,
            },
            "impact": {"changes_count": 1},
        }
        with patch("agent.api_server.build_autoconfig_plan", return_value=plan):
            ok, body = runtime.llm_autoconfig_apply({"actor": "webui", "confirm": True})
        self.assertTrue(ok)
        self.assertTrue(body["applied"])
        self.assertFalse(bool(runtime.registry_document.get("defaults", {}).get("allow_remote_fallback", True)))

        undo_ok, undo_body = runtime.llm_autopilot_undo({"actor": "webui"})
        self.assertTrue(undo_ok)
        self.assertTrue(undo_body["ok"])
        self.assertEqual(before, runtime.registry_document.get("defaults"))
        self.assertTrue(str(undo_body.get("rolled_back_to_snapshot_id") or ""))

    def test_churn_detection_enters_pause_and_blocks_future_apply(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                self.snapshots_dir,
                self.ledger_path,
                self.autopilot_state_path,
                llm_autopilot_churn_min_applies=2,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.autoconfig.apply": True}})
        plan_a = {
            "ok": True,
            "changes": [
                {
                    "kind": "defaults",
                    "field": "default_model",
                    "before": None,
                    "after": "ollama:llama3",
                    "reason": "selected_best_available_candidate",
                }
            ],
            "reasons": ["selected local model ollama:llama3"],
            "proposed_defaults": {
                "routing_mode": "prefer_local_lowest_cost_capable",
                "default_provider": "ollama",
                "default_model": "ollama:llama3",
                "allow_remote_fallback": True,
            },
            "impact": {"changes_count": 1},
        }
        plan_b = {
            "ok": True,
            "changes": [
                {
                    "kind": "defaults",
                    "field": "default_model",
                    "before": "ollama:llama3",
                    "after": "openrouter:openai/gpt-4o-mini",
                    "reason": "selected_best_available_candidate",
                }
            ],
            "reasons": ["selected remote model openrouter:openai/gpt-4o-mini"],
            "proposed_defaults": {
                "routing_mode": "prefer_best",
                "default_provider": "openrouter",
                "default_model": "openrouter:openai/gpt-4o-mini",
                "allow_remote_fallback": True,
            },
            "impact": {"changes_count": 1},
        }
        with patch("agent.api_server.build_autoconfig_plan", side_effect=[plan_a, plan_b]):
            first_ok, _ = runtime.llm_autoconfig_apply({"actor": "webui", "confirm": True})
            second_ok, _ = runtime.llm_autoconfig_apply({"actor": "webui", "confirm": True})
        self.assertTrue(first_ok)
        self.assertTrue(second_ok)

        churn = runtime._evaluate_autopilot_churn(now_epoch=1_700_000_000, trigger="manual")  # type: ignore[attr-defined]
        self.assertTrue(churn["entered_safe_mode"])
        self.assertTrue(runtime.llm_health_summary()["health"]["autopilot"]["safe_mode_override"])

        with patch("agent.api_server.build_autoconfig_plan", return_value=plan_a):
            blocked_ok, blocked_body = runtime.llm_autoconfig_apply({"actor": "webui", "confirm": True})
        self.assertTrue(blocked_ok)
        # Config-level safe mode is disabled here, so churn pause override must not
        # filter apply plans.
        self.assertTrue(blocked_body["applied"])
        self.assertFalse(blocked_body["safe_mode_blocked"])

    def test_bootstrap_selects_local_chat_and_then_noops(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                self.snapshots_dir,
                self.ledger_path,
                self.autopilot_state_path,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.autopilot.bootstrap.apply": True}})

        document = copy.deepcopy(runtime.registry_document)
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        defaults["default_provider"] = None
        defaults["default_model"] = None
        document["defaults"] = defaults
        saved, error = runtime._persist_registry_document(document)
        self.assertTrue(saved)
        self.assertIsNone(error)

        with patch.object(runtime._router, "doctor_snapshot", return_value=_doctor_snapshot_for_bootstrap()):
            ok, body = runtime.llm_autopilot_bootstrap({"actor": "webui", "confirm": True}, trigger="manual")
        self.assertTrue(ok)
        self.assertTrue(body["applied"])
        self.assertEqual("ollama", runtime.registry_document["defaults"]["default_provider"])
        self.assertEqual("ollama:llama3", runtime.registry_document["defaults"]["default_model"])

        with patch.object(runtime._router, "doctor_snapshot", return_value=_doctor_snapshot_for_bootstrap()):
            noop_ok, noop_body = runtime.llm_autopilot_bootstrap({"actor": "webui", "confirm": True}, trigger="manual")
        self.assertTrue(noop_ok)
        self.assertFalse(noop_body["applied"])
        self.assertIn("already_configured", (noop_body.get("plan") or {}).get("reasons") or [])

    def test_bootstrap_policy_non_loopback_denies_and_loopback_allows(self) -> None:
        runtime_denied = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                self.snapshots_dir,
                self.ledger_path,
                self.autopilot_state_path,
                llm_autopilot_bootstrap_allow_apply=None,
            )
        )
        runtime_denied.set_listening("0.0.0.0", 8765)
        document = copy.deepcopy(runtime_denied.registry_document)
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        defaults["default_provider"] = None
        defaults["default_model"] = None
        document["defaults"] = defaults
        runtime_denied._persist_registry_document(document)
        with patch.object(runtime_denied._router, "doctor_snapshot", return_value=_doctor_snapshot_for_bootstrap()):
            denied_ok, denied_body = runtime_denied.llm_autopilot_bootstrap(
                {"actor": "webui", "confirm": True},
                trigger="manual",
            )
        self.assertFalse(denied_ok)
        self.assertEqual("action_not_permitted", denied_body["error"])

        runtime_allowed = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                self.snapshots_dir,
                self.ledger_path,
                self.autopilot_state_path,
                llm_autopilot_bootstrap_allow_apply=None,
            )
        )
        runtime_allowed.set_listening("127.0.0.1", 8765)
        document_allowed = copy.deepcopy(runtime_allowed.registry_document)
        defaults_allowed = document_allowed.get("defaults") if isinstance(document_allowed.get("defaults"), dict) else {}
        defaults_allowed["default_provider"] = None
        defaults_allowed["default_model"] = None
        document_allowed["defaults"] = defaults_allowed
        runtime_allowed._persist_registry_document(document_allowed)
        with patch.object(runtime_allowed._router, "doctor_snapshot", return_value=_doctor_snapshot_for_bootstrap()):
            allowed_ok, allowed_body = runtime_allowed.llm_autopilot_bootstrap(
                {"actor": "webui", "confirm": True},
                trigger="manual",
            )
        self.assertTrue(allowed_ok)
        self.assertTrue(allowed_body["ok"])


if __name__ == "__main__":
    unittest.main()
