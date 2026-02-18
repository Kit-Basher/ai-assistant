from __future__ import annotations

import json
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
        autopilot_notify_enabled=True,
        autopilot_notify_rate_limit_seconds=0,
        autopilot_notify_dedupe_window_seconds=0,
        llm_self_heal_allow_apply=None,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _registry_doc_with_drift() -> dict[str, object]:
    return {
        "schema_version": 2,
        "providers": {
            "ollama": {
                "provider_type": "openai_compat",
                "base_url": "http://127.0.0.1:11434",
                "chat_path": "/v1/chat/completions",
                "api_key_source": None,
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": True,
            },
            "openrouter": {
                "provider_type": "openai_compat",
                "base_url": "https://openrouter.ai/api/v1",
                "chat_path": "/chat/completions",
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": False,
            },
        },
        "models": {
            "ollama:qwen2.5:3b-instruct": {
                "provider": "ollama",
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "quality_rank": 2,
                "cost_rank": 0,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
                "max_context_tokens": 8192,
            },
            "openrouter:openai/gpt-4o-mini": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "quality_rank": 6,
                "cost_rank": 3,
                "pricing": {"input_per_million_tokens": 0.15, "output_per_million_tokens": 0.6},
                "max_context_tokens": 128000,
            },
        },
        "defaults": {
            "routing_mode": "prefer_local_lowest_cost_capable",
            "default_provider": "ollama",
            "default_model": "ollama:missing",
            "allow_remote_fallback": True,
        },
    }


def _healthy_snapshot() -> dict[str, object]:
    return {
        "providers": [
            {"id": "ollama", "available": True, "health": {"status": "ok"}},
            {"id": "openrouter", "available": True, "health": {"status": "ok"}},
        ],
        "models": [
            {
                "id": "ollama:qwen2.5:3b-instruct",
                "provider": "ollama",
                "available": True,
                "enabled": True,
                "capabilities": ["chat"],
                "routable": True,
                "health": {"status": "ok"},
            },
            {
                "id": "openrouter:openai/gpt-4o-mini",
                "provider": "openrouter",
                "available": True,
                "enabled": True,
                "capabilities": ["chat"],
                "routable": True,
                "health": {"status": "ok"},
            },
        ],
    }


def _healthy_summary() -> dict[str, object]:
    return {
        "providers": [
            {"id": "ollama", "status": "ok"},
            {"id": "openrouter", "status": "ok"},
        ],
        "models": [
            {"id": "ollama:qwen2.5:3b-instruct", "status": "ok"},
            {"id": "openrouter:openai/gpt-4o-mini", "status": "ok"},
        ],
    }


def _notify_state(default_model: str) -> dict[str, object]:
    return {
        "defaults": {
            "routing_mode": "prefer_local_lowest_cost_capable",
            "default_provider": "ollama",
            "default_model": default_model,
            "allow_remote_fallback": True,
        },
        "providers": {
            "ollama": {
                "enabled": True,
                "available": True,
                "health": {
                    "status": "ok",
                    "cooldown_until": None,
                    "down_since": None,
                    "failure_streak": 0,
                },
            }
        },
        "models": {
            "ollama:qwen2.5:3b-instruct": {
                "enabled": True,
                "available": True,
                "routable": True,
                "health": {
                    "status": "ok",
                    "cooldown_until": None,
                    "down_since": None,
                    "failure_streak": 0,
                },
            }
        },
    }


class TestLLMSelfHealAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        with open(self.registry_path, "w", encoding="utf-8") as handle:
            json.dump(_registry_doc_with_drift(), handle)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_scheduler_loopback_auto_apply_allows_self_heal_without_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_self_heal_allow_apply=None))
        runtime.set_listening("127.0.0.1", 8765)
        with patch.object(runtime._health_monitor, "summary", return_value=_healthy_summary()), patch.object(
            runtime._router, "doctor_snapshot", return_value=_healthy_snapshot()
        ):
            ok, body = runtime.llm_self_heal_apply({"actor": "scheduler"}, trigger="scheduler")
        self.assertTrue(ok)
        self.assertTrue(body["applied"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", body["defaults"]["default_model"])

    def test_scheduler_non_loopback_denies_without_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_self_heal_allow_apply=None))
        runtime.set_listening("0.0.0.0", 8765)
        with patch.object(runtime._health_monitor, "summary", return_value=_healthy_summary()), patch.object(
            runtime._router, "doctor_snapshot", return_value=_healthy_snapshot()
        ):
            ok, body = runtime.llm_self_heal_apply({"actor": "scheduler"}, trigger="scheduler")
        self.assertFalse(ok)
        self.assertEqual("action_not_permitted", body["error"])

    def test_scheduler_order_runs_self_heal_before_autoconfig(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_automation_enabled=True))
        order: list[str] = []
        notify_calls: list[dict[str, object]] = []
        runtime._scheduler_next_run = {
            "refresh": 0.0,
            "catalog": 0.0,
            "capabilities_reconcile": 0.0,
            "health": 0.0,
            "hygiene": 0.0,
            "cleanup": 0.0,
            "self_heal": 0.0,
            "autoconfig": 0.0,
            "model_scout": 999999999.0,
        }

        with patch.object(runtime._scheduler_stop, "wait", side_effect=[False, True]), patch.object(
            runtime, "_autopilot_notify_state_snapshot", return_value={}
        ), patch.object(
            runtime,
            "_process_scheduler_notification_cycle",
            side_effect=lambda **kwargs: (notify_calls.append(kwargs), {"ok": True})[1],
        ), patch.object(
            runtime, "refresh_models", side_effect=lambda _payload: (order.append("refresh"), (True, {"changed": False}))[1]
        ), patch.object(
            runtime,
            "run_llm_catalog_refresh",
            side_effect=lambda trigger="scheduler": (
                order.append("catalog"),
                (
                    True,
                    {
                        "changed": True,
                        "notable_changes": [
                            "Catalog: pricing updated for openrouter:openai/gpt-4o-mini",
                            "Catalog: added ollama:qwen2.5:3b-instruct",
                            "Catalog: pricing updated for openrouter:openai/gpt-4o-mini",
                        ],
                    },
                ),
            )[1],
        ), patch.object(
            runtime,
            "llm_capabilities_reconcile_apply",
            side_effect=lambda _payload, trigger="scheduler": (
                order.append("reconcile"),
                (
                    True,
                    {
                        "applied": True,
                        "plan": {
                            "reasons": ["inferred_capability_mismatch"],
                            "changes": [
                                {
                                    "kind": "model",
                                    "id": "ollama:nomic-embed-text:latest",
                                    "field": "capabilities",
                                    "after": ["embedding"],
                                    "reason": "inferred_capability_mismatch",
                                }
                            ],
                        },
                    },
                ),
            )[1],
        ), patch.object(
            runtime, "run_llm_health", side_effect=lambda **_kwargs: (order.append("health"), (True, {"ok": True}))[1]
        ), patch.object(
            runtime, "llm_hygiene_apply", side_effect=lambda _payload: (order.append("hygiene"), (True, {"applied": False}))[1]
        ), patch.object(
            runtime,
            "llm_cleanup_apply",
            side_effect=lambda _payload, trigger="scheduler": (
                order.append("cleanup"),
                (
                    True,
                    {
                        "applied": True,
                        "plan": {
                            "reasons": ["missing_from_catalog"],
                            "changes": [
                                {
                                    "kind": "model",
                                    "id": "ollama:llama3",
                                    "field": "available",
                                    "after": False,
                                    "reason": "missing_from_catalog",
                                }
                            ],
                        },
                    },
                ),
            )[1],
        ), patch.object(
            runtime, "_current_drift_report", return_value={"has_drift": True, "reasons": ["default_model_missing"], "details": {}}
        ), patch.object(
            runtime,
            "llm_self_heal_apply",
            side_effect=lambda _payload, trigger="scheduler": (
                order.append("self_heal"),
                (
                    True,
                    {
                        "applied": True,
                        "plan": {
                            "reasons": [
                                "drift_repair(default_model_missing); picked ollama:qwen2.5:3b-instruct because same_provider_local+chat+routable+health_ok+cost=unknown"
                            ],
                            "changes": [{"kind": "defaults", "field": "default_model", "after": "ollama:qwen2.5:3b-instruct"}],
                        },
                    },
                ),
            )[1],
        ), patch.object(
            runtime, "llm_autoconfig_apply", side_effect=lambda _payload: (order.append("autoconfig"), (True, {"applied": False}))[1]
        ):
            runtime._scheduler_loop()
        self.assertEqual(
            ["refresh", "catalog", "reconcile", "health", "hygiene", "cleanup", "self_heal", "autoconfig"],
            order,
        )
        self.assertEqual(1, len(notify_calls))
        extra = list(notify_calls[0]["extra_changes"])
        if extra and isinstance(extra[0], str) and extra[0].startswith(
            "Autopilot paused applies after churn detection"
        ):
            extra = extra[1:]
        self.assertEqual(
            [
                "Capabilities: ollama:nomic-embed-text:latest updated capabilities",
                "Catalog: added ollama:qwen2.5:3b-instruct",
                "Catalog: pricing updated for openrouter:openai/gpt-4o-mini",
            ],
            extra,
        )
        self.assertIn("missing_from_catalog", notify_calls[0]["reasons"])
        self.assertIn("refreshed_model_catalog", notify_calls[0]["reasons"])

    def test_autoconfig_noops_after_self_heal_apply(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions({"mode": "auto", "actions": {"llm.self_heal.apply": True}})
        with patch.object(runtime._health_monitor, "summary", return_value=_healthy_summary()), patch.object(
            runtime._router, "doctor_snapshot", return_value=_healthy_snapshot()
        ):
            ok, body = runtime.llm_self_heal_apply({"actor": "test", "confirm": True}, trigger="manual")
        self.assertTrue(ok)
        self.assertTrue(body["applied"])

        ok_plan, plan_payload = runtime.llm_autoconfig_plan({"actor": "test"})
        self.assertTrue(ok_plan)
        self.assertEqual(0, int((plan_payload.get("plan") or {}).get("impact", {}).get("changes_count") or 0))

    def test_notification_reason_includes_drift_and_rationale(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        with patch("agent.api_server.time.time", return_value=77_000), patch.object(
            runtime, "_resolve_telegram_target", return_value=(None, None)
        ):
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=_notify_state("ollama:missing"),
                after_state=_notify_state("ollama:qwen2.5:3b-instruct"),
                reasons=[
                    "drift_repair(default_model_missing,default_model_unroutable); picked ollama:qwen2.5:3b-instruct because same_provider_local+chat+routable+health_ok+cost=unknown"
                ],
                trigger="scheduler",
            )
        self.assertEqual("sent", result["reason"])
        rows = runtime.llm_notifications(limit=1)["notifications"]
        self.assertEqual(1, len(rows))
        self.assertIn("Reason: drift_repair(default_model_missing,default_model_unroutable)", rows[0]["message"])
        self.assertIn("picked ollama:qwen2.5:3b-instruct", rows[0]["message"])


if __name__ == "__main__":
    unittest.main()
