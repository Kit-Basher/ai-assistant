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
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _registry_document() -> dict[str, object]:
    return {
        "defaults": {
            "routing_mode": "auto",
            "default_provider": "openrouter",
            "default_model": "openrouter:jamba-large",
            "allow_remote_fallback": True,
        },
        "providers": {
            "openrouter": {"enabled": True, "local": False, "available": True},
        },
        "models": {
            "openrouter:jamba-large": {
                "provider": "openrouter",
                "model": "jamba-large",
                "enabled": True,
                "available": True,
                "quality_rank": 10,
                "max_context_tokens": 32768,
                "pricing": {
                    "input_per_million_tokens": 0.6,
                    "output_per_million_tokens": 1.2,
                },
            },
            "openrouter:base-chat": {
                "provider": "openrouter",
                "model": "base-chat",
                "enabled": True,
                "available": True,
                "quality_rank": 3,
                "max_context_tokens": 8192,
                "pricing": {
                    "input_per_million_tokens": 0.08,
                    "output_per_million_tokens": 0.12,
                },
            },
            "openrouter:premium-reasoner": {
                "provider": "openrouter",
                "model": "premium-reasoner",
                "enabled": True,
                "available": True,
                "quality_rank": 9,
                "max_context_tokens": 131072,
                "pricing": {
                    "input_per_million_tokens": 2.0,
                    "output_per_million_tokens": 4.0,
                },
            },
        },
    }


def _doctor_snapshot() -> dict[str, object]:
    return {
        "models": [
            {"id": "openrouter:jamba-large", "routable": True, "health": {"status": "ok"}},
            {"id": "openrouter:base-chat", "routable": True, "health": {"status": "ok"}},
            {"id": "openrouter:premium-reasoner", "routable": True, "health": {"status": "ok"}},
        ]
    }


class TestValueOptimalSelection(unittest.TestCase):
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

    def test_escalation_trigger_selects_premium_model(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                premium_policy={
                    "cost_cap_per_1m": 20.0,
                    "allowlist": ["openrouter:premium-reasoner"],
                    "quality_weight": 1.35,
                    "price_weight": 0.02,
                    "latency_weight": 0.2,
                    "instability_weight": 0.45,
                },
            )
        )
        runtime.registry_document = _registry_document()
        calls: list[dict[str, object]] = []

        def _fake_route_inference(**kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            return {
                "ok": True,
                "text": "ok",
                "provider": kwargs.get("provider_override"),
                "model": kwargs.get("model_override"),
                "fallback_used": False,
                "attempts": [],
                "duration_ms": 1,
                "error_class": None,
            }

        with patch.object(runtime._router, "doctor_snapshot", return_value=_doctor_snapshot()), patch(
            "agent.api_server.route_inference",
            side_effect=_fake_route_inference,
        ):
            ok, body = runtime.chat(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "This is a high-stakes production security issue. Please reason deeply.",
                        }
                    ]
                }
            )

        self.assertTrue(ok)
        self.assertEqual("openrouter:premium-reasoner", calls[0].get("model_override"))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        policy_meta = meta.get("selection_policy") if isinstance(meta.get("selection_policy"), dict) else {}
        self.assertEqual("openrouter:premium-reasoner", policy_meta.get("premium_selected"))

    def test_over_cap_escalation_returns_wizard_prompt(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                premium_policy={
                    "cost_cap_per_1m": 1.0,
                    "allowlist": ["openrouter:premium-reasoner"],
                    "quality_weight": 1.35,
                    "price_weight": 0.02,
                    "latency_weight": 0.2,
                    "instability_weight": 0.45,
                },
            )
        )
        runtime.registry_document = _registry_document()

        with patch.object(runtime._router, "doctor_snapshot", return_value=_doctor_snapshot()), patch(
            "agent.api_server.route_inference",
        ) as route_mock:
            ok, body = runtime.chat(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Please use premium model for deep reasoning on this legal analysis.",
                        }
                    ]
                }
            )

        self.assertTrue(ok)
        route_mock.assert_not_called()
        assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
        self.assertIn("over the cost cap", str(assistant.get("content") or ""))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        policy_meta = meta.get("selection_policy") if isinstance(meta.get("selection_policy"), dict) else {}
        self.assertEqual("premium_over_cap", policy_meta.get("mode"))
        wizard_state = runtime._llm_fixit_store.state  # type: ignore[attr-defined]
        self.assertTrue(bool(wizard_state.get("active")))
        self.assertEqual("premium_over_cap", wizard_state.get("issue_code"))

    def test_api_chat_leaves_default_model_selection_to_canonical_inference(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                default_policy={
                    "cost_cap_per_1m": 10.0,
                    "allowlist": ["openrouter:base-chat"],
                    "quality_weight": 1.0,
                    "price_weight": 0.04,
                    "latency_weight": 0.25,
                    "instability_weight": 0.5,
                },
            )
        )
        runtime.registry_document = _registry_document()
        calls: list[dict[str, object]] = []

        def _fake_route_inference(**kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            return {
                "ok": True,
                "text": "ok",
                "provider": "openrouter",
                "model": "openrouter:base-chat",
                "fallback_used": False,
                "attempts": [],
                "duration_ms": 1,
                "error_class": None,
            }

        with patch.object(runtime._router, "doctor_snapshot", return_value=_doctor_snapshot()), patch(
            "agent.api_server.route_inference",
            side_effect=_fake_route_inference,
        ):
            ok, body = runtime.chat({"messages": [{"role": "user", "content": "hello"}]})

        self.assertTrue(ok)
        self.assertIsNone(calls[0].get("model_override"))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("openrouter:base-chat", meta.get("model"))


if __name__ == "__main__":
    unittest.main()
