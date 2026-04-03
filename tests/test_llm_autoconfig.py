from __future__ import annotations

import unittest

from agent.config import Config
from agent.llm.autoconfig import build_autoconfig_plan


def _config() -> Config:
    return Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path="/tmp/agent.db",
        log_path="/tmp/agent.log",
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
        llm_registry_path=None,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path="/tmp/usage.json",
        llm_health_state_path="/tmp/health.json",
        llm_automation_enabled=False,
        model_scout_state_path="/tmp/scout.json",
        autopilot_notify_store_path="/tmp/notify.json",
    )


def _registry_document() -> dict[str, object]:
    return {
        "providers": {
            "ollama": {
                "enabled": True,
                "local": True,
                "api_key_source": None,
            },
            "openrouter": {
                "enabled": True,
                "local": False,
                "api_key_source": {
                    "type": "env",
                    "name": "OPENROUTER_API_KEY",
                },
            },
        },
        "models": {
            "ollama:llama3": {
                "provider": "ollama",
                "model": "llama3",
                "capabilities": ["chat"],
                "quality_rank": 3,
                "cost_rank": 1,
                "enabled": True,
                "available": True,
            },
            "openrouter:openai/gpt-4o-mini": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat", "json", "tools"],
                "quality_rank": 6,
                "cost_rank": 3,
                "enabled": True,
                "available": True,
                "pricing": {"input_per_million_tokens": 0.1, "output_per_million_tokens": 0.1},
            },
        },
        "defaults": {
            "routing_mode": "auto",
            "default_provider": None,
            "default_model": None,
            "allow_remote_fallback": True,
        },
    }


class TestLLMAutoconfig(unittest.TestCase):
    def test_plan_prefers_local_when_healthy(self) -> None:
        summary = {
            "providers": [
                {"id": "ollama", "status": "ok"},
                {"id": "openrouter", "status": "ok"},
            ],
            "models": [
                {"id": "ollama:llama3", "status": "ok"},
                {"id": "openrouter:openai/gpt-4o-mini", "status": "ok"},
            ],
        }
        plan = build_autoconfig_plan(
            _registry_document(),
            summary,
            config=_config(),
            env={"OPENROUTER_API_KEY": "sk-test"},
        )
        proposed = plan["proposed_defaults"]
        self.assertEqual("ollama", proposed["default_provider"])
        self.assertEqual("ollama:llama3", proposed["default_model"])
        self.assertEqual("prefer_local_lowest_cost_capable", proposed["routing_mode"])

    def test_plan_falls_back_to_remote_and_can_disable_auth_failed_provider(self) -> None:
        doc = _registry_document()
        doc["models"]["ollama:llama3"]["available"] = False  # type: ignore[index]
        summary = {
            "providers": [
                {"id": "ollama", "status": "down", "last_error_kind": "provider_unavailable"},
                {"id": "openrouter", "status": "down", "last_error_kind": "auth_error", "status_code": 401},
            ],
            "models": [
                {"id": "ollama:llama3", "status": "down"},
                {"id": "openrouter:openai/gpt-4o-mini", "status": "ok"},
            ],
        }

        plan = build_autoconfig_plan(
            doc,
            summary,
            config=_config(),
            env={"OPENROUTER_API_KEY": "sk-test"},
            disable_auth_failed_providers=True,
        )
        provider_disable_changes = [
            row
            for row in plan["changes"]
            if row.get("kind") == "provider" and row.get("id") == "openrouter" and row.get("field") == "enabled"
        ]
        self.assertEqual(1, len(provider_disable_changes))
        self.assertFalse(provider_disable_changes[0]["after"])

    def test_plan_selects_remote_when_local_is_down(self) -> None:
        summary = {
            "providers": [
                {"id": "ollama", "status": "down", "last_error_kind": "provider_unavailable"},
                {"id": "openrouter", "status": "ok"},
            ],
            "models": [
                {"id": "ollama:llama3", "status": "down"},
                {"id": "openrouter:openai/gpt-4o-mini", "status": "ok"},
            ],
        }
        plan = build_autoconfig_plan(
            _registry_document(),
            summary,
            config=_config(),
            env={"OPENROUTER_API_KEY": "sk-test"},
        )
        proposed = plan["proposed_defaults"]
        self.assertEqual("openrouter", proposed["default_provider"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", proposed["default_model"])
        self.assertEqual("prefer_best", proposed["routing_mode"])

    def test_plan_keeps_existing_healthy_defaults(self) -> None:
        doc = _registry_document()
        doc["defaults"] = {
            "routing_mode": "prefer_local_lowest_cost_capable",
            "default_provider": "ollama",
            "default_model": "ollama:llama3",
            "allow_remote_fallback": True,
        }
        summary = {
            "providers": [
                {"id": "ollama", "status": "ok"},
                {"id": "openrouter", "status": "ok"},
            ],
            "models": [
                {"id": "ollama:llama3", "status": "ok"},
                {"id": "openrouter:openai/gpt-4o-mini", "status": "ok"},
            ],
        }
        plan = build_autoconfig_plan(
            doc,
            summary,
            config=_config(),
            env={"OPENROUTER_API_KEY": "sk-test"},
        )
        self.assertEqual([], plan["changes"])
        self.assertIn("keep_current_default_model", plan["reasons"])


if __name__ == "__main__":
    unittest.main()
