from __future__ import annotations

import os
import tempfile
import unittest

from agent.config import Config
from agent.llm.default_model_policy import choose_best_default_chat_candidate


def _config(tmpdir: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=os.path.join(tmpdir, "agent.db"),
        log_path=os.path.join(tmpdir, "agent.log"),
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
        llm_usage_stats_path=os.path.join(tmpdir, "usage.json"),
        llm_health_state_path=os.path.join(tmpdir, "health.json"),
        llm_automation_enabled=False,
        model_scout_state_path=os.path.join(tmpdir, "scout.json"),
        autopilot_notify_store_path=os.path.join(tmpdir, "notify.json"),
        default_policy={
            "cost_cap_per_1m": 1.0,
            "allowlist": [],
            "quality_weight": 1.0,
            "price_weight": 0.04,
            "latency_weight": 0.25,
            "instability_weight": 0.5,
        },
        default_switch_cheap_remote_cap_per_1m=0.5,
    )
    return base.__class__(**{**base.__dict__, **overrides})


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
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
            },
        },
        "models": {
            "ollama:cheap-local": {
                "provider": "ollama",
                "model": "cheap-local",
                "capabilities": ["chat"],
                "quality_rank": 3,
                "cost_rank": 1,
                "enabled": True,
                "available": True,
                "max_context_tokens": 8192,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            },
            "ollama:strong-local": {
                "provider": "ollama",
                "model": "strong-local",
                "capabilities": ["chat"],
                "quality_rank": 8,
                "cost_rank": 3,
                "enabled": True,
                "available": True,
                "max_context_tokens": 65536,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            },
            "ollama:context-local": {
                "provider": "ollama",
                "model": "context-local",
                "capabilities": ["chat"],
                "quality_rank": 3,
                "cost_rank": 2,
                "enabled": True,
                "available": True,
                "max_context_tokens": 65536,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            },
            "ollama:slight-local": {
                "provider": "ollama",
                "model": "slight-local",
                "capabilities": ["chat"],
                "quality_rank": 3,
                "cost_rank": 1,
                "enabled": True,
                "available": True,
                "max_context_tokens": 12000,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            },
            "openrouter:free-remote": {
                "provider": "openrouter",
                "model": "free-remote",
                "capabilities": ["chat"],
                "quality_rank": 7,
                "cost_rank": 2,
                "enabled": True,
                "available": True,
                "max_context_tokens": 65536,
                "pricing": {"input_per_million_tokens": 0.0, "output_per_million_tokens": 0.0},
            },
            "openrouter:cheap-remote": {
                "provider": "openrouter",
                "model": "cheap-remote",
                "capabilities": ["chat"],
                "quality_rank": 9,
                "cost_rank": 3,
                "enabled": True,
                "available": True,
                "max_context_tokens": 131072,
                "pricing": {"input_per_million_tokens": 0.1, "output_per_million_tokens": 0.1},
            },
            "openrouter:mid-remote": {
                "provider": "openrouter",
                "model": "mid-remote",
                "capabilities": ["chat"],
                "quality_rank": 10,
                "cost_rank": 4,
                "enabled": True,
                "available": True,
                "max_context_tokens": 131072,
                "pricing": {"input_per_million_tokens": 0.3, "output_per_million_tokens": 0.3},
            },
            "openrouter:expensive-remote": {
                "provider": "openrouter",
                "model": "expensive-remote",
                "capabilities": ["chat"],
                "quality_rank": 10,
                "cost_rank": 5,
                "enabled": True,
                "available": True,
                "max_context_tokens": 131072,
                "pricing": {"input_per_million_tokens": 2.0, "output_per_million_tokens": 4.0},
            },
        },
        "defaults": {
            "routing_mode": "auto",
            "default_provider": None,
            "default_model": None,
            "allow_remote_fallback": True,
        },
    }


def _health_summary(*, local_status: str = "ok", free_status: str = "ok", cheap_status: str = "ok", expensive_status: str = "ok") -> dict[str, object]:
    return {
        "providers": [
            {"id": "ollama", "status": local_status},
            {"id": "openrouter", "status": "ok"},
        ],
        "models": [
            {"id": "ollama:cheap-local", "status": local_status},
            {"id": "ollama:strong-local", "status": local_status},
            {"id": "ollama:context-local", "status": local_status},
            {"id": "ollama:slight-local", "status": local_status},
            {"id": "openrouter:free-remote", "status": free_status},
            {"id": "openrouter:cheap-remote", "status": cheap_status},
            {"id": "openrouter:mid-remote", "status": cheap_status},
            {"id": "openrouter:expensive-remote", "status": expensive_status},
        ],
    }


class TestDefaultModelPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.env = {"OPENROUTER_API_KEY": "sk-test"}

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_strongest_healthy_local_beats_cheaper_weaker_local(self) -> None:
        result = choose_best_default_chat_candidate(
            config=_config(self.tmpdir.name),
            registry_document=_registry_document(),
            health_summary=_health_summary(),
            env=self.env,
        )
        self.assertTrue(result["switch_recommended"])
        self.assertEqual("ollama:strong-local", result["recommended_candidate"]["model_id"])
        self.assertEqual("local", result["recommended_candidate"]["tier"])

    def test_healthy_local_beats_free_remote_when_good_enough(self) -> None:
        result = choose_best_default_chat_candidate(
            config=_config(self.tmpdir.name),
            registry_document=_registry_document(),
            health_summary=_health_summary(),
            env=self.env,
        )
        self.assertEqual("ollama:strong-local", result["recommended_candidate"]["model_id"])

    def test_free_remote_beats_paid_remote_when_no_local_is_suitable(self) -> None:
        result = choose_best_default_chat_candidate(
            config=_config(self.tmpdir.name),
            registry_document=_registry_document(),
            health_summary=_health_summary(local_status="down"),
            env=self.env,
        )
        self.assertEqual("openrouter:free-remote", result["recommended_candidate"]["model_id"])
        self.assertEqual("free_remote", result["recommended_candidate"]["tier"])

    def test_cheap_remote_can_be_selected_when_no_free_remote_exists(self) -> None:
        result = choose_best_default_chat_candidate(
            config=_config(self.tmpdir.name),
            registry_document=_registry_document(),
            health_summary=_health_summary(local_status="down", free_status="down"),
            env=self.env,
        )
        self.assertEqual("openrouter:cheap-remote", result["recommended_candidate"]["model_id"])
        self.assertEqual("cheap_remote", result["recommended_candidate"]["tier"])
        self.assertEqual(0.5, result["cheap_remote_cap_per_1m"])

    def test_strict_cheap_remote_cap_blocks_remote_that_general_cap_would_allow(self) -> None:
        document = _registry_document()
        document["models"].pop("openrouter:free-remote")  # type: ignore[index]
        document["models"].pop("openrouter:cheap-remote")  # type: ignore[index]
        result = choose_best_default_chat_candidate(
            config=_config(self.tmpdir.name),
            registry_document=document,
            health_summary=_health_summary(local_status="down", free_status="down"),
            env=self.env,
        )
        self.assertFalse(result["switch_recommended"])
        self.assertIsNone(result["recommended_candidate"])
        self.assertEqual("cheap_remote_cap_exceeded", result["rejected_candidates"][0]["reason"])
        self.assertEqual(0.5, result["cheap_remote_cap_per_1m"])
        self.assertEqual(1.0, result["general_remote_cap_per_1m"])

    def test_over_cap_expensive_remote_is_not_selected(self) -> None:
        document = _registry_document()
        document["models"].pop("openrouter:free-remote")  # type: ignore[index]
        document["models"].pop("openrouter:cheap-remote")  # type: ignore[index]
        result = choose_best_default_chat_candidate(
            config=_config(self.tmpdir.name),
            registry_document=document,
            health_summary=_health_summary(local_status="down"),
            env=self.env,
        )
        self.assertFalse(result["switch_recommended"])
        self.assertIsNone(result["recommended_candidate"])
        self.assertEqual("no_candidate", result["decision_reason"])

    def test_current_healthy_default_is_retained_when_challenger_is_not_materially_better(self) -> None:
        document = _registry_document()
        document["defaults"] = {
            "routing_mode": "auto",
            "default_provider": "ollama",
            "default_model": "ollama:cheap-local",
            "allow_remote_fallback": True,
        }
        result = choose_best_default_chat_candidate(
            config=_config(self.tmpdir.name),
            registry_document=document,
            health_summary=_health_summary(),
            env=self.env,
            candidate_model_ids=["ollama:cheap-local", "ollama:slight-local"],
        )
        self.assertFalse(result["switch_recommended"])
        self.assertEqual("ollama:cheap-local", result["selected_candidate"]["model_id"])

    def test_clear_quality_upgrade_triggers_switch(self) -> None:
        document = _registry_document()
        document["defaults"] = {
            "routing_mode": "auto",
            "default_provider": "ollama",
            "default_model": "ollama:cheap-local",
            "allow_remote_fallback": True,
        }
        result = choose_best_default_chat_candidate(
            config=_config(self.tmpdir.name),
            registry_document=document,
            health_summary=_health_summary(),
            env=self.env,
        )
        self.assertTrue(result["switch_recommended"])
        self.assertEqual("quality_upgrade", result["decision_reason"])
        self.assertEqual("ollama:strong-local", result["recommended_candidate"]["model_id"])


if __name__ == "__main__":
    unittest.main()
