from __future__ import annotations

import json
import os
import tempfile
import unittest

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
        llm_catalog_path=os.path.join(os.path.dirname(db_path), "llm_catalog.json"),
        llm_automation_enabled=False,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _registry_doc() -> dict[str, object]:
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
                "chat_path": "/v1/chat/completions",
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": False,
            },
        },
        "models": {
            "ollama:nomic-embed-text:latest": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest",
                "capabilities": ["chat"],
                "quality_rank": 1,
                "cost_rank": 0,
                "default_for": ["chat"],
                "enabled": True,
                "available": True,
                "pricing": {
                    "input_per_million_tokens": None,
                    "output_per_million_tokens": None,
                },
                "max_context_tokens": 8192,
            }
        },
        "defaults": {
            "routing_mode": "prefer_local_lowest_cost_capable",
            "default_provider": "ollama",
            "default_model": "ollama:nomic-embed-text:latest",
            "allow_remote_fallback": True,
            "fallback_chain": [],
        },
    }


def _catalog_doc() -> dict[str, object]:
    return {
        "schema_version": 1,
        "last_run_at": 100,
        "providers": {
            "ollama": {
                "provider_id": "ollama",
                "source": "ollama_tags",
                "last_refresh_at": 100,
                "last_error_kind": None,
                "last_error_at": None,
                "models": [
                    {
                        "id": "ollama:nomic-embed-text:latest",
                        "provider_id": "ollama",
                        "model": "nomic-embed-text:latest",
                        "capabilities": ["chat"],
                        "max_context_tokens": 8192,
                        "input_cost_per_million_tokens": None,
                        "output_cost_per_million_tokens": None,
                        "source": "ollama_tags",
                    }
                ],
            }
        },
    }


class TestLLMCapabilitiesAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_keys = (
            "AGENT_AUDIT_LOG_PATH",
            "AGENT_PERMISSIONS_PATH",
            "AGENT_SECRET_STORE_PATH",
        )
        self._env_backup = {key: os.environ.get(key) for key in self._env_keys}
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.json")
        with open(self.registry_path, "w", encoding="utf-8") as handle:
            json.dump(_registry_doc(), handle, ensure_ascii=True, indent=2)
        with open(os.path.join(self.tmpdir.name, "llm_catalog.json"), "w", encoding="utf-8") as handle:
            json.dump(_catalog_doc(), handle, ensure_ascii=True, indent=2)

    def tearDown(self) -> None:
        for key, previous in self._env_backup.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self.tmpdir.cleanup()

    def test_manual_apply_requires_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        ok, body = runtime.llm_capabilities_reconcile_apply({"actor": "test", "confirm": True}, trigger="manual")
        self.assertFalse(ok)
        self.assertEqual("action_not_permitted", body["error"])

    def test_manual_apply_updates_registry_when_permitted(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_permissions(
            {
                "mode": "auto",
                "actions": {"llm.capabilities.reconcile.apply": True},
            }
        )
        ok_plan, plan_payload = runtime.llm_capabilities_reconcile_plan({"actor": "test"})
        self.assertTrue(ok_plan)
        self.assertGreater(int((plan_payload.get("plan") or {}).get("impact", {}).get("changes_count") or 0), 0)

        ok_apply, apply_payload = runtime.llm_capabilities_reconcile_apply(
            {"actor": "test", "confirm": True},
            trigger="manual",
        )
        self.assertTrue(ok_apply)
        self.assertTrue(apply_payload["applied"])
        self.assertTrue(str(apply_payload.get("snapshot_id") or "").strip())
        models_rows = runtime.models().get("models") if isinstance(runtime.models().get("models"), list) else []
        row = [item for item in models_rows if item.get("id") == "ollama:nomic-embed-text:latest"][0]
        self.assertEqual(["embedding"], row["capabilities"])

    def test_scheduler_non_loopback_denies_without_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("0.0.0.0", 8765)
        ok, body = runtime.llm_capabilities_reconcile_apply({"actor": "scheduler"}, trigger="scheduler")
        self.assertFalse(ok)
        self.assertEqual("action_not_permitted", body["error"])

    def test_scheduler_loopback_auto_allows_without_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        ok, body = runtime.llm_capabilities_reconcile_apply({"actor": "scheduler"}, trigger="scheduler")
        self.assertTrue(ok)
        self.assertTrue(body["applied"])


if __name__ == "__main__":
    unittest.main()
