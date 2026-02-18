from __future__ import annotations

import json
import tempfile
import unittest

from agent.llm.hygiene import apply_hygiene_plan, build_hygiene_plan
from agent.llm.registry import RegistryStore


def _registry_document() -> dict[str, object]:
    return {
        "schema_version": 2,
        "providers": {
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
            "oldremote": {
                "provider_type": "openai_compat",
                "base_url": "https://old.example/v1/",
                "chat_path": "v1/chat/completions",
                "api_key_source": None,
                "default_headers": {},
                "default_query_params": {},
                "enabled": False,
                "local": False,
            },
        },
        "models": {
            "openrouter:openai/gpt-4o-mini": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat", "json", "tools"],
                "enabled": True,
                "available": True,
                "quality_rank": 6,
                "cost_rank": 3,
                "default_for": ["chat"],
                "pricing": {
                    "input_per_million_tokens": 0.15,
                    "output_per_million_tokens": 0.6,
                },
                "max_context_tokens": 128000,
            }
        },
        "defaults": {
            "routing_mode": "prefer_best",
            "default_provider": "openrouter",
            "default_model": "openai/gpt-4o-mini",
            "allow_remote_fallback": True,
        },
    }


class TestLLMHygiene(unittest.TestCase):
    def test_plan_is_deterministic_and_enforces_fully_qualified_default_model(self) -> None:
        down_since = 1
        summary = {
            "models": [
                {
                    "id": "openrouter:openai/gpt-4o-mini",
                    "status": "down",
                    "down_since": down_since,
                }
            ]
        }

        plan_a = build_hygiene_plan(
            _registry_document(),
            summary,
            unavailable_days=1,
            remove_empty_disabled_providers=True,
        )
        plan_b = build_hygiene_plan(
            _registry_document(),
            summary,
            unavailable_days=1,
            remove_empty_disabled_providers=True,
        )

        self.assertEqual(plan_a["changes"], plan_b["changes"])
        change_fields = {(row["kind"], row.get("id"), row["field"]) for row in plan_a["changes"]}
        self.assertIn(("defaults", None, "default_model"), change_fields)
        self.assertIn(("model", "openrouter:openai/gpt-4o-mini", "available"), change_fields)
        self.assertIn(("provider", "openrouter", "base_url"), change_fields)
        self.assertIn(("provider", "oldremote", "deleted"), change_fields)

    def test_apply_and_persist_round_trip(self) -> None:
        summary = {
            "models": [
                {
                    "id": "openrouter:openai/gpt-4o-mini",
                    "status": "ok",
                    "down_since": None,
                }
            ]
        }
        plan = build_hygiene_plan(
            _registry_document(),
            summary,
            unavailable_days=30,
            remove_empty_disabled_providers=True,
        )
        updated = apply_hygiene_plan(_registry_document(), plan)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/registry.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(_registry_document(), handle)

            store = RegistryStore(path)
            store.write_document(updated)
            reloaded = store.read_document()

        self.assertEqual("openrouter:openai/gpt-4o-mini", reloaded["defaults"]["default_model"])
        self.assertNotIn("oldremote", reloaded["providers"])
        self.assertEqual("https://openrouter.ai/api", reloaded["providers"]["openrouter"]["base_url"])

    def test_hygiene_marks_stale_ollama_models_unavailable_from_inventory(self) -> None:
        document = {
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
                }
            },
            "models": {
                "ollama:llama3": {
                    "provider": "ollama",
                    "model": "llama3",
                    "capabilities": ["chat"],
                    "enabled": True,
                    "available": True,
                },
                "ollama:qwen2.5:3b-instruct": {
                    "provider": "ollama",
                    "model": "qwen2.5:3b-instruct",
                    "capabilities": ["chat"],
                    "enabled": True,
                    "available": True,
                },
            },
            "defaults": {
                "routing_mode": "prefer_local_lowest_cost_capable",
                "default_provider": "ollama",
                "default_model": "ollama:qwen2.5:3b-instruct",
                "allow_remote_fallback": True,
            },
        }
        plan = build_hygiene_plan(
            document,
            {"providers": [], "models": []},
            unavailable_days=7,
            remove_empty_disabled_providers=False,
            provider_inventory={
                "ollama": {
                    "authoritative": True,
                    "models": ["qwen2.5:3b-instruct"],
                }
            },
        )
        changes = [row for row in plan["changes"] if row.get("kind") == "model" and row.get("id") == "ollama:llama3"]
        self.assertEqual(1, len(changes))
        self.assertEqual("available", changes[0]["field"])
        self.assertFalse(changes[0]["after"])


if __name__ == "__main__":
    unittest.main()
