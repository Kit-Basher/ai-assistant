from __future__ import annotations

import json
import tempfile
import unittest

from agent.llm.registry import load_registry_document


class TestLLMRegistry(unittest.TestCase):
    def test_v1_document_is_migrated_to_v2(self) -> None:
        legacy = {
            "providers": {
                "openai": {
                    "provider_type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "auth_env_var": "OPENAI_API_KEY",
                    "enabled": True,
                },
                "ollama": {
                    "provider_type": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "auth_env_var": None,
                    "enabled": True,
                },
            },
            "models": {
                "openai:gpt-4.1-mini": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "capabilities": ["chat", "json", "tools"],
                    "quality_rank": 7,
                    "cost_rank": 4,
                    "default_for": ["chat"],
                    "enabled": True,
                },
                "ollama:llama3": {
                    "provider": "ollama",
                    "model": "llama3",
                    "capabilities": ["chat"],
                    "quality_rank": 3,
                    "cost_rank": 1,
                    "default_for": ["cheap_local"],
                    "enabled": True,
                },
            },
            "routing": {
                "mode": "prefer_cheap",
                "fallback_chain": ["ollama:llama3", "openai:gpt-4.1-mini"],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/llm_registry_v1.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(legacy, handle)

            migrated = load_registry_document(path)

        self.assertEqual(2, migrated["schema_version"])
        self.assertEqual("prefer_cheap", migrated["defaults"]["routing_mode"])
        self.assertEqual("openai_compat", migrated["providers"]["openai"]["provider_type"])
        self.assertEqual("env", migrated["providers"]["openai"]["api_key_source"]["type"])
        self.assertTrue(migrated["providers"]["ollama"]["local"])
        self.assertIn("pricing", migrated["models"]["openai:gpt-4.1-mini"])

    def test_default_document_prefers_local_lowest_cost_mode(self) -> None:
        default_doc = load_registry_document(path=None)
        self.assertEqual(2, default_doc["schema_version"])
        self.assertEqual(
            "prefer_local_lowest_cost_capable",
            default_doc["defaults"]["routing_mode"],
        )


if __name__ == "__main__":
    unittest.main()
