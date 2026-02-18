from __future__ import annotations

import unittest

from agent.llm.autoconfig import build_autoconfig_plan


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
            env={"OPENROUTER_API_KEY": "sk-test"},
        )
        self.assertEqual([], plan["changes"])
        self.assertIn("keep_current_default_model", plan["reasons"])


if __name__ == "__main__":
    unittest.main()
