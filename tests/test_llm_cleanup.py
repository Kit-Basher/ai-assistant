from __future__ import annotations

import unittest

from agent.llm.cleanup import build_registry_cleanup_plan


def _registry_document() -> dict[str, object]:
    return {
        "schema_version": 2,
        "providers": {
            "openrouter": {
                "provider_type": "openai_compat",
                "base_url": "https://openrouter.ai/api/v1",
                "chat_path": "/chat/completions",
                "api_key_source": None,
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": False,
            },
            "stale": {
                "provider_type": "openai_compat",
                "base_url": "https://stale.example",
                "chat_path": "/v1/chat/completions",
                "api_key_source": None,
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": False,
            },
        },
        "models": {
            "openrouter:openai/gpt-4o-mini": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
            },
            "openrouter:ghost-model": {
                "provider": "openrouter",
                "model": "ghost-model",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
            },
            "stale:old-model": {
                "provider": "stale",
                "model": "old-model",
                "capabilities": ["chat"],
                "enabled": False,
                "available": False,
            },
        },
        "defaults": {
            "routing_mode": "auto",
            "default_provider": "openrouter",
            "default_model": "openrouter:openai/gpt-4o-mini",
            "allow_remote_fallback": True,
        },
    }


def _catalog_snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "last_run_at": 100,
        "providers": {
            "openrouter": {
                "provider_id": "openrouter",
                "source": "openrouter_models",
                "last_refresh_at": 100,
                "last_error_kind": None,
                "last_error_at": None,
                "models": [
                    {
                        "id": "openrouter:openai/gpt-4o-mini",
                        "provider_id": "openrouter",
                        "model": "openai/gpt-4o-mini",
                        "capabilities": ["chat"],
                        "max_context_tokens": 128000,
                        "input_cost_per_million_tokens": 0.15,
                        "output_cost_per_million_tokens": 0.6,
                        "source": "openrouter_models",
                    }
                ],
            },
            "stale": {
                "provider_id": "stale",
                "source": "openai_models",
                "last_refresh_at": 100,
                "last_error_kind": None,
                "last_error_at": None,
                "models": [],
            },
        },
    }


class TestLLMCleanup(unittest.TestCase):
    def test_plan_is_deterministic_and_marks_missing_catalog_models_unavailable(self) -> None:
        summary = {
            "providers": [{"id": "openrouter", "status": "ok"}, {"id": "stale", "status": "ok"}],
            "models": [
                {"id": "openrouter:openai/gpt-4o-mini", "status": "ok"},
                {"id": "openrouter:ghost-model", "status": "ok"},
            ],
        }
        plan_a = build_registry_cleanup_plan(
            _registry_document(),
            usage_stats_snapshot={},
            health_summary=summary,
            catalog_snapshot=_catalog_snapshot(),
            unused_days=30,
            disable_failing_provider=False,
            apply_prune=False,
        )
        plan_b = build_registry_cleanup_plan(
            _registry_document(),
            usage_stats_snapshot={},
            health_summary=summary,
            catalog_snapshot=_catalog_snapshot(),
            unused_days=30,
            disable_failing_provider=False,
            apply_prune=False,
        )
        self.assertEqual(plan_a["changes"], plan_b["changes"])
        target = [
            row
            for row in plan_a["changes"]
            if row.get("kind") == "model"
            and row.get("id") == "openrouter:ghost-model"
            and row.get("field") == "available"
        ]
        self.assertEqual(1, len(target))
        self.assertFalse(target[0]["after"])
        self.assertEqual("missing_from_catalog", target[0]["reason"])

    def test_apply_prune_removes_disabled_unused_entries(self) -> None:
        summary = {
            "providers": [
                {"id": "openrouter", "status": "ok", "failure_streak": 0},
                {"id": "stale", "status": "down", "failure_streak": 9, "down_since": 1},
            ],
            "models": [],
        }
        plan = build_registry_cleanup_plan(
            _registry_document(),
            usage_stats_snapshot={},
            health_summary=summary,
            catalog_snapshot=_catalog_snapshot(),
            unused_days=30,
            disable_failing_provider=True,
            provider_failure_streak=8,
            apply_prune=True,
        )
        changes = plan["changes"]
        self.assertTrue(any(row.get("id") == "stale" and row.get("field") == "enabled" for row in changes))
        self.assertTrue(any(row.get("id") == "stale:old-model" and row.get("field") == "deleted" for row in changes))
        self.assertNotIn("stale", plan["updated_document"]["providers"])


if __name__ == "__main__":
    unittest.main()
