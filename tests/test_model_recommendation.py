from __future__ import annotations

import unittest

from agent.model_recommendation import RecommendationContext, rank_candidates


class TestModelRecommendation(unittest.TestCase):
    def test_scoring_is_deterministic_and_sorted(self) -> None:
        candidates = [
            {
                "provider_id": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "capabilities": ["chat", "json", "tools"],
                "local": False,
                "availability": True,
                "params_b": None,
                "context_tokens": 128000,
                "price_in": 0.15,
                "price_out": 0.60,
            },
            {
                "provider_id": "openrouter",
                "model_id": "acme/no-meta-model",
                "capabilities": ["chat"],
                "local": False,
                "availability": True,
                "params_b": None,
                "context_tokens": None,
                "price_in": None,
                "price_out": None,
            },
        ]
        context = RecommendationContext(
            purpose="chat",
            default_model=None,
            allow_remote_fallback=True,
            enabled_providers=frozenset({"openrouter"}),
            vram_gb=None,
        )
        first = rank_candidates(candidates, context)
        second = rank_candidates(candidates, context)

        first_ids = [row.canonical_model_id for row in first.ranked]
        second_ids = [row.canonical_model_id for row in second.ranked]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual("openrouter:openai/gpt-4o-mini", first_ids[0])

    def test_missing_fields_are_explicit_tradeoffs(self) -> None:
        candidates = [
            {
                "provider_id": "openrouter",
                "model_id": "acme/no-meta-model",
                "capabilities": ["chat"],
                "local": False,
                "availability": True,
                "params_b": None,
                "context_tokens": None,
                "price_in": None,
                "price_out": None,
            }
        ]
        context = RecommendationContext(
            purpose="chat",
            default_model=None,
            allow_remote_fallback=True,
            enabled_providers=frozenset({"openrouter"}),
            vram_gb=None,
        )
        ranked = rank_candidates(candidates, context)
        self.assertEqual(1, len(ranked.ranked))
        row = ranked.ranked[0]
        tradeoffs = list(row.tradeoffs)
        self.assertIn("missing:context_length", tradeoffs)
        self.assertIn("missing:pricing", tradeoffs)
        self.assertIn("missing:params_b", tradeoffs)


if __name__ == "__main__":
    unittest.main()
