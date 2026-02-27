from __future__ import annotations

import unittest

from agent.modelops.discovery import ModelInfo
from agent.modelops.recommend import recommend_models


class TestModelOpsRecommend(unittest.TestCase):
    def test_recommendations_are_deterministic_for_code_and_chat(self) -> None:
        available = [
            ModelInfo(
                provider="ollama",
                model_id="deepseek-coder:6.7b",
                display_name="deepseek-coder:6.7b",
                context_tokens=16384,
                tags=["code", "chat"],
                created_at=None,
                metadata={},
            ),
            ModelInfo(
                provider="ollama",
                model_id="qwen2.5:7b-instruct",
                display_name="qwen2.5:7b-instruct",
                context_tokens=32768,
                tags=["chat"],
                created_at=None,
                metadata={},
            ),
            ModelInfo(
                provider="openrouter",
                model_id="openai/gpt-4o-mini",
                display_name="openai/gpt-4o-mini",
                context_tokens=128000,
                tags=["chat"],
                created_at=None,
                metadata={},
            ),
        ]
        current = {"provider": "openrouter", "model_id": "openrouter:openai/gpt-4o-mini"}
        recommendations = recommend_models(
            available=available,
            current=current,
            purposes=["code", "chat"],
            prefer_local=True,
        )
        code_rows = recommendations["code"]
        chat_rows = recommendations["chat"]
        self.assertTrue(code_rows)
        self.assertTrue(chat_rows)
        self.assertEqual("ollama", code_rows[0].provider)
        self.assertIn("coder", code_rows[0].model_id)
        self.assertGreaterEqual(float(code_rows[0].score), float(code_rows[-1].score))
        self.assertEqual(
            [f"{row.provider}:{row.model_id}" for row in code_rows],
            [f"{row.provider}:{row.model_id}" for row in recommendations["code"]],
        )
        self.assertGreaterEqual(float(chat_rows[0].score), float(chat_rows[-1].score))


if __name__ == "__main__":
    unittest.main()
