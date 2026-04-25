from __future__ import annotations

import unittest

from agent.llm.model_latency import resolve_speed_class, telegram_text_model_gate


class TestModelLatency(unittest.TestCase):
    def test_qwen7b_is_fast_and_allowed_for_telegram_chat(self) -> None:
        self.assertEqual("fast", resolve_speed_class(model_id="ollama:qwen2.5:7b-instruct"))
        allowed, reason = telegram_text_model_gate(
            channel="telegram",
            task_type="chat",
            required_capabilities={"chat"},
            model_id="ollama:qwen2.5:7b-instruct",
            model_name="qwen2.5:7b-instruct",
            capabilities=["chat"],
            speed_class="fast",
            latency_fallback=False,
        )
        self.assertTrue(allowed)
        self.assertIsNone(reason)
