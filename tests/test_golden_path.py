from __future__ import annotations

import unittest

from agent.golden_path import (
    bootstrap_guidance,
    bootstrap_needed,
    is_runtime_ready,
    next_step_for_failure,
    user_safe_summary,
)


class TestGoldenPath(unittest.TestCase):
    def test_is_runtime_ready_from_ready_payload(self) -> None:
        self.assertTrue(is_runtime_ready(ready_payload={"ready": True}))
        self.assertFalse(is_runtime_ready(ready_payload={"ready": False}))

    def test_bootstrap_needed_for_unavailable_llm(self) -> None:
        self.assertTrue(bootstrap_needed(llm_available=False))
        self.assertTrue(bootstrap_needed(availability_reason="provider_unhealthy"))
        self.assertFalse(bootstrap_needed(llm_available=True, availability_reason="ok"))

    def test_next_step_for_failure_has_deterministic_mapping(self) -> None:
        self.assertEqual(
            "Run: python -m agent.secrets set telegram:bot_token",
            next_step_for_failure("telegram_token_missing"),
        )
        self.assertEqual("Run: python -m agent doctor", next_step_for_failure("llm_unavailable"))

    def test_user_safe_summary(self) -> None:
        ready = user_safe_summary(ready=True, provider="ollama", model="qwen2.5:3b-instruct")
        self.assertIn("Agent is ready.", ready)
        degraded = user_safe_summary(ready=False, failure_code="llm_unavailable")
        self.assertIn("Agent is starting or degraded.", degraded)
        setup = user_safe_summary(ready=False, bootstrap=True, failure_code="no_chat_model")
        self.assertIn("Setup needed.", setup)

    def test_bootstrap_guidance_is_stable(self) -> None:
        text = bootstrap_guidance()
        self.assertIn("No chat model available", text)
        self.assertIn("Start Ollama", text)


if __name__ == "__main__":
    unittest.main()

