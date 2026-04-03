import os
import unittest


class TestLegacyLLMStackRemoved(unittest.TestCase):
    def test_legacy_modules_are_deleted(self) -> None:
        deleted_paths = (
            "agent/llm_client.py",
            "agent/llm/broker.py",
            "agent/llm/broker_policy.py",
            "agent/llm/providers/openrouter_provider.py",
            "agent/llm/providers/openai_provider.py",
            "agent/llm/providers/ollama_provider.py",
        )
        for path in deleted_paths:
            with self.subTest(path=path):
                self.assertFalse(os.path.exists(path))

    def test_runtime_sources_do_not_reference_legacy_stack(self) -> None:
        source_checks = {
            "telegram_adapter/bot.py": ("build_llm_broker", "agent.llm_client"),
            "agent/orchestrator.py": ("llm_broker", "llm_broker_error"),
            "agent/llm/router.py": ("LLMNarrationRouter",),
        }
        for path, needles in source_checks.items():
            with self.subTest(path=path):
                with open(path, "r", encoding="utf-8") as handle:
                    source = handle.read()
                for needle in needles:
                    self.assertNotIn(needle, source)


if __name__ == "__main__":
    unittest.main()
