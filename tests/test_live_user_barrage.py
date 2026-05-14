from __future__ import annotations

import unittest

from scripts.live_user_barrage import PromptCase, classify_barrage_response, first_line


class TestLiveUserBarrageClassifier(unittest.TestCase):
    def _classify(self, category: str, prompt: str, text: str) -> list[str]:
        return classify_barrage_response(
            PromptCase(category=category, prompt=prompt),
            {
                "text": text,
                "first_line": first_line(text),
                "route": "generic_chat",
                "used_llm": False,
                "used_runtime_state": True,
            },
        )

    def test_accepts_useful_status_answer(self) -> None:
        failures = self._classify(
            "runtime_status",
            "what model am i using",
            "Chat is currently using the configured model ollama:qwen2.5:7b-instruct on Ollama.",
        )
        self.assertEqual([], failures)

    def test_rejects_empty_and_placeholder_answers(self) -> None:
        self.assertIn("empty response", self._classify("open_chat", "hello", ""))
        self.assertIn("stale whole-answer placeholder", self._classify("open_chat", "do it", "OK"))
        self.assertIn("stale whole-answer placeholder", self._classify("open_chat", "do it", "Done."))

    def test_rejects_internal_leaks(self) -> None:
        cases = {
            "source_surface": "source_surface=telegram",
            "thread_id": "thread_id: api:default",
            "runtime_payload": "runtime_payload follows",
            "read-only guard": "Read-only guard / NL path refused",
            "nl path refused": "NL path refused",
            "traceback": "Traceback (most recent call last):",
        }
        for expected, text in cases.items():
            with self.subTest(expected=expected):
                failures = self._classify("runtime_status", "status", text)
                self.assertTrue(any(expected in failure for failure in failures), failures)

    def test_rejects_ready_to_help_for_frustration(self) -> None:
        failures = self._classify(
            "frustration",
            "why arent you working",
            "I’m here and ready to help. What can I do for you?",
        )
        self.assertIn("diagnostic/frustration prompt got stale ready-to-help wording", failures)

    def test_allows_ready_to_help_for_normal_open_chat(self) -> None:
        failures = self._classify(
            "open_chat",
            "say hello",
            "I’m here and ready to help. What can I do for you?",
        )
        self.assertEqual([], failures)

    def test_rejects_temporary_switch_default_update(self) -> None:
        failures = self._classify(
            "model_switch",
            "use ollama:qwen3.6:35b-a3b for this chat session only",
            "Default model updated to ollama:qwen3.6:35b-a3b.",
        )
        self.assertIn("temporary switch claimed default was updated", failures)

    def test_allows_temporary_switch_default_warning(self) -> None:
        failures = self._classify(
            "model_switch",
            "use ollama:qwen3.6:35b-a3b for this chat session only",
            "Temporary chat model switched to ollama:qwen3.6:35b-a3b. This does not change your default model.",
        )
        self.assertEqual([], failures)

    def test_rejects_browse_skill_as_model_acquisition(self) -> None:
        failures = self._classify(
            "skill_install",
            "install a skill that lets you browse",
            "Which model do you want me to acquire?",
        )
        self.assertIn("browse skill request was treated as model acquisition", failures)

    def test_allows_browse_skill_preview(self) -> None:
        failures = self._classify(
            "skill_install",
            "install a skill that lets you browse",
            "I can preview browser-related skill options and explain what approval is needed.",
        )
        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
