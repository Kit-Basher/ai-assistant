from __future__ import annotations

import unittest

from scripts import live_user_barrage
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

    def test_rejects_direct_questions_routed_to_assistant_clarification(self) -> None:
        failures = classify_barrage_response(
            PromptCase(category="app_setup", prompt="how do i open the web UI"),
            {
                "text": "I was following: model status. Do you want me to continue?",
                "first_line": "I was following: model status. Do you want me to continue?",
                "route": "assistant_clarification",
                "used_llm": False,
                "used_runtime_state": False,
            },
        )
        self.assertIn("clear direct question routed to assistant_clarification", failures)

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

    def test_rejects_skill_install_as_interpretation_followup_or_apt(self) -> None:
        cases = [
            ("interpretation_followup", "Likely cause: brave is using memory.", "stale interpretation"),
            ("action_tool", "I will install a using apt-get install -y a.", "OS package install"),
        ]
        for route, text, expected in cases:
            with self.subTest(route=route):
                failures = classify_barrage_response(
                    PromptCase(category="skill_install", prompt="add a capability for reading webpages"),
                    {
                        "text": text,
                        "first_line": first_line(text),
                        "route": route,
                        "used_llm": False,
                        "used_runtime_state": False,
                    },
                )
                self.assertTrue(any(expected in failure for failure in failures), failures)

    def test_ready_gate_accepts_core_and_chat_ready_with_optional_surface_warning(self) -> None:
        original = live_user_barrage.request_json
        try:
            live_user_barrage.request_json = lambda *_args, **_kwargs: (
                200,
                {
                    "ok": True,
                    "ready": True,
                    "core_ready": True,
                    "chat_ready": True,
                    "surfaces": {
                        "telegram": {
                            "state": "stopped",
                            "required": False,
                            "warning": "Telegram is stopped.",
                        }
                    },
                },
            )
            payload = live_user_barrage.require_ready("http://127.0.0.1:8765", timeout=1)
        finally:
            live_user_barrage.request_json = original
        self.assertTrue(payload["core_ready"])
        self.assertTrue(payload["chat_ready"])

    def test_ready_gate_rejects_when_core_split_says_chat_not_ready(self) -> None:
        original = live_user_barrage.request_json
        try:
            live_user_barrage.request_json = lambda *_args, **_kwargs: (
                200,
                {"ok": True, "ready": False, "core_ready": True, "chat_ready": False, "runtime_mode": "DEGRADED"},
            )
            with self.assertRaisesRegex(RuntimeError, "chat-ready"):
                live_user_barrage.require_ready("http://127.0.0.1:8765", timeout=1)
        finally:
            live_user_barrage.request_json = original


if __name__ == "__main__":
    unittest.main()
