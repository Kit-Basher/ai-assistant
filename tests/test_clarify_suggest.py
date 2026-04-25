from __future__ import annotations

import unittest

from agent.ux.clarify_suggest import (
    build_clarify_message,
    build_suggest_message,
    classify_ambiguity,
)


class TestClarifySuggest(unittest.TestCase):
    def test_short_ambiguous_message_detected(self) -> None:
        verdict = classify_ambiguity("fix it")
        self.assertTrue(verdict.ambiguous)
        self.assertEqual("known_vague_phrase", verdict.reason)

    def test_known_smalltalk_not_marked_ambiguous(self) -> None:
        verdict = classify_ambiguity("hello")
        self.assertFalse(verdict.ambiguous)

    def test_short_chat_request_not_marked_ambiguous(self) -> None:
        verdict = classify_ambiguity("say hi")
        self.assertFalse(verdict.ambiguous)
        self.assertEqual("short_chat_turn", verdict.reason)

    def test_clarify_message_has_one_question_and_ab_examples(self) -> None:
        message = build_clarify_message("help")
        self.assertEqual(4, len(message.splitlines()))
        self.assertIn("Do you mean:", message)
        self.assertIn("A)", message)
        self.assertIn("B)", message)
        self.assertEqual(1, message.count("?"))

    def test_suggest_message_has_constraint_and_three_options(self) -> None:
        message = build_suggest_message(availability_reason="provider_unhealthy")
        lines = message.splitlines()
        self.assertGreaterEqual(len(lines), 5)
        self.assertIn("LLM unavailable", lines[0])
        self.assertTrue(lines[1].startswith("1) "))
        self.assertTrue(lines[2].startswith("2) "))
        self.assertTrue(lines[3].startswith("3) "))
        self.assertEqual("Reply 1, 2, or 3.", lines[-1])


if __name__ == "__main__":
    unittest.main()
