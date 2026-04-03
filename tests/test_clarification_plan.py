from __future__ import annotations

import unittest

from agent.intent.clarification import build_clarification_plan, build_thread_integrity_plan


class TestClarificationPlan(unittest.TestCase):
    def test_table_driven_plan_messages(self) -> None:
        cases = [
            ("empty", "", "What should I help you with?"),
            ("no_semantic_tokens", "???", "I didn’t catch a request there"),
            ("repetition_spam", "ffffffff", "specific task"),
            ("too_short", "help", "what should I help with specifically"),
            ("underspecified", "hello", "What should I do with: “hello”?"),
        ]
        for reason, norm_text, expected_substring in cases:
            with self.subTest(reason=reason, norm_text=norm_text):
                plan = build_clarification_plan(
                    raw_text=norm_text,
                    norm_text=norm_text,
                    detector_reason=reason,
                    intent="chat",
                )
                self.assertTrue(plan.message.strip())
                self.assertEqual(plan.message, plan.next_question)
                self.assertIn(expected_substring, plan.message)
                self.assertEqual(plan.reason, reason)
                self.assertIsInstance(plan.hints, list)
                self.assertIsInstance(plan.suggested_intents, list)
                self.assertEqual(plan.message, build_clarification_plan(
                    raw_text=norm_text,
                    norm_text=norm_text,
                    detector_reason=reason,
                    intent="chat",
                ).message)

    def test_thread_integrity_plan_messages(self) -> None:
        explicit = build_thread_integrity_plan(
            user_text_norm="switch topics",
            drift_reason="explicit_switch",
            intent="chat",
        )
        self.assertEqual("thread_integrity", explicit.reason)
        self.assertIn("start a new thread", explicit.message)
        self.assertEqual(explicit.message, explicit.next_question)

        shifted = build_thread_integrity_plan(
            user_text_norm="recipe request",
            drift_reason="topic_shift",
            intent="chat",
        )
        self.assertEqual("thread_integrity", shifted.reason)
        self.assertIn("continuing the current thread", shifted.message)
        self.assertEqual([], shifted.suggested_intents)


if __name__ == "__main__":
    unittest.main()
