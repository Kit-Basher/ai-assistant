from __future__ import annotations

import unittest

from agent.intent.assessment import assess_intent_deterministic


class TestIntentAssessment(unittest.TestCase):
    def test_modelops_check_is_top_candidate(self) -> None:
        assessment = assess_intent_deterministic(
            user_text_raw="check for new models on ollama",
            user_text_norm="check for new models on ollama",
            route_intent="chat",
            context={},
        )
        self.assertEqual("proceed", assessment.decision)
        self.assertTrue(assessment.candidates)
        self.assertEqual("modelops_check", assessment.candidates[0].intent)
        self.assertGreaterEqual(float(assessment.candidates[0].score), 0.85)

    def test_ambiguous_help_clarifies(self) -> None:
        assessment = assess_intent_deterministic(
            user_text_raw="help",
            user_text_norm="help",
            route_intent="chat",
            context={},
        )
        self.assertEqual("clarify", assessment.decision)
        self.assertTrue(str(assessment.next_question or "").strip())

    def test_ordering_is_deterministic(self) -> None:
        first = assess_intent_deterministic(
            user_text_raw="switch to another model",
            user_text_norm="switch to another model",
            route_intent="chat",
            context={},
        )
        second = assess_intent_deterministic(
            user_text_raw="switch to another model",
            user_text_norm="switch to another model",
            route_intent="chat",
            context={},
        )
        first_rows = [(row.intent, row.score, row.reason) for row in first.candidates]
        second_rows = [(row.intent, row.score, row.reason) for row in second.candidates]
        self.assertEqual(first_rows, second_rows)


if __name__ == "__main__":
    unittest.main()
