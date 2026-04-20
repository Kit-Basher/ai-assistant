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

    def test_general_real_world_question_defaults_to_chat(self) -> None:
        assessment = assess_intent_deterministic(
            user_text_raw="I'm downloading a file and it's taking forever, can you tell why?",
            user_text_norm="i'm downloading a file and it's taking forever, can you tell why?",
            route_intent="chat",
            context={},
        )
        self.assertEqual("proceed", assessment.decision)
        self.assertIsNone(assessment.next_question)
        self.assertTrue(assessment.candidates)
        self.assertNotIn("chat, ask, or model check/switch", str(assessment.next_question or ""))

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
