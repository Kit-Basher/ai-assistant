from __future__ import annotations

import json
import unittest

from agent.epistemics.contract import parse_candidate_json, validate_candidate


class TestEpistemicsContract(unittest.TestCase):
    def test_rejects_non_json_and_missing_keys(self) -> None:
        parsed, errors = parse_candidate_json("not-json")
        self.assertIsNone(parsed)
        self.assertEqual(("NON_JSON",), errors)

        parsed2, errors2 = parse_candidate_json("{}")
        self.assertIsNone(parsed2)
        self.assertTrue(any(err.startswith("MISSING_KEY:kind") for err in errors2))
        self.assertTrue(any(err.startswith("MISSING_KEY:final_answer") for err in errors2))

    def test_assumptions_require_clarify(self) -> None:
        payload = {
            "kind": "answer",
            "final_answer": "Here is the result.",
            "clarifying_question": None,
            "claims": [],
            "assumptions": ["Assumed missing detail"],
            "unresolved_refs": [],
            "thread_refs": [],
        }
        parsed, errors = parse_candidate_json(json.dumps(payload))
        self.assertEqual(tuple(), errors)
        self.assertIsNotNone(parsed)
        validation = validate_candidate(parsed)
        self.assertIn("ASSUMPTIONS_REQUIRE_CLARIFY", validation)
        self.assertIn("ASSUMPTIONS_REQUIRE_QUESTION", validation)

    def test_clarify_final_answer_leakage_rejected(self) -> None:
        payload = {
            "kind": "clarify",
            "final_answer": "You should do this next.",
            "clarifying_question": "Which item should I use?",
            "claims": [],
            "assumptions": [],
            "unresolved_refs": [],
            "thread_refs": [],
        }
        parsed, errors = parse_candidate_json(json.dumps(payload))
        self.assertEqual(tuple(), errors)
        self.assertIsNotNone(parsed)
        validation = validate_candidate(parsed)
        self.assertIn("CLARIFY_FINAL_ANSWER_NOT_ALLOWED", validation)


if __name__ == "__main__":
    unittest.main()
