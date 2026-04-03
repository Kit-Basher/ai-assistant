from __future__ import annotations

import unittest

from agent.epistemics.canary import evaluate_case
from agent.epistemics.canary_cases import MUST_INTERCEPT, MUST_PASS


def _assert_intercept_shape(test: unittest.TestCase, text: str) -> None:
    lines = text.splitlines()
    test.assertEqual(3, len(lines))
    test.assertEqual("I’m not sure.", lines[0])
    test.assertEqual("", lines[1])
    test.assertEqual(lines[2], lines[2].strip())
    test.assertTrue(lines[2].endswith("?"))
    test.assertEqual(1, text.count("?"))


class TestEpistemicsCanary(unittest.TestCase):
    def test_must_intercept_cases(self) -> None:
        for case in MUST_INTERCEPT:
            result = evaluate_case(case, must_intercept=True)
            self.assertTrue(result.passed, case["name"])
            self.assertTrue(result.decision.intercepted, case["name"])
            _assert_intercept_shape(self, result.decision.user_text)

    def test_must_pass_cases_and_provenance_invariants(self) -> None:
        for case in MUST_PASS:
            result = evaluate_case(case, must_intercept=False)
            self.assertTrue(result.passed, case["name"])
            self.assertFalse(result.decision.intercepted, case["name"])
            for claim in result.candidate.claims:
                provenance_count = sum(
                    [
                        1 if claim.user_turn_id else 0,
                        1 if claim.memory_id is not None else 0,
                        1 if claim.tool_event_id else 0,
                    ]
                )
                if claim.support == "none":
                    self.assertEqual(0, provenance_count, case["name"])
                    continue
                self.assertEqual(1, provenance_count, case["name"])
                if claim.support == "user":
                    self.assertIn(claim.user_turn_id, result.ctx.recent_turn_ids, case["name"])
                if claim.support == "memory":
                    self.assertIn(str(claim.memory_id), result.ctx.in_scope_memory_ids, case["name"])
                if claim.support == "tool":
                    self.assertIn(claim.tool_event_id, result.ctx.tool_event_ids, case["name"])


if __name__ == "__main__":
    unittest.main()

