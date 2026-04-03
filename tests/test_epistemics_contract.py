from __future__ import annotations

import json
import unittest

from agent.epistemics.contract import parse_candidate_json, validate_candidate
from agent.epistemics.types import CandidateContract, Claim, ContextPack


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

    def test_provenance_invariant_requires_matching_field(self) -> None:
        candidate = CandidateContract(
            kind="answer",
            final_answer="ok",
            clarifying_question=None,
            claims=(Claim(text="claim", support="memory"),),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        validation = validate_candidate(candidate)
        self.assertIn("PROVENANCE_REQUIRED_FOR_SUPPORTED_CLAIM", validation)
        self.assertIn("MEMORY_PROVENANCE_REQUIRED", validation)

    def test_context_bound_provenance_validation(self) -> None:
        ctx = ContextPack(
            user_id="u1",
            active_thread_id="thread-1",
            recent_turn_ids=("thread-1:u:1",),
            in_scope_memory_ids=("mem:1",),
            tool_event_ids=("audit:1",),
        )
        candidate = CandidateContract(
            kind="answer",
            final_answer="ok",
            clarifying_question=None,
            claims=(
                Claim(text="user", support="user", user_turn_id="thread-1:u:2"),
                Claim(text="memory", support="memory", memory_id="mem:2"),
                Claim(text="tool", support="tool", tool_event_id="audit:2"),
            ),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        validation = validate_candidate(candidate, ctx)
        self.assertIn("USER_PROVENANCE_NOT_IN_CONTEXT", validation)
        self.assertIn("MEMORY_PROVENANCE_NOT_IN_CONTEXT", validation)
        self.assertIn("TOOL_PROVENANCE_NOT_IN_CONTEXT", validation)


if __name__ == "__main__":
    unittest.main()
