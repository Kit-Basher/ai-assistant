from __future__ import annotations

import unittest

from agent.epistemics.contract import build_plain_answer_candidate
from agent.epistemics.gate import apply_epistemic_gate
from agent.epistemics.question_selector import select_one_question
from agent.epistemics.types import CandidateContract, ContextPack, ThreadRef


def _base_ctx(**kwargs) -> ContextPack:
    payload = {
        "user_id": "user-1",
        "active_thread_id": "thread-1",
        "recent_messages": tuple(),
        "memory_hits": tuple(),
        "memory_ambiguous": tuple(),
        "memory_miss": False,
        "tools_available": ("core",),
        "tool_failures": tuple(),
        "referents": tuple(),
    }
    payload.update(kwargs)
    return ContextPack(**payload)


def _assert_intercept_shape(test: unittest.TestCase, text: str) -> None:
    lines = text.splitlines()
    test.assertEqual(3, len(lines))
    test.assertEqual("I’m not sure.", lines[0])
    test.assertEqual("", lines[1])
    test.assertEqual(lines[2], lines[2].strip())
    test.assertTrue(lines[2].endswith("?"))
    test.assertEqual(1, text.count("?"))


class TestEpistemicGate(unittest.TestCase):
    def test_intercepts_unresolved_reference_with_two_referents(self) -> None:
        ctx = _base_ctx(referents=("[1] Task A", "[2] Task B"))
        candidate = build_plain_answer_candidate("Done.")
        decision = apply_epistemic_gate("do that again", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("UNRESOLVED_REFERENCE", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)

    def test_intercepts_missing_slot_for_schedule(self) -> None:
        ctx = _base_ctx()
        candidate = build_plain_answer_candidate("Scheduled.")
        decision = apply_epistemic_gate("schedule it", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("MISSING_REQUIRED_SLOT", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)

    def test_intercepts_cross_thread_reference_without_confirmation(self) -> None:
        ctx = _base_ctx(active_thread_id="thread-1")
        candidate = CandidateContract(
            kind="answer",
            final_answer="Pulled details from another thread.",
            clarifying_question=None,
            claims=tuple(),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=(ThreadRef(target_thread_id="thread-2", needs_confirmation=True),),
            raw_json=None,
        )
        decision = apply_epistemic_gate("use that previous thread", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("CROSS_THREAD_RISK", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)

    def test_memory_miss_intercepts_without_reconstruction(self) -> None:
        ctx = _base_ctx(memory_miss=True)
        candidate = build_plain_answer_candidate("I reconstructed the missing memory.")
        decision = apply_epistemic_gate("what did we decide last week", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("MEMORY_MISS", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)

    def test_tool_failure_intercepts(self) -> None:
        ctx = _base_ctx(tool_failures=("runner_not_configured",))
        candidate = build_plain_answer_candidate("Executed successfully.")
        decision = apply_epistemic_gate("run the operation", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("TOOL_FAILURE_OR_UNAVAILABLE", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)

    def test_selector_normalizes_multiple_question_marks(self) -> None:
        ctx = _base_ctx()
        candidate = CandidateContract(
            kind="clarify",
            final_answer="",
            clarifying_question="Which one?? and why?",
            claims=tuple(),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        question = select_one_question("pick one", ctx, candidate, tuple())
        self.assertEqual(1, question.count("?"))
        self.assertTrue(question.endswith("?"))
        self.assertNotIn("\n", question)

    def test_soft_cross_thread_phrase_triggers_intercept(self) -> None:
        ctx = _base_ctx(active_thread_id="thread-1")
        candidate = build_plain_answer_candidate("As we discussed earlier, continue with the same plan.")
        decision = apply_epistemic_gate("summarize this", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("CROSS_THREAD_RISK", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)


if __name__ == "__main__":
    unittest.main()
