from __future__ import annotations

import unittest
import os
from unittest.mock import patch

from agent.epistemics.contract import build_plain_answer_candidate
from agent.epistemics.gate import apply_epistemic_gate
from agent.epistemics.question_selector import select_one_question
from agent.epistemics.types import CandidateContract, Claim, ContextPack, ThreadRef


def _base_ctx(**kwargs) -> ContextPack:
    payload = {
        "user_id": "user-1",
        "active_thread_id": "thread-1",
        "recent_messages": tuple(),
        "recent_turn_ids": tuple(),
        "memory_hits": tuple(),
        "memory_ambiguous": tuple(),
        "memory_miss": False,
        "in_scope_memory": tuple(),
        "in_scope_memory_ids": tuple(),
        "out_of_scope_memory": tuple(),
        "out_of_scope_relevant_memory": False,
        "thread_turn_count": 0,
        "tools_available": ("core",),
        "tool_event_ids": tuple(),
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

    def test_env_threshold_override_changes_intercept_decision(self) -> None:
        ctx = _base_ctx()
        candidate = build_plain_answer_candidate("OK")
        user_text = "plan and copy notes"

        with patch.dict(os.environ, {"PASS_SCORE_THRESHOLD": "0.90"}, clear=False):
            decision_high = apply_epistemic_gate(user_text, ctx, candidate)
        self.assertFalse(decision_high.intercepted)
        self.assertIn("MULTI_INTENT", decision_high.reasons)

        with patch.dict(os.environ, {"PASS_SCORE_THRESHOLD": "0.20"}, clear=False):
            decision_low = apply_epistemic_gate(user_text, ctx, candidate)
        self.assertTrue(decision_low.intercepted)
        self.assertIn("MULTI_INTENT", decision_low.reasons)
        _assert_intercept_shape(self, decision_low.user_text)

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

    def test_selector_unresolved_reference_uses_ab_form(self) -> None:
        ctx = _base_ctx(referents=("[1] Build", "[2] Test"))
        candidate = build_plain_answer_candidate("Done.")
        question = select_one_question("do that", ctx, candidate, ("UNRESOLVED_REFERENCE",))
        self.assertEqual("Which do you mean: [1] Build or [2] Test?", question)

    def test_selector_missing_slot_date_and_time_questions(self) -> None:
        ctx = _base_ctx()
        candidate = build_plain_answer_candidate("Scheduled.")
        question_date = select_one_question("schedule at 09:30 UTC", ctx, candidate, ("MISSING_REQUIRED_SLOT",))
        question_time = select_one_question("schedule on 2026-03-01", ctx, candidate, ("MISSING_REQUIRED_SLOT",))
        self.assertEqual("What date should I use?", question_date)
        self.assertEqual("What time should I use?", question_time)

    def test_selector_cross_thread_text_remains_locked(self) -> None:
        ctx = _base_ctx(active_thread_id="thread-1")
        candidate = build_plain_answer_candidate("Done.")
        question = select_one_question("previously we discussed this", ctx, candidate, ("CROSS_THREAD_RISK",))
        self.assertEqual("Do you want me to use information from another conversation?", question)

    def test_selector_questions_are_single_line_single_question_and_bounded_length(self) -> None:
        ctx = _base_ctx(referents=("A" * 200, "B" * 200))
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
        reason_inputs = (
            ("UNRESOLVED_REFERENCE", "do that"),
            ("MISSING_REQUIRED_SLOT", "schedule on 2026-03-01 at 09:30"),
            ("MULTI_INTENT", "schedule and copy"),
            ("CROSS_THREAD_RISK", "use previous thread"),
            ("MEMORY_MISS", "recall it"),
            ("TOOL_FAILURE_OR_UNAVAILABLE", "run operation"),
            ("UNSUPPORTED_CLAIM", "claim"),
        )
        for reason_code, user_text in reason_inputs:
            question = select_one_question(user_text, ctx, candidate, (reason_code,))
            self.assertEqual(1, question.count("?"), reason_code)
            self.assertTrue(question.endswith("?"), reason_code)
            self.assertNotIn("\n", question, reason_code)
            self.assertLessEqual(len(question), 120, reason_code)

    def test_soft_cross_thread_phrase_triggers_intercept(self) -> None:
        ctx = _base_ctx(active_thread_id="thread-1")
        candidate = build_plain_answer_candidate("As we discussed earlier, continue with the same plan.")
        decision = apply_epistemic_gate("summarize this", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("CROSS_THREAD_RISK", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)

    def test_cross_thread_memory_reference_intercepts(self) -> None:
        ctx = _base_ctx(
            active_thread_id="thread-1",
            in_scope_memory=("global:response_style",),
            out_of_scope_memory=("mem:thread-2-plan",),
            out_of_scope_relevant_memory=True,
            thread_turn_count=2,
        )
        candidate = CandidateContract(
            kind="answer",
            final_answer="Using your previous plan from another conversation.",
            clarifying_question=None,
            claims=(
                Claim(
                    text="You planned to launch this month.",
                    support="memory",
                    ref="mem:thread-2-plan",
                    memory_id="mem:thread-2-plan",
                ),
            ),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        decision = apply_epistemic_gate("what should I do now", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("CROSS_THREAD_RISK", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)
        self.assertEqual(
            "Do you want me to use information from another conversation?",
            decision.user_text.splitlines()[2],
        )

    def test_global_memory_claim_allowed(self) -> None:
        ctx = _base_ctx(
            active_thread_id="thread-1",
            in_scope_memory=("global:response_style",),
            in_scope_memory_ids=("global:response_style",),
            out_of_scope_memory=tuple(),
            out_of_scope_relevant_memory=False,
            thread_turn_count=1,
        )
        candidate = CandidateContract(
            kind="answer",
            final_answer="You prefer concise answers.",
            clarifying_question=None,
            claims=(
                Claim(
                    text="Preference is concise answers.",
                    support="memory",
                    ref="global:response_style",
                    memory_id="global:response_style",
                ),
            ),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        decision = apply_epistemic_gate("what do you remember about my style", ctx, candidate)
        self.assertFalse(decision.intercepted)

    def test_new_thread_summary_phrase_intercepts(self) -> None:
        ctx = _base_ctx(active_thread_id="thread-2", thread_turn_count=0)
        candidate = build_plain_answer_candidate("We've been working on your project, so continue as usual.")
        decision = apply_epistemic_gate("status", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("CROSS_THREAD_RISK", decision.reasons)
        _assert_intercept_shape(self, decision.user_text)

    def test_new_thread_resets_drift_risk(self) -> None:
        candidate = build_plain_answer_candidate("As usual, keep the same cadence.")
        existing_thread_ctx = _base_ctx(active_thread_id="thread-1", thread_turn_count=3)
        existing_thread_decision = apply_epistemic_gate("status", existing_thread_ctx, candidate)
        self.assertFalse(existing_thread_decision.intercepted)

        new_thread_ctx = _base_ctx(active_thread_id="thread-2", thread_turn_count=0)
        new_thread_decision = apply_epistemic_gate("status", new_thread_ctx, candidate)
        self.assertTrue(new_thread_decision.intercepted)
        self.assertIn("CROSS_THREAD_RISK", new_thread_decision.reasons)

    def test_memory_support_missing_memory_id_rejected(self) -> None:
        ctx = _base_ctx(in_scope_memory_ids=("mem:1",))
        candidate = CandidateContract(
            kind="answer",
            final_answer="Loaded memory detail.",
            clarifying_question=None,
            claims=(Claim(text="From memory", support="memory"),),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        decision = apply_epistemic_gate("recall that", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("CONTRACT_INVALID", decision.reasons)
        self.assertIn("MEMORY_PROVENANCE_REQUIRED", decision.contract_errors)
        _assert_intercept_shape(self, decision.user_text)

    def test_memory_id_not_in_scope_rejected(self) -> None:
        ctx = _base_ctx(in_scope_memory_ids=("mem:1",))
        candidate = CandidateContract(
            kind="answer",
            final_answer="Loaded memory detail.",
            clarifying_question=None,
            claims=(Claim(text="From memory", support="memory", memory_id="mem:2"),),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        decision = apply_epistemic_gate("recall that", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("CONTRACT_INVALID", decision.reasons)
        self.assertIn("MEMORY_PROVENANCE_NOT_IN_CONTEXT", decision.contract_errors)

    def test_user_provenance_must_be_in_recent_turn_ids(self) -> None:
        ctx = _base_ctx(recent_turn_ids=("thread-1:u:1",))
        candidate = CandidateContract(
            kind="answer",
            final_answer="You asked about uptime.",
            clarifying_question=None,
            claims=(Claim(text="User asked uptime", support="user", user_turn_id="thread-1:u:99"),),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        decision = apply_epistemic_gate("what did I ask", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("USER_PROVENANCE_NOT_IN_CONTEXT", decision.contract_errors)

    def test_tool_provenance_must_be_in_tool_event_ids(self) -> None:
        ctx = _base_ctx(tool_event_ids=("audit:1",))
        candidate = CandidateContract(
            kind="answer",
            final_answer="Tool returned success.",
            clarifying_question=None,
            claims=(Claim(text="Tool output ok", support="tool", tool_event_id="audit:2"),),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        decision = apply_epistemic_gate("run the check", ctx, candidate)
        self.assertTrue(decision.intercepted)
        self.assertIn("TOOL_PROVENANCE_NOT_IN_CONTEXT", decision.contract_errors)

    def test_valid_candidate_with_provenance_passes_unchanged(self) -> None:
        ctx = _base_ctx(
            recent_turn_ids=("thread-1:u:2",),
            in_scope_memory_ids=("mem:1",),
            tool_event_ids=("audit:9",),
        )
        candidate = CandidateContract(
            kind="answer",
            final_answer="Confirmed.",
            clarifying_question=None,
            claims=(
                Claim(text="From user", support="user", user_turn_id="thread-1:u:2"),
                Claim(text="From memory", support="memory", memory_id="mem:1"),
                Claim(text="From tool", support="tool", tool_event_id="audit:9"),
            ),
            assumptions=tuple(),
            unresolved_refs=tuple(),
            thread_refs=tuple(),
            raw_json=None,
        )
        decision = apply_epistemic_gate("confirm", ctx, candidate)
        self.assertFalse(decision.intercepted)
        self.assertEqual("Confirmed.", decision.user_text)


if __name__ == "__main__":
    unittest.main()
