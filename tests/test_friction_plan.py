from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.epistemics.canary import run_canary_suite
from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestFrictionPlan(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )

    def _candidate_response(self, final_answer: str, claims: list[dict[str, object]]) -> OrchestratorResponse:
        payload = {
            "kind": "answer",
            "final_answer": final_answer,
            "clarifying_question": None,
            "claims": claims,
            "assumptions": [],
            "unresolved_refs": [],
            "thread_refs": [],
        }
        return OrchestratorResponse(
            "fallback",
            {
                "thread_id": "thread-1",
                "epistemic_candidate_json": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            },
        )

    def test_planning_phrase_triggers_plan(self) -> None:
        orch = self._orchestrator()
        body = "Create docs/HANDOFF.md.\nRun pytest -q.\nCommit docs update."
        response = orch._apply_epistemic_layer(
            "user1",
            "how should I break this down",
            self._candidate_response(
                body,
                [{"text": body, "support": "user"}],
            ),
        )
        self.assertIn("\n\nPlan:\n", response.text)
        self.assertIn("1. Create docs/HANDOFF.md", response.text)
        self.assertIn("2. Run pytest -q", response.text)

    def test_non_planning_reply_does_not_include_plan(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                "Run pytest -q.",
                [{"text": "Run pytest -q.", "support": "user"}],
            ),
        )
        self.assertNotIn("\n\nPlan:\n", response.text)

    def test_intercept_reply_never_includes_plan(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer("user1", "schedule it", OrchestratorResponse("Scheduled."))
        self.assertIn("I’m not sure.", response.text)
        self.assertNotIn("Plan:", response.text)

    def test_plan_steps_are_ordered_limited_and_truncated(self) -> None:
        orch = self._orchestrator()
        long_step = "Implement " + ("very-long-step-content-" * 8)
        body = "\n".join(
            [
                "Create alpha.",
                "Add beta.",
                "Run pytest -q.",
                "Write gamma notes.",
                long_step + ".",
                "Verify zeta.",
            ]
        )
        response = orch._apply_epistemic_layer(
            "user1",
            "plan this",
            self._candidate_response(
                body,
                [{"text": body, "support": "user"}],
            ),
        )
        self.assertIn("\n\nPlan:\n", response.text)
        plan_block = response.text.split("\n\nPlan:\n", 1)[1].split("\n\n", 1)[0]
        plan_lines = [line for line in plan_block.splitlines() if line.strip()]
        self.assertEqual(5, len(plan_lines))
        self.assertEqual("1. Create alpha", plan_lines[0])
        self.assertEqual("2. Add beta", plan_lines[1])
        self.assertEqual("3. Run pytest -q", plan_lines[2])
        self.assertEqual("4. Write gamma notes", plan_lines[3])
        step_text = plan_lines[4].split(". ", 1)[1]
        self.assertLessEqual(len(step_text), 100)

    def test_plan_appears_before_next(self) -> None:
        orch = self._orchestrator()
        body = (
            "Create docs/HANDOFF.md.\n"
            "Run pytest -q.\n"
            "Commit docs update.\n"
            "Use `pytest -q` to verify."
        )
        response = orch._apply_epistemic_layer(
            "user1",
            "how do I do this",
            self._candidate_response(
                body,
                [{"text": body, "support": "user"}],
            ),
        )
        self.assertIn("\n\nPlan:\n", response.text)
        self.assertIn("\n\nNext: ", response.text)
        self.assertLess(response.text.index("\n\nPlan:\n"), response.text.index("\n\nNext: "))

    def test_terse_mode_with_summary_suppresses_plan(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("/prefs_set terse_mode on", "user1")
        body = "\n".join(
            [
                "Create alpha.",
                "Add beta.",
                "Run pytest -q.",
                "Write gamma.",
                "Define delta.",
                "Set epsilon.",
                "Commit zeta.",
                "Inspect eta.",
                "Verify theta.",
            ]
        )
        response = orch._apply_epistemic_layer(
            "user1",
            "plan this",
            self._candidate_response(
                body,
                [{"text": body, "support": "user"}],
            ),
        )
        self.assertIn("In short:", response.text)
        self.assertNotIn("\n\nPlan:\n", response.text)

    def test_env_disable_turns_off_plan(self) -> None:
        orch = self._orchestrator()
        body = "Create docs/HANDOFF.md.\nRun pytest -q.\nCommit docs update."
        with patch.dict(os.environ, {"FRICTION_PLAN": "0"}, clear=False):
            response = orch._apply_epistemic_layer(
                "user1",
                "how should I break this down",
                self._candidate_response(
                    body,
                    [{"text": body, "support": "user"}],
                ),
            )
        self.assertNotIn("\n\nPlan:\n", response.text)

    def test_imperative_count_trigger_is_deterministic(self) -> None:
        orch = self._orchestrator()
        body = "Create docs/HANDOFF.md. Run pytest -q. Commit docs update."
        first = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        ).text
        second = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        ).text
        self.assertEqual(first, second)
        self.assertIn("\n\nPlan:\n", first)

    def test_canary_suite_still_passes(self) -> None:
        results = run_canary_suite()
        self.assertTrue(all(result.passed for result in results))


if __name__ == "__main__":
    unittest.main()
