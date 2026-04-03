from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.epistemics.canary import run_canary_suite
from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestFrictionNextAction(unittest.TestCase):
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

    def test_intercept_reply_never_includes_next(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer("user1", "schedule it", OrchestratorResponse("Scheduled."))
        self.assertIn("I’m not sure.", response.text)
        self.assertNotIn("Next:", response.text)

    def test_pass_reply_includes_next_when_command_present(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer(
            "user1",
            "run tests",
            self._candidate_response(
                "Use `pytest -q` to verify.",
                [{"text": "User requested a test run.", "support": "user"}],
            ),
        )
        self.assertIn("Next: Run pytest -q", response.text)

    def test_pass_reply_no_next_without_extractable_action(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                "Everything looks stable.",
                [{"text": "User requested status.", "support": "user"}],
            ),
        )
        self.assertNotIn("Next:", response.text)

    def test_next_line_shape_is_single_line_without_question_and_max_120(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer(
            "user1",
            "run tests",
            self._candidate_response(
                "Use `pytest -q` to verify.",
                [{"text": "User requested a test run.", "support": "user"}],
            ),
        )
        next_lines = [line for line in response.text.splitlines() if line.startswith("Next: ")]
        self.assertEqual(1, len(next_lines))
        step = next_lines[0][len("Next: "):]
        self.assertNotIn("?", step)
        self.assertNotIn("\n", step)
        self.assertLessEqual(len(step), 120)

    def test_env_disable_turns_off_next_line(self) -> None:
        orch = self._orchestrator()
        with patch.dict(os.environ, {"FRiction_NEXT_ACTION": "0"}, clear=False):
            response = orch._apply_epistemic_layer(
                "user1",
                "run tests",
                self._candidate_response(
                    "Use `pytest -q` to verify.",
                    [{"text": "User requested a test run.", "support": "user"}],
                ),
            )
        self.assertNotIn("Next:", response.text)

    def test_canary_suite_still_passes(self) -> None:
        results = run_canary_suite()
        self.assertTrue(all(result.passed for result in results))


if __name__ == "__main__":
    unittest.main()

