from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.epistemics.canary import run_canary_suite
from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestFrictionSummary(unittest.TestCase):
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

    def test_intercept_reply_never_includes_summary(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer("user1", "schedule it", OrchestratorResponse("Scheduled."))
        self.assertIn("I’m not sure.", response.text)
        self.assertNotIn("In short:", response.text)

    def test_long_pass_reply_includes_summary(self) -> None:
        orch = self._orchestrator()
        long_body = "\n".join(
            [
                "Line 1",
                "Line 2",
                "Line 3",
                "Line 4",
                "Line 5",
                "Line 6",
                "Line 7",
                "Line 8",
                "Line 9",
            ]
        )
        response = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                long_body,
                [{"text": "Status lines were produced from the current request context.", "support": "user"}],
            ),
        )
        self.assertTrue(response.text.startswith("In short: "))
        self.assertIn("\n\nLine 1", response.text)

    def test_short_pass_reply_does_not_include_summary(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                "Line 1\nLine 2",
                [{"text": "Status lines were produced from the current request context.", "support": "user"}],
            ),
        )
        self.assertNotIn("In short:", response.text)

    def test_summary_line_shape(self) -> None:
        orch = self._orchestrator()
        long_claim = (
            "This claim is intentionally very long to ensure deterministic truncation to meet strict summary "
            "limits without introducing uncertainty or extra punctuation"
        )
        long_body = "\n".join(f"Line {idx}" for idx in range(1, 11))
        response = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(long_body, [{"text": long_claim, "support": "user"}]),
        )
        summary_line = response.text.splitlines()[0]
        self.assertTrue(summary_line.startswith("In short: "))
        self.assertLessEqual(len(summary_line), 140)
        self.assertNotIn("?", summary_line)

    def test_summary_env_disable(self) -> None:
        orch = self._orchestrator()
        long_body = "\n".join(f"Line {idx}" for idx in range(1, 11))
        with patch.dict(os.environ, {"FRICTION_SUMMARY": "0"}, clear=False):
            response = orch._apply_epistemic_layer(
                "user1",
                "status update",
                self._candidate_response(
                    long_body,
                    [{"text": "Status lines were produced from the current request context.", "support": "user"}],
                ),
            )
        self.assertNotIn("In short:", response.text)

    def test_canary_suite_still_passes(self) -> None:
        results = run_canary_suite()
        self.assertTrue(all(result.passed for result in results))


if __name__ == "__main__":
    unittest.main()

