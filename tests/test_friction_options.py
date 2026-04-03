from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.epistemics.canary import run_canary_suite
from agent.friction.canary import run_friction_canaries
from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestFrictionOptions(unittest.TestCase):
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

    def _extract_options_block(self, text: str) -> str | None:
        marker = "\n\nOptions:\n"
        if marker not in text:
            return None
        block = text.split(marker, 1)[1]
        if "\n\n" in block:
            block = block.split("\n\n", 1)[0]
        return block

    def test_options_triggered_by_distinct_imperative_approaches(self) -> None:
        orch = self._orchestrator()
        body = "Open docs/HANDOFF.md. Use the release template. Check current tests."
        response = orch._apply_epistemic_layer(
            "user1",
            "compare options",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        )
        block = self._extract_options_block(response.text)
        self.assertIsNotNone(block)
        assert block is not None
        self.assertIn("A) Open docs/HANDOFF.md", block)
        self.assertIn("B) Use the release template", block)
        self.assertIn("Choose A, B, or C.", block)

    def test_no_options_when_only_one_approach(self) -> None:
        orch = self._orchestrator()
        body = "Open docs/HANDOFF.md."
        response = orch._apply_epistemic_layer(
            "user1",
            "which should I use",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        )
        self.assertNotIn("\n\nOptions:\n", response.text)

    def test_no_options_on_intercepted_reply(self) -> None:
        orch = self._orchestrator()
        response = orch._apply_epistemic_layer("user1", "schedule it", OrchestratorResponse("Scheduled."))
        self.assertIn("I’m not sure.", response.text)
        self.assertNotIn("Options:", response.text)

    def test_max_three_options(self) -> None:
        orch = self._orchestrator()
        body = "\n".join(
            [
                "Open alpha.",
                "Use beta.",
                "Check gamma.",
                "Update delta.",
                "Verify epsilon.",
            ]
        )
        response = orch._apply_epistemic_layer(
            "user1",
            "compare",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        )
        block = self._extract_options_block(response.text)
        self.assertIsNotNone(block)
        assert block is not None
        option_lines = [line for line in block.splitlines() if line.startswith(("A)", "B)", "C)"))]
        self.assertLessEqual(len(option_lines), 3)

    def test_options_block_has_no_question_marks(self) -> None:
        orch = self._orchestrator()
        body = "Open docs/HANDOFF.md?. Use the release template?. Check current tests?"
        response = orch._apply_epistemic_layer(
            "user1",
            "options",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        )
        block = self._extract_options_block(response.text)
        self.assertIsNotNone(block)
        assert block is not None
        self.assertNotIn("?", block)

    def test_deterministic_ordering(self) -> None:
        orch = self._orchestrator()
        body = "Open alpha. Use beta. Check gamma."
        first = orch._apply_epistemic_layer(
            "user1",
            "options",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        ).text
        second = orch._apply_epistemic_layer(
            "user1",
            "options",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        ).text
        self.assertEqual(self._extract_options_block(first), self._extract_options_block(second))

    def test_options_placement_before_next(self) -> None:
        orch = self._orchestrator()
        body = (
            "Open docs/HANDOFF.md.\n"
            "Check release notes.\n"
            "Use `pytest -q` to verify."
        )
        response = orch._apply_epistemic_layer(
            "user1",
            "which should I use",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        )
        self.assertIn("\n\nOptions:\n", response.text)
        self.assertIn("\n\nNext: ", response.text)
        self.assertLess(response.text.index("\n\nOptions:\n"), response.text.index("\n\nNext: "))

    def test_terse_mode_with_plan_skips_options(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("/prefs_set terse_mode on", "user1")
        body = "Create alpha. Add beta. Inspect gamma."
        response = orch._apply_epistemic_layer(
            "user1",
            "plan this",
            self._candidate_response(body, [{"text": body, "support": "user"}]),
        )
        self.assertIn("\n\nPlan:\n", response.text)
        self.assertNotIn("\n\nOptions:\n", response.text)

    def test_env_disable_turns_off_options(self) -> None:
        orch = self._orchestrator()
        body = "Open alpha. Use beta. Check gamma."
        with patch.dict(os.environ, {"FRICTION_OPTIONS": "0"}, clear=False):
            response = orch._apply_epistemic_layer(
                "user1",
                "options",
                self._candidate_response(body, [{"text": body, "support": "user"}]),
            )
        self.assertNotIn("\n\nOptions:\n", response.text)

    def test_canary_suites_still_pass(self) -> None:
        epistemics = run_canary_suite()
        self.assertTrue(all(result.passed for result in epistemics))
        friction = run_friction_canaries()
        self.assertEqual(0, friction["failed"])


if __name__ == "__main__":
    unittest.main()
