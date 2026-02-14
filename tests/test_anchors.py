from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestAnchors(unittest.TestCase):
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

    def _candidate_response(self, thread_id: str) -> OrchestratorResponse:
        payload = {
            "kind": "answer",
            "final_answer": "ack",
            "clarifying_question": None,
            "claims": [{"text": "User provided current thread context.", "support": "user"}],
            "assumptions": [],
            "unresolved_refs": [],
            "thread_refs": [],
        }
        return OrchestratorResponse(
            "fallback",
            {
                "thread_id": thread_id,
                "epistemic_candidate_json": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            },
        )

    def _set_active_thread(self, orch: Orchestrator, user_id: str, thread_id: str) -> None:
        orch._apply_epistemic_layer(user_id, "hello", self._candidate_response(thread_id))

    def test_anchor_stores_per_thread_only(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        save = orch.handle_message("/anchor Sprint check\n- Ship docs\n- Run tests", "user1")
        self.assertEqual("Saved checkpoint 1.", save.text)

        anchors_a = orch.handle_message("/anchors", "user1")
        self.assertIn("Anchors (thread thread-a):", anchors_a.text)
        self.assertIn("#1 ", anchors_a.text)
        self.assertIn("  - Ship docs", anchors_a.text)

        self._set_active_thread(orch, "user1", "thread-b")
        anchors_b = orch.handle_message("/anchors", "user1")
        self.assertEqual("Anchors (thread thread-b):\n(none)", anchors_b.text)

    def test_list_formatting_deterministic_newest_first(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message("/anchor First\n- one", "user1")
        orch.handle_message("/anchor Second\n- two", "user1")
        response = orch.handle_message("/anchors", "user1")
        lines = response.text.splitlines()
        self.assertEqual("Anchors (thread thread-a):", lines[0])
        self.assertRegex(lines[1], r"^#2 .+ — Second$")
        self.assertEqual("  - two", lines[2])
        self.assertRegex(lines[3], r"^#1 .+ — First$")
        self.assertEqual("  - one", lines[4])
        self.assertTrue(all("?" not in line for line in lines))

    def test_open_line_normalization_strips_question_mark(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message("/anchor Release\n- Build docs\nOpen: decide publish time?", "user1")
        rows = self.db.list_thread_anchors("thread-a", limit=5)
        self.assertEqual(1, len(rows))
        self.assertEqual("Open: decide publish time", rows[0]["open_line"])
        self.assertNotIn("?", rows[0]["open_line"])

    def test_bullets_normalization_and_max_three_enforced(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message(
            "/checkpoint Checkpoint?\n- one?\n* two?\nthree?\nfour?\nOpen: pending?",
            "user1",
        )
        rows = self.db.list_thread_anchors("thread-a", limit=5)
        self.assertEqual(1, len(rows))
        bullets = json.loads(rows[0]["bullets"])
        self.assertEqual(["one", "two", "three"], bullets)
        self.assertNotIn("?", "".join(bullets))
        self.assertEqual("Open: pending", rows[0]["open_line"])

    def test_anchors_reset_clears_thread(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message("/anchor A\n- one", "user1")
        reset = orch.handle_message("/anchors_reset", "user1")
        self.assertEqual("Cleared anchors for this thread.", reset.text)
        listed = orch.handle_message("/anchors", "user1")
        self.assertEqual("Anchors (thread thread-a):\n(none)", listed.text)

    def test_anchor_command_replies_are_not_friction_decorated(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        save = orch.handle_message("/anchor Status\nCreate alpha. Add beta. Run tests.", "user1")
        list_resp = orch.handle_message("/anchors", "user1")
        for response in (save, list_resp):
            self.assertNotIn("In short:", response.text)
            self.assertNotIn("Plan:", response.text)
            self.assertNotIn("Options:", response.text)
            self.assertNotIn("Next:", response.text)
            self.assertEqual(0, response.text.count("?"))


if __name__ == "__main__":
    unittest.main()
