from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestThreadPrefs(unittest.TestCase):
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

    def _candidate_response(self, thread_id: str, final_answer: str, claims: list[dict[str, object]]) -> OrchestratorResponse:
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
                "thread_id": thread_id,
                "epistemic_candidate_json": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            },
        )

    def _set_active_thread(self, orch: Orchestrator, user_id: str, thread_id: str) -> None:
        orch._apply_epistemic_layer(
            user_id,
            "hello",
            self._candidate_response(
                thread_id,
                "ack",
                [{"text": "User initiated current thread context.", "support": "user"}],
            ),
        )

    def test_default_effective_prefs_follow_defaults_then_global(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        defaults = orch.handle_message("/prefs_thread", "user1")
        self.assertEqual(
            [
                "show_next_action: on (source: default)",
                "show_summary: on (source: default)",
                "terse_mode: off (source: default)",
                "commands_in_codeblock: off (source: default)",
            ],
            defaults.text.splitlines(),
        )

        orch.handle_message("/prefs_set show_summary off", "user1")
        global_applied = orch.handle_message("/prefs_thread", "user1")
        self.assertIn("show_summary: off (source: global)", global_applied.text)

    def test_thread_override_beats_global_only_for_that_thread(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("/prefs_set show_summary off", "user1")
        long_body = "\n".join(
            [
                "Status report:",
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "line 5",
                "line 6",
                "line 7",
                "line 8",
            ]
        )

        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message("/prefs_thread_set show_summary on", "user1")
        thread_a_view = orch.handle_message("/prefs_thread", "user1")
        self.assertIn("show_summary: on (source: thread)", thread_a_view.text)
        thread_a_reply = orch._apply_epistemic_layer(
            "user1",
            "status",
            self._candidate_response(
                "thread-a",
                long_body,
                [{"text": "Status report was produced for this request context.", "support": "user"}],
            ),
        )
        self.assertIn("In short:", thread_a_reply.text)

        self._set_active_thread(orch, "user1", "thread-b")
        thread_b_view = orch.handle_message("/prefs_thread", "user1")
        self.assertIn("show_summary: off (source: global)", thread_b_view.text)
        thread_b_reply = orch._apply_epistemic_layer(
            "user1",
            "status",
            self._candidate_response(
                "thread-b",
                long_body,
                [{"text": "Status report was produced for this request context.", "support": "user"}],
            ),
        )
        self.assertNotIn("In short:", thread_b_reply.text)

    def test_thread_reset_returns_to_global(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("/prefs_set show_summary off", "user1")
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message("/prefs_thread_set show_summary on", "user1")
        before_reset = orch.handle_message("/prefs_thread", "user1")
        self.assertIn("show_summary: on (source: thread)", before_reset.text)

        reset_resp = orch.handle_message("/prefs_thread_reset", "user1")
        self.assertEqual("Thread preferences reset.", reset_resp.text)
        after_reset = orch.handle_message("/prefs_thread", "user1")
        self.assertIn("show_summary: off (source: global)", after_reset.text)

    def test_prefs_thread_output_order_is_deterministic(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-x")
        response = orch.handle_message("/prefs_thread", "user1")
        self.assertEqual(
            [
                "show_next_action: on (source: default)",
                "show_summary: on (source: default)",
                "terse_mode: off (source: default)",
                "commands_in_codeblock: off (source: default)",
            ],
            response.text.splitlines(),
        )

    def test_intercepts_unaffected_by_thread_prefs(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message("/prefs_thread_set show_summary on", "user1")
        orch.handle_message("/prefs_thread_set show_next_action on", "user1")
        response = orch._apply_epistemic_layer("user1", "schedule it", OrchestratorResponse("Scheduled."))
        self.assertEqual("I’m not sure.", response.text.splitlines()[0])
        self.assertNotIn("In short:", response.text)
        self.assertNotIn("Next:", response.text)
        self.assertEqual(1, response.text.count("?"))


if __name__ == "__main__":
    unittest.main()
