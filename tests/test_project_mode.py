from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestProjectMode(unittest.TestCase):
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
                [{"text": "User initiated this thread context.", "support": "user"}],
            ),
        )

    def test_project_mode_default_off_status(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        response = orch.handle_message("/project_mode", "user1")
        self.assertEqual("Project mode: off", response.text)
        self.assertNotIn("?", response.text)

    def test_project_mode_forces_summary_off_and_codeblock_on(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        on_resp = orch.handle_message("/project_mode on", "user1")
        self.assertEqual("Project mode enabled for thread-a.", on_resp.text)

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
                "pytest -q",
            ]
        )
        response = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                "thread-a",
                long_body,
                [{"text": "Status report lines were produced from this request.", "support": "user"}],
            ),
        )
        self.assertNotIn("In short:", response.text)
        self.assertIn("```bash\npytest -q\n```", response.text)

    def test_project_mode_lowers_plan_threshold_to_one_imperative(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        body = "Run pytest -q."
        before = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                "thread-a",
                body,
                [{"text": "Run pytest -q.", "support": "user"}],
            ),
        )
        self.assertNotIn("Plan:", before.text)

        orch.handle_message("/project_mode on", "user1")
        after = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                "thread-a",
                body,
                [{"text": "Run pytest -q.", "support": "user"}],
            ),
        )
        self.assertIn("Plan:", after.text)
        self.assertIn("1. Run pytest -q", after.text)

    def test_project_mode_off_restores_existing_prefs(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message("/prefs_thread_set show_summary on", "user1")
        orch.handle_message("/prefs_thread_set commands_in_codeblock off", "user1")
        orch.handle_message("/project_mode on", "user1")

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
                "pytest -q",
            ]
        )
        on_reply = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                "thread-a",
                long_body,
                [{"text": "Status report lines were produced from this request.", "support": "user"}],
            ),
        )
        self.assertNotIn("In short:", on_reply.text)
        self.assertIn("```bash\npytest -q\n```", on_reply.text)

        off_cmd = orch.handle_message("/project_mode off", "user1")
        self.assertEqual("Project mode disabled for thread-a.", off_cmd.text)
        off_reply = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                "thread-a",
                long_body,
                [{"text": "Status report lines were produced from this request.", "support": "user"}],
            ),
        )
        self.assertIn("In short:", off_reply.text)
        self.assertNotIn("```bash\npytest -q\n```", off_reply.text)

    def test_project_mode_status_and_intercepts_unaffected(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        self.assertEqual("Project mode: off", orch.handle_message("/project_mode", "user1").text)
        self.assertEqual("Project mode enabled for thread-a.", orch.handle_message("/project_mode on", "user1").text)
        self.assertEqual("Project mode: on", orch.handle_message("/project_mode", "user1").text)

        intercept = orch._apply_epistemic_layer("user1", "schedule it", OrchestratorResponse("Scheduled."))
        self.assertEqual("I’m not sure.", intercept.text.splitlines()[0])
        self.assertNotIn("In short:", intercept.text)
        self.assertNotIn("Plan:", intercept.text)
        self.assertNotIn("Next:", intercept.text)

    def test_project_mode_command_outputs_no_question_and_not_decorated(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        responses = [
            orch.handle_message("/project_mode", "user1"),
            orch.handle_message("/project_mode on", "user1"),
            orch.handle_message("/project_mode off", "user1"),
            orch.handle_message("/project_mode invalid", "user1"),
        ]
        for response in responses:
            self.assertNotIn("?", response.text)
            self.assertNotIn("In short:", response.text)
            self.assertNotIn("Plan:", response.text)
            self.assertNotIn("Options:", response.text)
            self.assertNotIn("Next:", response.text)


if __name__ == "__main__":
    unittest.main()
