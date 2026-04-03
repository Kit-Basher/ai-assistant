from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestPrefs(unittest.TestCase):
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

    def test_default_prefs_keep_summary_and_next_behavior(self) -> None:
        orch = self._orchestrator()
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
                "Use `pytest -q` to verify.",
            ]
        )
        response = orch._apply_epistemic_layer(
            "user1",
            "run tests",
            self._candidate_response(
                long_body,
                [{"text": "Status report was prepared for this request.", "support": "user"}],
            ),
        )
        self.assertIn("In short:", response.text)
        self.assertIn("Next: Run pytest -q", response.text)

    def test_prefs_set_show_next_action_off_disables_next(self) -> None:
        orch = self._orchestrator()
        set_resp = orch.handle_message("/prefs_set show_next_action off", "user1")
        self.assertIn("show_next_action: off", set_resp.text)
        response = orch._apply_epistemic_layer(
            "user1",
            "run tests",
            self._candidate_response(
                "Use `pytest -q` to verify.",
                [{"text": "User requested a test run.", "support": "user"}],
            ),
        )
        self.assertNotIn("Next:", response.text)

    def test_prefs_set_show_summary_off_disables_summary(self) -> None:
        orch = self._orchestrator()
        set_resp = orch.handle_message("/prefs_set show_summary off", "user1")
        self.assertIn("show_summary: off", set_resp.text)
        long_body = "\n".join(f"Line {idx}" for idx in range(1, 10))
        response = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                long_body,
                [{"text": "Status lines were produced from the current request.", "support": "user"}],
            ),
        )
        self.assertNotIn("In short:", response.text)

    def test_terse_mode_trims_pass_reply_and_not_intercept(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("/prefs_set terse_mode on", "user1")
        body = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        pass_response = orch._apply_epistemic_layer(
            "user1",
            "status update",
            self._candidate_response(
                body,
                [{"text": "Status paragraph came from this request.", "support": "user"}],
            ),
        )
        self.assertIn("Paragraph one.", pass_response.text)
        self.assertNotIn("Paragraph two.", pass_response.text)

        intercept_response = orch._apply_epistemic_layer("user1", "schedule it", OrchestratorResponse("Scheduled."))
        self.assertEqual("I’m not sure.", intercept_response.text.splitlines()[0])
        self.assertEqual(1, intercept_response.text.count("?"))

    def test_commands_in_codeblock_wraps_standalone_command_line(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("/prefs_set commands_in_codeblock on", "user1")
        response = orch._apply_epistemic_layer(
            "user1",
            "run tests",
            self._candidate_response(
                "Run this:\npytest -q\nDone.",
                [{"text": "User requested a test run.", "support": "user"}],
            ),
        )
        self.assertIn("```bash\npytest -q\n```", response.text)
        self.assertEqual(0, response.text.split("```bash\npytest -q\n```")[0].count("?"))

    def test_prefs_output_ordering_and_reset(self) -> None:
        orch = self._orchestrator()
        orch.handle_message("/prefs_set show_summary off", "user1")
        prefs_response = orch.handle_message("/prefs", "user1")
        self.assertEqual(
            [
                "show_next_action: on",
                "show_summary: off",
                "terse_mode: off",
                "commands_in_codeblock: off",
            ],
            prefs_response.text.splitlines(),
        )
        reset_resp = orch.handle_message("/prefs_reset", "user1")
        self.assertEqual("Preferences reset to defaults.", reset_resp.text)
        prefs_after_reset = orch.handle_message("/prefs", "user1")
        self.assertEqual(
            [
                "show_next_action: on",
                "show_summary: on",
                "terse_mode: off",
                "commands_in_codeblock: off",
            ],
            prefs_after_reset.text.splitlines(),
        )


if __name__ == "__main__":
    unittest.main()

