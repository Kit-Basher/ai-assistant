from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestFocusNode(unittest.TestCase):
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
            "claims": [{"text": "Thread context set.", "support": "user"}],
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

    def test_focus_node_set_by_alias_canonicalizes(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node core_api "Core API"', "user1")
        orch.handle_message("/node_alias core_api api", "user1")
        response = orch.handle_message("/focus_node api", "user1")
        self.assertEqual("Focus node set to core_api.", response.text)
        self.assertEqual("core_api", self.db.get_thread_focus_node("thread-a"))
        self.assertNotIn("?", response.text)

    def test_focus_node_show_with_label(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node planner "Planner Node"', "user1")
        orch.handle_message("/focus_node planner", "user1")
        response = orch.handle_message("/focus_node", "user1")
        self.assertEqual("Focus node: planner (Planner Node)", response.text)
        self.assertNotIn("?", response.text)

    def test_focus_node_clear(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node planner "Planner Node"', "user1")
        orch.handle_message("/focus_node planner", "user1")
        cleared = orch.handle_message("/focus_node_clear", "user1")
        self.assertEqual("Focus node cleared.", cleared.text)
        status = orch.handle_message("/focus_node", "user1")
        self.assertEqual("Focus node: (none)", status.text)
        self.assertNotIn("?", cleared.text)
        self.assertNotIn("?", status.text)

    def test_resume_includes_related_nodes_when_focus_set(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/anchor Sprint\n- one', "user1")
        orch.handle_message('/node a "A"', "user1")
        orch.handle_message('/node b "B"', "user1")
        orch.handle_message('/node c "C"', "user1")
        orch.handle_message("/link a alpha b", "user1")
        orch.handle_message("/link a beta c", "user1")
        orch.handle_message("/focus_node a", "user1")
        resume = orch.handle_message("/resume", "user1").text
        self.assertIn("Related nodes:", resume)
        self.assertIn("- b (B)", resume)
        self.assertIn("- c (C)", resume)
        self.assertNotIn("?", resume)

    def test_resume_related_nodes_none_when_focus_has_no_outgoing(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/anchor Sprint\n- one', "user1")
        orch.handle_message('/node solo "Solo"', "user1")
        orch.handle_message("/focus_node solo", "user1")
        resume = orch.handle_message("/resume", "user1").text
        self.assertIn("Related nodes:\n(none)", resume)
        self.assertNotIn("?", resume)

    def test_resume_unchanged_when_focus_not_set(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/anchor Sprint\n- one', "user1")
        resume = orch.handle_message("/resume", "user1").text
        self.assertNotIn("Related nodes:", resume)
        self.assertNotIn("?", resume)

    def test_related_nodes_order_deterministic_relation_then_to_node(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/anchor Sprint\n- one', "user1")
        orch.handle_message('/node root "Root"', "user1")
        orch.handle_message('/node zed "Zed"', "user1")
        orch.handle_message('/node alpha "Alpha"', "user1")
        orch.handle_message('/node beta "Beta"', "user1")
        orch.handle_message("/link root z_rel zed", "user1")
        orch.handle_message("/link root a_rel beta", "user1")
        orch.handle_message("/link root a_rel alpha", "user1")
        orch.handle_message("/focus_node root", "user1")
        resume = orch.handle_message("/resume", "user1").text
        related_lines = [line for line in resume.splitlines() if line.startswith("- ") and "(" in line and ")" in line]
        self.assertEqual(
            [
                "- alpha (Alpha)",
                "- beta (Beta)",
                "- zed (Zed)",
            ],
            related_lines,
        )

    def test_thread_isolation_for_focus_and_related_nodes(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/anchor A\n- one', "user1")
        orch.handle_message('/node a "A"', "user1")
        orch.handle_message('/node b "B"', "user1")
        orch.handle_message("/link a rel b", "user1")
        orch.handle_message("/focus_node a", "user1")

        self._set_active_thread(orch, "user1", "thread-b")
        orch.handle_message('/anchor B\n- one', "user1")
        orch.handle_message('/node x "X"', "user1")
        orch.handle_message("/focus_node x", "user1")
        resume_b = orch.handle_message("/resume", "user1").text
        self.assertIn("Resume (thread thread-b):", resume_b)
        self.assertIn("Related nodes:\n(none)", resume_b)
        self.assertNotIn("- b (B)", resume_b)
        self.assertNotIn("?", resume_b)


if __name__ == "__main__":
    unittest.main()
