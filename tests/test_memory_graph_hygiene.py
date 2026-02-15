from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestMemoryGraphHygiene(unittest.TestCase):
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

    def test_node_rename_by_node_id_updates_label(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Old Label"', "user1")
        renamed = orch.handle_message('/node_rename alpha "New Label??"', "user1")
        self.assertEqual("Node alpha renamed.", renamed.text)
        graph = orch.handle_message("/graph", "user1").text
        self.assertIn("  - alpha: New Label", graph)
        self.assertNotIn("?", graph)

    def test_node_rename_by_alias_resolves(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Old Label"', "user1")
        orch.handle_message("/node_alias alpha a1", "user1")
        renamed = orch.handle_message('/node_rename a1 "Aliased Rename"', "user1")
        self.assertEqual("Node alpha renamed.", renamed.text)
        graph = orch.handle_message("/graph", "user1").text
        self.assertIn("  - alpha: Aliased Rename", graph)

    def test_alias_add_remove_and_sorted_listing(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha"', "user1")
        orch.handle_message('/node beta "Beta"', "user1")
        orch.handle_message("/node_alias beta zed", "user1")
        orch.handle_message("/node_alias alpha alpha_alias", "user1")
        orch.handle_message("/node_alias beta beta_alias", "user1")
        graph = orch.handle_message("/graph", "user1").text
        alias_lines = [line for line in graph.splitlines() if line.startswith("  - ") and " -> " in line]
        self.assertEqual(
            [
                "  - alpha_alias -> alpha",
                "  - beta_alias -> beta",
                "  - zed -> beta",
            ],
            alias_lines,
        )
        removed = orch.handle_message("/node_unalias zed", "user1")
        self.assertEqual("Alias zed removed.", removed.text)
        graph_after = orch.handle_message("/graph", "user1").text
        self.assertNotIn("  - zed -> beta", graph_after)

    def test_link_accepts_aliases(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node first "First"', "user1")
        orch.handle_message('/node second "Second"', "user1")
        orch.handle_message("/node_alias first f", "user1")
        orch.handle_message("/node_alias second s", "user1")
        linked = orch.handle_message("/link f depends s", "user1")
        self.assertEqual("Link created.", linked.text)
        graph = orch.handle_message("/graph", "user1").text
        self.assertIn("  - first --depends--> second", graph)

    def test_node_delete_removes_edges_and_aliases(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha"', "user1")
        orch.handle_message('/node beta "Beta"', "user1")
        orch.handle_message("/node_alias alpha a1", "user1")
        orch.handle_message("/link alpha uses beta", "user1")
        deleted = orch.handle_message("/node_delete a1", "user1")
        self.assertEqual("Node alpha deleted.", deleted.text)
        graph = orch.handle_message("/graph", "user1").text
        self.assertNotIn("  - alpha: Alpha", graph)
        self.assertNotIn("  - a1 -> alpha", graph)
        self.assertNotIn("  - alpha --uses--> beta", graph)
        self.assertIn("Edges:\n  - (none)", graph)

    def test_error_messages_no_question_marks(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha"', "user1")
        responses = [
            orch.handle_message('/node_rename missing "X"', "user1"),
            orch.handle_message("/node_alias alpha !!!", "user1"),
            orch.handle_message("/node_unalias missing", "user1"),
            orch.handle_message("/node_delete missing", "user1"),
        ]
        expected = [
            "Node not found.",
            "Invalid alias.",
            "Alias not found.",
            "Node not found.",
        ]
        self.assertEqual(expected, [response.text for response in responses])
        for response in responses:
            self.assertNotIn("?", response.text)

    def test_graph_includes_alias_section_none_when_empty(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha"', "user1")
        graph = orch.handle_message("/graph", "user1").text
        self.assertIn("Aliases:\n  (none)", graph)
        self.assertNotIn("?", graph)

    def test_determinism_alias_and_edges_order(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node a "A"', "user1")
        orch.handle_message('/node b "B"', "user1")
        orch.handle_message('/node c "C"', "user1")
        orch.handle_message("/node_alias c z", "user1")
        orch.handle_message("/node_alias a aa", "user1")
        orch.handle_message("/node_alias b m", "user1")
        orch.handle_message("/link b rel c", "user1")
        orch.handle_message("/link a rel b", "user1")
        graph = orch.handle_message("/graph", "user1").text
        alias_lines = [line for line in graph.splitlines() if line.startswith("  - ") and " -> " in line]
        edge_lines = [line for line in graph.splitlines() if line.startswith("  - ") and "--" in line]
        self.assertEqual(
            [
                "  - aa -> a",
                "  - m -> b",
                "  - z -> c",
            ],
            alias_lines,
        )
        self.assertEqual(
            [
                "  - a --rel--> b",
                "  - b --rel--> c",
            ],
            edge_lines,
        )


if __name__ == "__main__":
    unittest.main()
