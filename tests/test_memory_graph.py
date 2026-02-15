from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestMemoryGraph(unittest.TestCase):
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

    def test_create_nodes_and_list_sorted(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node zeta "Zeta node"', "user1")
        orch.handle_message('/node alpha "Alpha node"', "user1")
        orch.handle_message('/node beta "Beta node"', "user1")
        graph = orch.handle_message("/graph", "user1").text
        self.assertIn("Nodes:", graph)
        self.assertLess(graph.index("  - alpha: Alpha node"), graph.index("  - beta: Beta node"))
        self.assertLess(graph.index("  - beta: Beta node"), graph.index("  - zeta: Zeta node"))
        self.assertNotIn("?", graph)

    def test_create_edge_and_list_deterministically(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node a "A"', "user1")
        orch.handle_message('/node b "B"', "user1")
        orch.handle_message('/node c "C"', "user1")
        orch.handle_message("/link b relates c", "user1")
        orch.handle_message("/link a depends_on b", "user1")
        orch.handle_message("/link a blocks c", "user1")
        graph = orch.handle_message("/graph", "user1").text
        edge_lines = [line for line in graph.splitlines() if line.startswith("  - ") and "--" in line]
        self.assertEqual(
            [
                "  - a --blocks--> c",
                "  - a --depends_on--> b",
                "  - b --relates--> c",
            ],
            edge_lines,
        )
        self.assertNotIn("?", graph)

    def test_reject_link_if_nodes_missing(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        response = orch.handle_message("/link missing uses also_missing", "user1")
        self.assertEqual("Cannot create link.", response.text)
        graph = orch.handle_message("/graph", "user1").text
        self.assertIn("Edges:", graph)
        self.assertIn("  - (none)", graph.split("Edges:\n", 1)[1])
        self.assertNotIn("?", graph)

    def test_normalization_rules_applied(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        node_resp = orch.handle_message('/node NoDe-1!? " Label??   with   spaces  "', "user1")
        self.assertEqual("Node node1 created.", node_resp.text)
        orch.handle_message('/node node_two "Node Two"', "user1")
        orch.handle_message('/link NoDe-1!? ReL??AtioN-Type!? node_two', "user1")
        graph = orch.handle_message("/graph", "user1").text
        self.assertIn("  - node1: Label with spaces", graph)
        self.assertIn("  - node1 --relation-type!--> node_two", graph)
        self.assertNotIn("?", graph)

    def test_graph_clear_removes_all(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha"', "user1")
        orch.handle_message('/node beta "Beta"', "user1")
        orch.handle_message("/link alpha uses beta", "user1")
        cleared = orch.handle_message("/graph_clear", "user1")
        self.assertEqual("Graph cleared for this thread.", cleared.text)
        graph = orch.handle_message("/graph", "user1").text
        self.assertIn("Nodes:\n  - (none)", graph)
        self.assertIn("Edges:\n  - (none)", graph)
        self.assertNotIn("?", graph)

    def test_graph_data_isolated_per_thread(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha"', "user1")
        graph_a = orch.handle_message("/graph", "user1").text
        self.assertIn("Graph (thread thread-a):", graph_a)
        self.assertIn("  - alpha: Alpha", graph_a)

        self._set_active_thread(orch, "user1", "thread-b")
        graph_b = orch.handle_message("/graph", "user1").text
        self.assertIn("Graph (thread thread-b):", graph_b)
        self.assertIn("Nodes:\n  - (none)", graph_b)

    def test_graph_command_outputs_not_decorated(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        responses = [
            orch.handle_message('/node alpha "Alpha"', "user1"),
            orch.handle_message("/graph", "user1"),
            orch.handle_message("/graph_clear", "user1"),
        ]
        for response in responses:
            self.assertNotIn("In short:", response.text)
            self.assertNotIn("Plan:", response.text)
            self.assertNotIn("Options:", response.text)
            self.assertNotIn("Next:", response.text)
            self.assertNotIn("?", response.text)


if __name__ == "__main__":
    unittest.main()
