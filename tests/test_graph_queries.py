from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestGraphQueries(unittest.TestCase):
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

    def _assert_clean_output(self, text: str) -> None:
        self.assertNotIn("?", text)
        self.assertNotIn("In short:", text)
        self.assertNotIn("Plan:", text)
        self.assertNotIn("Options:", text)
        self.assertNotIn("Next:", text)

    def test_graph_out_groups_and_sorts_deterministically(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        orch.handle_message('/node c "C"', "u1")
        orch.handle_message('/node d "D"', "u1")
        orch.handle_message("/link a beta c", "u1")
        orch.handle_message("/link a alpha d", "u1")
        orch.handle_message("/link a alpha b", "u1")

        response = orch.handle_message("/graph_out a", "u1")
        self._assert_clean_output(response.text)
        self.assertEqual(
            [
                "Graph out (thread thread-a):",
                "Node: a (A)",
                "alpha:",
                "  - b (B)",
                "  - d (D)",
                "beta:",
                "  - c (C)",
            ],
            response.text.splitlines(),
        )

    def test_graph_in_groups_and_sorts_deterministically(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node s1 "S1"', "u1")
        orch.handle_message('/node s2 "S2"', "u1")
        orch.handle_message('/node t "T"', "u1")
        orch.handle_message("/link s2 alpha t", "u1")
        orch.handle_message("/link s1 alpha t", "u1")
        orch.handle_message("/link s1 beta t", "u1")

        response = orch.handle_message("/graph_in t", "u1")
        self._assert_clean_output(response.text)
        self.assertEqual(
            [
                "Graph in (thread thread-a):",
                "Node: t (T)",
                "alpha:",
                "  - s1 (S1)",
                "  - s2 (S2)",
                "beta:",
                "  - s1 (S1)",
            ],
            response.text.splitlines(),
        )

    def test_graph_path_shortest_and_deterministic_neighbor_order(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node start "Start"', "u1")
        orch.handle_message('/node n1 "N1"', "u1")
        orch.handle_message('/node n2 "N2"', "u1")
        orch.handle_message('/node target "Target"', "u1")
        orch.handle_message("/link start alpha n2", "u1")
        orch.handle_message("/link start alpha n1", "u1")
        orch.handle_message("/link n1 step target", "u1")
        orch.handle_message("/link n2 step target", "u1")

        response = orch.handle_message("/graph_path start target", "u1")
        self._assert_clean_output(response.text)
        self.assertEqual(
            [
                "Graph path (thread thread-a):",
                "From: start (Start)",
                "To: target (Target)",
                "Depth: 2",
                "Path:",
                "  1) start (Start)",
                "  2) --alpha--> n1 (N1)",
                "  3) --step--> target (Target)",
            ],
            response.text.splitlines(),
        )

    def test_graph_path_no_path_and_max_limit(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        orch.handle_message('/node c "C"', "u1")
        orch.handle_message('/node isolated "Isolated"', "u1")
        orch.handle_message("/link a rel b", "u1")
        orch.handle_message("/link b rel c", "u1")

        capped = orch.handle_message("/graph_path a c --max 1", "u1")
        self._assert_clean_output(capped.text)
        self.assertEqual(
            "Graph path (thread thread-a):\nNo path found.",
            capped.text,
        )

        no_path = orch.handle_message("/graph_path a isolated", "u1")
        self._assert_clean_output(no_path.text)
        self.assertEqual(
            "Graph path (thread thread-a):\nNo path found.",
            no_path.text,
        )

    def test_alias_resolution_for_all_graph_query_commands(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node src "Src"', "u1")
        orch.handle_message('/node hub "Hub"', "u1")
        orch.handle_message('/node dst "Dst"', "u1")
        orch.handle_message("/node_alias src s", "u1")
        orch.handle_message("/node_alias hub h", "u1")
        orch.handle_message("/node_alias dst d", "u1")
        orch.handle_message("/link src rel hub", "u1")
        orch.handle_message("/link hub rel dst", "u1")

        out_resp = orch.handle_message("/graph_out h", "u1")
        self._assert_clean_output(out_resp.text)
        self.assertIn("Node: hub (Hub)", out_resp.text)
        self.assertIn("  - dst (Dst)", out_resp.text)

        in_resp = orch.handle_message("/graph_in h", "u1")
        self._assert_clean_output(in_resp.text)
        self.assertIn("Node: hub (Hub)", in_resp.text)
        self.assertIn("  - src (Src)", in_resp.text)

        path_resp = orch.handle_message("/graph_path s d", "u1")
        self._assert_clean_output(path_resp.text)
        self.assertIn("From: src (Src)", path_resp.text)
        self.assertIn("To: dst (Dst)", path_resp.text)
        self.assertIn("Depth: 2", path_resp.text)

    def test_node_not_found_message(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        responses = [
            orch.handle_message("/graph_out missing", "u1"),
            orch.handle_message("/graph_in missing", "u1"),
            orch.handle_message("/graph_path missing other", "u1"),
        ]
        for response in responses:
            self.assertEqual("Node not found.", response.text)
            self._assert_clean_output(response.text)


if __name__ == "__main__":
    unittest.main()
