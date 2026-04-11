from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestGraphExportImport(unittest.TestCase):
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

    @staticmethod
    def _payload_text(payload: dict[str, object]) -> str:
        return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=False)

    def _stable_graph_state(self, thread_id: str) -> dict[str, object]:
        exported = self.db.export_graph(thread_id)
        return {
            "nodes": exported.get("nodes", []),
            "aliases": exported.get("aliases", []),
            "edges": exported.get("edges", []),
            "focus_node": exported.get("focus_node"),
        }

    def test_graph_export_shape_and_ordering(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node zeta "Zeta"', "user1")
        orch.handle_message('/node alpha "Alpha"', "user1")
        orch.handle_message("/node_alias zeta z", "user1")
        orch.handle_message("/node_alias alpha a", "user1")
        orch.handle_message("/link alpha depends zeta", "user1")
        orch.handle_message("/focus_node alpha", "user1")

        response = orch.handle_message("/graph_export", "user1")
        self.assertNotIn("?", response.text)
        self.assertNotIn("In short:", response.text)
        self.assertNotIn("Next:", response.text)
        self.assertTrue(response.text.startswith("{\n \"thread_id\": "))
        self.assertIn("\n \"nodes\": [", response.text)
        self.assertIn("\n \"aliases\": [", response.text)
        self.assertIn("\n \"edges\": [", response.text)
        self.assertIn("\n \"focus_node\": ", response.text)

        root_pairs = json.loads(response.text, object_pairs_hook=list)
        self.assertEqual(
            ["thread_id", "exported_at", "nodes", "aliases", "edges", "focus_node"],
            [key for key, _ in root_pairs],
        )
        root = dict(root_pairs)

        nodes = root["nodes"]
        self.assertEqual(["alpha", "zeta"], [dict(node)["node_id"] for node in nodes])
        self.assertEqual(["node_id", "label", "created_at"], [key for key, _ in nodes[0]])

        aliases = root["aliases"]
        self.assertEqual(["a", "z"], [dict(alias)["alias"] for alias in aliases])
        self.assertEqual(["alias", "node_id", "created_at"], [key for key, _ in aliases[0]])

        edges = root["edges"]
        self.assertEqual([("alpha", "depends", "zeta")], [(dict(edge)["from"], dict(edge)["relation"], dict(edge)["to"]) for edge in edges])
        self.assertEqual(["from", "relation", "to", "created_at"], [key for key, _ in edges[0]])

        focus_node = root["focus_node"]
        self.assertIsInstance(focus_node, list)
        self.assertEqual(["node_id", "updated_at"], [key for key, _ in focus_node])
        self.assertEqual("alpha", dict(focus_node)["node_id"])

    def test_graph_import_replace_clears_and_restores(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node old "Old"', "user1")
        orch.handle_message("/focus_node old", "user1")

        payload = {
            "thread_id": "ignored-thread",
            "exported_at": "2026-02-15T07:00:00+00:00",
            "nodes": [
                {"node_id": "node_b", "label": "Node B", "created_at": "2026-02-15T07:01:00+00:00"},
                {"node_id": "node_a", "label": "Node A", "created_at": "2026-02-15T07:00:00+00:00"},
            ],
            "aliases": [
                {"alias": "b", "node_id": "node_b", "created_at": "2026-02-15T07:02:00+00:00"},
            ],
            "edges": [
                {
                    "from": "node_a",
                    "relation": "uses",
                    "to": "node_b",
                    "created_at": "2026-02-15T07:03:00+00:00",
                },
            ],
            "focus_node": {"node_id": "node_a", "updated_at": "2026-02-15T07:04:00+00:00"},
        }

        response = orch.handle_message("/graph_import " + self._payload_text(payload), "user1")
        self.assertEqual("Graph imported.", response.text)
        self.assertNotIn("?", response.text)
        graph = orch.handle_message("/graph", "user1").text
        self.assertNotIn("old", graph)
        self.assertIn(" - node_a: Node A", graph)
        self.assertIn(" - node_b: Node B", graph)
        self.assertIn(" - b -> node_b", graph)
        self.assertIn(" - node_a --uses--> node_b", graph)
        self.assertEqual("node_a", self.db.get_thread_focus_node("thread-a"))

    def test_graph_import_merge_conflict_rules(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha Existing"', "user1")
        orch.handle_message('/node beta "Beta"', "user1")
        orch.handle_message("/node_alias alpha a", "user1")
        orch.handle_message("/focus_node alpha", "user1")

        payload = {
            "thread_id": "thread-a",
            "exported_at": "2026-02-15T08:00:00+00:00",
            "nodes": [
                {"node_id": "alpha", "label": "Alpha Incoming", "created_at": "2026-02-15T08:00:01+00:00"},
                {"node_id": "gamma", "label": "Gamma", "created_at": "2026-02-15T08:00:02+00:00"},
            ],
            "aliases": [
                {"alias": "a", "node_id": "gamma", "created_at": "2026-02-15T08:00:03+00:00"},
                {"alias": "g", "node_id": "gamma", "created_at": "2026-02-15T08:00:04+00:00"},
            ],
            "edges": [
                {
                    "from": "alpha",
                    "relation": "depends",
                    "to": "gamma",
                    "created_at": "2026-02-15T08:00:05+00:00",
                },
            ],
            "focus_node": {"node_id": "gamma", "updated_at": "2026-02-15T08:00:06+00:00"},
        }

        response = orch.handle_message("/graph_import --merge " + self._payload_text(payload), "user1")
        self.assertEqual("Graph merged.", response.text)
        self.assertNotIn("?", response.text)

        alpha = self.db.get_graph_node("thread-a", "alpha")
        gamma = self.db.get_graph_node("thread-a", "gamma")
        self.assertIsNotNone(alpha)
        self.assertIsNotNone(gamma)
        self.assertEqual("Alpha Existing", str((alpha or {}).get("label")))
        self.assertEqual("alpha", self.db.resolve_graph_ref("thread-a", "a"))
        self.assertEqual("gamma", self.db.resolve_graph_ref("thread-a", "g"))
        edges = self.db.list_graph_edges("thread-a")
        edge_triples = [(str(edge["from_node"]), str(edge["relation"]), str(edge["to_node"])) for edge in edges]
        self.assertIn(("alpha", "depends", "gamma"), edge_triples)
        self.assertEqual("alpha", self.db.get_thread_focus_node("thread-a"))

    def test_invalid_graph_import_does_not_change_state(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha"', "user1")
        orch.handle_message('/node beta "Beta"', "user1")
        orch.handle_message("/link alpha rel beta", "user1")
        orch.handle_message("/focus_node alpha", "user1")
        before = self._stable_graph_state("thread-a")

        invalid_payload = {
            "thread_id": "thread-a",
            "exported_at": "2026-02-15T09:00:00+00:00",
            "nodes": [{"node_id": "alpha", "label": "Alpha", "created_at": "2026-02-15T09:00:01+00:00"}],
            "aliases": [],
            "edges": [
                {
                    "from": "alpha",
                    "relation": "rel",
                    "to": "missing",
                    "created_at": "2026-02-15T09:00:02+00:00",
                }
            ],
            "focus_node": None,
        }
        response = orch.handle_message("/graph_import " + self._payload_text(invalid_payload), "user1")
        self.assertEqual("Import failed.", response.text)
        self.assertNotIn("?", response.text)
        self.assertNotIn("In short:", response.text)
        self.assertNotIn("Next:", response.text)

        after = self._stable_graph_state("thread-a")
        self.assertEqual(before, after)

    def test_import_fails_when_node_cap_exceeded_and_graph_unchanged(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node base "Base"', "user1")
        orch.handle_message("/focus_node base", "user1")
        before = self._stable_graph_state("thread-a")

        nodes = [
            {
                "node_id": f"n_{idx:03d}",
                "label": f"Node {idx:03d}",
                "created_at": "2026-02-15T10:00:00+00:00",
            }
            for idx in range(201)
        ]
        payload = {
            "thread_id": "thread-a",
            "exported_at": "2026-02-15T10:00:00+00:00",
            "nodes": nodes,
            "aliases": [],
            "edges": [],
            "focus_node": None,
        }
        payload_text = self._payload_text(payload)
        for command in ("/graph_import ", "/graph_import --merge "):
            with self.subTest(command=command.strip()):
                response = orch.handle_message(command + payload_text, "user1")
                self.assertEqual("Import failed.", response.text)
                self.assertNotIn("?", response.text)
                after = self._stable_graph_state("thread-a")
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
