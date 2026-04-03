from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestGraphPacks(unittest.TestCase):
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

    def test_graph_pack_export_default_active_thread_only(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node alpha "Alpha"', "user1")
        orch.handle_message("/focus_node alpha", "user1")

        response = orch.handle_message("/graph_pack_export", "user1")
        self.assertNotIn("?", response.text)
        self.assertNotIn("In short:", response.text)
        self.assertNotIn("Next:", response.text)

        root_pairs = json.loads(response.text, object_pairs_hook=list)
        self.assertEqual(["pack_version", "exported_at", "threads"], [key for key, _ in root_pairs])
        root = dict(root_pairs)
        self.assertEqual(1, root["pack_version"])

        threads = root["threads"]
        self.assertEqual(1, len(threads))
        entry_pairs = threads[0]
        self.assertEqual(["thread_id", "graph"], [key for key, _ in entry_pairs])
        self.assertEqual("thread-a", dict(entry_pairs)["thread_id"])
        graph_pairs = dict(entry_pairs)["graph"]
        self.assertEqual(
            ["exported_at", "nodes", "aliases", "edges", "focus_node"],
            [key for key, _ in graph_pairs],
        )

    def test_graph_pack_export_threads_sorted(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-b")
        orch.handle_message('/node bnode "B"', "user1")
        self._set_active_thread(orch, "user1", "thread-a")
        orch.handle_message('/node anode "A"', "user1")

        response = orch.handle_message("/graph_pack_export --threads thread-b,thread-a", "user1")
        self.assertNotIn("?", response.text)
        root = json.loads(response.text, object_pairs_hook=list)
        threads = dict(root)["threads"]
        thread_ids = [dict(entry)["thread_id"] for entry in threads]
        self.assertEqual(["thread-a", "thread-b"], thread_ids)

    def test_graph_pack_import_replace_multiple_threads(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-home")

        payload = {
            "pack_version": 1,
            "exported_at": "2026-02-15T11:00:00+00:00",
            "threads": [
                {
                    "thread_id": "thread-a",
                    "graph": {
                        "exported_at": "2026-02-15T11:00:01+00:00",
                        "nodes": [
                            {"node_id": "a1", "label": "A1", "created_at": "2026-02-15T11:00:02+00:00"},
                            {"node_id": "a2", "label": "A2", "created_at": "2026-02-15T11:00:03+00:00"},
                        ],
                        "aliases": [
                            {"alias": "a", "node_id": "a1", "created_at": "2026-02-15T11:00:04+00:00"},
                        ],
                        "edges": [
                            {
                                "from": "a1",
                                "relation": "uses",
                                "to": "a2",
                                "created_at": "2026-02-15T11:00:05+00:00",
                            }
                        ],
                        "focus_node": {"node_id": "a1", "updated_at": "2026-02-15T11:00:06+00:00"},
                    },
                },
                {
                    "thread_id": "thread-b",
                    "graph": {
                        "exported_at": "2026-02-15T11:00:07+00:00",
                        "nodes": [
                            {"node_id": "b1", "label": "B1", "created_at": "2026-02-15T11:00:08+00:00"},
                        ],
                        "aliases": [],
                        "edges": [],
                        "focus_node": None,
                    },
                },
            ],
        }
        response = orch.handle_message("/graph_pack_import " + self._payload_text(payload), "user1")
        self.assertEqual("Pack imported.", response.text)
        self.assertNotIn("?", response.text)
        nodes_a = [row["node_id"] for row in self.db.list_graph_nodes("thread-a")]
        nodes_b = [row["node_id"] for row in self.db.list_graph_nodes("thread-b")]
        self.assertEqual(["a1", "a2"], nodes_a)
        self.assertEqual(["b1"], nodes_b)
        self.assertEqual("a1", self.db.get_thread_focus_node("thread-a"))

    def test_graph_pack_import_merge_conflict_rules(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-home")
        self.db.create_graph_node("thread-a", "core", "Core Existing")
        self.db.add_graph_alias("thread-a", "c", "core")
        self.db.set_thread_focus_node("thread-a", "core")

        payload = {
            "pack_version": 1,
            "exported_at": "2026-02-15T12:00:00+00:00",
            "threads": [
                {
                    "thread_id": "thread-a",
                    "graph": {
                        "exported_at": "2026-02-15T12:00:01+00:00",
                        "nodes": [
                            {"node_id": "core", "label": "Core Incoming", "created_at": "2026-02-15T12:00:02+00:00"},
                            {"node_id": "addon", "label": "Addon", "created_at": "2026-02-15T12:00:03+00:00"},
                        ],
                        "aliases": [
                            {"alias": "c", "node_id": "addon", "created_at": "2026-02-15T12:00:04+00:00"},
                            {"alias": "a", "node_id": "addon", "created_at": "2026-02-15T12:00:05+00:00"},
                        ],
                        "edges": [
                            {
                                "from": "core",
                                "relation": "depends",
                                "to": "addon",
                                "created_at": "2026-02-15T12:00:06+00:00",
                            }
                        ],
                        "focus_node": {"node_id": "addon", "updated_at": "2026-02-15T12:00:07+00:00"},
                    },
                }
            ],
        }
        response = orch.handle_message("/graph_pack_import --merge " + self._payload_text(payload), "user1")
        self.assertEqual("Pack merged.", response.text)
        self.assertNotIn("?", response.text)
        core = self.db.get_graph_node("thread-a", "core")
        self.assertIsNotNone(core)
        self.assertEqual("Core Existing", str((core or {}).get("label")))
        self.assertEqual("core", self.db.resolve_graph_ref("thread-a", "c"))
        self.assertEqual("addon", self.db.resolve_graph_ref("thread-a", "a"))
        edges = self.db.list_graph_edges("thread-a")
        self.assertIn(("core", "depends", "addon"), [(e["from_node"], e["relation"], e["to_node"]) for e in edges])
        self.assertEqual("core", self.db.get_thread_focus_node("thread-a"))

    def test_graph_pack_import_caps_enforced_without_partial_writes(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "user1", "thread-home")
        self.db.create_graph_node("thread-a", "keep_a", "Keep A")
        self.db.create_graph_node("thread-b", "keep_b", "Keep B")
        before_a = self._stable_graph_state("thread-a")
        before_b = self._stable_graph_state("thread-b")

        oversized_nodes = [
            {
                "node_id": f"n_{idx:03d}",
                "label": f"Node {idx:03d}",
                "created_at": "2026-02-15T13:00:00+00:00",
            }
            for idx in range(201)
        ]
        payload = {
            "pack_version": 1,
            "exported_at": "2026-02-15T13:00:00+00:00",
            "threads": [
                {
                    "thread_id": "thread-a",
                    "graph": {
                        "exported_at": "2026-02-15T13:00:01+00:00",
                        "nodes": [{"node_id": "new_a", "label": "New A", "created_at": "2026-02-15T13:00:02+00:00"}],
                        "aliases": [],
                        "edges": [],
                        "focus_node": None,
                    },
                },
                {
                    "thread_id": "thread-b",
                    "graph": {
                        "exported_at": "2026-02-15T13:00:03+00:00",
                        "nodes": oversized_nodes,
                        "aliases": [],
                        "edges": [],
                        "focus_node": None,
                    },
                },
            ],
        }
        payload_text = self._payload_text(payload)
        for command in ("/graph_pack_import ", "/graph_pack_import --merge "):
            with self.subTest(command=command.strip()):
                response = orch.handle_message(command + payload_text, "user1")
                self.assertEqual("Import failed.", response.text)
                self.assertNotIn("?", response.text)
                self.assertEqual(before_a, self._stable_graph_state("thread-a"))
                self.assertEqual(before_b, self._stable_graph_state("thread-b"))

    def test_graph_clone_replace_and_merge(self) -> None:
        orch = self._orchestrator()

        self.db.create_graph_node("thread-source", "alpha", "Source Alpha")
        self.db.create_graph_node("thread-source", "beta", "Source Beta")
        self.db.create_graph_edge("thread-source", "alpha", "beta", "uses")
        self.db.add_graph_alias("thread-source", "a", "alpha")
        self.db.set_thread_focus_node("thread-source", "alpha")

        self._set_active_thread(orch, "user1", "thread-target")
        self.db.create_graph_node("thread-target", "old", "Old")

        replace_resp = orch.handle_message("/graph_clone thread-source", "user1")
        self.assertEqual("Graph cloned.", replace_resp.text)
        self.assertNotIn("?", replace_resp.text)
        target_nodes = [row["node_id"] for row in self.db.list_graph_nodes("thread-target")]
        self.assertEqual(["alpha", "beta"], target_nodes)
        self.assertEqual("alpha", self.db.get_thread_focus_node("thread-target"))

        self.db.set_graph_node_label("thread-target", "alpha", "Target Alpha")
        merge_payload = {
            "pack_version": 1,
            "exported_at": "2026-02-15T14:00:00+00:00",
            "threads": [
                {
                    "thread_id": "thread-source",
                    "graph": {
                        "exported_at": "2026-02-15T14:00:01+00:00",
                        "nodes": [
                            {"node_id": "alpha", "label": "Source Alpha New", "created_at": "2026-02-15T14:00:02+00:00"},
                            {"node_id": "gamma", "label": "Gamma", "created_at": "2026-02-15T14:00:03+00:00"},
                        ],
                        "aliases": [
                            {"alias": "a", "node_id": "gamma", "created_at": "2026-02-15T14:00:04+00:00"},
                            {"alias": "g", "node_id": "gamma", "created_at": "2026-02-15T14:00:05+00:00"},
                        ],
                        "edges": [
                            {
                                "from": "alpha",
                                "relation": "depends",
                                "to": "gamma",
                                "created_at": "2026-02-15T14:00:06+00:00",
                            }
                        ],
                        "focus_node": {"node_id": "gamma", "updated_at": "2026-02-15T14:00:07+00:00"},
                    },
                }
            ],
        }
        import_resp = orch.handle_message("/graph_pack_import " + self._payload_text(merge_payload), "user1")
        self.assertEqual("Pack imported.", import_resp.text)

        merge_resp = orch.handle_message("/graph_clone thread-source --merge", "user1")
        self.assertEqual("Graph merged from thread-source.", merge_resp.text)
        self.assertNotIn("?", merge_resp.text)
        alpha = self.db.get_graph_node("thread-target", "alpha")
        self.assertEqual("Target Alpha", str((alpha or {}).get("label")))
        self.assertEqual("alpha", self.db.resolve_graph_ref("thread-target", "a"))
        self.assertEqual("gamma", self.db.resolve_graph_ref("thread-target", "g"))


if __name__ == "__main__":
    unittest.main()
