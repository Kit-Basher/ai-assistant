from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestGraphConstraints(unittest.TestCase):
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

    def _stable_graph_state(self, thread_id: str) -> dict[str, object]:
        exported = self.db.export_graph(thread_id)
        return {
            "nodes": exported.get("nodes", []),
            "aliases": exported.get("aliases", []),
            "edges": exported.get("edges", []),
            "focus_node": exported.get("focus_node"),
        }

    @staticmethod
    def _payload_text(payload: dict[str, object]) -> str:
        return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=False)

    def test_add_remove_list_constraints_deterministic(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        a = orch.handle_message("/relation_constraint_add zeta acyclic", "u1")
        b = orch.handle_message("/relation_constraint_add alpha acyclic", "u1")
        listed = orch.handle_message("/relation_constraints", "u1")
        removed = orch.handle_message("/relation_constraint_remove alpha acyclic", "u1")
        listed_after = orch.handle_message("/relation_constraints", "u1")

        self.assertEqual("Constraint added: zeta acyclic.", a.text)
        self.assertEqual("Constraint added: alpha acyclic.", b.text)
        self.assertEqual(
            [
                "Relation constraints (thread thread-a):",
                "- alpha acyclic",
                "- zeta acyclic",
            ],
            listed.text.splitlines(),
        )
        self.assertEqual("Constraint removed: alpha acyclic.", removed.text)
        self.assertEqual(
            [
                "Relation constraints (thread thread-a):",
                "- zeta acyclic",
            ],
            listed_after.text.splitlines(),
        )
        self._assert_clean_output(a.text)
        self._assert_clean_output(b.text)
        self._assert_clean_output(listed.text)
        self._assert_clean_output(removed.text)
        self._assert_clean_output(listed_after.text)

    def test_constraint_add_strict_mode_requires_declared_relation(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message("/relation_mode strict", "u1")
        response = orch.handle_message("/relation_constraint_add depends acyclic", "u1")
        self.assertEqual("Relation type not found.", response.text)
        self._assert_clean_output(response.text)

    def test_link_prevented_when_acyclic_cycle_created(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        orch.handle_message("/relation_constraint_add depends acyclic", "u1")
        first = orch.handle_message("/link a depends b", "u1")
        second = orch.handle_message("/link b depends a", "u1")
        self.assertEqual("Link created.", first.text)
        self.assertEqual("Cannot create link.", second.text)
        graph = orch.handle_message("/graph", "u1").text
        self.assertIn(" - a --depends--> b", graph)
        self.assertNotIn(" - b --depends--> a", graph)

    def test_link_allows_non_cycle_for_constrained_relation(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        orch.handle_message('/node c "C"', "u1")
        orch.handle_message("/relation_constraint_add depends acyclic", "u1")
        r1 = orch.handle_message("/link a depends b", "u1")
        r2 = orch.handle_message("/link b depends c", "u1")
        self.assertEqual("Link created.", r1.text)
        self.assertEqual("Link created.", r2.text)

    def test_constraint_applies_only_to_target_relation(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        orch.handle_message("/relation_constraint_add depends acyclic", "u1")
        orch.handle_message("/link a depends b", "u1")
        back = orch.handle_message("/link b relates a", "u1")
        self.assertEqual("Link created.", back.text)
        graph = orch.handle_message("/graph", "u1").text
        self.assertIn(" - a --depends--> b", graph)
        self.assertIn(" - b --relates--> a", graph)

    def test_import_replace_rejects_constrained_cycle_and_unchanged(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node keep "Keep"', "u1")
        orch.handle_message("/relation_constraint_add depends acyclic", "u1")
        before = self._stable_graph_state("thread-a")
        payload = {
            "thread_id": "thread-a",
            "exported_at": "2026-02-15T10:00:00+00:00",
            "nodes": [
                {"node_id": "a", "label": "A", "created_at": "2026-02-15T10:00:01+00:00"},
                {"node_id": "b", "label": "B", "created_at": "2026-02-15T10:00:02+00:00"},
            ],
            "aliases": [],
            "edges": [
                {
                    "from": "a",
                    "relation": "depends",
                    "to": "b",
                    "created_at": "2026-02-15T10:00:03+00:00",
                },
                {
                    "from": "b",
                    "relation": "depends",
                    "to": "a",
                    "created_at": "2026-02-15T10:00:04+00:00",
                },
            ],
            "focus_node": None,
        }
        response = orch.handle_message("/graph_import " + self._payload_text(payload), "u1")
        self.assertEqual("Import failed.", response.text)
        self._assert_clean_output(response.text)
        self.assertEqual(before, self._stable_graph_state("thread-a"))

    def test_import_merge_rejects_cycle_introduction_and_unchanged(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        orch.handle_message("/relation_constraint_add depends acyclic", "u1")
        orch.handle_message("/link a depends b", "u1")
        before = self._stable_graph_state("thread-a")

        payload = {
            "thread_id": "thread-a",
            "exported_at": "2026-02-15T11:00:00+00:00",
            "nodes": [],
            "aliases": [],
            "edges": [
                {
                    "from": "b",
                    "relation": "depends",
                    "to": "a",
                    "created_at": "2026-02-15T11:00:01+00:00",
                }
            ],
            "focus_node": None,
        }
        response = orch.handle_message("/graph_import --merge " + self._payload_text(payload), "u1")
        self.assertEqual("Import failed.", response.text)
        self._assert_clean_output(response.text)
        self.assertEqual(before, self._stable_graph_state("thread-a"))

    def test_constraint_none_output(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        response = orch.handle_message("/relation_constraints", "u1")
        self.assertEqual(
            [
                "Relation constraints (thread thread-a):",
                "(none)",
            ],
            response.text.splitlines(),
        )
        self._assert_clean_output(response.text)


if __name__ == "__main__":
    unittest.main()
