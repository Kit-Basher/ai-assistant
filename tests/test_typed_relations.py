from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestTypedRelations(unittest.TestCase):
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

    def test_add_list_remove_relation_types_sorted(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")

        add_beta = orch.handle_message("/relation_add beta", "u1")
        add_alpha = orch.handle_message("/relation_add alpha", "u1")
        self.assertEqual("Relation type added: beta.", add_beta.text)
        self.assertEqual("Relation type added: alpha.", add_alpha.text)

        listed = orch.handle_message("/relations", "u1")
        self._assert_clean_output(listed.text)
        self.assertEqual(
            [
                "Relations (thread thread-a):",
                "Mode: open",
                "Types:",
                " - alpha",
                " - beta",
            ],
            listed.text.splitlines(),
        )

        removed = orch.handle_message("/relation_remove alpha", "u1")
        self.assertEqual("Relation type removed: alpha.", removed.text)
        listed_after = orch.handle_message("/relations", "u1")
        self.assertIn(" - beta", listed_after.text)
        self.assertNotIn(" - alpha", listed_after.text)

    def test_strict_mode_default_open(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        listed = orch.handle_message("/relations", "u1")
        self.assertIn("Mode: open", listed.text)
        self._assert_clean_output(listed.text)

    def test_strict_mode_on_rejects_undeclared_relation(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        mode = orch.handle_message("/relation_mode strict", "u1")
        self.assertEqual("Relation mode set to strict.", mode.text)
        link = orch.handle_message("/link a uses b", "u1")
        self.assertEqual("Cannot create link.", link.text)
        graph = orch.handle_message("/graph", "u1")
        self.assertIn("Edges:\n - (none)", graph.text)

    def test_strict_mode_on_allows_declared_relation(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        orch.handle_message("/relation_add uses", "u1")
        orch.handle_message("/relation_mode strict", "u1")
        link = orch.handle_message("/link a uses b", "u1")
        self.assertEqual("Link created.", link.text)
        graph = orch.handle_message("/graph", "u1")
        self.assertIn(" - a --uses--> b", graph.text)
        self.assertIn("Mode: strict", graph.text)
        self.assertIn("Declared relations: 1", graph.text)

    def test_strict_mode_off_allows_any_relation(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        orch.handle_message('/node a "A"', "u1")
        orch.handle_message('/node b "B"', "u1")
        orch.handle_message("/relation_mode strict", "u1")
        orch.handle_message("/relation_mode open", "u1")
        link = orch.handle_message("/link a free_form_relation b", "u1")
        self.assertEqual("Link created.", link.text)
        graph = orch.handle_message("/graph", "u1")
        self.assertIn(" - a --free_form_relation--> b", graph.text)
        self.assertIn("Mode: open", graph.text)

    def test_relations_output_none_when_empty(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")
        listed = orch.handle_message("/relations", "u1")
        self.assertEqual(
            [
                "Relations (thread thread-a):",
                "Mode: open",
                "Types:",
                " (none)",
            ],
            listed.text.splitlines(),
        )
        self._assert_clean_output(listed.text)

    def test_relation_normalization_and_limits(self) -> None:
        orch = self._orchestrator()
        self._set_active_thread(orch, "u1", "thread-a")

        resp = orch.handle_message("/relation_add  ReL   Type!?  ", "u1")
        self.assertEqual("Relation type added: rel_type.", resp.text)

        invalid = orch.handle_message("/relation_add !!!", "u1")
        self.assertEqual("Invalid relation.", invalid.text)

        long_raw = "a" * 60
        long_resp = orch.handle_message(f"/relation_add {long_raw}", "u1")
        expected = "a" * 40
        self.assertEqual(f"Relation type added: {expected}.", long_resp.text)

        dup = orch.handle_message("/relation_add rel_type", "u1")
        self.assertEqual("Relation type already exists.", dup.text)

        all_text = orch.handle_message("/relations", "u1").text
        self.assertIn(" - rel_type", all_text)
        self.assertIn(f" - {expected}", all_text)
        self._assert_clean_output(all_text)


if __name__ == "__main__":
    unittest.main()
