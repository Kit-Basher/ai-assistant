from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class TestThreadLabels(unittest.TestCase):
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

    def _insert_epistemic_turn(self, ts: str, thread_id: str) -> None:
        payload = {
            "user_id": "user1",
            "thread_id": thread_id,
            "turn_id": f"{thread_id}:u:1",
            "role": "user",
            "text": "hello",
        }
        self.db._conn.execute(
            "INSERT INTO activity_log (ts, type, payload_json) VALUES (?, 'epistemic_turn', ?)",
            (ts, json.dumps(payload, ensure_ascii=True, separators=(",", ":"))),
        )
        self.db._conn.commit()

    def test_thread_label_sets_and_threads_shows_label(self) -> None:
        orch = self._orchestrator()
        self._insert_epistemic_turn("2099-01-02T00:00:00+00:00", "thread-a")
        orch.handle_message("/thread_use thread-a", "user1")
        label_resp = orch.handle_message("/thread_label My Sprint Focus", "user1")
        self.assertEqual("Label set for thread-a.", label_resp.text)
        threads = orch.handle_message("/threads", "user1")
        self.assertIn("1) thread-a  2099-01-02T00:00:00+00:00  Label: My Sprint Focus  Focus: (none)", threads.text)
        self.assertNotIn("?", threads.text)

    def test_thread_unlabel_clears_label(self) -> None:
        orch = self._orchestrator()
        self._insert_epistemic_turn("2099-01-02T00:00:00+00:00", "thread-a")
        orch.handle_message("/thread_use thread-a", "user1")
        orch.handle_message("/thread_label Focus Alpha", "user1")
        clear_resp = orch.handle_message("/thread_unlabel", "user1")
        self.assertEqual("Label cleared for thread-a.", clear_resp.text)
        threads = orch.handle_message("/threads", "user1")
        self.assertIn("Label: (none)", threads.text)
        self.assertNotIn("?", threads.text)

    def test_label_normalization_trim_collapse_strip_and_truncate(self) -> None:
        orch = self._orchestrator()
        self._insert_epistemic_turn("2099-01-02T00:00:00+00:00", "thread-a")
        orch.handle_message("/thread_use thread-a", "user1")
        raw = "   Big   Launch   Plan ???  with    many     spaces and extra suffix 1234567890abcdef   "
        orch.handle_message(f"/thread_label {raw}", "user1")
        stored = self.db.get_thread_label("thread-a")
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertNotIn("?", stored)
        self.assertLessEqual(len(stored), 60)
        self.assertEqual(stored, "Big Launch Plan with many spaces and extra suffix 1234567890")

    def test_threads_ordering_preserved_with_label_field(self) -> None:
        orch = self._orchestrator()
        self._insert_epistemic_turn("2099-01-02T00:00:00+00:00", "thread-b")
        self._insert_epistemic_turn("2099-01-01T00:00:00+00:00", "thread-c")
        self._insert_epistemic_turn("2099-01-01T00:00:00+00:00", "thread-a")
        self.db.set_thread_label("thread-a", "Alpha")
        self.db.set_thread_label("thread-b", "Beta")
        threads = orch.handle_message("/threads", "user1")
        lines = threads.text.splitlines()
        self.assertEqual("Threads:", lines[0])
        self.assertEqual("1) thread-b  2099-01-02T00:00:00+00:00  Label: Beta  Focus: (none)", lines[1])
        self.assertEqual("2) thread-a  2099-01-01T00:00:00+00:00  Label: Alpha  Focus: (none)", lines[2])
        self.assertEqual("3) thread-c  2099-01-01T00:00:00+00:00  Label: (none)  Focus: (none)", lines[3])
        self.assertNotIn("?", threads.text)

    def test_thread_label_replies_not_decorated(self) -> None:
        orch = self._orchestrator()
        self._insert_epistemic_turn("2099-01-02T00:00:00+00:00", "thread-a")
        orch.handle_message("/thread_use thread-a", "user1")
        set_resp = orch.handle_message("/thread_label Alpha", "user1")
        clear_resp = orch.handle_message("/thread_unlabel", "user1")
        for response in (set_resp, clear_resp):
            self.assertNotIn("In short:", response.text)
            self.assertNotIn("Plan:", response.text)
            self.assertNotIn("Options:", response.text)
            self.assertNotIn("Next:", response.text)
            self.assertNotIn("?", response.text)


if __name__ == "__main__":
    unittest.main()
