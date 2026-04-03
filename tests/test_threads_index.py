from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class TestThreadsIndex(unittest.TestCase):
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

    def test_threads_lists_recent_order_with_focus(self) -> None:
        orch = self._orchestrator()
        self._insert_epistemic_turn("2099-01-02T00:00:00+00:00", "thread-b")
        self._insert_epistemic_turn("2099-01-01T00:00:00+00:00", "thread-c")
        self._insert_epistemic_turn("2099-01-01T00:00:00+00:00", "thread-a")
        self.db.add_thread_anchor(
            "thread-a",
            "Alpha focus",
            json.dumps(["note"], ensure_ascii=True, separators=(",", ":")),
            "",
        )
        response = orch.handle_message("/threads", "user1")
        lines = response.text.splitlines()
        self.assertEqual("Threads:", lines[0])
        self.assertEqual("1) thread-b  2099-01-02T00:00:00+00:00  Label: (none)  Focus: (none)", lines[1])
        self.assertEqual("2) thread-a  2099-01-01T00:00:00+00:00  Label: (none)  Focus: Alpha focus", lines[2])
        self.assertEqual("3) thread-c  2099-01-01T00:00:00+00:00  Label: (none)  Focus: (none)", lines[3])
        self.assertNotIn("?", response.text)

    def test_thread_use_switches_active_thread_for_resume(self) -> None:
        orch = self._orchestrator()
        self.db.add_thread_anchor(
            "thread-target",
            "Target focus",
            json.dumps(["first note", "second note"], ensure_ascii=True, separators=(",", ":")),
            "Open: ship release",
        )
        switch = orch.handle_message("/thread_use thread-target", "user1")
        self.assertEqual("Active thread set to thread-target.", switch.text)
        resume = orch.handle_message("/resume", "user1")
        self.assertTrue(resume.text.startswith("Resume (thread thread-target):"))
        self.assertIn("Focus: Target focus", resume.text)
        self.assertNotIn("?", resume.text)

    def test_thread_use_unknown_thread(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message("/thread_use missing-thread", "user1")
        self.assertEqual("Unknown thread: missing-thread.", response.text)
        self.assertNotIn("?", response.text)

    def test_thread_commands_are_not_friction_decorated(self) -> None:
        orch = self._orchestrator()
        self._insert_epistemic_turn("2099-01-01T00:00:00+00:00", "thread-a")
        self.db.add_thread_anchor(
            "thread-a",
            "Alpha focus",
            json.dumps(["note"], ensure_ascii=True, separators=(",", ":")),
            "",
        )
        threads_response = orch.handle_message("/threads", "user1")
        use_response = orch.handle_message("/thread_use thread-a", "user1")
        for response in (threads_response, use_response):
            self.assertNotIn("In short:", response.text)
            self.assertNotIn("Plan:", response.text)
            self.assertNotIn("Options:", response.text)
            self.assertNotIn("Next:", response.text)
            self.assertNotIn("?", response.text)


if __name__ == "__main__":
    unittest.main()
