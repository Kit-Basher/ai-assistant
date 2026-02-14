from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class TestThreadNew(unittest.TestCase):
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

    @staticmethod
    def _extract_thread_id(text: str) -> str:
        for line in text.splitlines():
            if line.startswith("Thread: "):
                return line[len("Thread: "):].strip()
        return ""

    def test_thread_new_sets_active_thread_and_label(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message('/thread_new "Launch Plan"', "user1")
        thread_id = self._extract_thread_id(response.text)
        self.assertTrue(thread_id.startswith("user:user1:t"))
        self.assertEqual("Launch Plan", self.db.get_thread_label(thread_id))
        resume = orch.handle_message("/resume", "user1")
        self.assertTrue(resume.text.startswith(f"Resume (thread {thread_id}):"))
        self.assertIn("No checkpoints yet.", resume.text)
        self.assertNotIn("?", response.text)

    def test_thread_new_flags_set_thread_prefs(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message(
            '/thread_new "Flagged" --terse on --summary off --next off --codeblock on',
            "user1",
        )
        thread_id = self._extract_thread_id(response.text)
        self.assertEqual("on", self.db.get_thread_pref(thread_id, "terse_mode"))
        self.assertEqual("off", self.db.get_thread_pref(thread_id, "show_summary"))
        self.assertEqual("off", self.db.get_thread_pref(thread_id, "show_next_action"))
        self.assertEqual("on", self.db.get_thread_pref(thread_id, "commands_in_codeblock"))
        self.assertIn("Prefs: terse=on summary=off next=off codeblock=on", response.text)
        self.assertNotIn("?", response.text)

    def test_thread_new_anchor_created_when_body_present(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message(
            '/thread_new "Anchorized"\n- bullet one\n- bullet two\nOpen: next step?',
            "user1",
        )
        thread_id = self._extract_thread_id(response.text)
        rows = self.db.list_thread_anchors(thread_id, limit=5)
        self.assertEqual(1, len(rows))
        bullets = json.loads(rows[0]["bullets"])
        self.assertEqual(["bullet one", "bullet two"], bullets)
        self.assertEqual("Open: next step", rows[0]["open_line"])
        self.assertIn("Anchor initialized.", response.text)
        self.assertNotIn("?", response.text)

    def test_thread_new_no_anchor_when_no_body(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message('/thread_new "No Anchor"', "user1")
        thread_id = self._extract_thread_id(response.text)
        rows = self.db.list_thread_anchors(thread_id, limit=5)
        self.assertEqual([], rows)
        self.assertNotIn("Anchor initialized.", response.text)
        self.assertNotIn("?", response.text)

    def test_thread_new_reply_not_friction_decorated(self) -> None:
        orch = self._orchestrator()
        response = orch.handle_message('/thread_new "Decorations Off" --summary off', "user1")
        self.assertNotIn("In short:", response.text)
        self.assertNotIn("Plan:", response.text)
        self.assertNotIn("Options:", response.text)
        self.assertNotIn("Next:", response.text)
        self.assertNotIn("?", response.text)

    def test_existing_threads_unaffected(self) -> None:
        orch = self._orchestrator()
        first = orch.handle_message('/thread_new "First" --summary on', "user1")
        first_thread = self._extract_thread_id(first.text)
        second = orch.handle_message('/thread_new "Second" --summary off', "user1")
        second_thread = self._extract_thread_id(second.text)
        self.assertNotEqual(first_thread, second_thread)
        self.assertEqual("on", self.db.get_thread_pref(first_thread, "show_summary"))
        self.assertEqual("off", self.db.get_thread_pref(second_thread, "show_summary"))
        self.assertEqual("First", self.db.get_thread_label(first_thread))
        self.assertEqual("Second", self.db.get_thread_label(second_thread))
        self.assertNotIn("?", first.text)
        self.assertNotIn("?", second.text)


if __name__ == "__main__":
    unittest.main()
