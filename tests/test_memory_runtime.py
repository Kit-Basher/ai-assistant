from __future__ import annotations

import os
import tempfile
import unittest

from agent.memory_runtime import MemoryRuntime
from memory.db import MemoryDB


class TestMemoryRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.runtime = MemoryRuntime(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_pending_lifecycle_marks_expired(self) -> None:
        self.runtime.set_thread_state("u1", thread_id="thread-a")
        self.runtime.add_pending_item(
            "u1",
            {
                "pending_id": "p1",
                "kind": "followup",
                "origin_tool": "compare_now",
                "question": "Run compare?",
                "options": ["yes", "no"],
                "created_at": 1,
                "expires_at": 2,
                "thread_id": "thread-a",
                "status": "READY_TO_RESUME",
            },
        )
        changed = self.runtime.clear_expired_pending_items("u1", now_ts=10)
        self.assertEqual(1, changed)
        rows = self.runtime.list_pending_items("u1", thread_id="thread-a", include_expired=True, now_ts=10)
        self.assertEqual("EXPIRED", rows[0]["status"])

    def test_followup_is_ambiguous_when_multiple_pending_in_same_thread(self) -> None:
        self.runtime.set_thread_state("u1", thread_id="thread-a")
        for pending_id in ("p1", "p2"):
            self.runtime.add_pending_item(
                "u1",
                {
                    "pending_id": pending_id,
                    "kind": "clarification",
                    "origin_tool": "ask_query",
                    "question": "Pick one.",
                    "options": ["a", "b"],
                    "created_at": 1,
                    "expires_at": 9999999999,
                    "thread_id": "thread-a",
                    "status": "WAITING_FOR_USER",
                },
            )
        result = self.runtime.resolve_followup("u1", "yes", "thread-a")
        self.assertEqual("ambiguous", result["type"])
        self.assertEqual("multiple_pending", result["reason"])

    def test_followup_does_not_mix_threads(self) -> None:
        self.runtime.set_thread_state("u1", thread_id="thread-a")
        self.runtime.add_pending_item(
            "u1",
            {
                "pending_id": "p1",
                "kind": "followup",
                "origin_tool": "compare_now",
                "question": "A",
                "options": ["yes", "no"],
                "created_at": 1,
                "expires_at": 9999999999,
                "thread_id": "thread-a",
                "status": "READY_TO_RESUME",
            },
        )
        self.runtime.add_pending_item(
            "u1",
            {
                "pending_id": "p2",
                "kind": "followup",
                "origin_tool": "compare_now",
                "question": "B",
                "options": ["yes", "no"],
                "created_at": 2,
                "expires_at": 9999999999,
                "thread_id": "thread-b",
                "status": "READY_TO_RESUME",
            },
        )
        result = self.runtime.resolve_followup("u1", "yes", "thread-a")
        self.assertEqual("match", result["type"])
        pending_item = result.get("pending_item") if isinstance(result.get("pending_item"), dict) else {}
        self.assertEqual("p1", pending_item.get("pending_id"))

    def test_memory_summary_reports_resumable(self) -> None:
        self.runtime.set_thread_state("u1", thread_id="thread-a", current_topic="model setup")
        self.runtime.record_user_request("u1", "setup ollama")
        self.runtime.record_agent_action("u1", "Asked one question")
        self.runtime.add_pending_item(
            "u1",
            {
                "pending_id": "p1",
                "kind": "clarification",
                "origin_tool": "setup",
                "question": "Which size?",
                "options": ["small", "medium"],
                "created_at": 1,
                "expires_at": 9999999999,
                "thread_id": "thread-a",
                "status": "WAITING_FOR_USER",
            },
        )
        summary = self.runtime.get_memory_summary("u1", "thread-a")
        self.assertEqual("model setup", summary["current_topic"])
        self.assertEqual(1, summary["pending_count"])
        self.assertTrue(summary["resumable"])

    def test_record_agent_action_skips_meta_actions(self) -> None:
        wrote = self.runtime.record_agent_action("u1", "Memory summary text", action_kind="memory")
        self.assertFalse(wrote)
        summary = self.runtime.get_memory_summary("u1")
        self.assertIsNone(summary.get("last_agent_action"))

        wrote_meaningful = self.runtime.record_agent_action("u1", "Ran brief report", action_kind="brief")
        self.assertTrue(wrote_meaningful)
        summary_after = self.runtime.get_memory_summary("u1")
        self.assertEqual("Ran brief report", summary_after.get("last_agent_action"))

    def test_resolve_followup_reports_expired_when_only_expired_items_exist(self) -> None:
        self.runtime.set_thread_state("u1", thread_id="thread-a")
        self.runtime.add_pending_item(
            "u1",
            {
                "pending_id": "p-exp",
                "kind": "followup",
                "origin_tool": "compare_now",
                "question": "Run compare?",
                "options": ["yes", "no"],
                "created_at": 1,
                "expires_at": 2,
                "thread_id": "thread-a",
                "status": "READY_TO_RESUME",
            },
        )
        result = self.runtime.resolve_followup("u1", "yes", "thread-a")
        self.assertEqual("expired", result["type"])
        self.assertEqual("pending_expired", result["reason"])


if __name__ == "__main__":
    unittest.main()
