from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.memory_runtime import MemoryRuntime
from agent.working_memory import WorkingMemoryState, append_turn
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

    def test_followup_accepts_natural_confirmation_variants(self) -> None:
        self.runtime.set_thread_state("u1", thread_id="thread-a")
        self.runtime.add_pending_item(
            "u1",
            {
                "pending_id": "p1",
                "kind": "confirmation",
                "origin_tool": "printer_cups",
                "question": "Run diagnostics?",
                "options": ["yes", "no"],
                "created_at": 1,
                "expires_at": 9999999999,
                "thread_id": "thread-a",
                "status": "WAITING_FOR_USER",
            },
        )
        for text in ("yes please", "yes do it", "sure go ahead", "please do it"):
            result = self.runtime.resolve_followup("u1", text, "thread-a")
            self.assertEqual("match", result["type"], text)
            self.assertEqual("accept", result["intent"], text)

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

    def test_working_memory_state_inspects_and_resets_cleanly(self) -> None:
        state = WorkingMemoryState()
        append_turn(state, role="user", text="Please keep replies concise.")
        append_turn(state, role="assistant", text="I will keep them concise.")
        saved = self.runtime.save_working_memory_state("u1", state)
        self.assertTrue(saved)

        inspect = self.runtime.inspect_user_state("u1")
        working_memory = inspect.get("working_memory") if isinstance(inspect.get("working_memory"), dict) else {}
        self.assertTrue(bool(working_memory.get("healthy")))
        summary = working_memory.get("summary") if isinstance(working_memory.get("summary"), dict) else {}
        self.assertEqual(2, summary.get("hot_turn_count"))

        reset = self.runtime.reset_user_state("u1")
        self.assertIn("memory_runtime:u1:working_memory_state", reset.get("deleted_keys"))

    def test_corrupt_working_memory_state_is_visible_and_not_silently_loaded(self) -> None:
        self.db.set_user_pref("memory_runtime:u1:working_memory_state", "{bad json")

        inspect = self.runtime.inspect_user_state("u1")
        working_memory = inspect.get("working_memory") if isinstance(inspect.get("working_memory"), dict) else {}
        self.assertFalse(bool(working_memory.get("healthy")))
        self.assertEqual("corrupt_json", working_memory.get("status"))

        state, issue = self.runtime.load_working_memory_state("u1")
        self.assertEqual([], state.hot_turns)
        self.assertIsInstance(issue, dict)
        self.assertEqual("corrupt_json", issue.get("status"))

        new_state = WorkingMemoryState()
        append_turn(new_state, role="user", text="hello")
        saved = self.runtime.save_working_memory_state("u1", new_state, refuse_if_corrupt=True)
        self.assertFalse(saved)

    def test_working_memory_legacy_revision_zero_restores_and_moves_to_revisioned_writes(self) -> None:
        legacy_state = WorkingMemoryState()
        append_turn(legacy_state, role="user", text="legacy preference")
        payload = json.dumps(
            {
                "hot_turns": [
                    {
                        "turn_id": legacy_state.hot_turns[0].turn_id,
                        "role": "user",
                        "text": "legacy preference",
                        "token_count": legacy_state.hot_turns[0].token_count,
                        "pinned": False,
                        "created_at": legacy_state.hot_turns[0].created_at,
                        "metadata": {},
                    }
                ]
            },
            ensure_ascii=True,
        )
        self.db._conn.execute(  # noqa: SLF001
            "INSERT INTO user_prefs (key, value, updated_at, revision) VALUES (?, ?, ?, 0)",
            ("memory_runtime:u1:working_memory_state", payload, "2026-01-01T00:00:00+00:00"),
        )
        self.db._conn.commit()  # noqa: SLF001

        state, issue = self.runtime.load_working_memory_state("u1")
        self.assertIsNone(issue)
        self.assertEqual(1, len(state.hot_turns))

        append_turn(state, role="assistant", text="acknowledged")
        saved = self.runtime.save_working_memory_state("u1", state)
        self.assertTrue(saved)
        entry = self.db.get_user_pref_entry("memory_runtime:u1:working_memory_state")
        self.assertEqual(1, int((entry or {}).get("revision") or -1))

    def test_working_memory_save_increments_revision(self) -> None:
        state = WorkingMemoryState()
        append_turn(state, role="user", text="first")

        self.assertTrue(self.runtime.save_working_memory_state("u1", state))
        entry_one = self.db.get_user_pref_entry("memory_runtime:u1:working_memory_state")
        self.assertEqual(1, int((entry_one or {}).get("revision") or -1))

        reloaded, issue = self.runtime.load_working_memory_state("u1")
        self.assertIsNone(issue)
        append_turn(reloaded, role="assistant", text="second")
        self.assertTrue(self.runtime.save_working_memory_state("u1", reloaded))
        entry_two = self.db.get_user_pref_entry("memory_runtime:u1:working_memory_state")
        self.assertEqual(2, int((entry_two or {}).get("revision") or -1))

    def test_managed_continuity_records_carry_revision_metadata_after_save(self) -> None:
        self.runtime.set_thread_state("u1", thread_id="thread-a", current_topic="revisions")
        self.runtime.add_pending_item(
            "u1",
            {
                "pending_id": "p1",
                "kind": "followup",
                "origin_tool": "compare_now",
                "question": "continue?",
                "options": ["yes", "no"],
                "created_at": 1,
                "expires_at": 9999999999,
                "thread_id": "thread-a",
                "status": "WAITING_FOR_USER",
            },
        )
        self.runtime.record_user_request("u1", "remember revisions")
        self.runtime.record_agent_action("u1", "tracked revision")
        state = WorkingMemoryState()
        append_turn(state, role="user", text="revision tracked working memory")
        self.assertTrue(self.runtime.save_working_memory_state("u1", state))

        entries = {
            row["key"]: row
            for row in self.db.list_user_prefs()
            if str(row.get("key") or "").startswith("memory_runtime:u1:")
        }
        for key in (
            "memory_runtime:u1:thread_state",
            "memory_runtime:u1:pending_items",
            "memory_runtime:u1:last_meaningful_user_request",
            "memory_runtime:u1:last_agent_action",
            "memory_runtime:u1:working_memory_state",
            "memory_runtime:u1:persistence_status",
        ):
            self.assertIn(key, entries)
            self.assertGreaterEqual(int((entries[key] or {}).get("revision") or 0), 1)

    def test_stale_working_memory_save_is_rejected_and_conflict_is_observable(self) -> None:
        runtime_a = MemoryRuntime(self.db)
        runtime_b = MemoryRuntime(self.db)

        state_a, issue_a = runtime_a.load_working_memory_state("u1")
        state_b, issue_b = runtime_b.load_working_memory_state("u1")
        self.assertIsNone(issue_a)
        self.assertIsNone(issue_b)

        append_turn(state_a, role="user", text="runtime a wins")
        self.assertTrue(runtime_a.save_working_memory_state("u1", state_a))
        entry_after_a = self.db.get_user_pref_entry("memory_runtime:u1:working_memory_state")
        self.assertEqual(1, int((entry_after_a or {}).get("revision") or -1))

        append_turn(state_b, role="user", text="runtime b stale write")
        self.assertFalse(runtime_b.save_working_memory_state("u1", state_b))
        entry_after_b = self.db.get_user_pref_entry("memory_runtime:u1:working_memory_state")
        self.assertEqual(1, int((entry_after_b or {}).get("revision") or -1))

        final_state, final_issue = self.runtime.load_working_memory_state("u1")
        self.assertIsNone(final_issue)
        self.assertEqual(1, len(final_state.hot_turns))
        self.assertEqual("runtime a wins", final_state.hot_turns[0].text)

        inspect = runtime_b.inspect_user_state("u1")
        persistence = inspect.get("persistence") if isinstance(inspect.get("persistence"), dict) else {}
        last_attempted_write = (
            persistence.get("last_attempted_write")
            if isinstance(persistence.get("last_attempted_write"), dict)
            else {}
        )
        last_successful_write = (
            persistence.get("last_successful_write")
            if isinstance(persistence.get("last_successful_write"), dict)
            else {}
        )
        last_conflict = persistence.get("last_conflict") if isinstance(persistence.get("last_conflict"), dict) else {}
        self.assertEqual("revision_conflict", last_attempted_write.get("status"))
        self.assertEqual("ok", last_successful_write.get("status"))
        self.assertEqual("stale_write_conflict", last_conflict.get("reason"))
        self.assertTrue(bool(persistence.get("active_conflict")))
        self.assertEqual(1, ((persistence.get("current_revisions") or {}).get("working_memory_state") or 0))

    def test_success_after_conflict_supersedes_active_conflict_without_erasing_history(self) -> None:
        runtime_a = MemoryRuntime(self.db)
        runtime_b = MemoryRuntime(self.db)
        runtime_c = MemoryRuntime(self.db)

        state_a, _ = runtime_a.load_working_memory_state("u1")
        state_b, _ = runtime_b.load_working_memory_state("u1")
        append_turn(state_a, role="user", text="runtime a wins first")
        self.assertTrue(runtime_a.save_working_memory_state("u1", state_a))
        append_turn(state_b, role="user", text="runtime b stale write")
        self.assertFalse(runtime_b.save_working_memory_state("u1", state_b))

        state_c, issue_c = runtime_c.load_working_memory_state("u1")
        self.assertIsNone(issue_c)
        append_turn(state_c, role="assistant", text="runtime c recovered write")
        self.assertTrue(runtime_c.save_working_memory_state("u1", state_c))

        inspect = runtime_c.inspect_user_state("u1")
        persistence = inspect.get("persistence") if isinstance(inspect.get("persistence"), dict) else {}
        last_attempted_write = (
            persistence.get("last_attempted_write")
            if isinstance(persistence.get("last_attempted_write"), dict)
            else {}
        )
        last_successful_write = (
            persistence.get("last_successful_write")
            if isinstance(persistence.get("last_successful_write"), dict)
            else {}
        )
        last_conflict = persistence.get("last_conflict") if isinstance(persistence.get("last_conflict"), dict) else {}
        self.assertEqual("ok", last_attempted_write.get("status"))
        self.assertEqual("ok", last_successful_write.get("status"))
        self.assertEqual("stale_write_conflict", last_conflict.get("reason"))
        self.assertFalse(bool(persistence.get("active_conflict")))
        self.assertLess(
            int(last_conflict.get("stored_revision") or 0),
            int(last_successful_write.get("stored_revision") or 0),
        )

    def test_managed_working_memory_save_does_not_fall_back_to_blind_set_user_pref(self) -> None:
        state = WorkingMemoryState()
        append_turn(state, role="user", text="guard against blind overwrite")

        with patch.object(self.db, "set_user_pref", side_effect=AssertionError("blind set_user_pref should not be used")):
            self.assertTrue(self.runtime.save_working_memory_state("u1", state))


if __name__ == "__main__":
    unittest.main()
