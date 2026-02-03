import os
import tempfile
import unittest

from agent.intent_router import route_message
from memory.db import MemoryDB


class TestIntentRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.context = {
            "db": self.db,
            "chat_id": "chat-1",
            "timezone": "America/Regina",
            "now_ts": "2026-02-03T12:00:00+00:00",
        }

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_remember_note(self) -> None:
        decision = route_message("user1", "remember buy milk #groceries", self.context)
        self.assertEqual(decision["type"], "skill_call")
        self.assertEqual(decision["function"], "remember_note")
        self.assertEqual(decision["args"]["text"], "buy milk #groceries")

    def test_next_best_task(self) -> None:
        decision = route_message(
            "user1",
            "what should I do next? 30 minutes low energy",
            self.context,
        )
        self.assertEqual(decision["type"], "skill_call")
        self.assertEqual(decision["function"], "next_best_task")
        self.assertEqual(decision["args"]["minutes"], 30)
        self.assertEqual(decision["args"]["energy"], "low")

    def test_daily_plan(self) -> None:
        decision = route_message("user1", "plan my evening, 2 hours high", self.context)
        self.assertEqual(decision["type"], "skill_call")
        self.assertEqual(decision["function"], "daily_plan")
        self.assertEqual(decision["args"]["minutes"], 120)
        self.assertEqual(decision["args"]["energy"], "high")

    def test_set_reminder(self) -> None:
        decision = route_message(
            "user1",
            "remind me 2026-02-05 15:00 to call dentist",
            self.context,
        )
        self.assertEqual(decision["type"], "skill_call")
        self.assertEqual(decision["function"], "set_reminder")
        self.assertEqual(decision["args"]["when_ts"], "2026-02-05 15:00")
        self.assertEqual(decision["args"]["text"], "call dentist")

    def test_ambiguous_reminder(self) -> None:
        decision = route_message("user1", "remind me tomorrow", self.context)
        self.assertEqual(decision["type"], "clarify")
        self.assertEqual(
            decision["options"],
            ["2026-02-04 09:00", "2026-02-04 17:00", "2026-02-04 20:00"],
        )

    def test_no_intent(self) -> None:
        decision = route_message("user1", "hello there", self.context)
        self.assertEqual(decision["type"], "noop")

    def test_next_week_not_next_task(self) -> None:
        decision = route_message("user1", "next week", self.context)
        self.assertEqual(decision["type"], "noop")

    def test_clarification_flow(self) -> None:
        decision = route_message("user1", "what should I do next", self.context)
        self.assertEqual(decision["type"], "clarify")

        follow_up = route_message("user1", "30 low", self.context)
        self.assertEqual(follow_up["type"], "skill_call")
        self.assertEqual(follow_up["function"], "next_best_task")
        self.assertEqual(follow_up["args"]["minutes"], 30)
        self.assertEqual(follow_up["args"]["energy"], "low")

    def test_time_only_reminder_follow_up(self) -> None:
        decision = route_message("user1", "remind me to go to sasktel", self.context)
        self.assertEqual(decision["type"], "clarify")

        follow_up = route_message("user1", "2:00", self.context)
        self.assertEqual(follow_up["type"], "clarify")
        self.assertEqual(
            follow_up["options"],
            ["2026-02-04 02:00", "2026-02-04 14:00"],
        )

        confirmed = route_message("user1", "2026-02-04 14:00", self.context)
        self.assertEqual(confirmed["type"], "skill_call")
        self.assertEqual(confirmed["function"], "set_reminder")
        self.assertEqual(confirmed["args"]["when_ts"], "2026-02-04 14:00")
        self.assertEqual(confirmed["args"]["text"], "go to sasktel")

    def test_past_timestamp_reclarifies(self) -> None:
        decision = route_message(
            "user1",
            "remind me 2026-02-02 09:00 to test",
            self.context,
        )
        self.assertEqual(decision["type"], "clarify")
        self.assertIn("past", decision["question"].lower())


if __name__ == "__main__":
    unittest.main()
