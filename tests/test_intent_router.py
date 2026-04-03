import os
import tempfile
import unittest

from agent.intent_router import route_message
from agent.knowledge_cache import KnowledgeQueryCache
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
        self.assertEqual(decision["type"], "greeting")

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

    def test_opinion_followup_requires_cache(self) -> None:
        decision = route_message("user1", "opinion", self.context)
        self.assertEqual(decision["type"], "noop")
        self.assertIn("need a report first", decision["text"].lower())

    def test_opinion_followup_uses_cache(self) -> None:
        cache = KnowledgeQueryCache()
        cache.set(
            "user1",
            "what changed this week",
            {"mounts": [{"mountpoint": "/", "delta_used": 10}]},
            {"name": "time_window_storage_changes"},
        )
        context = dict(self.context)
        context["knowledge_cache"] = cache
        decision = route_message("user1", "opinion", context)
        self.assertEqual(decision["type"], "skill_call")
        self.assertEqual(decision["function"], "opinion_on_report")
        self.assertIn("facts", decision["args"])

    def test_knowledge_query_routes(self) -> None:
        decision = route_message("user1", "what changed this week", self.context)
        self.assertEqual(decision["type"], "skill_call")
        self.assertEqual(decision["function"], "knowledge_query")
        self.assertEqual(decision["args"]["query"], "what changed this week")

    def test_knowledge_query_anomalies_routes(self) -> None:
        decision = route_message("user1", "any anomalies lately?", self.context)
        self.assertEqual(decision["type"], "skill_call")
        self.assertEqual(decision["function"], "knowledge_query")

    def test_knowledge_query_action_verbs_block(self) -> None:
        decision = route_message("user1", "clean up disk space", self.context)
        self.assertNotEqual(decision.get("function"), "knowledge_query")

    def test_knowledge_query_advice_block(self) -> None:
        decision = route_message("user1", "what should I do about disk usage", self.context)
        self.assertNotEqual(decision.get("function"), "knowledge_query")

    def test_knowledge_query_ambiguous_clarifies(self) -> None:
        decision = route_message("user1", "disk usage", self.context)
        self.assertEqual(decision["type"], "clarify")

    def test_storage_report_alias(self) -> None:
        decision = route_message("user1", "show me my last disk report", self.context)
        self.assertEqual(decision["type"], "skill_call")
        self.assertEqual(decision["skill"], "storage_governor")
        self.assertEqual(decision["function"], "storage_report")

    def test_brief_routes_and_avoids_ask_timeframe(self) -> None:
        phrases = [
            "anything new on my pc?",
            "anything different?",
            "what changed?",
            "what's new on my computer",
            "is my pc okay?",
        ]
        for text in phrases:
            decision = route_message("user1", text, self.context)
            self.assertEqual(decision.get("type"), "brief")

    def test_affirmation_after_brief_offer_routes_to_brief(self) -> None:
        ctx = dict(self.context)
        ctx["last_topic"] = "brief_offer"
        decision = route_message("user1", "yes please", ctx)
        self.assertEqual(decision["type"], "brief")


if __name__ == "__main__":
    unittest.main()
