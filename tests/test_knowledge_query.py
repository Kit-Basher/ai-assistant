import os
import tempfile
import unittest

from memory.db import MemoryDB
from skills.knowledge_query import handler


class TestKnowledgeQuery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.context = {"db": self.db, "timezone": "UTC"}

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_parse_intent_latest(self) -> None:
        parsed = handler.parse_intent("latest disk report", "UTC")
        self.assertEqual(parsed.intent, "latest_snapshot_summary")

    def test_parse_intent_time_window(self) -> None:
        parsed = handler.parse_intent("changes over 2026-02-01 to 2026-02-07", "UTC")
        self.assertEqual(parsed.intent, "time_window_storage_changes")
        self.assertEqual(parsed.start_date, "2026-02-01")
        self.assertEqual(parsed.end_date, "2026-02-07")

    def test_parse_intent_top_growth(self) -> None:
        parsed = handler.parse_intent("largest directory growth in /home", "UTC")
        self.assertEqual(parsed.intent, "top_growth_paths")
        self.assertEqual(parsed.path_filter, "/home")

    def test_parse_intent_anomalies(self) -> None:
        parsed = handler.parse_intent("any anomalies lately?", "UTC")
        self.assertEqual(parsed.intent, "anomalies_in_window")

    def test_unsupported_query_returns_clarification(self) -> None:
        result = handler.knowledge_query(self.context, "hello there")
        self.assertIn("Examples:", result["text"])
        self.assertIn("clarification_required", result["data"]["limits"]["notes"])

    def test_read_only_handler_no_db_writes(self) -> None:
        before = self.db._conn.total_changes
        result = handler.knowledge_query(self.context, "latest disk report")
        after = self.db._conn.total_changes
        self.assertEqual(before, after)
        self.assertIn("Knowledge query", result["text"])

    def test_storage_changes_fact_delta(self) -> None:
        self.db.insert_disk_snapshot(
            taken_at="2026-02-01T00:00:00",
            snapshot_local_date="2026-02-01",
            hostname="host",
            mountpoint="/",
            filesystem="ext4",
            total_bytes=1000,
            used_bytes=400,
            free_bytes=600,
        )
        self.db.insert_disk_snapshot(
            taken_at="2026-02-07T00:00:00",
            snapshot_local_date="2026-02-07",
            hostname="host",
            mountpoint="/",
            filesystem="ext4",
            total_bytes=1000,
            used_bytes=500,
            free_bytes=500,
        )
        result = handler.knowledge_query(
            self.context, "changes over 2026-02-01 to 2026-02-07"
        )
        facts = result["data"]["facts"]
        mount = facts["mounts"][0]
        self.assertEqual(mount["delta_used"], 100)

    def test_top_growth_unavailable_without_samples(self) -> None:
        result = handler.knowledge_query(self.context, "largest directory growth in /home")
        facts = result["data"]["facts"]
        self.assertFalse(facts.get("available"))
        self.assertEqual(facts.get("reason"), "dir_growth_not_stored")

    def test_anomalies_query(self) -> None:
        self.db.insert_anomaly_events(
            "user1",
            "2026-02-03T10:00:00Z",
            [
                {
                    "source": "disk_anomalies",
                    "anomaly_key": "root_usage_high",
                    "severity": "warn",
                    "message": "Root usage high",
                    "context": {"mount": "/"},
                }
            ],
        )
        context = {"db": self.db, "timezone": "UTC", "user_id": "user1"}
        result = handler.knowledge_query(context, "any anomalies this week?")
        facts = result["data"]["facts"]
        self.assertTrue(facts.get("available"))
        self.assertEqual(len(facts.get("anomalies", [])), 1)


if __name__ == "__main__":
    unittest.main()
