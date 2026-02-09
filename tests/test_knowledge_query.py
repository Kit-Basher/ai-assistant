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
        self.assertEqual(facts.get("snapshot_count"), 2)

    def test_storage_changes_no_snapshots_in_window(self) -> None:
        result = handler.knowledge_query(
            self.context, "changes over 2026-02-01 to 2026-02-07"
        )
        facts = result["data"]["facts"]
        self.assertFalse(facts.get("available"))
        self.assertEqual(facts.get("reason"), "no_snapshots")
        self.assertEqual(facts.get("snapshot_count"), 0)
        self.assertIn("not available", result["text"])

    def test_storage_changes_single_snapshot_fallback(self) -> None:
        self.db.insert_disk_snapshot(
            taken_at="2026-02-03T00:00:00",
            snapshot_local_date="2026-02-03",
            hostname="host",
            mountpoint="/",
            filesystem="ext4",
            total_bytes=1000,
            used_bytes=400,
            free_bytes=600,
        )
        result = handler.knowledge_query(
            self.context, "changes over 2026-02-01 to 2026-02-07"
        )
        facts = result["data"]["facts"]
        self.assertTrue(facts.get("available"))
        self.assertEqual(facts.get("snapshot_count"), 1)
        mount = facts["mounts"][0]
        self.assertIsNone(mount["delta_used"])
        self.assertEqual(mount["latest_used_bytes"], 400)

    def test_storage_changes_mount_mismatch(self) -> None:
        self.db.insert_disk_snapshot(
            taken_at="2026-02-01T00:00:00",
            snapshot_local_date="2026-02-01",
            hostname="host",
            mountpoint="/mnt/data",
            filesystem="ext4",
            total_bytes=2000,
            used_bytes=1000,
            free_bytes=1000,
        )
        self.db.insert_disk_snapshot(
            taken_at="2026-02-07T00:00:00",
            snapshot_local_date="2026-02-07",
            hostname="host",
            mountpoint="/mnt/data",
            filesystem="ext4",
            total_bytes=2000,
            used_bytes=1100,
            free_bytes=900,
        )
        result = handler.knowledge_query(
            self.context, "changes over 2026-02-01 to 2026-02-07"
        )
        facts = result["data"]["facts"]
        self.assertTrue(facts.get("available"))
        mount = next(item for item in facts["mounts"] if item["mountpoint"] == "/data")
        self.assertEqual(mount["delta_used"], 100)
        self.assertEqual(mount["mount_resolution"], "matched_by_basename")

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
        result = handler.knowledge_query(context, "any anomalies 2026-02-01 to 2026-02-07?")
        facts = result["data"]["facts"]
        self.assertTrue(facts.get("available"))
        self.assertEqual(len(facts.get("anomalies", [])), 1)


if __name__ == "__main__":
    unittest.main()
