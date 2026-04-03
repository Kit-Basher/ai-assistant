import os
import tempfile
import unittest

from memory.db import MemoryDB


class TestAnomalyEvents(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_schema_has_anomaly_events(self) -> None:
        cur = self.db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='anomaly_events'"
        )
        row = cur.fetchone()
        self.assertIsNotNone(row)

    def test_insert_dedupe(self) -> None:
        events = [
            {
                "source": "disk_anomalies",
                "anomaly_key": "root_usage_high",
                "severity": "warn",
                "message": "Root usage high",
                "context": {"mount": "/"},
            }
        ]
        self.db.insert_anomaly_events("user1", "2026-02-01T10:00:00Z", events)
        self.db.insert_anomaly_events("user1", "2026-02-01T10:00:00Z", events)
        cur = self.db._conn.execute("SELECT COUNT(*) AS cnt FROM anomaly_events")
        count = cur.fetchone()["cnt"]
        self.assertEqual(count, 1)

    def test_window_query(self) -> None:
        events = [
            {
                "source": "disk_anomalies",
                "anomaly_key": "root_usage_high",
                "severity": "warn",
                "message": "Root usage high",
                "context": {"mount": "/"},
            }
        ]
        self.db.insert_anomaly_events("user1", "2026-02-01T10:00:00Z", events)
        self.db.insert_anomaly_events("user1", "2026-02-10T10:00:00Z", events)
        rows = self.db.get_anomalies("user1", "2026-02-01", "2026-02-05")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["anomaly_key"], "root_usage_high")


if __name__ == "__main__":
    unittest.main()
