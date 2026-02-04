import os
import tempfile
import unittest
from unittest import mock

from memory.db import MemoryDB
from skills.resource_governor import handler


class TestResourceGovernor(unittest.TestCase):
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

    def test_same_day_uniqueness(self) -> None:
        day = "2026-02-04"
        self.db.insert_resource_snapshot(
            taken_at=f"{day}T09:00:00-06:00",
            snapshot_local_date=day,
            hostname="host-a",
            load_1m=0.5,
            load_5m=0.6,
            load_15m=0.7,
            mem_total=100,
            mem_used=40,
            mem_free=60,
            swap_total=10,
            swap_used=2,
        )
        self.db.insert_resource_snapshot(
            taken_at=f"{day}T09:05:00-06:00",
            snapshot_local_date=day,
            hostname="host-a",
            load_1m=1.5,
            load_5m=1.6,
            load_15m=1.7,
            mem_total=100,
            mem_used=50,
            mem_free=50,
            swap_total=10,
            swap_used=3,
        )

        cur = self.db._conn.execute(
            "SELECT COUNT(*) AS count FROM resource_snapshots WHERE snapshot_local_date = ? AND hostname = ?",
            (day, "host-a"),
        )
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 1)

        cur = self.db._conn.execute(
            "SELECT taken_at, load_1m, mem_used FROM resource_snapshots WHERE snapshot_local_date = ? AND hostname = ?",
            (day, "host-a"),
        )
        row = cur.fetchone()
        self.assertEqual(row["taken_at"], f"{day}T09:05:00-06:00")
        self.assertEqual(float(row["load_1m"]), 1.5)
        self.assertEqual(int(row["mem_used"]), 50)

    def test_audit_hard_fail_rolls_back_snapshot(self) -> None:
        def fake_collect(db, timezone, top_n=None):
            db.insert_resource_snapshot(
                taken_at="2026-02-04T09:00:00-06:00",
                snapshot_local_date="2026-02-04",
                hostname="host-a",
                load_1m=0.5,
                load_5m=0.6,
                load_15m=0.7,
                mem_total=100,
                mem_used=40,
                mem_free=60,
                swap_total=10,
                swap_used=2,
            )
            return {"taken_at": "2026-02-04T09:00:00-06:00"}

        with mock.patch.object(handler.collector, "collect_and_persist_snapshot", side_effect=fake_collect):
            with mock.patch.object(self.db, "log_activity", side_effect=Exception("fail")):
                result = handler.resource_snapshot(
                    {"db": self.db, "timezone": "America/Regina"},
                    top_n=5,
                    user_id="tester",
                )

        self.assertEqual(result.get("message"), handler.AUDIT_HARD_FAIL_MSG)

        cur = self.db._conn.execute("SELECT COUNT(*) AS count FROM resource_snapshots")
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 0)


if __name__ == "__main__":
    unittest.main()
