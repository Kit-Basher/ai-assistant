import os
import tempfile
import unittest
from unittest import mock

from memory.db import MemoryDB
from skills.storage_governor import handler


class TestStorageGovernor(unittest.TestCase):
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

    def test_same_day_duplicate_prevention(self) -> None:
        taken_at_1 = "2026-02-04T09:00:00-06:00"
        taken_at_2 = "2026-02-04T09:05:00-06:00"
        snapshot_date = "2026-02-04"

        self.db.insert_disk_snapshot(
            taken_at=taken_at_1,
            snapshot_local_date=snapshot_date,
            hostname="host-a",
            mountpoint="/",
            filesystem="/dev/root",
            total_bytes=100,
            used_bytes=60,
            free_bytes=40,
        )
        self.db.insert_disk_snapshot(
            taken_at=taken_at_2,
            snapshot_local_date=snapshot_date,
            hostname="host-a",
            mountpoint="/",
            filesystem="/dev/root",
            total_bytes=100,
            used_bytes=70,
            free_bytes=30,
        )

        cur = self.db._conn.execute(
            "SELECT COUNT(*) AS count FROM disk_snapshots WHERE mountpoint = ?",
            ("/",),
        )
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 1)

        cur = self.db._conn.execute(
            "SELECT taken_at, used_bytes FROM disk_snapshots WHERE mountpoint = ?",
            ("/",),
        )
        row = cur.fetchone()
        self.assertEqual(row["taken_at"], taken_at_2)
        self.assertEqual(int(row["used_bytes"]), 70)

    def test_audit_hard_fail_rolls_back_snapshot(self) -> None:
        def fake_collect(db, timezone, top_n=None, home_path=None, mountpoints=None):
            db.insert_disk_snapshot(
                taken_at="2026-02-04T09:00:00-06:00",
                snapshot_local_date="2026-02-04",
                hostname="host-a",
                mountpoint="/",
                filesystem="/dev/root",
                total_bytes=100,
                used_bytes=60,
                free_bytes=40,
            )
            db.insert_dir_size_samples(
                "2026-02-04T09:00:00-06:00",
                "root_top",
                [("/var", 123)],
            )
            return {"taken_at": "2026-02-04T09:00:00-06:00"}

        with mock.patch.object(handler.collector, "collect_and_persist_snapshot", side_effect=fake_collect):
            with mock.patch.object(self.db, "audit_log_update_status", side_effect=Exception("fail")):
                result = handler.storage_snapshot(
                    {"db": self.db, "timezone": "America/Regina"},
                    top_n=5,
                    user_id="tester",
                )

        self.assertEqual(result.get("message"), handler.AUDIT_HARD_FAIL_MSG)

        cur = self.db._conn.execute("SELECT COUNT(*) AS count FROM disk_snapshots")
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 0)

        cur = self.db._conn.execute("SELECT COUNT(*) AS count FROM dir_size_samples")
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 0)


if __name__ == "__main__":
    unittest.main()
