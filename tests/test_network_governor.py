import os
import tempfile
import unittest
from unittest import mock

from memory.db import MemoryDB
from skills.network_governor import handler


class TestNetworkGovernor(unittest.TestCase):
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
        self.db.insert_network_snapshot(
            taken_at=f"{day}T09:00:00-06:00",
            snapshot_local_date=day,
            hostname="host-a",
            default_iface="eth0",
            default_gateway="192.168.1.1",
        )
        self.db.insert_network_snapshot(
            taken_at=f"{day}T09:05:00-06:00",
            snapshot_local_date=day,
            hostname="host-a",
            default_iface="eth1",
            default_gateway="192.168.1.2",
        )

        cur = self.db._conn.execute(
            "SELECT COUNT(*) AS count FROM network_snapshots WHERE snapshot_local_date = ? AND hostname = ?",
            (day, "host-a"),
        )
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 1)

        cur = self.db._conn.execute(
            "SELECT taken_at, default_iface, default_gateway FROM network_snapshots WHERE snapshot_local_date = ? AND hostname = ?",
            (day, "host-a"),
        )
        row = cur.fetchone()
        self.assertEqual(row["taken_at"], f"{day}T09:05:00-06:00")
        self.assertEqual(row["default_iface"], "eth1")
        self.assertEqual(row["default_gateway"], "192.168.1.2")

    def test_audit_hard_fail_rolls_back_snapshot(self) -> None:
        def fake_collect(db, timezone):
            db.insert_network_snapshot(
                taken_at="2026-02-04T09:00:00-06:00",
                snapshot_local_date="2026-02-04",
                hostname="host-a",
                default_iface="eth0",
                default_gateway="192.168.1.1",
            )
            db.replace_network_interfaces(
                "2026-02-04T09:00:00-06:00",
                [("eth0", "up", 1, 2, 0, 0)],
            )
            db.replace_network_nameservers(
                "2026-02-04T09:00:00-06:00",
                ["1.1.1.1"],
            )
            return {"taken_at": "2026-02-04T09:00:00-06:00"}

        with mock.patch.object(handler.collector, "collect_and_persist_snapshot", side_effect=fake_collect):
            with mock.patch.object(self.db, "log_activity", side_effect=Exception("fail")):
                result = handler.network_snapshot(
                    {"db": self.db, "timezone": "America/Regina"},
                    user_id="tester",
                )

        self.assertEqual(result.get("message"), handler.AUDIT_HARD_FAIL_MSG)

        cur = self.db._conn.execute("SELECT COUNT(*) AS count FROM network_snapshots")
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 0)

        cur = self.db._conn.execute("SELECT COUNT(*) AS count FROM network_interfaces")
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 0)

        cur = self.db._conn.execute("SELECT COUNT(*) AS count FROM network_nameservers")
        row = cur.fetchone()
        self.assertEqual(int(row["count"]), 0)


if __name__ == "__main__":
    unittest.main()
