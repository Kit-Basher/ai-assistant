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

    def test_resource_report_prefers_live_probe_and_reports_memory_breakdown(self) -> None:
        live = {
            "taken_at": "2026-02-04T09:00:00-06:00",
            "hostname": "host-a",
            "loadavg": (1.23, 0.99, 0.75),
            "mem": {
                "total": int(67.4 * 1024**3),
                "used": int(21.3 * 1024**3),
                "available": int(46.1 * 1024**3),
                "free": int(12.4 * 1024**3),
                "buffers": int(512 * 1024**2),
                "cached": int(8 * 1024**3),
                "shared": int(256 * 1024**2),
                "used_pct": 31.6,
            },
            "swap": {"total": 0, "used": 0},
            "top_cpu": [{"pid": 11, "name": "python", "cpu_ticks": 42, "rss_bytes": int(2.8 * 1024**3)}],
            "top_rss": [
                {"pid": 22, "name": "chrome", "cpu_ticks": 18, "rss_bytes": int(4.2 * 1024**3)},
                {"pid": 11, "name": "python", "cpu_ticks": 42, "rss_bytes": int(2.8 * 1024**3)},
            ],
            "proc_stats": {"procs_scanned": 128, "errors_skipped": 2},
        }

        with mock.patch.object(handler.collector, "collect_live_snapshot", return_value=live):
            result = handler.resource_report(
                {"db": self.db, "timezone": "America/Regina", "read_only_mode": True},
                user_id="tester",
            )

        self.assertEqual("ok", result.get("status"))
        self.assertEqual("live", result.get("payload", {}).get("source"))
        self.assertIn("Live memory probe", result.get("text", ""))
        self.assertIn("Memory used: 21.3G / total 67.4G (31.6%)", result.get("text", ""))
        self.assertIn("Top memory processes", result.get("text", ""))
        self.assertIn("chrome", result.get("text", ""))
        self.assertIn("MemAvailable includes reclaimable cache/buffers/shared memory", result.get("text", ""))
        self.assertIn("Likely cause:", result.get("cards_payload", {}).get("summary", ""))
        self.assertEqual(1.0, result.get("cards_payload", {}).get("confidence"))

    def test_resource_report_without_live_probe_and_no_snapshots_is_explicit(self) -> None:
        with mock.patch.object(handler.collector, "collect_live_snapshot", side_effect=OSError("probe unavailable")):
            result = handler.resource_report(
                {"db": self.db, "timezone": "America/Regina", "read_only_mode": True},
                user_id="tester",
            )

        text = str(result.get("text") or "")
        self.assertIn("couldn't get a live memory probe", text.lower())
        self.assertIn("no stored resource snapshots yet", text.lower())
        self.assertNotIn("0.0%", text)
        self.assertEqual(0.0, result.get("cards_payload", {}).get("confidence"))

    def test_resource_report_invalid_snapshot_does_not_fabricate_memory(self) -> None:
        day = "2026-02-04"
        self.db.insert_resource_snapshot(
            taken_at=f"{day}T09:00:00-06:00",
            snapshot_local_date=day,
            hostname="host-a",
            load_1m=0.5,
            load_5m=0.6,
            load_15m=0.7,
            mem_total=0,
            mem_used=0,
            mem_free=0,
            swap_total=0,
            swap_used=0,
        )

        with mock.patch.object(handler.collector, "collect_live_snapshot", side_effect=OSError("probe unavailable")):
            result = handler.resource_report(
                {"db": self.db, "timezone": "America/Regina", "read_only_mode": True},
                user_id="tester",
            )

        text = str(result.get("text") or "")
        self.assertIn("incomplete or invalid", text.lower())
        self.assertNotIn("0.0%", text)
        self.assertNotIn("memory 0", text.lower())
        self.assertEqual(0.0, result.get("cards_payload", {}).get("confidence"))


if __name__ == "__main__":
    unittest.main()
