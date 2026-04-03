import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from agent.report_followups import resource_followup
from memory.db import MemoryDB


class TestResourceFollowups(unittest.TestCase):
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

    def _store_report(self, ts: str, mem_used: int, load_1m: float) -> None:
        payload = {
            "taken_at": ts,
            "loads": {"1m": load_1m, "5m": 0.0, "15m": 0.0},
            "memory": {"total": 100, "used": mem_used, "free": 100 - mem_used},
            "swap": {"total": 0, "used": 0},
            "cpu_samples": [{"pid": 1, "name": "cpuproc", "cpu_ticks": 10}],
            "rss_samples": [{"pid": 2, "name": "memproc", "rss_bytes": 1024}],
        }
        self.db.upsert_last_report("user1", "resource_report", ts, payload, audit_ref=ts)

    def test_followup_requires_report(self) -> None:
        text = resource_followup(self.db, "user1", "top_memory", "UTC")
        self.assertIn("Run /resource_report first", text)

    def test_followup_top_memory(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._store_report(ts, 80, 1.0)
        text = resource_followup(self.db, "user1", "top_memory", "UTC")
        self.assertIn("Top memory processes", text)

    def test_compare_to_previous(self) -> None:
        base = datetime.now(timezone.utc)
        self._store_report(base.isoformat(), 80, 1.0)
        self._store_report((base + timedelta(minutes=10)).isoformat(), 90, 2.0)
        text = resource_followup(self.db, "user1", "compare", "UTC")
        self.assertIn("Memory used delta", text)


if __name__ == "__main__":
    unittest.main()
