import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from agent.report_followups import resource_followup
from memory.db import MemoryDB
from skills.resource_governor import collector


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

    def _store_report(
        self,
        ts: str,
        mem_used: int,
        load_1m: float,
        *,
        process_pid: int = 2,
        process_name: str = "memproc",
    ) -> None:
        payload = {
            "taken_at": ts,
            "loads": {"1m": load_1m, "5m": 0.0, "15m": 0.0},
            "memory": {"total": 100, "used": mem_used, "free": 100 - mem_used},
            "swap": {"total": 0, "used": 0},
            "cpu_samples": [{"pid": 1, "name": "cpuproc", "cpu_ticks": 10}],
            "rss_samples": [{"pid": process_pid, "name": process_name, "rss_bytes": 1024}],
        }
        self.db.upsert_last_report("user1", "resource_report", ts, payload, audit_ref=ts)

    def _live_snapshot(
        self,
        *,
        used: int,
        total: int = 100,
        available: int | None = None,
        load_1m: float = 0.0,
        top_cpu: list[dict[str, object]] | None = None,
        top_rss: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        if available is None:
            available = max(total - used, 0)
        return {
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_local_date": datetime.now(timezone.utc).date().isoformat(),
            "hostname": "testhost",
            "cpu_count": 8,
            "loadavg": (load_1m, 0.0, 0.0),
            "mem": {
                "total": total,
                "used": used,
                "available": available,
                "free": available,
                "buffers": 0,
                "cached": 0,
                "shared": 0,
                "used_pct": round((used / float(total)) * 100.0, 2) if total else 0.0,
            },
            "swap": {"total": 0, "used": 0},
            "top_cpu": top_cpu or [],
            "top_rss": top_rss or [],
            "proc_stats": {"procs_scanned": 1, "errors_skipped": 0},
        }

    def test_followup_requires_fresh_probe_when_live_unavailable(self) -> None:
        with patch.object(collector, "collect_live_snapshot", return_value=None):
            text = resource_followup(self.db, "user1", "top_memory", "UTC")
        self.assertIn("fresh live probe", text.lower())
        self.assertIn("cannot verify", text.lower())

    def test_followup_top_memory_uses_fresh_probe(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._store_report(ts, 80, 1.0)
        live = self._live_snapshot(
            used=42,
            top_rss=[{"pid": 77, "name": "freshproc", "rss_bytes": 42, "cpu_ticks": 7}],
        )
        with patch.object(collector, "collect_live_snapshot", return_value=live):
            text = resource_followup(self.db, "user1", "top_memory", "UTC", question="what is using memory")
        self.assertIn("freshproc", text)
        self.assertIn("Top memory processes", text)
        self.assertNotIn("oldproc", text)

    def test_compare_to_previous(self) -> None:
        base = datetime.now(timezone.utc)
        self._store_report(base.isoformat(), 80, 1.0)
        live = self._live_snapshot(used=90)
        with patch.object(collector, "collect_live_snapshot", return_value=live):
            text = resource_followup(self.db, "user1", "compare", "UTC")
        self.assertIn("fresh state", text.lower())
        self.assertIn("memory used changed from 80 to 90 bytes", text.lower())

    def test_process_killed_followup_says_gone(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._store_report(ts, 900, 1.0, process_pid=3367, process_name="windows-thing")
        live = self._live_snapshot(used=600)
        with patch.object(collector, "collect_live_snapshot", return_value=live), patch.object(
            collector,
            "collect_live_process_index",
            return_value=[],
        ):
            text = resource_followup(
                self.db,
                "user1",
                "process_state",
                "UTC",
                question="is that windows thing still running?",
            )
        self.assertIn("no longer running", text.lower())
        self.assertIn("memory changed from 900 bytes to 600 bytes", text.lower())

    def test_stale_pid_reference_detects_missing_process(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._store_report(ts, 800, 1.0, process_pid=3367, process_name="windows-thing")
        live = self._live_snapshot(used=500)
        with patch.object(collector, "collect_live_snapshot", return_value=live), patch.object(
            collector,
            "collect_live_process_index",
            return_value=[{"pid": 999, "name": "windows-thing"}],
        ):
            text = resource_followup(
                self.db,
                "user1",
                "process_state",
                "UTC",
                question="is pid 3367 still running?",
            )
        self.assertIn("pid 3367 is no longer running", text.lower())
        self.assertNotIn("pid 999", text.lower())

    def test_kill_followup_detects_live_process_and_gives_guidance(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._store_report(ts, 800, 1.0, process_pid=4123, process_name="brave")
        live = self._live_snapshot(used=450)
        with patch.object(collector, "collect_live_snapshot", return_value=live), patch.object(
            collector,
            "collect_live_process_index",
            return_value=[{"pid": 4123, "name": "brave"}],
        ):
            text = resource_followup(
                self.db,
                "user1",
                "process_state",
                "UTC",
                question="kill brave",
            )
        self.assertIn("still running", text.lower())
        self.assertIn("safe next action", text.lower())
        self.assertIn("leave it alone", text.lower())

    def test_followup_after_system_change_uses_fresh_probe_not_cached(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._store_report(ts, 20, 1.0)
        first_live = self._live_snapshot(
            used=30,
            top_rss=[{"pid": 11, "name": "oldproc", "rss_bytes": 30, "cpu_ticks": 5}],
        )
        second_live = self._live_snapshot(
            used=70,
            top_rss=[{"pid": 12, "name": "newproc", "rss_bytes": 70, "cpu_ticks": 5}],
        )
        with patch.object(collector, "collect_live_snapshot", side_effect=[first_live, second_live]):
            first = resource_followup(self.db, "user1", "top_memory", "UTC", question="what is using memory")
            second = resource_followup(self.db, "user1", "top_memory", "UTC", question="what is using memory")
        self.assertIn("oldproc", first)
        self.assertIn("newproc", second)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
