from __future__ import annotations

import os
import subprocess
import tempfile
import unittest

from agent.perception.collector import _collect_gpu_metrics, collect_snapshot
from agent.perception.diagnostics import analyze_snapshot
from memory.db import MemoryDB


class TestPerception(unittest.TestCase):
    def test_snapshot_returns_required_keys(self) -> None:
        snapshot = collect_snapshot(roots=[])
        self.assertIn("ts", snapshot)
        self.assertIn("cpu", snapshot)
        self.assertIn("memory", snapshot)
        self.assertIn("disk", snapshot)
        self.assertIn("gpu", snapshot)
        self.assertIn("system_health", snapshot)
        self.assertIn("usage_pct", snapshot["cpu"])
        self.assertIn("per_core_usage_pct", snapshot["cpu"])
        self.assertIn("freq_mhz", snapshot["cpu"])
        self.assertIn("load_avg", snapshot["cpu"])
        self.assertIn("used", snapshot["memory"])
        self.assertIn("available", snapshot["memory"])
        self.assertIn("root", snapshot["disk"])
        self.assertIn("used_pct", snapshot["disk"]["root"])

    def test_diagnostics_trigger_expected_events(self) -> None:
        snapshot = {
            "cpu": {"usage_pct": 92.0, "freq_mhz": 1200.0, "freq_max_mhz": 3200.0},
            "gpu": {"usage_pct": 10.0, "temperature_c": 84.0},
            "memory": {"available": 500 * 1024 * 1024},
            "disk": {"root": {"used_pct": 90.0}},
        }
        events = analyze_snapshot(snapshot)
        kinds = {event.kind for event in events}
        self.assertIn("cpu_bound", kinds)
        self.assertIn("thermal_throttle_suspected", kinds)
        self.assertIn("disk_pressure", kinds)
        self.assertIn("oom_risk", kinds)

    def test_sqlite_insert_and_read_for_perception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "agent.db")
            schema_path = os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
            db = MemoryDB(db_path)
            db.init_schema(schema_path)

            snapshot = {
                "ts": 1700000000,
                "cpu": {"usage_pct": 12.5, "freq_mhz": 2400.0},
                "memory": {"used": 1000, "available": 2000},
                "disk": {"root": {"used_pct": 45.2}},
                "gpu": {"usage_pct": 22.0, "memory_used_mb": 512, "temperature_c": 65.0},
            }
            snapshot_id = db.insert_metrics_snapshot(snapshot)
            self.assertGreater(snapshot_id, 0)
            latest = db.get_latest_metrics_snapshot()
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(1700000000, int(latest["ts"]))
            self.assertAlmostEqual(12.5, float(latest["cpu_usage"]))

            event_id = db.insert_event(
                1700000000,
                "disk_pressure",
                "warning",
                "Root filesystem is under pressure.",
                {"root_disk_used_pct": 91.0},
            )
            self.assertGreater(event_id, 0)
            recent = db.list_recent_events(limit=5)
            self.assertTrue(recent)
            self.assertEqual("disk_pressure", recent[0]["kind"])
            self.assertIsInstance(recent[0]["evidence_json"], dict)
            db.close()

    def test_gpu_missing_nvidia_smi_is_graceful(self) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            cmd = args[0]
            if isinstance(cmd, list) and cmd and cmd[0] == "nvidia-smi":
                raise FileNotFoundError("nvidia-smi")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        metrics = _collect_gpu_metrics(fake_run)
        self.assertFalse(metrics["available"])
        self.assertIsNotNone(metrics["error"])


if __name__ == "__main__":
    unittest.main()
