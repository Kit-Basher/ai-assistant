import os
import tempfile
import unittest
import json

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class TestBriefCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self) -> Orchestrator:
        os.environ["UI_MODE"] = "conversational"
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )

    def test_brief_smoke_output_shape(self) -> None:
        orch = self._orchestrator()

        def _insert_system_facts(snapshot_id: str, taken_at: str, load_1m: float, mem_used: int, disk_used: int) -> None:
            facts = {
                "schema": {"name": "system_facts", "version": 1},
                "snapshot": {
                    "snapshot_id": snapshot_id,
                    "taken_at": taken_at,
                    "timezone": "UTC",
                    "collector": {
                        "agent_version": "0.6.0",
                        "hostname": "host",
                        "boot_id": "boot",
                        "uptime_s": 1,
                        "collection_duration_ms": 1,
                        "partial": False,
                        "errors": [],
                    },
                    "provenance": {"sources": []},
                },
                "os": {"kernel": {"release": "6.0.0", "arch": "x86_64"}},
                "cpu": {"load": {"load_1m": load_1m, "load_5m": load_1m, "load_15m": load_1m}},
                "memory": {
                    "ram_bytes": {
                        "total": 16 * 1024**3,
                        "used": mem_used,
                        "free": 0,
                        "available": (16 * 1024**3) - mem_used,
                        "buffers": 0,
                        "cached": 0,
                    },
                    "swap_bytes": {"total": 0, "free": 0, "used": 0},
                    "pressure": {"psi_supported": False, "memory_some_avg10": None, "io_some_avg10": None, "cpu_some_avg10": None},
                },
                "filesystems": {
                    "mounts": [
                        {
                            "mountpoint": "/",
                            "device": "/dev/sda1",
                            "fstype": "ext4",
                            "total_bytes": 100 * 1024**3,
                            "used_bytes": disk_used,
                            "avail_bytes": 100 * 1024**3 - disk_used,
                            "used_pct": (float(disk_used) / float(100 * 1024**3)) * 100.0,
                            "inodes": {"total": None, "used": None, "avail": None, "used_pct": None},
                        }
                    ]
                },
                "process_summary": {"top_cpu": [], "top_mem": [{"pid": 1, "name": "proc", "cpu_pct": None, "rss_bytes": mem_used // 4}]},
                "integrity": {"content_hash_sha256": "0" * 64, "signed": False, "signature": None},
            }
            facts_json = json.dumps(facts, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            self.db.insert_system_facts_snapshot(
                id=snapshot_id,
                user_id="user1",
                taken_at=taken_at,
                boot_id="boot",
                schema_version=1,
                facts_json=facts_json,
                content_hash_sha256="0" * 64,
                partial=False,
                errors_json="[]",
            )

        def observe_handler_1(ctx, user_id=None):
            _insert_system_facts("snap-1", "2026-02-06T00:00:00+00:00", load_1m=0.1, mem_used=2 * 1024**3, disk_used=60 * 1024**3)
            return {"text": "Snapshot taken: 2026-02-06T00:00:00+00:00 (UTC)", "payload": {}}

        orch.skills["observe_now"].functions["observe_now"].handler = observe_handler_1

        first = orch.handle_message("/brief", "user1")
        self.assertIn("baseline created", first.text.lower())
        self.assertNotIn("/resource_report", first.text)
        self.assertNotIn("/storage_report", first.text)
        self.assertNotIn("/hardware_report", first.text)
        self.assertLessEqual(len(first.text.splitlines()), 10)

        def observe_handler_2(ctx, user_id=None):
            _insert_system_facts("snap-2", "2026-02-07T00:00:00+00:00", load_1m=0.6, mem_used=4 * 1024**3, disk_used=70 * 1024**3)
            return {"text": "Snapshot taken: 2026-02-07T00:00:00+00:00 (UTC)", "payload": {}}

        orch.skills["observe_now"].functions["observe_now"].handler = observe_handler_2

        second = orch.handle_message("/brief", "user1")
        self.assertIn("- ", second.text)
        self.assertIn("Disk /:", second.text)
        self.assertIn("RAM available:", second.text)
        self.assertNotIn("/resource_report", second.text)
        self.assertLessEqual(len(second.text.splitlines()), 12)


if __name__ == "__main__":
    unittest.main()

