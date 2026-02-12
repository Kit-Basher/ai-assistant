import os
import tempfile
import unittest
import json

from agent.knowledge_cache import facts_hash
from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_handle_message_no_longer_raises(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        response = orchestrator.handle_message("hello there", "user1")
        self.assertIsInstance(response, OrchestratorResponse)

    def test_knowledge_query_cache_and_cta(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        response = orchestrator.handle_message("what changed this week", "user1")
        self.assertIn("Want my opinion", response.text)
        entry = orchestrator._knowledge_cache.get_recent("user1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.facts_hash, facts_hash(entry.facts))

    def test_opinion_followup_uses_cached_facts(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        orchestrator.handle_message("what changed this week", "user1")
        entry = orchestrator._knowledge_cache.get_recent("user1")
        response = orchestrator.handle_message("opinion", "user1")
        self.assertIn("source", response.data.get("data", {}))
        self.assertEqual(response.data["data"]["facts_hash"], entry.facts_hash)

    def test_greeting_then_affirmation_runs_brief(self) -> None:
        orch = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )

        def _insert_system_facts(snapshot_id: str, taken_at: str) -> None:
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
                "cpu": {"load": {"load_1m": 0.1, "load_5m": 0.1, "load_15m": 0.1}},
                "memory": {
                    "ram_bytes": {
                        "total": 16 * 1024**3,
                        "used": 2 * 1024**3,
                        "free": 0,
                        "available": 14 * 1024**3,
                        "buffers": 0,
                        "cached": 0,
                    },
                    "swap_bytes": {"total": 0, "free": 0, "used": 0},
                    "pressure": {
                        "psi_supported": False,
                        "memory_some_avg10": None,
                        "io_some_avg10": None,
                        "cpu_some_avg10": None,
                    },
                },
                "filesystems": {
                    "mounts": [
                        {
                            "mountpoint": "/",
                            "device": "/dev/sda1",
                            "fstype": "ext4",
                            "total_bytes": 100 * 1024**3,
                            "used_bytes": 60 * 1024**3,
                            "avail_bytes": 40 * 1024**3,
                            "used_pct": 60.0,
                            "inodes": {"total": None, "used": None, "avail": None, "used_pct": None},
                        }
                    ]
                },
                "process_summary": {"top_cpu": [], "top_mem": []},
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

        def observe_handler(ctx, user_id=None):
            _insert_system_facts("snap-1", "2026-02-06T00:00:00+00:00")
            return {"text": "Snapshot taken", "payload": {}}

        orch.skills["observe_now"].functions["observe_now"].handler = observe_handler

        first = orch.handle_message("hello", "user1")
        self.assertIn("want a quick /brief", first.text.lower())

        second = orch.handle_message("yes please", "user1")
        self.assertIn("baseline created", second.text.lower())


if __name__ == "__main__":
    unittest.main()
