import os
import tempfile
import unittest

from agent.intent_router import route_message
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB
from skills.recall import handler as recall_handler


class TestAskQuery(unittest.TestCase):
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

    def _seed_minimal_data(self) -> None:
        day1 = "2026-02-01"
        day2 = "2026-02-07"

        # Storage
        for mount, first_used, last_used in [("/", 100, 110), ("/data", 200, 250), ("/data2", 300, 330)]:
            self.db.insert_disk_snapshot(
                taken_at=f"{day1}T09:00:00-06:00",
                snapshot_local_date=day1,
                hostname="host-a",
                mountpoint=mount,
                filesystem="/dev/root",
                total_bytes=1000,
                used_bytes=first_used,
                free_bytes=900,
            )
            self.db.insert_disk_snapshot(
                taken_at=f"{day2}T09:00:00-06:00",
                snapshot_local_date=day2,
                hostname="host-a",
                mountpoint=mount,
                filesystem="/dev/root",
                total_bytes=1000,
                used_bytes=last_used,
                free_bytes=900,
            )

        self.db.insert_dir_size_samples(
            f"{day1}T09:00:00-06:00",
            "root_top",
            [("/var", 10)],
        )
        self.db.insert_dir_size_samples(
            f"{day2}T09:00:00-06:00",
            "root_top",
            [("/var", 20)],
        )
        self.db.insert_dir_size_samples(
            f"{day1}T09:00:00-06:00",
            "home_top",
            [("/home/user/Downloads", 5)],
        )
        self.db.insert_dir_size_samples(
            f"{day2}T09:00:00-06:00",
            "home_top",
            [("/home/user/Downloads", 9)],
        )
        self.db.insert_storage_scan_stats(f"{day1}T09:00:00-06:00", "root_top", 10, 1)
        self.db.insert_storage_scan_stats(f"{day2}T09:00:00-06:00", "home_top", 10, 2)

        # Resources
        self.db.insert_resource_snapshot(
            taken_at=f"{day1}T09:00:00-06:00",
            snapshot_local_date=day1,
            hostname="host-a",
            load_1m=1.0,
            load_5m=1.2,
            load_15m=1.4,
            mem_total=800,
            mem_used=100,
            mem_free=700,
            swap_total=0,
            swap_used=0,
        )
        self.db.insert_resource_snapshot(
            taken_at=f"{day2}T09:00:00-06:00",
            snapshot_local_date=day2,
            hostname="host-a",
            load_1m=3.0,
            load_5m=2.2,
            load_15m=2.4,
            mem_total=800,
            mem_used=300,
            mem_free=500,
            swap_total=0,
            swap_used=0,
        )
        self.db.replace_resource_process_samples(
            f"{day2}T09:00:00-06:00",
            "rss",
            [(123, "procA", 10, 400)],
        )

        # Network
        self.db.insert_network_snapshot(
            taken_at=f"{day1}T09:00:00-06:00",
            snapshot_local_date=day1,
            hostname="host-a",
            default_iface="eth0",
            default_gateway="10.0.0.1",
        )
        self.db.insert_network_snapshot(
            taken_at=f"{day2}T09:00:00-06:00",
            snapshot_local_date=day2,
            hostname="host-a",
            default_iface="eth0",
            default_gateway="10.0.0.2",
        )
        self.db.replace_network_interfaces(
            f"{day1}T09:00:00-06:00",
            [("eth0", "up", 10, 20, 0, 0)],
        )
        self.db.replace_network_interfaces(
            f"{day2}T09:00:00-06:00",
            [("eth0", "up", 40, 70, 0, 0)],
        )
        self.db.replace_network_nameservers(f"{day1}T09:00:00-06:00", ["1.1.1.1"])
        self.db.replace_network_nameservers(f"{day2}T09:00:00-06:00", ["8.8.8.8"])

    def test_advice_rejection(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="America/Regina",
            llm_client=None,
            enable_writes=False,
        )
        response = orchestrator.handle_message("/ask should i clean my disk", "user1")
        self.assertIn("factual recall", response.text)

    def test_deterministic_output(self) -> None:
        self._seed_minimal_data()
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result = recall_handler.ask_query(
            {"db": self.db, "timezone": "America/Regina"},
            question="status lately",
            timeframe=timeframe,
        )
        expected = "\n".join(
            [
                "Question Restated: status lately",
                "Timeframe: last 7 days (2026-02-01 to 2026-02-07, America/Regina)",
                "Storage:",
                "- / used change: 10 (+10B)",
                "- /data used change: 50 (+50B)",
                "- /data2 used change: 30 (+30B)",
                "- / largest dir growth: /var (+10B)",
                "- home largest dir growth: /home/user/Downloads (+4B)",
                "- storage scan errors_skipped (window): 3",
                "CPU/Memory:",
                "- load_1m min/avg/max: 1.00/2.00/3.00",
                "- load_5m min/avg/max: 1.20/1.70/2.20",
                "- load_15m min/avg/max: 1.40/1.90/2.40",
                "- mem_used min/avg/max: 100B/200B/300B",
                "- swap_used: no swap recorded",
                "- top processes by RSS (latest day):",
                "  pid=123 procA rss=400B",
                "Network:",
                "- default gateway changes:",
                "  2026-02-07: 10.0.0.1 -> 10.0.0.2",
                "- interface rx/tx change (first -> last):",
                "  eth0 rx=+30B tx=+50B",
                "- nameserver changes: 2026-02-07",
                "Limits:",
                "- Facts are based on stored snapshots only.",
                "- Missing days or domains reduce coverage for this timeframe.",
            ]
        )
        self.assertEqual(result.get("text"), expected)

    def test_multi_domain_order(self) -> None:
        self._seed_minimal_data()
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result = recall_handler.ask_query(
            {"db": self.db, "timezone": "America/Regina"},
            question="status lately",
            timeframe=timeframe,
        )
        text = result.get("text", "")
        storage_idx = text.find("Storage:")
        cpu_idx = text.find("CPU/Memory:")
        net_idx = text.find("Network:")
        self.assertTrue(0 <= storage_idx < cpu_idx < net_idx)

    def test_missing_data_handling(self) -> None:
        day1 = "2026-02-01"
        day2 = "2026-02-07"
        self.db.insert_disk_snapshot(
            taken_at=f"{day1}T09:00:00-06:00",
            snapshot_local_date=day1,
            hostname="host-a",
            mountpoint="/",
            filesystem="/dev/root",
            total_bytes=1000,
            used_bytes=100,
            free_bytes=900,
        )
        self.db.insert_disk_snapshot(
            taken_at=f"{day2}T09:00:00-06:00",
            snapshot_local_date=day2,
            hostname="host-a",
            mountpoint="/",
            filesystem="/dev/root",
            total_bytes=1000,
            used_bytes=200,
            free_bytes=800,
        )
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result = recall_handler.ask_query(
            {"db": self.db, "timezone": "America/Regina"},
            question="cpu recently",
            timeframe=timeframe,
        )
        text = result.get("text", "")
        self.assertIn("CPU/Memory:\n- insufficient data", text)

    def test_single_clarification_question(self) -> None:
        # Seed one snapshot so timeframe parsing can anchor.
        self.db.insert_disk_snapshot(
            taken_at="2026-02-04T09:00:00-06:00",
            snapshot_local_date="2026-02-04",
            hostname="host-a",
            mountpoint="/",
            filesystem="/dev/root",
            total_bytes=1000,
            used_bytes=100,
            free_bytes=900,
        )
        decision = route_message(
            "user1",
            "what happened last",
            {"db": self.db, "timezone": "America/Regina", "chat_id": "user1"},
        )
        self.assertEqual(decision.get("type"), "clarify")
        from datetime import datetime, timezone

        pending = self.db.get_pending_clarification(
            "user1",
            "user1",
            datetime.now(timezone.utc).isoformat(),
        )
        self.assertIsNotNone(pending)


if __name__ == "__main__":
    unittest.main()
