import os
import tempfile
import unittest

from agent.intent_router import route_message
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB
from skills.opinion import handler as opinion_handler


class TestOpinionQuery(unittest.TestCase):
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

        # seed minimal data
        day1 = "2026-02-01"
        day2 = "2026-02-07"
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

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_opinion_vocab_enforced(self) -> None:
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system",
            timeframe=timeframe,
            trigger="what do you think",
        )
        text = result.get("text", "")
        for forbidden in opinion_handler.FORBIDDEN_WORDS:
            self.assertNotIn(forbidden, text.lower())

    def test_deterministic_output(self) -> None:
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result1 = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my storage lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        result2 = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my storage lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        self.assertEqual(result1.get("text"), result2.get("text"))

    def test_opinion_only_with_trigger(self) -> None:
        decision = route_message(
            "user1",
            "status lately",
            {"db": self.db, "timezone": "America/Regina", "chat_id": "user1"},
        )
        self.assertNotEqual(decision.get("skill"), "opinion")

    def test_advice_rejected(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="America/Regina",
            llm_client=None,
            enable_writes=False,
        )
        response = orchestrator.handle_message("/ask_opinion should i clean my disk", "user1")
        self.assertIn("not advice", response.text)

    def test_numeric_basis_present(self) -> None:
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        text = result.get("text", "")
        lines = [line for line in text.splitlines() if line.startswith("- ") and "(basis:" in line]
        self.assertTrue(lines)
        for line in lines:
            has_digit = any(ch.isdigit() for ch in line)
            self.assertTrue(has_digit)


if __name__ == "__main__":
    unittest.main()
