import os
import tempfile
import unittest
from datetime import datetime, timezone

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class TestHealthAndBriefStatus(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)
        self.orch = Orchestrator(
            db=self.db,
            skills_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_health_payload_contains_expected_cards(self) -> None:
        response = self.orch.handle_message("/health", "user-1")
        self.assertIn("Bot status", response.text)
        self.assertIn("Observe scheduler", response.text)
        self.assertIn("schema_version", response.text)
        self.assertIn("Daily brief config", response.text)

    def test_daily_brief_status_explains_already_sent(self) -> None:
        self.db.set_preference("daily_brief_enabled", "on")
        self.db.set_preference("daily_brief_time", "00:00")
        self.db.set_preference("daily_brief_last_sent_date", datetime.now(timezone.utc).date().isoformat())
        self.orch.build_daily_brief_cards = lambda _user: {  # type: ignore[assignment]
            "daily_brief_signals": {
                "disk_delta_mb": 10.0,
                "service_unhealthy": False,
                "due_open_loops_count": 0,
            }
        }
        response = self.orch.handle_message("/daily_brief_status", "user-1")
        self.assertIn("Daily brief status", response.text)
        self.assertIn("already_sent_today", response.text)

    def test_daily_brief_status_explains_quiet_suppression(self) -> None:
        self.db.set_preference("daily_brief_enabled", "on")
        self.db.set_preference("daily_brief_time", "00:00")
        self.db.set_preference("daily_brief_quiet_mode", "on")
        self.db.set_preference("disk_delta_threshold_mb", "300")
        self.orch.build_daily_brief_cards = lambda _user: {  # type: ignore[assignment]
            "daily_brief_signals": {
                "disk_delta_mb": 10.0,
                "service_unhealthy": False,
                "due_open_loops_count": 0,
            }
        }
        response = self.orch.handle_message("daily brief status", "user-1")
        self.assertIn("quiet_no_signals", response.text)
        self.assertIn("disk delta below threshold", response.text)


if __name__ == "__main__":
    unittest.main()
