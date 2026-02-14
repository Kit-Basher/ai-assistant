import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from agent.scheduled_daily_brief import run_once
from memory.db import MemoryDB


class TestScheduledDailyBriefEntrypoint(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(self.db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)
        self.db.set_preference("telegram_chat_id", "chat-1")
        self.db.set_preference("daily_brief_enabled", "off")

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _update_task(
        self,
        task_id: int,
        *,
        due_date: str | None,
        impact: int,
        effort: int,
        created_at: str,
        status: str = "todo",
    ) -> None:
        self.db._conn.execute(
            """
            UPDATE tasks
            SET due_date = ?, impact_1to5 = ?, effort_mins = ?, created_at = ?, status = ?
            WHERE id = ?
            """,
            (due_date, impact, effort, created_at, status, task_id),
        )
        self.db._conn.commit()

    def _update_open_loop(
        self,
        loop_id: int,
        *,
        due_date: str | None,
        priority: int,
        created_at: str,
        status: str = "open",
    ) -> None:
        self.db._conn.execute(
            """
            UPDATE open_loops
            SET due_date = ?, priority = ?, created_at = ?, status = ?
            WHERE id = ?
            """,
            (due_date, priority, created_at, status, loop_id),
        )
        self.db._conn.commit()

    def test_run_once_disabled_returns_zero_and_does_not_send(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENT_DB_PATH": self.db_path,
                "AGENT_TIMEZONE": "UTC",
            },
            clear=False,
        ), patch("agent.scheduled_daily_brief._send_telegram_message") as mocked_send:
            result = run_once(now_utc=datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc))

        self.assertEqual(0, result)
        self.assertFalse(mocked_send.called)
        self.assertIsNone(self.db.get_preference("daily_brief_last_sent_date"))

    def test_brief_output_is_deterministic_and_sorts_null_due_last(self) -> None:
        self.db.set_preference("daily_brief_enabled", "on")
        self.db.set_preference("daily_brief_time", "00:00")

        task_no_due = self.db.add_task(None, "Task no due", 30, 5)
        task_due_low = self.db.add_task(None, "Task due soon low", 20, 3)
        task_due_high = self.db.add_task(None, "Task due soon high", 25, 5)

        self._update_task(
            task_no_due,
            due_date=None,
            impact=5,
            effort=30,
            created_at="2026-02-09T08:00:00+00:00",
        )
        self._update_task(
            task_due_low,
            due_date="2026-02-15",
            impact=3,
            effort=20,
            created_at="2026-02-10T08:00:00+00:00",
        )
        self._update_task(
            task_due_high,
            due_date="2026-02-15",
            impact=5,
            effort=25,
            created_at="2026-02-11T08:00:00+00:00",
        )

        loop_due_low = self.db.add_open_loop("Loop due tomorrow low priority")
        loop_due_high = self.db.add_open_loop("Loop due soon high priority")
        loop_no_due = self.db.add_open_loop("Loop no due")

        self._update_open_loop(
            loop_due_low,
            due_date="2026-02-16",
            priority=1,
            created_at="2026-02-12T08:00:00+00:00",
        )
        self._update_open_loop(
            loop_due_high,
            due_date="2026-02-15",
            priority=3,
            created_at="2026-02-11T08:00:00+00:00",
        )
        self._update_open_loop(
            loop_no_due,
            due_date=None,
            priority=2,
            created_at="2026-02-10T08:00:00+00:00",
        )

        with patch.dict(
            os.environ,
            {
                "AGENT_DB_PATH": self.db_path,
                "AGENT_TIMEZONE": "UTC",
                "TELEGRAM_BOT_TOKEN": "token",
            },
            clear=False,
        ), patch("agent.scheduled_daily_brief._send_telegram_message") as mocked_send:
            result = run_once(now_utc=datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc))

        self.assertEqual(0, result)
        self.assertEqual(1, mocked_send.call_count)
        msg = str(mocked_send.call_args.args[2])

        expected = "\n".join(
            [
                "**Today:** 2026-02-14",
                "",
                "**Top tasks:**",
                f"- [{task_due_high}] Task due soon high (due 2026-02-15 | impact 5 | effort 25m)",
                f"- [{task_due_low}] Task due soon low (due 2026-02-15 | impact 3 | effort 20m)",
                f"- [{task_no_due}] Task no due (due none | impact 5 | effort 30m)",
                "",
                "**Due soon:**",
                f"- [{task_due_high}] Task due soon high (due 2026-02-15 | impact 5 | effort 25m)",
                f"- [{task_due_low}] Task due soon low (due 2026-02-15 | impact 3 | effort 20m)",
                "",
                "**Open loops:**",
                f"- [{loop_due_high}] Loop due soon high priority (due 2026-02-15 | priority 3)",
                f"- [{loop_due_low}] Loop due tomorrow low priority (due 2026-02-16 | priority 1)",
                f"- [{loop_no_due}] Loop no due (due none | priority 2)",
                "",
                f"**Nudge:** [{task_no_due}] Task no due",
            ]
        )

        self.assertEqual(expected, msg)

    def test_no_duplicate_send_same_day(self) -> None:
        self.db.set_preference("daily_brief_enabled", "on")
        self.db.set_preference("daily_brief_time", "00:00")
        self.db.add_task(None, "Single task", 10, 1)

        now = datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc)
        with patch.dict(
            os.environ,
            {
                "AGENT_DB_PATH": self.db_path,
                "AGENT_TIMEZONE": "UTC",
                "TELEGRAM_BOT_TOKEN": "token",
            },
            clear=False,
        ), patch("agent.scheduled_daily_brief._send_telegram_message") as mocked_send:
            first = run_once(now_utc=now)
            second = run_once(now_utc=now)

        self.assertEqual(0, first)
        self.assertEqual(0, second)
        self.assertEqual(1, mocked_send.call_count)
        self.assertEqual("2026-02-14", self.db.get_preference("daily_brief_last_sent_date"))


if __name__ == "__main__":
    unittest.main()
