import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from memory.db import MemoryDB
from telegram_adapter.bot import _check_reminders


class FakeBot:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent = []

    async def send_message(self, chat_id: str, text: str):
        if self.fail:
            raise RuntimeError("send_failed")
        self.sent.append((chat_id, text))


class FakeApplication:
    def __init__(self, bot_data):
        self.bot_data = bot_data


class FakeContext:
    def __init__(self, bot, bot_data):
        self.bot = bot
        self.application = FakeApplication(bot_data)


class TestReminderDelivery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.db.set_preference("telegram_chat_id", "chat-1")

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _run_check(self, bot):
        ctx = FakeContext(
            bot,
            {"db": self.db, "log_path": self.log_path},
        )
        asyncio.run(_check_reminders(ctx))

    def test_due_reminder_sent_once(self) -> None:
        now = datetime.now(timezone.utc)
        due_ts = (now - timedelta(minutes=1)).isoformat()
        reminder_id = self.db.add_reminder(due_ts, "ping")

        bot = FakeBot()
        self._run_check(bot)
        self.assertEqual(len(bot.sent), 1)

        self._run_check(bot)
        self.assertEqual(len(bot.sent), 1)

        cur = self.db._conn.execute("SELECT status FROM reminders WHERE id = ?", (reminder_id,))
        status = cur.fetchone()["status"]
        self.assertEqual(status, "sent")

    def test_claim_gate_blocks_double_send(self) -> None:
        now = datetime.now(timezone.utc)
        due_ts = (now - timedelta(minutes=1)).isoformat()
        self.db.add_reminder(due_ts, "ping")

        bot = FakeBot()
        self._run_check(bot)
        self._run_check(bot)
        self.assertEqual(len(bot.sent), 1)

    def test_failed_send_marks_failed(self) -> None:
        now = datetime.now(timezone.utc)
        due_ts = (now - timedelta(minutes=1)).isoformat()
        reminder_id = self.db.add_reminder(due_ts, "ping")

        bot = FakeBot(fail=True)
        self._run_check(bot)
        self.assertEqual(len(bot.sent), 0)

        cur = self.db._conn.execute(
            "SELECT status, last_error FROM reminders WHERE id = ?",
            (reminder_id,),
        )
        row = cur.fetchone()
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["last_error"])

        bot_ok = FakeBot()
        self._run_check(bot_ok)
        self.assertEqual(len(bot_ok.sent), 0)

    def test_due_query_filters(self) -> None:
        now = datetime.now(timezone.utc)
        past_ts = (now - timedelta(minutes=1)).isoformat()
        future_ts = (now + timedelta(minutes=10)).isoformat()
        due_id = self.db.add_reminder(past_ts, "due")
        self.db.add_reminder(future_ts, "future")
        self.db._conn.execute(
            "UPDATE reminders SET status = 'sent' WHERE id = ?",
            (due_id,),
        )
        self.db._conn.commit()
        due = self.db.list_due_reminders(now.isoformat())
        self.assertEqual(due, [])


if __name__ == "__main__":
    unittest.main()
