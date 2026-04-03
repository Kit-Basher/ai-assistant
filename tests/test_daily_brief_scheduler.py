import asyncio
import os
import tempfile
import unittest

from memory.db import MemoryDB
from telegram_adapter.bot import _scheduled_daily_brief


class _StubOrchestrator:
    def build_daily_brief_cards(self, user_id: str) -> dict:
        return {
            "cards": [{"title": "Disk usage (/)", "lines": ["used 50%"], "severity": "ok"}],
            "raw_available": True,
            "summary": "Daily brief",
            "confidence": 0.9,
            "next_questions": [],
            "daily_brief_signals": {
                "disk_delta_mb": 300.0,
                "service_unhealthy": False,
                "due_open_loops_count": 0,
            },
        }


class _BotFailOnce:
    def __init__(self) -> None:
        self.calls = 0

    async def send_message(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return {"ok": True}


class _BotAlwaysFail:
    def __init__(self) -> None:
        self.calls = 0

    async def send_message(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError("down")


class _App:
    def __init__(self, bot_data: dict) -> None:
        self.bot_data = bot_data


class _Ctx:
    def __init__(self, app: _App, bot) -> None:
        self.application = app
        self.bot = bot


class TestDailyBriefScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)
        self.db.set_preference("telegram_chat_id", "chat-1")
        self.db.set_preference("daily_brief_enabled", "on")
        self.db.set_preference("daily_brief_time", "00:00")

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_retry_once_and_update_last_sent_on_success_only(self) -> None:
        bot = _BotFailOnce()
        ctx = _Ctx(
            _App({"db": self.db, "timezone": "UTC", "orchestrator": _StubOrchestrator()}),
            bot,
        )
        asyncio.run(_scheduled_daily_brief(ctx))
        self.assertEqual(2, bot.calls)
        self.assertIsNotNone(self.db.get_preference("daily_brief_last_sent_date"))
        asyncio.run(_scheduled_daily_brief(ctx))
        self.assertEqual(2, bot.calls)

    def test_no_last_sent_update_on_total_failure(self) -> None:
        bot = _BotAlwaysFail()
        ctx = _Ctx(
            _App({"db": self.db, "timezone": "UTC", "orchestrator": _StubOrchestrator()}),
            bot,
        )
        asyncio.run(_scheduled_daily_brief(ctx))
        self.assertEqual(2, bot.calls)
        self.assertIsNone(self.db.get_preference("daily_brief_last_sent_date"))


if __name__ == "__main__":
    unittest.main()
