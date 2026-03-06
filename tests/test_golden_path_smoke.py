from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent import cli
from agent.doctor import DoctorCheck, DoctorReport
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB
from telegram_adapter.bot import _handle_message, _send_reply


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.data = None


class _FakeOrchestrator:
    def __init__(self, reply_text: str = "LLM_CHAT_REPLY") -> None:
        self.reply_text = reply_text
        self.calls: list[tuple[str, str]] = []

    def handle_message(self, text: str, *, user_id: str) -> _FakeResponse:
        self.calls.append((text, user_id))
        return _FakeResponse(self.reply_text)


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.message_id = 1
        self.replies: list[dict[str, str | None]] = []

    async def reply_text(self, text: str, parse_mode: str | None = None, **_kwargs: object) -> None:
        self.replies.append({"text": text, "parse_mode": parse_mode})


class _FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _FakeUpdate:
    def __init__(self, chat_id: int, text: str) -> None:
        self.effective_chat = _FakeChat(chat_id)
        self.effective_message = _FakeMessage(text)


class _FakeContext:
    def __init__(self, bot_data: dict[str, object]) -> None:
        self.application = type("App", (), {"bot_data": bot_data})()


class _BadRequest(Exception):
    pass


class _BadThenGoodMessage:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def reply_text(self, text: str, parse_mode: str | None = None, **kwargs: object) -> None:
        self.calls.append({"text": text, "parse_mode": parse_mode, "kwargs": dict(kwargs)})
        if len(self.calls) == 1:
            raise _BadRequest("can't parse entities")
        return


class TestGoldenPathSmoke(unittest.TestCase):
    def test_cli_smoke_commands(self) -> None:
        with patch("agent.cli.doctor_main", return_value=0):
            self.assertEqual(0, cli.main(["doctor"]))
        with patch("agent.cli._http_json", return_value=(True, {"ready": True, "message": "Ready.", "telegram": {"state": "running"}})):
            self.assertEqual(0, cli.main(["status"]))
        with patch("agent.cli._resolve_git_commit", return_value="abc1234"):
            self.assertEqual(0, cli.main(["version"]))

    def test_telegram_routing_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = _FakeOrchestrator(reply_text="LLM_CHAT_REPLY")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": SimpleNamespace(set_preference=lambda *_args, **_kwargs: None),
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": SimpleNamespace(load=lambda: {"active": False}, save=lambda _x: _x, clear=lambda: None),
                    "audit_log": None,
                }
            )

            hello = _FakeUpdate(42, "hello")
            asyncio.run(_handle_message(hello, context))
            self.assertIn("LLM_CHAT_REPLY", str(hello.effective_message.replies[-1]["text"] or ""))

            help_update = _FakeUpdate(42, "help")
            asyncio.run(_handle_message(help_update, context))
            self.assertIn("Available commands:", str(help_update.effective_message.replies[-1]["text"] or ""))

            report = DoctorReport(
                trace_id="doctor-1",
                generated_at="2026-03-06T00:00:00+00:00",
                summary_status="OK",
                checks=[DoctorCheck("a", "OK", "ok")],
                next_action=None,
                fixes_applied=[],
                support_bundle_path=None,
            )
            with patch("telegram_adapter.bot.run_doctor_report", return_value=report):
                doctor = _FakeUpdate(42, "doctor")
                asyncio.run(_handle_message(doctor, context))
            self.assertIn("Doctor: OK", str(doctor.effective_message.replies[-1]["text"] or ""))

            status = _FakeUpdate(42, "show me the status")
            asyncio.run(_handle_message(status, context))
            self.assertIn(("/status", "42"), orchestrator.calls)

            health = _FakeUpdate(42, "how is the bot health")
            asyncio.run(_handle_message(health, context))
            self.assertIn(("/health", "42"), orchestrator.calls)

            brief = _FakeUpdate(42, "what changed on my pc?")
            asyncio.run(_handle_message(brief, context))
            self.assertIn(("/brief", "42"), orchestrator.calls)

    def test_llm_unavailable_bootstrap_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDB(os.path.join(tmpdir, "agent.db"))
            schema = os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
            db.init_schema(os.path.abspath(schema))
            skills = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
            orch = Orchestrator(db=db, skills_path=skills, log_path=os.path.join(tmpdir, "agent.log"), timezone="UTC", llm_client=None)
            response = orch.handle_message("hello", "u1")
            self.assertIn("No chat model available", response.text)
            self.assertNotIn("Anthropic", response.text)
            db.close()

    def test_send_safety_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "agent.log")
            msg = _BadThenGoodMessage()
            delivered = asyncio.run(
                _send_reply(
                    message=msg,
                    log_path=log_path,
                    chat_id="42",
                    route="health",
                    text="*" * 6000,
                    trace_id="tg-42-1",
                    parse_mode="Markdown",
                )
            )
            self.assertLessEqual(len(delivered), 3900)
            self.assertEqual(2, len(msg.calls))
            self.assertIsNone(msg.calls[-1]["parse_mode"])

            with open(log_path, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
            out_rows = [row for row in rows if row.get("type") == "telegram.out"]
            self.assertTrue(out_rows)


if __name__ == "__main__":
    unittest.main()

