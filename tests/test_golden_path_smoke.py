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
from agent.public_chat import build_no_llm_public_message
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
        with patch(
            "agent.cli._load_ready_status_payload",
            return_value=(
                True,
                {
                    "ok": True,
                    "ready": True,
                    "message": "Ready.",
                    "runtime_mode": "READY",
                    "runtime_status": {
                        "runtime_mode": "READY",
                        "summary": "Ready.",
                    },
                    "telegram": {"state": "running"},
                },
            ),
        ), patch(
            "agent.cli._load_runtime_status_payload",
            return_value=(False, "runtime_unavailable"),
        ), patch(
            "agent.cli._load_runtime_history_payload",
            return_value=(False, "runtime_history_unavailable"),
        ), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ):
            self.assertEqual(0, cli.main(["status"]))
        with patch("agent.cli._resolve_git_commit", return_value="abc1234"):
            self.assertEqual(0, cli.main(["version"]))

    def test_telegram_routing_smoke(self) -> None:
        class _Runtime:
            version = "0.2.0"
            git_commit = "abc123def456"

            def ready_status(self) -> dict[str, object]:
                return {
                    "ok": True,
                    "ready": True,
                    "runtime_mode": "READY",
                    "runtime_status": {
                        "runtime_mode": "READY",
                        "summary": "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
                    },
                    "telegram": {"state": "running"},
                }

            def llm_status(self) -> dict[str, object]:
                return {
                    "default_provider": "ollama",
                    "default_model": "ollama:qwen2.5:3b-instruct",
                    "resolved_default_model": "ollama:qwen2.5:3b-instruct",
                    "active_provider_health": {"status": "ok"},
                    "active_model_health": {"status": "ok"},
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = _FakeOrchestrator(reply_text="LLM_CHAT_REPLY")

            async def _fake_handle_telegram_text_via_local_api(**kwargs):  # type: ignore[no-untyped-def]
                text = str(kwargs.get("text") or "").strip().lower()
                chat_id = str(kwargs.get("chat_id") or "")
                if text == "help":
                    return {
                        "ok": True,
                        "text": "Available commands:\n\ndoctor\nsetup\nstatus\nhealth\nbrief\nmemory",
                        "route": "help",
                        "selected_route": "help",
                        "handler_name": "test_bridge",
                        "used_llm": False,
                        "used_memory": False,
                        "used_runtime_state": True,
                        "used_tools": [],
                        "legacy_compatibility": False,
                        "generic_fallback_used": False,
                        "generic_fallback_reason": None,
                    }
                if text == "doctor":
                    return {
                        "ok": True,
                        "text": "Doctor: OK",
                        "route": "doctor",
                        "selected_route": "doctor",
                        "handler_name": "test_bridge",
                        "used_llm": False,
                        "used_memory": False,
                        "used_runtime_state": True,
                        "used_tools": [],
                        "legacy_compatibility": False,
                        "generic_fallback_used": False,
                        "generic_fallback_reason": None,
                    }
                if "show me the status" in text:
                    return {
                        "ok": True,
                        "text": "Ready.\nruntime_mode: READY\ntelegram: running",
                        "route": "status",
                        "selected_route": "status",
                        "handler_name": "test_bridge",
                        "used_llm": False,
                        "used_memory": False,
                        "used_runtime_state": True,
                        "used_tools": [],
                        "legacy_compatibility": False,
                        "generic_fallback_used": False,
                        "generic_fallback_reason": None,
                    }
                if "health" in text:
                    orchestrator.handle_message("/health", user_id=chat_id)
                    return {
                        "ok": True,
                        "text": "Health route ok",
                        "route": "health",
                        "selected_route": "health",
                        "handler_name": "test_bridge",
                        "used_llm": False,
                        "used_memory": False,
                        "used_runtime_state": True,
                        "used_tools": [],
                        "legacy_compatibility": False,
                        "generic_fallback_used": False,
                        "generic_fallback_reason": None,
                    }
                if "what changed on my pc" in text:
                    orchestrator.handle_message("/brief", user_id=chat_id)
                    return {
                        "ok": True,
                        "text": "Brief route ok",
                        "route": "brief",
                        "selected_route": "brief",
                        "handler_name": "test_bridge",
                        "used_llm": False,
                        "used_memory": False,
                        "used_runtime_state": True,
                        "used_tools": [],
                        "legacy_compatibility": False,
                        "generic_fallback_used": False,
                        "generic_fallback_reason": None,
                    }
                return {
                    "ok": True,
                    "text": "LLM_CHAT_REPLY",
                    "route": "generic_chat",
                    "selected_route": "generic_chat",
                    "handler_name": "test_bridge",
                    "used_llm": True,
                    "used_memory": False,
                    "used_runtime_state": False,
                    "used_tools": [],
                    "legacy_compatibility": False,
                    "generic_fallback_used": False,
                    "generic_fallback_reason": None,
                }

            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": SimpleNamespace(set_preference=lambda *_args, **_kwargs: None),
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": SimpleNamespace(load=lambda: {"active": False}, save=lambda _x: _x, clear=lambda: None),
                    "audit_log": None,
                    "runtime": _Runtime(),
                }
            )

            with patch(
                "telegram_adapter.bot._handle_telegram_text_via_local_api",
                side_effect=_fake_handle_telegram_text_via_local_api,
            ):
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
                with patch("agent.telegram_bridge.run_doctor_report", return_value=report):
                    doctor = _FakeUpdate(42, "doctor")
                    asyncio.run(_handle_message(doctor, context))
                self.assertIn("Doctor: OK", str(doctor.effective_message.replies[-1]["text"] or ""))

                status = _FakeUpdate(42, "show me the status")
                asyncio.run(_handle_message(status, context))
                status_text = str(status.effective_message.replies[-1]["text"] or "")
                self.assertIn("runtime_mode: READY", status_text)
                self.assertNotIn("ENABLE_WRITES", status_text)

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
            self.assertEqual(build_no_llm_public_message(), response.text)
            self.assertNotIn("No chat model available", response.text)
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
