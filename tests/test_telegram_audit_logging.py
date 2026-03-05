from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import tempfile
import unittest

from agent.audit_log import AuditLog
from agent.ux.llm_fixit_wizard import LLMFixitWizardStore
from telegram_adapter.bot import _handle_brief_alias, _handle_message, _handle_status


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.data = None


class _FakeOrchestrator:
    def __init__(self, reply_text: str = "chat reply") -> None:
        self.reply_text = reply_text
        self.calls: list[tuple[str, str]] = []

    def handle_message(self, text: str, *, user_id: str) -> _FakeResponse:
        self.calls.append((text, user_id))
        return _FakeResponse(self.reply_text)


class _RaisingOrchestrator:
    def handle_message(self, text: str, *, user_id: str) -> _FakeResponse:
        _ = text
        _ = user_id
        raise RuntimeError("boom")


class _FakeDB:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set_preference(self, key: str, value: str) -> None:
        self.values[str(key)] = str(value)


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.message_id = 1
        self.replies: list[dict[str, str | None]] = []

    async def reply_text(self, text: str, parse_mode: str | None = None) -> None:
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


def _read_audit_rows(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines() if line.strip()]


class TestTelegramAuditLogging(unittest.TestCase):
    def test_received_then_handled_fixit_choice_routes_second_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            wizard_path = f"{tmpdir}/wizard.json"
            audit_log = AuditLog(path=audit_path)
            wizard_store = LLMFixitWizardStore(path=wizard_path)
            wizard_store.save(
                {
                    "active": True,
                    "issue_hash": "issue-2",
                    "issue_code": "openrouter_down",
                    "step": "awaiting_choice",
                    "question": "Choose an option",
                    "choices": [
                        {"id": "local_only", "label": "Use local-only", "recommended": True},
                        {"id": "repair_openrouter", "label": "Repair OpenRouter", "recommended": False},
                        {"id": "details", "label": "Show details", "recommended": False},
                    ],
                    "pending_plan": [],
                    "pending_confirm_token": None,
                    "pending_created_ts": None,
                    "pending_expires_ts": None,
                    "pending_issue_code": None,
                    "last_prompt_ts": 1,
                }
            )
            orchestrator = _FakeOrchestrator(reply_text="should not be used")
            db = _FakeDB()
            chat_id = "123456789"
            update = _FakeUpdate(int(chat_id), "2")
            captured_payload: dict[str, object] = {}

            def _llm_fixit(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
                captured_payload.update(payload)
                return True, {"ok": True, "message": "Repairing OpenRouter."}

            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": _llm_fixit,
                    "llm_fixit_store": wizard_store,
                    "audit_log": audit_log,
                }
            )

            asyncio.run(_handle_message(update, context))

            rows = _read_audit_rows(audit_path)
            actions = [str(row.get("action")) for row in rows if str(row.get("action", "")).startswith("telegram.message.")]
            self.assertEqual(["telegram.message.received", "telegram.message.handled"], actions)
            handled_row = [row for row in rows if row.get("action") == "telegram.message.handled"][-1]
            params = handled_row.get("params_redacted") if isinstance(handled_row.get("params_redacted"), dict) else {}
            self.assertEqual("fixit", params.get("route"))
            self.assertEqual("repair_openrouter", captured_payload.get("answer"))
            self.assertEqual([], orchestrator.calls)

    def test_received_then_handled_fixit_choice_route_with_numeric_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            wizard_path = f"{tmpdir}/wizard.json"
            audit_log = AuditLog(path=audit_path)
            wizard_store = LLMFixitWizardStore(path=wizard_path)
            wizard_store.save(
                {
                    "active": True,
                    "issue_hash": "issue-1",
                    "issue_code": "openrouter_down",
                    "step": "awaiting_choice",
                    "question": "Choose an option",
                    "choices": [
                        {"id": "local_only", "label": "Use local-only", "recommended": True},
                        {"id": "repair_openrouter", "label": "Repair OpenRouter", "recommended": False},
                        {"id": "details", "label": "Show details", "recommended": False},
                    ],
                    "pending_plan": [],
                    "pending_confirm_token": None,
                    "pending_created_ts": None,
                    "pending_expires_ts": None,
                    "pending_issue_code": None,
                    "last_prompt_ts": 1,
                }
            )
            orchestrator = _FakeOrchestrator(reply_text="should not be used")
            db = _FakeDB()
            chat_id = "123456789"
            update = _FakeUpdate(int(chat_id), "1")
            captured_payload: dict[str, object] = {}

            def _llm_fixit(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
                captured_payload.update(payload)
                return True, {"ok": True, "message": "Applied local-only."}

            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": _llm_fixit,
                    "llm_fixit_store": wizard_store,
                    "audit_log": audit_log,
                }
            )

            asyncio.run(_handle_message(update, context))

            rows = _read_audit_rows(audit_path)
            actions = [str(row.get("action")) for row in rows if str(row.get("action", "")).startswith("telegram.message.")]
            self.assertEqual(["telegram.message.received", "telegram.message.handled"], actions)
            handled_row = [row for row in rows if row.get("action") == "telegram.message.handled"][-1]
            params = handled_row.get("params_redacted") if isinstance(handled_row.get("params_redacted"), dict) else {}
            self.assertEqual("fixit", params.get("route"))

            self.assertEqual("local_only", captured_payload.get("answer"))
            self.assertEqual("telegram", captured_payload.get("actor"))
            self.assertEqual([], orchestrator.calls)

    def test_received_then_handled_fixit_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            wizard_path = f"{tmpdir}/wizard.json"
            audit_log = AuditLog(path=audit_path)
            wizard_store = LLMFixitWizardStore(path=wizard_path)
            wizard_store.save(
                {
                    "active": True,
                    "issue_hash": "abc",
                    "issue_code": "openrouter_down",
                    "step": "awaiting_confirm",
                    "question": "Apply?",
                    "choices": [],
                    "pending_plan": [{"id": "01", "kind": "safe_action", "action": "health.run", "reason": "test"}],
                    "pending_confirm_token": "token",
                    "pending_created_ts": 1,
                    "pending_expires_ts": 9_999_999_999,
                    "pending_issue_code": "openrouter_down",
                    "last_prompt_ts": 1,
                }
            )
            orchestrator = _FakeOrchestrator(reply_text="should not be used")
            db = _FakeDB()
            chat_id = "123456789"
            update = _FakeUpdate(int(chat_id), "yes")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "fixit applied"}),
                    "llm_fixit_store": wizard_store,
                    "audit_log": audit_log,
                }
            )

            asyncio.run(_handle_message(update, context))

            rows = _read_audit_rows(audit_path)
            actions = [str(row.get("action")) for row in rows if str(row.get("action", "")).startswith("telegram.message.")]
            self.assertEqual(["telegram.message.received", "telegram.message.handled"], actions)
            handled_row = [row for row in rows if row.get("action") == "telegram.message.handled"][-1]
            params = handled_row.get("params_redacted") if isinstance(handled_row.get("params_redacted"), dict) else {}
            self.assertEqual("fixit", params.get("route"))
            self.assertIn("chat_id_redacted", params)
            self.assertNotEqual(chat_id, str(params.get("chat_id_redacted") or ""))
            params_text = json.dumps(params, ensure_ascii=True)
            self.assertNotIn(chat_id, params_text)
            self.assertNotIn("yes", params_text.lower())
            self.assertEqual([], orchestrator.calls)

    def test_received_then_handled_chat_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            audit_log = AuditLog(path=audit_path)
            wizard_store = LLMFixitWizardStore(path=f"{tmpdir}/wizard.json")
            orchestrator = _FakeOrchestrator(reply_text="chat ok")
            db = _FakeDB()
            chat_id = "987654321"
            user_text = "show me task summary"
            update = _FakeUpdate(int(chat_id), user_text)
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "ignored"}),
                    "llm_fixit_store": wizard_store,
                    "audit_log": audit_log,
                }
            )

            asyncio.run(_handle_message(update, context))

            rows = _read_audit_rows(audit_path)
            actions = [str(row.get("action")) for row in rows if str(row.get("action", "")).startswith("telegram.message.")]
            self.assertEqual(["telegram.message.received", "telegram.message.handled"], actions)
            handled_row = [row for row in rows if row.get("action") == "telegram.message.handled"][-1]
            params = handled_row.get("params_redacted") if isinstance(handled_row.get("params_redacted"), dict) else {}
            self.assertEqual("chat", params.get("route"))
            self.assertIn("chat_id_redacted", params)
            self.assertNotEqual(chat_id, str(params.get("chat_id_redacted") or ""))
            params_text = json.dumps(params, ensure_ascii=True)
            self.assertNotIn(chat_id, params_text)
            self.assertNotIn(user_text.lower(), params_text.lower())
            self.assertEqual([(user_text, chat_id)], orchestrator.calls)

    def test_status_command_returns_status_with_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            audit_log = AuditLog(path=audit_path)
            wizard_store = LLMFixitWizardStore(path=f"{tmpdir}/wizard.json")
            orchestrator = _FakeOrchestrator(reply_text="unused")
            db = _FakeDB()
            chat_id = "12349876"
            update = _FakeUpdate(int(chat_id), "/status")
            runtime = type(
                "Runtime",
                (),
                {
                    "version": "0.2.0",
                    "git_commit": "abc123def456",
                    "started_at": datetime.now(timezone.utc) - timedelta(seconds=42),
                },
            )()
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "ignored"}),
                    "llm_fixit_store": wizard_store,
                    "audit_log": audit_log,
                    "runtime": runtime,
                }
            )

            asyncio.run(_handle_status(update, context))

            self.assertTrue(update.effective_message.replies)
            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("✅ Agent is running", reply_text)
            self.assertIn("commit abc123def456", reply_text)
            self.assertIn("uptime", reply_text)
            self.assertEqual([], orchestrator.calls)

    def test_hello_forwards_to_orchestrator_not_status_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            audit_log = AuditLog(path=audit_path)
            wizard_store = LLMFixitWizardStore(path=f"{tmpdir}/wizard.json")
            orchestrator = _FakeOrchestrator(reply_text="LLM_CHAT_REPLY")
            db = _FakeDB()
            chat_id = "246801357"
            update = _FakeUpdate(int(chat_id), "hello")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "ignored"}),
                    "llm_fixit_store": wizard_store,
                    "audit_log": audit_log,
                }
            )

            asyncio.run(_handle_message(update, context))

            self.assertTrue(update.effective_message.replies)
            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertEqual("LLM_CHAT_REPLY", reply_text)
            self.assertNotIn("✅ Agent is running", reply_text)
            self.assertEqual([("hello", chat_id)], orchestrator.calls)

    def test_help_returns_help_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            audit_log = AuditLog(path=audit_path)
            wizard_store = LLMFixitWizardStore(path=f"{tmpdir}/wizard.json")
            orchestrator = _FakeOrchestrator(reply_text="unused")
            db = _FakeDB()
            chat_id = "654321987"
            update = _FakeUpdate(int(chat_id), "help")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "ignored"}),
                    "llm_fixit_store": wizard_store,
                    "audit_log": audit_log,
                }
            )

            asyncio.run(_handle_message(update, context))

            self.assertTrue(update.effective_message.replies)
            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Try one of these:", reply_text)
            self.assertIn("/help", reply_text)
            self.assertIn("/brief", reply_text)
            self.assertEqual([], orchestrator.calls)

            rows = _read_audit_rows(audit_path)
            handled_row = [row for row in rows if row.get("action") == "telegram.message.handled"][-1]
            params = handled_row.get("params_redacted") if isinstance(handled_row.get("params_redacted"), dict) else {}
            self.assertEqual("help", params.get("route"))

    def test_unknown_orchestrator_reply_uses_improved_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            audit_log = AuditLog(path=audit_path)
            wizard_store = LLMFixitWizardStore(path=f"{tmpdir}/wizard.json")
            orchestrator = _FakeOrchestrator(
                reply_text="I didn’t understand that. Try /brief, or ask “anything new on my PC?”"
            )
            db = _FakeDB()
            chat_id = "1122334455"
            update = _FakeUpdate(int(chat_id), "asdkjhasdkjh")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "ignored"}),
                    "llm_fixit_store": wizard_store,
                    "audit_log": audit_log,
                }
            )

            asyncio.run(_handle_message(update, context))

            self.assertTrue(update.effective_message.replies)
            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("1)", reply_text)
            self.assertIn("2)", reply_text)
            self.assertIn("3)", reply_text)
            self.assertIn("Reply 1, 2, or 3.", reply_text)
            self.assertEqual([], orchestrator.calls)

            rows = _read_audit_rows(audit_path)
            handled_row = [row for row in rows if row.get("action") == "telegram.message.handled"][-1]
            params = handled_row.get("params_redacted") if isinstance(handled_row.get("params_redacted"), dict) else {}
            self.assertEqual("fallback", params.get("route"))

    def test_brief_alias_routes_to_brief_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            audit_log = AuditLog(path=audit_path)
            orchestrator = _FakeOrchestrator(reply_text="brief reply")
            db = _FakeDB()
            chat_id = "99887766"
            update = _FakeUpdate(int(chat_id), "/breif")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": db,
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": audit_log,
                }
            )

            asyncio.run(_handle_brief_alias(update, context))

            self.assertEqual([("/brief", chat_id)], orchestrator.calls)
            self.assertTrue(update.effective_message.replies)
            self.assertEqual("brief reply", update.effective_message.replies[-1]["text"])
            rows = _read_audit_rows(audit_path)
            handled_row = [row for row in rows if row.get("action") == "telegram.message.handled"][-1]
            params = handled_row.get("params_redacted") if isinstance(handled_row.get("params_redacted"), dict) else {}
            self.assertEqual("alias", params.get("route"))


if __name__ == "__main__":
    unittest.main()


def test_orchestrator_exception_logs_traceback_and_returns_fallback(caplog):  # type: ignore[no-untyped-def]
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_path = f"{tmpdir}/audit.jsonl"
        audit_log = AuditLog(path=audit_path)
        wizard_store = LLMFixitWizardStore(path=f"{tmpdir}/wizard.json")
        orchestrator = _RaisingOrchestrator()
        db = _FakeDB()
        chat_id = "9988776655"
        update = _FakeUpdate(int(chat_id), "hello")
        context = _FakeContext(
            {
                "orchestrator": orchestrator,
                "db": db,
                "log_path": f"{tmpdir}/agent.log",
                "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "ignored"}),
                "llm_fixit_store": wizard_store,
                "audit_log": audit_log,
            }
        )

        caplog.set_level(logging.ERROR, logger="telegram_adapter.bot")
        asyncio.run(_handle_message(update, context))

        assert update.effective_message.replies
        reply_text = str(update.effective_message.replies[-1]["text"] or "")
        assert "I hit an internal error" in reply_text
        assert "(trace " in reply_text

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert "boom" in logged
        assert "Traceback (most recent call last)" in logged
