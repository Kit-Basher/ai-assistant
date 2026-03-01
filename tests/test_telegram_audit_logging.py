from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from agent.audit_log import AuditLog
from agent.ux.llm_fixit_wizard import LLMFixitWizardStore
from telegram_adapter.bot import _handle_message


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
            user_text = "hello there"
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


if __name__ == "__main__":
    unittest.main()
