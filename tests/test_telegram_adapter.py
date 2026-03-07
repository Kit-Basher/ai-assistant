from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from telegram_adapter.bot import _handle_message


class _RaisingLLM:
    def chat(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("telegram adapter must not call llm directly")


class _FakeResponse:
    def __init__(self, text: str, data: dict[str, object] | None = None) -> None:
        self.text = text
        self.data = data


class _FakeOrchestrator:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []
        self.llm_client = _RaisingLLM()

    def handle_message(self, text: str, *, user_id: str) -> _FakeResponse:
        self.calls.append((text, user_id))
        return self.response


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

    async def reply_text(self, text: str, parse_mode: str | None = None, **_kwargs) -> None:  # type: ignore[no-untyped-def]
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


def _read_log_rows(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines() if line.strip()]


class TestTelegramAdapter(unittest.TestCase):
    def test_text_message_routes_through_orchestrator_and_sends_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            orchestrator = _FakeOrchestrator(_FakeResponse("adapter reply"))
            update = _FakeUpdate(42, "tell me a joke")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            asyncio.run(_handle_message(update, context))

            self.assertEqual([("tell me a joke", "42")], orchestrator.calls)
            self.assertEqual("adapter reply", str(update.effective_message.replies[-1]["text"] or ""))
            event_types = [str(row.get("type") or "") for row in _read_log_rows(log_path)]
            self.assertIn("incoming_message", event_types)
            self.assertIn("orchestrator_call", event_types)
            self.assertIn("response_sent", event_types)

    def test_error_response_from_orchestrator_gets_user_friendly_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            orchestrator = _FakeOrchestrator(
                _FakeResponse(
                    "",
                    data={
                        "ok": False,
                        "error_kind": "llm_unavailable",
                    },
                )
            )
            update = _FakeUpdate(42, "tell me a joke")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            asyncio.run(_handle_message(update, context))

            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Agent could not complete the request.", reply_text)
            self.assertIn("Reason: llm_unavailable", reply_text)


if __name__ == "__main__":
    unittest.main()
