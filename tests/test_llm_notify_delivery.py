from __future__ import annotations

import unittest

from agent.llm.notify_delivery import LocalTarget, TelegramTarget


class TestLLMNotifyDelivery(unittest.TestCase):
    def test_telegram_target_delivers_when_configured(self) -> None:
        calls: list[tuple[str, str, str]] = []

        def _send(token: str, chat_id: str, text: str) -> None:
            calls.append((token, chat_id, text))

        target = TelegramTarget(token="token", chat_id="chat", send_fn=_send, enabled=True)
        result = target.deliver({"message": "hello"})
        self.assertTrue(result.ok)
        self.assertEqual("telegram", result.delivered_to)
        self.assertEqual("sent", result.reason)
        self.assertEqual([("token", "chat", "hello")], calls)

    def test_telegram_target_reports_not_configured(self) -> None:
        target = TelegramTarget(token="", chat_id="", send_fn=lambda *_args: None, enabled=True)
        result = target.deliver({"message": "hello"})
        self.assertFalse(result.ok)
        self.assertEqual("telegram_not_configured_or_no_chat", result.reason)

    def test_local_target_always_delivers_when_enabled(self) -> None:
        target = LocalTarget(enabled=True)
        result = target.deliver({"message": "hello"})
        self.assertTrue(result.ok)
        self.assertEqual("local", result.delivered_to)
        self.assertEqual("sent_local", result.reason)


if __name__ == "__main__":
    unittest.main()
