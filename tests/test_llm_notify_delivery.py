from __future__ import annotations

import unittest

from agent.capability_policy import TrustedInvocationContext
from agent.llm.notify_delivery import LocalTarget, TelegramTarget


def _context(capability_id: str, executor_id: str) -> dict[str, object]:
    return TrustedInvocationContext(
        capability_id=capability_id,
        executor_id=executor_id,
        authorization_decision_id="authz-test123",
        plan_fingerprint="plan-fingerprint",
        operation_id="operation-test",
    ).to_dict()


class TestLLMNotifyDelivery(unittest.TestCase):
    def test_telegram_target_delivers_when_configured(self) -> None:
        calls: list[tuple[str, str, str]] = []

        def _send(token: str, chat_id: str, text: str) -> None:
            calls.append((token, chat_id, text))

        target = TelegramTarget(token="token", chat_id="chat", send_fn=_send, enabled=True)
        result = target.deliver(
            {"message": "hello"},
            trusted_invocation_context=_context("notification.external.send", "operator.notification.telegram.send.v1"),
            plan_fingerprint="plan-fingerprint",
        )
        self.assertTrue(result.ok)
        self.assertEqual("telegram", result.delivered_to)
        self.assertEqual("sent", result.reason)
        self.assertEqual([("token", "chat", "hello")], calls)

    def test_telegram_target_blocks_direct_send_without_trusted_context(self) -> None:
        calls: list[tuple[str, str, str]] = []
        target = TelegramTarget(token="token", chat_id="chat", send_fn=lambda *args: calls.append(args), enabled=True)
        result = target.deliver({"message": "hello"})
        self.assertFalse(result.ok)
        self.assertEqual("generic_bypass_blocked", result.reason)
        self.assertEqual([], calls)

    def test_telegram_target_reports_not_configured(self) -> None:
        target = TelegramTarget(token="", chat_id="", send_fn=lambda *_args: None, enabled=True)
        result = target.deliver({"message": "hello"})
        self.assertFalse(result.ok)
        self.assertEqual("telegram_not_configured_or_no_chat", result.reason)

    def test_local_target_always_delivers_when_enabled(self) -> None:
        target = LocalTarget(enabled=True)
        result = target.deliver(
            {"message": "hello"},
            trusted_invocation_context=_context("notification.local.send", "operator.notification.local.send.v1"),
            plan_fingerprint="plan-fingerprint",
        )
        self.assertTrue(result.ok)
        self.assertEqual("local", result.delivered_to)
        self.assertEqual("sent_local", result.reason)

    def test_local_target_blocks_direct_send_without_trusted_context(self) -> None:
        target = LocalTarget(enabled=True)
        result = target.deliver({"message": "hello"})
        self.assertFalse(result.ok)
        self.assertEqual("generic_bypass_blocked", result.reason)


if __name__ == "__main__":
    unittest.main()
