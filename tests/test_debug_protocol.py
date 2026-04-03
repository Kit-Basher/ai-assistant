import unittest
from datetime import datetime, timedelta, timezone

from agent.debug_protocol import DebugProtocol


class TestDebugProtocol(unittest.TestCase):
    def test_trigger_rate_limit(self) -> None:
        protocol = DebugProtocol(threshold=3, window_seconds=300, cooldown_seconds=600)
        base = datetime(2026, 2, 4, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(3):
            triggered = protocol.record_reminder("chat-1", "ping", base + timedelta(seconds=i))
            self.assertFalse(triggered)
        triggered = protocol.record_reminder("chat-1", "ping", base + timedelta(seconds=10))
        self.assertTrue(triggered)
        triggered = protocol.record_reminder("chat-1", "ping", base + timedelta(seconds=20))
        self.assertFalse(triggered)
        for i in range(3):
            triggered = protocol.record_reminder(
                "chat-1",
                "ping",
                base + timedelta(seconds=620 + i),
            )
            self.assertFalse(triggered)
        triggered = protocol.record_reminder(
            "chat-1",
            "ping",
            base + timedelta(seconds=630),
        )
        self.assertTrue(triggered)

    def test_audit_trigger(self) -> None:
        protocol = DebugProtocol(threshold=3, window_seconds=300, cooldown_seconds=600)
        base = datetime(2026, 2, 4, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(3):
            triggered = protocol.record_audit_event(
                "reminder_send",
                "1",
                "failed",
                base + timedelta(seconds=i),
            )
            self.assertFalse(triggered)
        triggered = protocol.record_audit_event(
            "reminder_send",
            "1",
            "failed",
            base + timedelta(seconds=30),
        )
        self.assertTrue(triggered)


if __name__ == "__main__":
    unittest.main()
