import unittest
from datetime import datetime, timezone

from agent.daily_brief import parse_hhmm, should_send_daily_brief


class TestDailyBriefLogic(unittest.TestCase):
    def test_parse_hhmm(self) -> None:
        self.assertEqual((9, 30), parse_hhmm("09:30"))
        self.assertIsNone(parse_hhmm("25:00"))

    def test_should_send_daily_brief(self) -> None:
        now = datetime(2026, 2, 12, 16, 5, tzinfo=timezone.utc)
        decision = should_send_daily_brief(
            now_utc=now,
            timezone_name="UTC",
            enabled=True,
            local_time_hhmm="16:00",
            last_sent_local_date=None,
        )
        self.assertTrue(decision.should_send)
        self.assertEqual("send", decision.reason)

        already = should_send_daily_brief(
            now_utc=now,
            timezone_name="UTC",
            enabled=True,
            local_time_hhmm="16:00",
            last_sent_local_date="2026-02-12",
        )
        self.assertFalse(already.should_send)
        self.assertEqual("already_sent_today", already.reason)

    def test_quiet_mode_and_service_gate(self) -> None:
        now = datetime(2026, 2, 12, 16, 5, tzinfo=timezone.utc)
        quiet_skip = should_send_daily_brief(
            now_utc=now,
            timezone_name="UTC",
            enabled=True,
            local_time_hhmm="16:00",
            last_sent_local_date=None,
            quiet_mode=True,
            disk_delta_mb=10.0,
            disk_delta_threshold_mb=100.0,
            service_unhealthy=False,
            has_due_open_loops=False,
        )
        self.assertFalse(quiet_skip.should_send)
        self.assertEqual("quiet_no_signals", quiet_skip.reason)

        svc_gate_skip = should_send_daily_brief(
            now_utc=now,
            timezone_name="UTC",
            enabled=True,
            local_time_hhmm="16:00",
            last_sent_local_date=None,
            only_send_if_service_unhealthy=True,
            service_unhealthy=False,
        )
        self.assertFalse(svc_gate_skip.should_send)
        self.assertEqual("service_healthy_gate", svc_gate_skip.reason)


if __name__ == "__main__":
    unittest.main()
