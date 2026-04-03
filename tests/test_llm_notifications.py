from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from agent.llm.notifications import (
    NotificationStore,
    build_notification_from_diff,
    build_notification_from_state_diff,
    sanitize_notification_text,
    should_send,
)


def _before_doc() -> dict[str, object]:
    return {
        "defaults": {
            "routing_mode": "auto",
            "default_provider": None,
            "default_model": None,
            "allow_remote_fallback": True,
        },
        "providers": {
            "openrouter": {
                "enabled": True,
                "local": False,
                "base_url": "https://openrouter.ai/api/v1",
                "chat_path": "/chat/completions",
                "default_headers": {},
                "default_query_params": {},
            }
        },
        "models": {
            "ollama:llama3": {
                "provider": "ollama",
                "enabled": True,
                "available": True,
                "capabilities": ["chat"],
            }
        },
    }


def _after_doc() -> dict[str, object]:
    return {
        "defaults": {
            "routing_mode": "prefer_local_lowest_cost_capable",
            "default_provider": "ollama",
            "default_model": "ollama:qwen2.5:3b-instruct",
            "allow_remote_fallback": True,
        },
        "providers": {
            "openrouter": {
                "enabled": False,
                "local": False,
                "base_url": "https://openrouter.ai/api",
                "chat_path": "/chat/completions",
                "default_headers": {},
                "default_query_params": {},
            }
        },
        "models": {
            "ollama:llama3": {
                "provider": "ollama",
                "enabled": True,
                "available": False,
                "capabilities": ["chat"],
            }
        },
    }


class TestLLMNotifications(unittest.TestCase):
    def test_no_changes_produces_no_notification(self) -> None:
        message, dedupe_hash, lines = build_notification_from_diff(
            _before_doc(),
            _before_doc(),
            reasons=["noop"],
            modified_ids=[],
        )
        self.assertEqual("", message)
        self.assertEqual("", dedupe_hash)
        self.assertEqual([], lines)

    def test_changes_produce_stable_message_and_single_record(self) -> None:
        message, dedupe_hash, _lines = build_notification_from_diff(
            _before_doc(),
            _after_doc(),
            reasons=["repaired missing defaults"],
            modified_ids=["defaults:default_model", "provider:openrouter", "model:ollama:llama3"],
        )
        self.assertIn("LLM Autopilot updated configuration", message)
        self.assertIn("Defaults: default_model -> ollama:qwen2.5:3b-instruct", message)
        self.assertIn("Provider openrouter: disabled", message)
        self.assertIn("Model ollama:llama3: marked unroutable", message)
        self.assertTrue(dedupe_hash)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotificationStore(path=f"{tmpdir}/notifications.json", max_recent=50, max_age_days=0)
            store.append(
                ts=1_000,
                message=message,
                dedupe_hash=dedupe_hash,
                delivered_to="none",
                deferred=False,
                outcome="skipped",
                reason="telegram_not_configured_or_no_chat",
                modified_ids=["defaults:default_model"],
                mark_sent=True,
            )
            rows = store.recent(limit=10)
            self.assertEqual(1, len(rows))
            self.assertEqual(dedupe_hash, rows[0]["dedupe_hash"])

    def test_rate_limit_prevents_second_send(self) -> None:
        first = should_send(
            now_epoch=1_000,
            last_sent_ts=900,
            last_sent_hash="abc",
            message_hash="def",
            enabled=True,
            rate_limit_seconds=300,
            dedupe_window_seconds=0,
            quiet_start_hour=None,
            quiet_end_hour=None,
            timezone_name="UTC",
        )
        self.assertFalse(first["send"])
        self.assertEqual("rate_limited", first["reason"])

    def test_dedupe_hash_prevents_repeat(self) -> None:
        decision = should_send(
            now_epoch=2_000,
            last_sent_ts=1_500,
            last_sent_hash="same-hash",
            message_hash="same-hash",
            enabled=True,
            rate_limit_seconds=0,
            dedupe_window_seconds=3_600,
            quiet_start_hour=None,
            quiet_end_hour=None,
            timezone_name="UTC",
        )
        self.assertFalse(decision["send"])
        self.assertEqual("dedupe_hash_match", decision["reason"])

    def test_quiet_hours_defer_delivery(self) -> None:
        decision = should_send(
            now_epoch=1_706_836_800,  # 2024-02-01T02:00:00Z
            last_sent_ts=None,
            last_sent_hash=None,
            message_hash="x",
            enabled=True,
            rate_limit_seconds=0,
            dedupe_window_seconds=0,
            quiet_start_hour=1,
            quiet_end_hour=6,
            timezone_name="UTC",
        )
        self.assertFalse(decision["send"])
        self.assertTrue(decision["deferred"])
        self.assertEqual("quiet_hours", decision["reason"])

    def test_prune_by_age_then_max_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch("agent.llm.notifications.time.time", return_value=2_000_000):
            store = NotificationStore(
                path=f"{tmpdir}/notifications.json",
                max_recent=50,
                max_items=2,
                max_age_days=30,
                compact=False,
            )
            # Too old: pruned by age first.
            store.append(
                ts=2_000_000 - (31 * 86400),
                message="old",
                dedupe_hash="old",
                delivered_to="local",
                deferred=False,
                outcome="sent",
                reason="sent_local",
                modified_ids=["defaults:default_model"],
            )
            store.append(
                ts=1_999_000,
                message="one",
                dedupe_hash="h1",
                delivered_to="local",
                deferred=False,
                outcome="sent",
                reason="sent_local",
                modified_ids=["defaults:default_model"],
            )
            store.append(
                ts=1_999_100,
                message="two",
                dedupe_hash="h2",
                delivered_to="local",
                deferred=False,
                outcome="sent",
                reason="sent_local",
                modified_ids=["defaults:default_model"],
            )
            store.append(
                ts=1_999_200,
                message="three",
                dedupe_hash="h3",
                delivered_to="local",
                deferred=False,
                outcome="sent",
                reason="sent_local",
                modified_ids=["defaults:default_model"],
            )
            rows = store.recent(limit=10)
            self.assertEqual(2, len(rows))
            self.assertEqual("h3", rows[0]["dedupe_hash"])
            self.assertEqual("h2", rows[1]["dedupe_hash"])
            status = store.status()
            self.assertEqual(2, status["stored_count"])
            self.assertTrue(status["pruned_count_total"] >= 2)

    def test_compaction_is_deterministic_for_hash_and_no_changes_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch("agent.llm.notifications.time.time", return_value=3_000_000):
            store = NotificationStore(
                path=f"{tmpdir}/notifications.json",
                max_recent=50,
                max_items=100,
                max_age_days=0,
                compact=True,
            )
            seed_rows = [
                {
                    "ts": 2_999_900 + idx,
                    "message": f"m{idx}",
                    "dedupe_hash": "same-hash",
                    "delivered_to": "local",
                    "deferred": False,
                    "outcome": "sent",
                    "reason": "sent_local",
                    "modified_ids": [],
                }
                for idx in range(8)
            ] + [
                {
                    "ts": 2_999_980 + idx,
                    "message": f"noop-{idx}",
                    "dedupe_hash": f"noop-{idx}",
                    "delivered_to": "none",
                    "deferred": False,
                    "outcome": "skipped",
                    "reason": "no_changes",
                    "modified_ids": [],
                }
                for idx in range(6)
            ]
            state = {
                "schema_version": 1,
                "notifications": seed_rows,
                "last_sent_ts": None,
                "last_sent_hash": None,
            }
            first = store.save(state)
            second = store.save({"schema_version": 1, "notifications": list(reversed(seed_rows))})
            self.assertEqual(first["notifications"], second["notifications"])
            grouped_hash_rows = [row for row in first["notifications"] if row["dedupe_hash"] == "same-hash"]
            grouped_no_changes = [row for row in first["notifications"] if row["reason"] == "no_changes"]
            self.assertEqual(3, len(grouped_hash_rows))
            self.assertEqual(3, len(grouped_no_changes))

    def test_sanitizer_redacts_tokens_and_query_secrets(self) -> None:
        raw = "Authorization: Bearer sk-abc12345678901234567890 and ?token=abc123&key=def456"
        redacted = sanitize_notification_text(raw)
        self.assertNotIn("sk-abc12345678901234567890", redacted)
        self.assertNotIn("?token=abc123", redacted)
        self.assertIn("Authorization: [REDACTED]", redacted)
        self.assertIn("?token=[REDACTED]", redacted)
        self.assertIn("&key=[REDACTED]", redacted)

    def test_build_notification_from_state_diff_redacts_reason_and_values(self) -> None:
        before = {
            "defaults": {
                "routing_mode": "auto",
                "default_provider": None,
                "default_model": None,
                "allow_remote_fallback": True,
            },
            "providers": {},
            "models": {},
        }
        after = {
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "openrouter?token=abc123",
                "default_model": "sk-abc12345678901234567890",
                "allow_remote_fallback": True,
            },
            "providers": {},
            "models": {},
        }
        built = build_notification_from_state_diff(
            before,
            after,
            reasons=["provider returned Bearer badtoken123456789"],
        )
        message = str(built["message"])
        self.assertNotIn("sk-abc12345678901234567890", message)
        self.assertNotIn("badtoken123456789", message)
        self.assertIn("[REDACTED]", message)


if __name__ == "__main__":
    unittest.main()
