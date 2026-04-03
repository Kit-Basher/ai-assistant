from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.config import Config
from memory.db import MemoryDB


class _FakeRegistry:
    def to_document(self):
        return {"providers": {}, "models": {}, "defaults": {}}


class _FakeRouter:
    def __init__(self, *_args, **_kwargs):
        self.registry = _FakeRegistry()

    def doctor_snapshot(self):
        return {"providers": [], "models": []}

    def usage_stats_snapshot(self):
        return {}


class _FakeScout:
    def __init__(self):
        self.last_notify_sender = None
        self.closed = False

    def run(self, *, registry_document, router_snapshot, usage_stats_snapshot, notify_sender):
        _ = registry_document
        _ = router_snapshot
        _ = usage_stats_snapshot
        self.last_notify_sender = notify_sender
        if notify_sender is not None:
            notify_sender("test message", [])
            notified = 1
        else:
            notified = 0
        return {
            "ok": True,
            "error": None,
            "fetched_trending": 0,
            "suggestions": [],
            "new_suggestions": [],
            "notified": notified,
        }

    def close(self):
        self.closed = True


def _config(db_path: str) -> Config:
    return Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path="/tmp/skills",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=None,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
    )


class TestScheduledModelScout(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        db = MemoryDB(self.db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        db.init_schema(schema_path)
        db.close()
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_run_once_skips_notifications_when_chat_not_configured(self) -> None:
        fake_scout = _FakeScout()
        os.environ["TELEGRAM_BOT_TOKEN"] = "123:token"

        with patch("agent.scheduled_model_scout.load_config", return_value=_config(self.db_path)), patch(
            "agent.scheduled_model_scout.build_model_scout", return_value=fake_scout
        ), patch("agent.scheduled_model_scout.LLMRouter", _FakeRouter), patch(
            "agent.scheduled_model_scout._send_telegram_message"
        ) as mocked_send:
            from agent.scheduled_model_scout import run_once

            result = run_once()

        self.assertEqual(0, result)
        self.assertTrue(fake_scout.closed)
        self.assertIsNone(fake_scout.last_notify_sender)
        self.assertFalse(mocked_send.called)

    def test_run_once_sends_notification_when_token_and_chat_exist(self) -> None:
        db = MemoryDB(self.db_path)
        db.set_preference("telegram_chat_id", "chat-1")
        db.close()

        fake_scout = _FakeScout()
        os.environ["TELEGRAM_BOT_TOKEN"] = "123:token"

        with patch("agent.scheduled_model_scout.load_config", return_value=_config(self.db_path)), patch(
            "agent.scheduled_model_scout.build_model_scout", return_value=fake_scout
        ), patch("agent.scheduled_model_scout.LLMRouter", _FakeRouter), patch(
            "agent.scheduled_model_scout._send_telegram_message"
        ) as mocked_send:
            from agent.scheduled_model_scout import run_once

            result = run_once()

        self.assertEqual(0, result)
        self.assertTrue(fake_scout.closed)
        self.assertIsNotNone(fake_scout.last_notify_sender)
        self.assertTrue(mocked_send.called)


if __name__ == "__main__":
    unittest.main()
