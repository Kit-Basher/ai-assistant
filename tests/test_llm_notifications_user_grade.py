from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config


def _config(registry_path: str, db_path: str, **overrides: object) -> Config:
    base = Config(
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
        llm_registry_path=registry_path,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
        llm_health_state_path=os.path.join(os.path.dirname(db_path), "llm_health_state.json"),
        llm_automation_enabled=False,
        model_scout_state_path=os.path.join(os.path.dirname(db_path), "model_scout_state.json"),
        autopilot_notify_store_path=os.path.join(os.path.dirname(db_path), "llm_notifications.json"),
        llm_notifications_max_age_days=0,
        llm_notifications_compact=False,
    )
    return base.__class__(**{**base.__dict__, **overrides})


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload or {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self.status_code = status
        self.content_type = "application/json"
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        _ = cache_control
        self.status_code = status
        self.content_type = content_type
        self.body = body

    def _read_json(self) -> dict[str, object]:
        return self._payload


class TestLLMNotificationsUserGrade(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_chat_meta_includes_autopilot_ops_summary(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._notification_store.append(  # type: ignore[attr-defined]
            ts=1_500,
            message="LLM Autopilot updated configuration\n- Defaults: default_model -> ollama:qwen2.5:3b-instruct",
            dedupe_hash="hash-new",
            delivered_to="local",
            deferred=False,
            outcome="sent",
            reason="sent_local",
            modified_ids=["defaults:default_model"],
            mark_sent=True,
        )
        router_result = {
            "ok": True,
            "text": "Grounded response",
            "provider": "test",
            "model": "test-model",
            "fallback_used": False,
            "attempts": [],
            "duration_ms": 1,
        }
        with patch("agent.api_server.route_inference", return_value=router_result), patch(
            "agent.api_server.time.time", return_value=1_000
        ):
            ok, body = runtime.chat({"messages": [{"role": "user", "content": "hello"}]})
        self.assertTrue(ok)
        autopilot = body["meta"]["autopilot"]
        self.assertEqual(1, autopilot["since_last_user_message"])
        self.assertEqual("hash-new", autopilot["last_notification"]["hash"])
        self.assertEqual("LLM Autopilot updated configuration", autopilot["last_notification"]["title"])
        self.assertNotIn("body", autopilot["last_notification"])

    def test_last_change_selects_latest_actionable_notification(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._notification_store.append(  # type: ignore[attr-defined]
            ts=10,
            message="LLM Autopilot updated configuration\n- noop",
            dedupe_hash="h-noop",
            delivered_to="none",
            deferred=False,
            outcome="skipped",
            reason="no_changes",
            modified_ids=[],
        )
        runtime._notification_store.append(  # type: ignore[attr-defined]
            ts=11,
            message="LLM Autopilot updated configuration\n- rate",
            dedupe_hash="h-rate",
            delivered_to="none",
            deferred=False,
            outcome="skipped",
            reason="rate_limited",
            modified_ids=[],
        )
        runtime._notification_store.append(  # type: ignore[attr-defined]
            ts=12,
            message=(
                "LLM Autopilot updated configuration\n"
                "- Defaults: default_model -> ollama:qwen2.5:3b-instruct\n"
                "- Provider openrouter: enabled -> false\n"
                "- Model ollama:llama3: available -> false"
            ),
            dedupe_hash="h-action",
            delivered_to="local",
            deferred=False,
            outcome="sent",
            reason="sent_local",
            modified_ids=["model:ollama:llama3", "defaults:default_model", "provider:openrouter"],
            mark_sent=True,
        )
        runtime._notification_store.append(  # type: ignore[attr-defined]
            ts=13,
            message="LLM Autopilot updated configuration\n- dedupe",
            dedupe_hash="h-dedupe",
            delivered_to="none",
            deferred=False,
            outcome="skipped",
            reason="dedupe_hash_match",
            modified_ids=[],
        )

        payload = runtime.llm_notifications_last_change()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["found"])
        row = payload["last_change"]
        self.assertEqual("h-action", row["hash"])
        self.assertEqual(["default_model"], row["diff_summary"]["defaults_changed"])
        self.assertEqual(["openrouter"], row["diff_summary"]["providers_changed"])
        self.assertEqual(["ollama:llama3"], row["diff_summary"]["models_changed"])
        self.assertEqual(
            [
                "Review defaults in the Setup tab to confirm the active provider/model.",
                "Open Providers and verify connectivity, auth, and enabled states.",
                "Run /llm/health/run to verify model health and routability.",
            ],
            row["suggested_next_steps"],
        )

    def test_mark_read_updates_unread_count_deterministically(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        for ts, dedupe_hash in [(101, "h1"), (102, "h2"), (103, "h3")]:
            runtime._notification_store.append(  # type: ignore[attr-defined]
                ts=ts,
                message=f"LLM Autopilot updated configuration\n- {dedupe_hash}",
                dedupe_hash=dedupe_hash,
                delivered_to="local",
                deferred=False,
                outcome="sent",
                reason="sent_local",
                modified_ids=[],
            )

        status_before = runtime.llm_notifications_status()["status"]
        self.assertEqual(3, status_before["unread_count"])
        ok, mark_payload = runtime.llm_notifications_mark_read({"hash": "h2"})
        self.assertTrue(ok)
        self.assertEqual("h2", mark_payload["status"]["last_read_hash"])
        self.assertEqual(1, mark_payload["status"]["unread_count"])
        status_after = runtime.llm_notifications_status()["status"]
        self.assertEqual(1, status_after["unread_count"])

        bad_ok, bad_payload = runtime.llm_notifications_mark_read({"hash": "missing"})
        self.assertFalse(bad_ok)
        self.assertEqual("hash_not_found", bad_payload["error"])

    def test_last_change_and_mark_read_endpoints(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._notification_store.append(  # type: ignore[attr-defined]
            ts=1_000,
            message="LLM Autopilot updated configuration\n- Defaults: default_model -> ollama:qwen2.5:3b",
            dedupe_hash="h-endpoint",
            delivered_to="local",
            deferred=False,
            outcome="sent",
            reason="sent_local",
            modified_ids=["defaults:default_model"],
        )
        get_handler = _HandlerForTest(runtime, "/llm/notifications/last_change")
        get_handler.do_GET()
        self.assertEqual(200, get_handler.status_code)
        get_payload = json.loads(get_handler.body.decode("utf-8"))
        self.assertTrue(get_payload["ok"])
        self.assertTrue(get_payload["found"])
        self.assertEqual("h-endpoint", get_payload["last_change"]["hash"])

        post_handler = _HandlerForTest(runtime, "/llm/notifications/mark_read", {"hash": "h-endpoint"})
        post_handler.do_POST()
        self.assertEqual(200, post_handler.status_code)
        post_payload = json.loads(post_handler.body.decode("utf-8"))
        self.assertTrue(post_payload["ok"])
        self.assertEqual("h-endpoint", post_payload["status"]["last_read_hash"])


if __name__ == "__main__":
    unittest.main()
