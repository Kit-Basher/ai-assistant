from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.error_kind import (
    INTERNAL_ERROR_SUPPORT_HINT,
    classify_error_kind,
)
from telegram_adapter.bot import _envelope_from_exception


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
    )
    return base.__class__(**{**base.__dict__, **overrides})


class _HandlerForPostTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object]) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {"Content-Length": "0"}
        self._payload = dict(payload)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


class _RawBodyPostHandler(APIServerHandler):
    def __init__(
        self,
        runtime_obj: AgentRuntime,
        path: str,
        raw_body: bytes,
        *,
        content_type: str,
    ) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {
            "Content-Length": str(len(raw_body)),
            "Content-Type": content_type,
        }
        self.rfile = io.BytesIO(raw_body)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


class TestErrorKindContract(unittest.TestCase):
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

    def test_chat_empty_payload_sets_needs_clarification(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        handler = _HandlerForPostTest(runtime, "/chat", {})
        handler.do_POST()

        self.assertEqual(200, handler.status_code)
        self.assertEqual(True, handler.response_payload.get("ok"))
        self.assertEqual("needs_clarification", handler.response_payload.get("error_kind"))
        self.assertEqual(0.0, handler.response_payload.get("confidence"))
        self.assertEqual(False, handler.response_payload.get("did_work"))
        self.assertTrue(str(handler.response_payload.get("message") or "").strip())
        envelope = handler.response_payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        self.assertEqual(["needs_clarification"], handler.response_payload.get("errors"))
        self.assertEqual(
            str((envelope or {}).get("message") or ""),
            str((envelope or {}).get("next_question") or ""),
        )

    def test_ask_empty_payload_sets_needs_clarification(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        handler = _HandlerForPostTest(runtime, "/ask", {})
        handler.do_POST()

        self.assertEqual(200, handler.status_code)
        self.assertEqual(True, handler.response_payload.get("ok"))
        self.assertEqual("needs_clarification", handler.response_payload.get("error_kind"))
        self.assertEqual("ask", handler.response_payload.get("intent"))
        self.assertEqual(["needs_clarification"], handler.response_payload.get("errors"))
        self.assertTrue(str(handler.response_payload.get("message") or "").strip())

    def test_done_invalid_payload_sets_bad_request_error_kind(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        handler = _HandlerForPostTest(runtime, "/done", {})
        handler.do_POST()

        self.assertEqual(400, handler.status_code)
        self.assertEqual(False, handler.response_payload.get("ok"))
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))
        self.assertTrue(str(handler.response_payload.get("message") or "").strip())

    def test_chat_internal_error_sets_support_hint_and_error_kind(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch.object(runtime, "chat", side_effect=RuntimeError("boom")):
            handler = _HandlerForPostTest(
                runtime,
                "/chat",
                {"messages": [{"role": "user", "content": "hello"}]},
            )
            handler.do_POST()

        self.assertEqual(500, handler.status_code)
        self.assertEqual(False, handler.response_payload.get("ok"))
        self.assertEqual("internal_error", handler.response_payload.get("error_kind"))
        self.assertIn(INTERNAL_ERROR_SUPPORT_HINT, str(handler.response_payload.get("message") or ""))

    def test_classify_upstream_down_from_health_and_attempt_reason(self) -> None:
        kind = classify_error_kind(
            payload={
                "ok": False,
                "meta": {
                    "provider": "openrouter",
                    "attempts": [{"reason": "provider_unavailable"}],
                    "error": "no_candidates",
                },
            },
            context={
                "provider": "openrouter",
                "health_state": {
                    "providers": {"openrouter": {"status": "down"}},
                    "models": {},
                },
            },
        )
        self.assertEqual("upstream_down", kind)

    def test_classify_timeout_rate_limit_and_policy(self) -> None:
        self.assertEqual("timeout", classify_error_kind(error=TimeoutError("timed out")))
        self.assertEqual("rate_limited", classify_error_kind(payload={"ok": False, "error": "rate_limit"}))
        self.assertEqual("payment_required", classify_error_kind(payload={"ok": False, "error": "payment_required"}))
        self.assertEqual("policy_blocked", classify_error_kind(payload={"ok": False, "error": "safe_mode_blocked"}))

    def test_classify_feature_disabled_from_memory_v2_disabled(self) -> None:
        self.assertEqual("feature_disabled", classify_error_kind(payload={"ok": False, "error": "memory_v2_disabled"}))
        self.assertEqual("feature_disabled", classify_error_kind(payload={"ok": False, "error": "delivery_disabled"}))

    def test_upstream_down_message_includes_cooldown_iso_and_remaining(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._health_monitor.state = {  # type: ignore[attr-defined]
            "providers": {
                "openrouter": {
                    "status": "down",
                    "cooldown_until": 1_700_000_100,
                }
            },
            "models": {
                "openrouter:openai/gpt-4o-mini": {
                    "status": "down",
                    "cooldown_until": 1_700_000_100,
                }
            },
        }
        with patch.object(
            runtime,
            "chat",
            return_value=(
                False,
                {
                    "ok": False,
                    "assistant": {"role": "assistant", "content": ""},
                    "meta": {
                        "provider": "openrouter",
                        "model": "openai/gpt-4o-mini",
                        "attempts": [{"provider": "openrouter", "model": "openai/gpt-4o-mini", "reason": "provider_unavailable"}],
                    },
                },
            ),
        ), patch("agent.api_server.time.time", return_value=1_700_000_000):
            handler = _HandlerForPostTest(
                runtime,
                "/chat",
                {"messages": [{"role": "user", "content": "hello"}]},
            )
            handler.do_POST()
        self.assertEqual(400, handler.status_code)
        self.assertEqual("upstream_down", handler.response_payload.get("error_kind"))
        message = str(handler.response_payload.get("message") or "")
        self.assertIn("Cooldown until 2023-11-14T22:15:00Z", message)
        self.assertIn("(100s remaining)", message)
        self.assertIn("Try again after 2023-11-14T22:15:00Z or switch provider.", message)

    def test_missing_content_type_returns_bad_request_next_question(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        raw_body = b'{"messages":[{"role":"user","content":"hello"}]}'
        handler = _RawBodyPostHandler(runtime, "/chat", raw_body, content_type="text/plain")
        handler.do_POST()
        self.assertEqual(400, handler.status_code)
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))
        self.assertIn("Content-Type: application/json", str(handler.response_payload.get("next_question") or ""))

    def test_invalid_json_body_returns_bad_request_next_question(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        raw_body = b'{"messages": invalid}'
        handler = _RawBodyPostHandler(runtime, "/chat", raw_body, content_type="application/json")
        handler.do_POST()
        self.assertEqual(400, handler.status_code)
        self.assertEqual("bad_request", handler.response_payload.get("error_kind"))
        self.assertIn('"messages"', str(handler.response_payload.get("next_question") or ""))

    def test_bootstrap_run_returns_feature_disabled_when_memory_v2_disabled(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, memory_v2_enabled=False))
        handler = _HandlerForPostTest(runtime, "/bootstrap/run", {"force": True})
        handler.do_POST()

        self.assertEqual(400, handler.status_code)
        self.assertEqual(False, handler.response_payload.get("ok"))
        self.assertEqual("feature_disabled", handler.response_payload.get("error_kind"))
        self.assertTrue(str(handler.response_payload.get("message") or "").strip())
        envelope = handler.response_payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        self.assertEqual("feature_disabled", str((envelope or {}).get("error_kind") or ""))
        errors = (envelope or {}).get("errors") if isinstance(envelope, dict) else []
        self.assertTrue(isinstance(errors, list))
        self.assertIn("memory_v2_disabled", errors)

    def test_telegram_fallback_uses_friendly_upstream_message_without_traceback(self) -> None:
        envelope = _envelope_from_exception(
            exc=RuntimeError("provider_unavailable"),
            intent="telegram.message",
            trace_id="tg-1",
            log_path=None,
        )
        self.assertEqual("upstream_down", envelope.get("error_kind"))
        self.assertIn("switch provider", str(envelope.get("message") or ""))
        self.assertNotIn("Traceback", str(envelope.get("message") or ""))


if __name__ == "__main__":
    unittest.main()
