from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import load_config


def _config(registry_path: str, db_path: str, *, telegram_enabled: bool = False):
    cfg = load_config(require_telegram_token=False)
    return replace(
        cfg,
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        llm_registry_path=registry_path,
        llm_automation_enabled=False,
        telegram_enabled=telegram_enabled,
    )


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""

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


class _FakeRunner:
    def __init__(self, status_payload: dict[str, object]) -> None:
        self._status_payload = dict(status_payload)

    def status(self) -> dict[str, object]:
        return dict(self._status_payload)


class TestReadyEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_ready_telegram_disabled_optional_reports_ready(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._telegram_runner = None
        runtime._telegram_configured_cached = False
        runtime._telegram_token_source_cached = "none"

        handler = _HandlerForTest(runtime, "/ready")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["ready"])
        self.assertIn(payload["runtime_mode"], {"READY", "BOOTSTRAP_REQUIRED"})
        self.assertFalse(bool(payload["telegram"]["enabled"]))
        self.assertEqual("disabled_optional", payload["telegram"]["state"])
        if payload["runtime_mode"] == "READY":
            self.assertIsNone(payload.get("next_action"))
        else:
            self.assertTrue(str(payload.get("next_action") or "").strip())
        self.assertIn("llm", payload)
        self.assertIn("known", payload["llm"])
        self.assertIn("onboarding", payload)
        self.assertIn("state", payload["onboarding"])
        self.assertIn("next_action", payload["onboarding"])
        self.assertIn("recovery", payload)
        self.assertIn("mode", payload["recovery"])
        self.assertEqual([], payload["telegram"]["recent_messages"])
        self.assertTrue(str(payload.get("message") or "").strip())

    def test_ready_missing_token_enabled_reports_disabled_missing_token(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, telegram_enabled=True))
        runtime._telegram_runner = None
        runtime._telegram_configured_cached = False
        runtime._telegram_token_source_cached = "none"

        handler = _HandlerForTest(runtime, "/ready")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(bool(payload["telegram"]["enabled"]))
        self.assertEqual("disabled_missing_token", payload["telegram"]["state"])

    def test_ready_reports_not_ready_when_telegram_crash_loop(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, telegram_enabled=True))
        runtime._telegram_runner = _FakeRunner(
            {
                "state": "crash_loop",
                "embedded_running": False,
                "last_event": "telegram.retry",
                "last_error": "RuntimeError: boom",
                "last_ts": 1.0,
                "last_ts_iso": "1970-01-01T00:00:01+00:00",
                "token_source": "secret_store",
                "consecutive_failures": 3,
            }
        )
        runtime._telegram_configured_cached = True
        runtime._telegram_token_source_cached = "secret_store"

        handler = _HandlerForTest(runtime, "/ready")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["ready"])
        self.assertEqual("DEGRADED", payload["runtime_mode"])
        self.assertTrue(str(payload.get("next_action") or "").strip())
        self.assertEqual("crash_loop", payload["telegram"]["state"])

    def test_ready_reports_ready_when_telegram_running(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, telegram_enabled=True))
        runtime._telegram_runner = _FakeRunner(
            {
                "state": "running",
                "embedded_running": True,
                "last_event": "telegram.started",
                "last_error": None,
                "last_ts": 2.0,
                "last_ts_iso": "1970-01-01T00:00:02+00:00",
                "token_source": "secret_store",
                "consecutive_failures": 0,
            }
        )
        runtime._telegram_configured_cached = True
        runtime._telegram_token_source_cached = "secret_store"

        handler = _HandlerForTest(runtime, "/ready")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["ready"])
        self.assertTrue(bool(payload["telegram"]["enabled"]))
        self.assertEqual("running", payload["telegram"]["state"])
        self.assertGreaterEqual(int(payload["api"]["uptime_seconds"]), 0)
        self.assertIn("version", payload["api"])
        self.assertIn("pid", payload["api"])

    def test_ready_includes_recent_telegram_messages_redacted(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, telegram_enabled=True))
        runtime._telegram_runner = _FakeRunner(
            {
                "state": "running",
                "embedded_running": True,
                "last_event": "telegram.started",
                "last_error": None,
                "last_ts": 2.0,
                "last_ts_iso": "1970-01-01T00:00:02+00:00",
                "token_source": "secret_store",
                "consecutive_failures": 0,
            }
        )
        runtime._telegram_configured_cached = True
        runtime._telegram_token_source_cached = "secret_store"
        runtime.audit_log.append(
            actor="telegram",
            action="telegram.message.received",
            params={
                "chat_id": "123456789",
                "chat_id_redacted": "***6789",
                "text": "ping",
                "route": "chat",
            },
            decision="allow",
            reason="chat:received",
            dry_run=False,
            outcome="received",
            error_kind=None,
            duration_ms=0,
        )
        runtime.audit_log.append(
            actor="telegram",
            action="telegram.message.handled",
            params={
                "chat_id": "123456789",
                "chat_id_redacted": "***6789",
                "message": "pong",
                "route": "chat",
            },
            decision="allow",
            reason="chat:handled",
            dry_run=False,
            outcome="handled",
            error_kind=None,
            duration_ms=0,
        )
        runtime.audit_log.append(
            actor="system",
            action="llm.autopilot",
            params={},
            decision="allow",
            reason="n/a",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=0,
        )

        handler = _HandlerForTest(runtime, "/ready")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        recent = payload["telegram"]["recent_messages"]
        self.assertEqual(2, len(recent))
        self.assertEqual("telegram.message.handled", recent[0]["action"])
        self.assertEqual("telegram.message.received", recent[1]["action"])
        serialized = json.dumps(recent, ensure_ascii=True)
        self.assertIn("***6789", serialized)
        self.assertNotIn("123456789", serialized)
        self.assertNotIn("ping", serialized.lower())
        self.assertNotIn("pong", serialized.lower())


if __name__ == "__main__":
    unittest.main()
