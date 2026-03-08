from __future__ import annotations

import inspect
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import load_config
from agent.runtime_contract import normalize_user_facing_status
from agent.secret_store import SecretStore
from agent.telegram_runtime_state import (
    TELEGRAM_SERVICE_NAME,
    get_telegram_runtime_state,
    read_telegram_enablement,
    write_telegram_enablement,
)


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


def _completed(args: list[str], returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def _canonical_llm_status_payload(
    *,
    provider: str = "ollama",
    model: str = "ollama:llama3.2",
    failure_code: str | None = None,
    active_provider_status: str = "ok",
    active_model_status: str = "ok",
) -> dict[str, object]:
    runtime_status = normalize_user_facing_status(
        ready=failure_code is None,
        bootstrap_required=False,
        failure_code=failure_code,
        phase=None,
        provider=provider,
        model=model,
        local_providers={provider},
    )
    return {
        "ok": True,
        "default_provider": provider,
        "default_model": model,
        "resolved_default_model": model,
        "chat_model": model,
        "allow_remote_fallback": True,
        "providers": [{"id": provider, "local": True}],
        "active_provider_health": {"status": active_provider_status},
        "active_model_health": {"status": active_model_status},
        "runtime_mode": runtime_status["runtime_mode"],
        "runtime_status": runtime_status,
    }


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

    def test_ready_telegram_disabled_optional_defers_to_canonical_llm_readiness(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch(
            "agent.api_server.get_telegram_runtime_state",
            return_value={
                "enabled": False,
                "token_configured": False,
                "token_source": "missing",
                "ready_state": "disabled_optional",
                "effective_state": "disabled_optional",
                "config_source": "default",
                "config_source_path": None,
                "service_installed": True,
                "service_active": False,
                "service_enabled": False,
                "lock_present": False,
                "lock_live": False,
                "lock_stale": False,
                "lock_path": None,
                "lock_pid": None,
                "next_action": "Run: python -m agent telegram_enable",
            },
        ):
            handler = _HandlerForTest(runtime, "/ready")
            handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertIn(payload["runtime_mode"], {"READY", "BOOTSTRAP_REQUIRED"})
        self.assertEqual(bool(payload["runtime_mode"] == "READY"), bool(payload["ready"]))
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
        with patch(
            "agent.api_server.get_telegram_runtime_state",
            return_value={
                "enabled": True,
                "token_configured": False,
                "token_source": "missing",
                "ready_state": "disabled_missing_token",
                "effective_state": "enabled_misconfigured",
                "config_source": "config",
                "config_source_path": "/tmp/override.conf",
                "service_installed": True,
                "service_active": False,
                "service_enabled": True,
                "lock_present": False,
                "lock_live": False,
                "lock_stale": False,
                "lock_path": None,
                "lock_pid": None,
                "next_action": "Run: python -m agent.secrets set telegram:bot_token",
            },
        ):
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
        with patch(
            "agent.api_server.get_telegram_runtime_state",
            return_value={
                "enabled": True,
                "token_configured": True,
                "token_source": "secret_store",
                "ready_state": "stopped",
                "effective_state": "enabled_stopped",
                "config_source": "config",
                "config_source_path": "/tmp/override.conf",
                "service_installed": True,
                "service_active": False,
                "service_enabled": True,
                "lock_present": False,
                "lock_live": False,
                "lock_stale": False,
                "lock_path": None,
                "lock_pid": None,
                "next_action": "Run: python -m agent telegram_enable",
            },
        ):
            handler = _HandlerForTest(runtime, "/ready")
            handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["ready"])
        self.assertEqual("DEGRADED", payload["runtime_mode"])
        self.assertTrue(str(payload.get("next_action") or "").strip())
        self.assertEqual("stopped", payload["telegram"]["state"])
        self.assertEqual("crash_loop", payload["telegram"]["embedded_state"])

    def test_ready_when_telegram_running_preserves_canonical_llm_readiness(self) -> None:
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
        with patch(
            "agent.api_server.get_telegram_runtime_state",
            return_value={
                "enabled": True,
                "token_configured": True,
                "token_source": "secret_store",
                "ready_state": "running",
                "effective_state": "enabled_running",
                "config_source": "config",
                "config_source_path": "/tmp/override.conf",
                "service_installed": True,
                "service_active": True,
                "service_enabled": True,
                "lock_present": False,
                "lock_live": False,
                "lock_stale": False,
                "lock_path": None,
                "lock_pid": None,
                "next_action": "No action needed.",
            },
        ):
            handler = _HandlerForTest(runtime, "/ready")
            handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(bool(payload["telegram"]["enabled"]))
        self.assertEqual("running", payload["telegram"]["state"])
        self.assertEqual(payload["runtime_status"], payload["llm"]["runtime_status"])
        self.assertEqual(bool(payload["llm"]["runtime_status"]["ready"]), bool(payload["ready"]))
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
        with patch(
            "agent.api_server.get_telegram_runtime_state",
            return_value={
                "enabled": True,
                "token_configured": True,
                "token_source": "secret_store",
                "ready_state": "running",
                "effective_state": "enabled_running",
                "config_source": "config",
                "config_source_path": "/tmp/override.conf",
                "service_installed": True,
                "service_active": True,
                "service_enabled": True,
                "lock_present": False,
                "lock_live": False,
                "lock_stale": False,
                "lock_path": None,
                "lock_pid": None,
                "next_action": "No action needed.",
            },
        ):
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

    def test_ready_and_telegram_status_follow_live_enablement_without_restart(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, telegram_enabled=False))
        home = Path(self.tmpdir.name)
        secret_path = os.environ["AGENT_SECRET_STORE_PATH"]
        SecretStore(path=secret_path).set_secret("telegram:bot_token", "123456:abcdef")
        write_telegram_enablement(False, home=home, env={})

        def _run(args, **_kwargs):  # type: ignore[no-untyped-def]
            cmd = list(args)
            enabled = bool(read_telegram_enablement(home=home, env={}).get("enabled", False))
            if cmd[-2:] == ["cat", TELEGRAM_SERVICE_NAME]:
                return _completed(cmd, 0, "unit")
            if cmd[-2:] == ["is-active", TELEGRAM_SERVICE_NAME]:
                return _completed(cmd, 0 if enabled else 3, "active\n" if enabled else "inactive\n")
            if cmd[-2:] == ["is-enabled", TELEGRAM_SERVICE_NAME]:
                return _completed(cmd, 0 if enabled else 1, "enabled\n" if enabled else "disabled\n")
            raise AssertionError(f"unexpected command: {cmd}")

        def _live_state(**kwargs: object) -> dict[str, object]:
            return get_telegram_runtime_state(
                home=home,
                env=kwargs.get("env"),
                run=_run,
                secret_store_path=secret_path,
            )

        with patch("agent.api_server.get_telegram_runtime_state", side_effect=_live_state):
            disabled_status = runtime.telegram_status()
            self.assertFalse(bool(disabled_status["enabled"]))
            self.assertEqual("disabled_optional", disabled_status["state"])

            write_telegram_enablement(True, home=home, env={})

            enabled_status = runtime.telegram_status()
            self.assertTrue(bool(enabled_status["enabled"]))
            self.assertEqual("running", enabled_status["state"])

            telegram_handler = _HandlerForTest(runtime, "/telegram/status")
            telegram_handler.do_GET()
            telegram_payload = json.loads(telegram_handler.body.decode("utf-8"))
            self.assertTrue(bool(telegram_payload["enabled"]))
            self.assertEqual("running", telegram_payload["state"])

            ready_handler = _HandlerForTest(runtime, "/ready")
            ready_handler.do_GET()
            ready_payload = json.loads(ready_handler.body.decode("utf-8"))
            self.assertTrue(bool(ready_payload["telegram"]["enabled"]))
            self.assertEqual("running", ready_payload["telegram"]["state"])

            write_telegram_enablement(False, home=home, env={})
            final_status = runtime.telegram_status()
            self.assertFalse(bool(final_status["enabled"]))
            self.assertEqual("disabled_optional", final_status["state"])

    def test_ready_reuses_canonical_llm_status_when_provider_is_unhealthy(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        canonical_status = _canonical_llm_status_payload(
            failure_code="provider_unhealthy",
            active_provider_status="down",
        )
        with patch.object(runtime, "llm_status", return_value=canonical_status), patch(
            "agent.api_server.get_telegram_runtime_state",
            return_value={
                "enabled": False,
                "token_configured": False,
                "token_source": "missing",
                "ready_state": "disabled_optional",
                "effective_state": "disabled_optional",
                "config_source": "default",
                "config_source_path": None,
                "service_installed": True,
                "service_active": False,
                "service_enabled": False,
                "lock_present": False,
                "lock_live": False,
                "lock_stale": False,
                "lock_path": None,
                "lock_pid": None,
                "next_action": "Run: python -m agent telegram_enable",
            },
        ):
            ready_handler = _HandlerForTest(runtime, "/ready")
            ready_handler.do_GET()
            llm_handler = _HandlerForTest(runtime, "/llm/status")
            llm_handler.do_GET()

        ready_payload = json.loads(ready_handler.body.decode("utf-8"))
        llm_payload = json.loads(llm_handler.body.decode("utf-8"))
        self.assertFalse(ready_payload["ready"])
        self.assertEqual(llm_payload["runtime_mode"], ready_payload["runtime_mode"])
        self.assertEqual(
            llm_payload["runtime_status"]["failure_code"],
            ready_payload["runtime_status"]["failure_code"],
        )
        self.assertEqual(llm_payload["runtime_status"], ready_payload["llm"]["runtime_status"])
        self.assertEqual("down", ready_payload["llm"]["active_provider_health"]["status"])

    def test_ready_preserves_canonical_llm_status_during_startup_overlay(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.startup_phase = "warming"
        with runtime._startup_warmup_lock:
            runtime._startup_warmup_remaining = ["router_reload"]
        canonical_status = _canonical_llm_status_payload()
        with patch.object(runtime, "llm_status", return_value=canonical_status), patch(
            "agent.api_server.get_telegram_runtime_state",
            return_value={
                "enabled": False,
                "token_configured": False,
                "token_source": "missing",
                "ready_state": "disabled_optional",
                "effective_state": "disabled_optional",
                "config_source": "default",
                "config_source_path": None,
                "service_installed": True,
                "service_active": False,
                "service_enabled": False,
                "lock_present": False,
                "lock_live": False,
                "lock_stale": False,
                "lock_path": None,
                "lock_pid": None,
                "next_action": "Run: python -m agent telegram_enable",
            },
        ):
            ready_handler = _HandlerForTest(runtime, "/ready")
            ready_handler.do_GET()
            llm_handler = _HandlerForTest(runtime, "/llm/status")
            llm_handler.do_GET()

        ready_payload = json.loads(ready_handler.body.decode("utf-8"))
        llm_payload = json.loads(llm_handler.body.decode("utf-8"))
        self.assertFalse(ready_payload["ready"])
        self.assertEqual("warming", ready_payload["phase"])
        self.assertEqual(["router_reload"], ready_payload["warmup_remaining"])
        self.assertEqual("DEGRADED", ready_payload["runtime_mode"])
        self.assertEqual("READY", llm_payload["runtime_mode"])
        self.assertEqual(llm_payload["runtime_status"], ready_payload["llm"]["runtime_status"])
        self.assertEqual("READY", ready_payload["llm"]["runtime_status"]["runtime_mode"])

    def test_ready_telegram_overlay_does_not_replace_canonical_llm_status(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, telegram_enabled=True))
        canonical_status = _canonical_llm_status_payload()
        with patch.object(runtime, "llm_status", return_value=canonical_status), patch(
            "agent.api_server.get_telegram_runtime_state",
            return_value={
                "enabled": True,
                "token_configured": True,
                "token_source": "secret_store",
                "ready_state": "stopped",
                "effective_state": "enabled_stopped",
                "config_source": "config",
                "config_source_path": "/tmp/override.conf",
                "service_installed": True,
                "service_active": False,
                "service_enabled": True,
                "lock_present": False,
                "lock_live": False,
                "lock_stale": False,
                "lock_path": None,
                "lock_pid": None,
                "next_action": "Run: python -m agent telegram_enable",
            },
        ):
            ready_handler = _HandlerForTest(runtime, "/ready")
            ready_handler.do_GET()
            llm_handler = _HandlerForTest(runtime, "/llm/status")
            llm_handler.do_GET()

        ready_payload = json.loads(ready_handler.body.decode("utf-8"))
        llm_payload = json.loads(llm_handler.body.decode("utf-8"))
        self.assertFalse(ready_payload["ready"])
        self.assertEqual("service_down", ready_payload["runtime_status"]["failure_code"])
        self.assertEqual("stopped", ready_payload["telegram"]["state"])
        self.assertEqual("READY", llm_payload["runtime_mode"])
        self.assertEqual(llm_payload["runtime_status"], ready_payload["llm"]["runtime_status"])
        self.assertEqual("READY", ready_payload["llm"]["runtime_status"]["runtime_mode"])

    def test_ready_status_uses_canonical_llm_status_helper(self) -> None:
        ready_source = inspect.getsource(AgentRuntime.ready_status)
        llm_helper_source = inspect.getsource(AgentRuntime._canonical_llm_ready_context)
        self.assertIn("self._canonical_llm_ready_context()", ready_source)
        self.assertIn("self.llm_status()", llm_helper_source)
        self.assertNotIn("provider_status_hint", ready_source)
        self.assertNotIn("model_status_hint", ready_source)
        self.assertNotIn("doctor_snapshot", ready_source)


if __name__ == "__main__":
    unittest.main()
