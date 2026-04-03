from __future__ import annotations

import copy
import inspect
import io
import json
import os
import threading
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.chat_response_serializer import SerializedChatResponse
from agent.config import Config
from agent.llm.chat_preflight import PreparedChatRequest
from agent.llm.model_manager import (
    CanonicalModelManager,
    model_manager_state_path_for_runtime,
    save_model_manager_state,
)
from agent.llm.types import LLMError, Response, Usage
from agent.model_watch_catalog import write_snapshot_atomic
from agent.modelops.discovery import ModelInfo
from agent.orchestrator import OrchestratorResponse


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


class TestAPIServerRuntime(unittest.TestCase):
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

    def test_health_includes_safe_mode_fields(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._autopilot_safety_state.enter_safe_mode(reason="churn_detected", now_epoch=1_700_000_000)
        runtime._scheduler_next_run["autoconfig"] = float(1_700_000_600)
        payload = runtime.health()
        safe_mode = payload.get("safe_mode") if isinstance(payload.get("safe_mode"), dict) else {}
        self.assertTrue(bool(safe_mode.get("paused")))
        self.assertEqual("churn_detected", safe_mode.get("reason"))
        self.assertEqual(1_700_000_600, safe_mode.get("next_retry"))
        self.assertEqual(1_700_000_000, safe_mode.get("last_transition_at"))
        self.assertIn("cooldown_until", safe_mode)

    def test_health_includes_runtime_contract_control_mode_and_blocked_fields(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._autopilot_safety_state.enter_safe_mode(reason="churn_detected", now_epoch=1_700_000_000)
        payload = runtime.health()
        runtime_status = payload.get("runtime_status") if isinstance(payload.get("runtime_status"), dict) else {}
        blocked = payload.get("blocked") if isinstance(payload.get("blocked"), dict) else {}
        control_mode = payload.get("control_mode") if isinstance(payload.get("control_mode"), dict) else {}

        self.assertIn("phase", payload)
        self.assertIn("startup_phase", payload)
        self.assertIn("warmup_remaining", payload)
        self.assertIn("runtime_mode", payload)
        self.assertEqual(payload["runtime_mode"], runtime_status.get("runtime_mode"))
        self.assertIn("failure_code", payload)
        self.assertIn("next_action", payload)
        self.assertTrue(str(payload.get("message") or "").strip())
        self.assertIn("safe_mode_target", payload)
        self.assertTrue(bool(blocked.get("blocked")))
        self.assertEqual("policy", blocked.get("kind"))
        self.assertEqual("churn_detected", blocked.get("reason"))
        self.assertEqual("controlled", control_mode.get("mode"))
        self.assertIn("allow_remote_switch", control_mode)
        self.assertIn("approval_required_actions", control_mode)

    def test_health_is_explicit_before_and_after_process_restart_with_deferred_warmup(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path), defer_bootstrap_warmup=True)
        try:
            payload = runtime.health()
            self.assertFalse(bool(payload.get("ready")))
            self.assertEqual("warmup", payload.get("phase"))
            self.assertEqual("starting", payload.get("startup_phase"))
            self.assertTrue(bool(payload.get("warmup_remaining")))
            self.assertEqual("DEGRADED", payload.get("runtime_mode"))
        finally:
            runtime.close()

        restarted = AgentRuntime(_config(self.registry_path, self.db_path), defer_bootstrap_warmup=True)
        try:
            payload = restarted.health()
            self.assertFalse(bool(payload.get("ready")))
            self.assertEqual("warmup", payload.get("phase"))
            self.assertEqual("starting", payload.get("startup_phase"))
            self.assertTrue(bool(payload.get("warmup_remaining")))
            self.assertEqual("DEGRADED", payload.get("runtime_mode"))
        finally:
            restarted.close()

    def test_health_reports_telegram_partial_failure_explicitly(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, telegram_enabled=True))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            "ollama": {"status": "ok", "last_checked_at": 123}
        }
        runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
            "ollama:qwen2.5:7b-instruct": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123}
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]
        runtime.startup_phase = "ready"

        class _FakeRunner:
            def status(self) -> dict[str, object]:
                return {
                    "state": "crash_loop",
                    "embedded_running": False,
                    "last_event": "telegram.retry",
                    "last_error": "RuntimeError: boom",
                    "last_ts": 1.0,
                    "last_ts_iso": "1970-01-01T00:00:01+00:00",
                    "token_source": "secret_store",
                    "consecutive_failures": 3,
                }

        runtime._telegram_runner = _FakeRunner()  # type: ignore[assignment]
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
            payload = runtime.health()

        telegram = payload.get("telegram") if isinstance(payload.get("telegram"), dict) else {}
        runtime_status = payload.get("runtime_status") if isinstance(payload.get("runtime_status"), dict) else {}
        self.assertFalse(bool(payload.get("ready")))
        self.assertEqual("DEGRADED", payload.get("runtime_mode"))
        self.assertEqual("service_down", payload.get("failure_code"))
        self.assertEqual("service_down", runtime_status.get("failure_code"))
        self.assertEqual("stopped", telegram.get("state"))
        self.assertEqual("crash_loop", telegram.get("embedded_state"))

    def test_runtime_snapshot_includes_runtime_status_control_mode_and_blocked_fields(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._autopilot_safety_state.enter_safe_mode(reason="churn_detected", now_epoch=1_700_000_000)
        payload = runtime.runtime_snapshot()
        runtime_status = payload.get("runtime_status") if isinstance(payload.get("runtime_status"), dict) else {}
        blocked = payload.get("blocked") if isinstance(payload.get("blocked"), dict) else {}
        control_mode = payload.get("control_mode") if isinstance(payload.get("control_mode"), dict) else {}

        self.assertIn("runtime_status", payload)
        self.assertEqual(payload["runtime_mode"], runtime_status.get("runtime_mode"))
        self.assertIn("message", payload)
        self.assertIn("safe_mode", payload)
        self.assertIn("safe_mode_target", payload)
        self.assertTrue(bool(blocked.get("blocked")))
        self.assertEqual("policy", blocked.get("kind"))
        self.assertEqual("controlled", control_mode.get("mode"))

    def test_add_provider_set_secret_and_test_provider(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        ok, added = runtime.add_provider(
            {
                "id": "acme",
                "provider_type": "openai_compat",
                "base_url": "https://acme.example",
                "chat_path": "/v1/chat/completions",
                "local": False,
                "enabled": True,
                "models": [
                    {
                        "id": "acme:chat",
                        "model": "chat-model",
                        "capabilities": ["chat"],
                        "pricing": {
                            "input_per_million_tokens": 0.2,
                            "output_per_million_tokens": 0.4,
                        },
                    }
                ],
            }
        )
        self.assertTrue(ok)
        self.assertEqual("acme", added["provider"]["id"])

        ok, secret_resp = runtime.set_provider_secret("acme", {"api_key": "sk-acme"})
        self.assertTrue(ok)
        self.assertEqual("acme", secret_resp["provider"])

        source = runtime.registry_document["providers"]["acme"]["api_key_source"]
        self.assertEqual("secret", source["type"])
        self.assertEqual("provider:acme:api_key", source["name"])
        self.assertEqual("sk-acme", runtime.secret_store.get_secret("provider:acme:api_key"))

        provider_impl = runtime._router._providers["acme"]  # type: ignore[attr-defined]
        original_chat = provider_impl.chat
        provider_impl.chat = lambda request, *, model, timeout_seconds: Response(  # type: ignore[assignment]
            text="PONG",
            provider="acme",
            model=model,
            usage=Usage(5, 2, 7),
        )
        try:
            ok, tested = runtime.test_provider("acme", {"model": "acme:chat"})
        finally:
            provider_impl.chat = original_chat  # type: ignore[assignment]

        self.assertTrue(ok)
        self.assertTrue(tested["ok"])
        self.assertEqual("acme", tested["provider"])
        self.assertEqual("chat-model", tested["model"])

    def test_test_provider_uses_lightweight_local_model_probe_for_ollama(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        provider_impl = runtime._router._providers["ollama"]  # type: ignore[attr-defined]
        original_chat = provider_impl.chat
        provider_impl.chat = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heavy chat probe should not run"))  # type: ignore[assignment]
        try:
            with patch.object(
                runtime,
                "_probe_provider_models",
                return_value={"ok": True, "models": ["qwen2.5:3b-instruct"], "count": 1},
            ), patch.object(
                runtime,
                "_probe_llm_model",
                return_value={
                    "status": "ok",
                    "error_kind": None,
                    "status_code": None,
                    "detail": "model probe ok",
                },
            ) as model_probe:
                ok, tested = runtime.test_provider("ollama", {"model": "ollama:qwen2.5:3b-instruct"})
        finally:
            provider_impl.chat = original_chat  # type: ignore[assignment]

        self.assertTrue(ok)
        self.assertTrue(tested["ok"])
        self.assertEqual("ollama", tested["provider"])
        self.assertEqual("qwen2.5:3b-instruct", tested["model"])
        self.assertGreaterEqual(model_probe.call_args.kwargs["timeout_seconds"], 15.0)

    def test_probe_llm_model_keeps_bare_ollama_model_name_intact(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        with patch(
            "agent.api_server.probe_model",
            return_value={"status": "ok", "error_kind": None, "status_code": None, "detail": "ok"},
        ) as probe:
            runtime._probe_llm_model("ollama", "qwen2.5:3b-instruct", 15.0)

        self.assertEqual("qwen2.5:3b-instruct", probe.call_args.args[1])

    def test_add_provider_accepts_openai_compat_defaults_and_config_fields(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        ok, added = runtime.add_provider(
            {
                "id": "routerx",
                "base_url": "https://routerx.example/api/v1",
                "chat_path": "/chat/completions",
                "local": False,
                "enabled": True,
                "default_headers": {"X-App": "personal-agent"},
                "default_query_params": {"api-version": "2024-06-01"},
                "requires_api_key": False,
            }
        )
        self.assertTrue(ok)
        self.assertEqual("openai_compat", added["provider"]["provider_type"])
        self.assertEqual("https://routerx.example/api/v1", added["provider"]["base_url"])
        self.assertEqual("/chat/completions", added["provider"]["chat_path"])
        self.assertEqual({"X-App": "personal-agent"}, added["provider"]["default_headers"])
        self.assertEqual({"api-version": "2024-06-01"}, added["provider"]["default_query_params"])
        self.assertIsNone(added["provider"]["api_key_source"])

    def test_add_provider_model_persists_for_manual_entry(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        ok, _added = runtime.add_provider(
            {
                "id": "manualx",
                "base_url": "https://manualx.example",
                "chat_path": "/v1/chat/completions",
                "requires_api_key": False,
            }
        )
        self.assertTrue(ok)

        ok, response = runtime.add_provider_model(
            "manualx",
            {
                "model": "custom-chat",
                "capabilities": ["chat", "json", "tools"],
            },
        )
        self.assertTrue(ok)
        self.assertEqual("manualx:custom-chat", response["model"]["id"])
        with open(self.registry_path, "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        self.assertIn("manualx:custom-chat", on_disk["models"])

    def test_provider_secret_flips_to_secret_source_and_persists(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        initial_source = runtime.registry_document["providers"]["openrouter"]["api_key_source"]
        self.assertEqual("env", initial_source["type"])

        ok, response = runtime.set_provider_secret("openrouter", {"api_key": "sk-openrouter"})
        self.assertTrue(ok)
        self.assertTrue(response["ok"])

        source = runtime.registry_document["providers"]["openrouter"]["api_key_source"]
        self.assertEqual("secret", source["type"])
        self.assertEqual("provider:openrouter:api_key", source["name"])

        with open(self.registry_path, "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        disk_source = on_disk["providers"]["openrouter"]["api_key_source"]
        self.assertEqual("secret", disk_source["type"])
        self.assertEqual("provider:openrouter:api_key", disk_source["name"])

        restarted = AgentRuntime(_config(self.registry_path, self.db_path))
        restarted_source = restarted.registry_document["providers"]["openrouter"]["api_key_source"]
        self.assertEqual("secret", restarted_source["type"])
        self.assertEqual("provider:openrouter:api_key", restarted_source["name"])

    def test_update_provider_persists_and_survives_restart(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        ok, updated = runtime.update_provider(
            "openrouter",
            {
                "base_url": "https://openrouter.example/api",
                "chat_path": "v2/chat/completions",
                "enabled": True,
                "local": False,
                "default_headers": {"X-Test": "1"},
            },
        )
        self.assertTrue(ok)
        self.assertEqual("https://openrouter.example/api", updated["provider"]["base_url"])
        self.assertEqual("/v2/chat/completions", updated["provider"]["chat_path"])

        with open(self.registry_path, "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        provider_disk = on_disk["providers"]["openrouter"]
        self.assertEqual("https://openrouter.example/api", provider_disk["base_url"])
        self.assertEqual("/v2/chat/completions", provider_disk["chat_path"])
        self.assertEqual({"X-Test": "1"}, provider_disk["default_headers"])

        restarted = AgentRuntime(_config(self.registry_path, self.db_path))
        provider_after_restart = restarted.registry_document["providers"]["openrouter"]
        self.assertEqual("https://openrouter.example/api", provider_after_restart["base_url"])
        self.assertEqual("/v2/chat/completions", provider_after_restart["chat_path"])
        self.assertEqual({"X-Test": "1"}, provider_after_restart["default_headers"])
        self.assertEqual(
            "https://openrouter.example/api",
            restarted._router.registry.providers["openrouter"].base_url,  # type: ignore[attr-defined]
        )

    def test_update_provider_returns_clear_error_when_registry_not_writable(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch.object(runtime.registry_store, "write_document", side_effect=PermissionError("denied")):
            ok, response = runtime.update_provider("openrouter", {"base_url": "https://example.invalid"})
        self.assertFalse(ok)
        self.assertIn("registry_path not writable:", response["error"])

    def test_set_provider_secret_returns_clear_error_when_registry_not_writable(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch.object(runtime.registry_store, "write_document", side_effect=PermissionError("denied")):
            ok, response = runtime.set_provider_secret("openrouter", {"api_key": "sk-should-not-save"})
        self.assertFalse(ok)
        self.assertIn("registry_path not writable:", response["error"])
        self.assertIsNone(runtime.secret_store.get_secret("provider:openrouter:api_key"))

    def test_provider_test_returns_structured_error_kinds(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        ok, _added = runtime.add_provider(
            {
                "id": "kindcheck",
                "base_url": "https://kindcheck.example",
                "chat_path": "/v1/chat/completions",
                "models": [{"id": "kindcheck:chat", "model": "chat", "capabilities": ["chat"]}],
                "requires_api_key": False,
            }
        )
        self.assertTrue(ok)

        provider_impl = runtime._router._providers["kindcheck"]  # type: ignore[attr-defined]
        original_chat = provider_impl.chat
        try:
            cases = [
                (
                    LLMError(
                        kind="auth_error",
                        retriable=False,
                        provider="kindcheck",
                        status_code=401,
                        message="unauthorized",
                    ),
                    "auth_error",
                    401,
                ),
                (
                    LLMError(
                        kind="rate_limit",
                        retriable=True,
                        provider="kindcheck",
                        status_code=429,
                        message="limited",
                    ),
                    "rate_limit",
                    429,
                ),
                (
                    LLMError(
                        kind="server_error",
                        retriable=True,
                        provider="kindcheck",
                        status_code=502,
                        message="bad_gateway",
                    ),
                    "server_error",
                    502,
                ),
                (
                    LLMError(
                        kind="provider_error",
                        retriable=False,
                        provider="kindcheck",
                        status_code=400,
                        message="invalid",
                    ),
                    "bad_request",
                    400,
                ),
                (
                    RuntimeError("boom"),
                    "server_error",
                    None,
                ),
            ]
            for raised, expected_kind, expected_status in cases:
                provider_impl.chat = lambda request, *, model, timeout_seconds, err=raised: (_ for _ in ()).throw(err)  # type: ignore[assignment]
                ok, response = runtime.test_provider("kindcheck", {"model": "kindcheck:chat"})
                self.assertFalse(ok)
                self.assertEqual(expected_kind, response["error"])
                self.assertEqual(expected_status, response["status_code"])
        finally:
            provider_impl.chat = original_chat  # type: ignore[assignment]

    def test_openrouter_provider_test_402_is_friendly_and_uses_small_budget(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        provider_impl = runtime._router._providers["openrouter"]  # type: ignore[attr-defined]
        original_chat = provider_impl.chat
        captured: dict[str, object] = {}

        openrouter_error_body = json.dumps(
            {
                "error": {
                    "message": "Insufficient credits for this request",
                    "code": 402,
                }
            },
            ensure_ascii=True,
        )

        def _raise_402(request, *, model, timeout_seconds):  # type: ignore[no-untyped-def]
            captured["request"] = request
            raise LLMError(
                kind="bad_request",
                retriable=False,
                provider="openrouter",
                status_code=402,
                message=openrouter_error_body,
            )

        provider_impl.chat = _raise_402  # type: ignore[assignment]
        try:
            with patch.object(
                runtime,
                "_probe_provider_models",
                return_value={"ok": True, "models": ["openrouter:openai/gpt-4o-mini"], "count": 1},
            ):
                ok, response = runtime.test_provider(
                    "openrouter",
                    {"model": "openrouter:openai/gpt-4o-mini"},
                )
        finally:
            provider_impl.chat = original_chat  # type: ignore[assignment]

        self.assertFalse(ok)
        self.assertEqual(402, response["status_code"])
        self.assertEqual("payment_required", response["error"])
        self.assertEqual("payment_required", response["error_kind"])
        self.assertIn("credits/limit issue", response["message"])
        self.assertIn("lower max_tokens", response["message"])
        self.assertIn("cheaper model", response["message"])
        self.assertNotIn('{"error"', response["message"])

        request_obj = captured.get("request")
        self.assertIsNotNone(request_obj)
        self.assertEqual(256, getattr(request_obj, "max_tokens", None))
        self.assertEqual(0.0, getattr(request_obj, "temperature", None))
        self.assertEqual("health", getattr(request_obj, "purpose", None))
        self.assertEqual("test", getattr(request_obj, "task_type", None))

    def test_defaults_endpoints(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        ok, updated = runtime.update_defaults(
            {
                "routing_mode": "prefer_local_lowest_cost_capable",
                "default_provider": "ollama",
                "default_model": "ollama:llama3",
                "allow_remote_fallback": False,
            }
        )
        self.assertTrue(ok)
        self.assertEqual("prefer_local_lowest_cost_capable", updated["routing_mode"])
        self.assertEqual("ollama", updated["default_provider"])
        self.assertEqual("ollama:llama3", updated["default_model"])
        self.assertEqual("ollama:llama3", updated["chat_model"])
        self.assertIn("resolved_default_model", updated)
        self.assertFalse(updated["allow_remote_fallback"])

        current = runtime.get_defaults()
        self.assertEqual("prefer_local_lowest_cost_capable", current["routing_mode"])
        self.assertEqual("ollama", current["default_provider"])
        self.assertEqual("ollama:llama3", current["default_model"])
        self.assertEqual("ollama:llama3", current["chat_model"])
        self.assertEqual("ollama:llama3", current["resolved_default_model"])
        self.assertFalse(current["allow_remote_fallback"])
        self.assertEqual("prefer_local_lowest_cost_capable", runtime._router.policy.mode)

    def test_llm_control_mode_status_and_ready_surface_reflect_explicit_override(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "quality_rank": 6,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": True,
            }
        )

        initial = runtime.llm_control_mode_status()
        self.assertEqual("safe", initial.get("mode"))
        self.assertEqual("config_default", initial.get("mode_source"))
        self.assertEqual("safe", initial.get("config_default_mode"))
        self.assertFalse(bool(initial.get("override_active")))

        ok, changed = runtime.llm_control_mode_set({"mode": "controlled", "confirm": True, "actor": "test"})
        self.assertTrue(ok)
        policy = changed.get("policy") if isinstance(changed.get("policy"), dict) else {}
        self.assertEqual("controlled", policy.get("mode"))
        self.assertEqual("explicit_override", policy.get("mode_source"))
        self.assertEqual("safe", policy.get("config_default_mode"))
        self.assertEqual("controlled", policy.get("override_mode"))
        self.assertTrue(bool(policy.get("override_active")))
        self.assertTrue(bool(policy.get("allow_remote_switch")))
        self.assertTrue(bool(policy.get("allow_install_pull")))

        ready = runtime.ready_status()
        llm = ready.get("llm") if isinstance(ready.get("llm"), dict) else {}
        ready_policy = llm.get("policy") if isinstance(llm.get("policy"), dict) else {}
        self.assertEqual("controlled", ready_policy.get("mode"))
        self.assertEqual("explicit_override", ready_policy.get("mode_source"))

        ok_reset, reset = runtime.llm_control_mode_set({"mode": "baseline", "confirm": True, "actor": "test"})
        self.assertTrue(ok_reset)
        reset_policy = reset.get("policy") if isinstance(reset.get("policy"), dict) else {}
        self.assertEqual("safe", reset_policy.get("mode"))
        self.assertEqual("config_default", reset_policy.get("mode_source"))
        self.assertFalse(bool(reset_policy.get("override_active")))

    def test_llm_control_mode_can_enable_controlled_mode_from_safe_baseline_and_return(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )
        os.environ["OPENROUTER_API_KEY"] = "sk-test"
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "quality_rank": 6,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "quality_rank": 10,
                "available": True,
                "max_context_tokens": 131072,
            },
        )
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": True,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
                "openrouter": {"status": "ok", "last_checked_at": 123},
            },
            "models": {
                "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "openrouter:openai/gpt-4o-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        blocked_ok, blocked = runtime.llm_models_switch(
            {
                "provider": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "purpose": "chat",
                "confirm": True,
            }
        )
        self.assertFalse(blocked_ok)
        self.assertEqual("safe_mode_remote_switch_blocked", blocked.get("error_kind"))

        ok_mode, mode_body = runtime.llm_control_mode_set({"mode": "controlled", "confirm": True, "actor": "test"})
        self.assertTrue(ok_mode)
        self.assertEqual("controlled", ((mode_body.get("policy") or {}).get("mode")))

        switch_ok, switched = runtime.llm_models_switch(
            {
                "provider": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "purpose": "chat",
                "confirm": True,
            }
        )
        self.assertTrue(switch_ok)
        self.assertIn("Switched to openrouter:openai/gpt-4o-mini", str(switched.get("message") or ""))

        ok_safe, safe_body = runtime.llm_control_mode_set({"mode": "safe", "confirm": True, "actor": "test"})
        self.assertTrue(ok_safe)
        self.assertEqual("safe", ((safe_body.get("policy") or {}).get("mode")))

        blocked_again_ok, blocked_again = runtime.llm_models_switch(
            {
                "provider": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "purpose": "chat",
                "confirm": True,
            }
        )
        self.assertFalse(blocked_again_ok)
        self.assertEqual("safe_mode_remote_switch_blocked", blocked_again.get("error_kind"))
        target_status = runtime.safe_mode_target_status()
        self.assertEqual("ollama:qwen3.5:4b", target_status.get("effective_model"))
        self.assertTrue(bool(target_status.get("effective_local")))

    def test_llm_control_mode_endpoint_is_loopback_only_and_requires_confirm(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, *, client_host: str, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/llm/control_mode"
                self.headers = {"Content-Length": "0"}
                self.client_address = (client_host, 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}
                self._payload = payload

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        forbidden = _HandlerForTest(runtime, client_host="10.0.0.8", payload={"mode": "controlled", "confirm": True})
        forbidden.do_POST()
        self.assertEqual(403, forbidden.status_code)
        self.assertEqual("forbidden", forbidden.response_payload.get("error_kind"))
        self.assertTrue(bool(forbidden.response_payload.get("operator_only")))
        self.assertIn("operator-only", str(forbidden.response_payload.get("message") or "").lower())
        self.assertIn("loopback", str(forbidden.response_payload.get("why") or "").lower())
        self.assertTrue(str(forbidden.response_payload.get("next_action") or "").strip())

        confirm = _HandlerForTest(runtime, client_host="127.0.0.1", payload={"mode": "controlled"})
        confirm.do_POST()
        self.assertEqual(400, confirm.status_code)
        self.assertEqual("confirm_required", confirm.response_payload.get("error_kind"))
        self.assertIn("was not changed", str(confirm.response_payload.get("message") or "").lower())
        self.assertIn("mutates runtime policy", str(confirm.response_payload.get("why") or "").lower())
        self.assertIn('"confirm": true', str(confirm.response_payload.get("next_action") or ""))

        allowed = _HandlerForTest(runtime, client_host="127.0.0.1", payload={"mode": "controlled", "confirm": True})
        allowed.do_POST()
        self.assertEqual(200, allowed.status_code)
        self.assertTrue(bool(allowed.response_payload.get("ok")))
        policy = allowed.response_payload.get("policy") if isinstance(allowed.response_payload.get("policy"), dict) else {}
        self.assertEqual("controlled", policy.get("mode"))

    def test_defaults_sets_last_chat_model_on_successful_chat_model_change(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        document = runtime.registry_document
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        models["ollama:qwen2.5:3b-instruct"] = {
            "provider": "ollama",
            "model": "qwen2.5:3b-instruct",
            "capabilities": ["chat"],
            "enabled": True,
            "available": True,
            "quality_rank": 3,
            "cost_rank": 1,
            "default_for": ["chat"],
            "pricing": {
                "input_per_million_tokens": None,
                "output_per_million_tokens": None,
            },
            "max_context_tokens": 32768,
        }
        document["models"] = models
        runtime._save_registry_document(document)
        runtime.update_defaults({"chat_model": "ollama:llama3"})

        ok, updated = runtime.update_defaults({"chat_model": "ollama:qwen2.5:3b-instruct"})
        self.assertTrue(ok)
        self.assertEqual("ollama:qwen2.5:3b-instruct", updated["chat_model"])
        self.assertEqual("ollama:llama3", updated["last_chat_model"])

    def test_defaults_rollback_swaps_chat_and_last_chat_model(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        document = runtime.registry_document
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        models["ollama:qwen2.5:3b-instruct"] = {
            "provider": "ollama",
            "model": "qwen2.5:3b-instruct",
            "capabilities": ["chat"],
            "enabled": True,
            "available": True,
            "quality_rank": 3,
            "cost_rank": 1,
            "default_for": ["chat"],
            "pricing": {
                "input_per_million_tokens": None,
                "output_per_million_tokens": None,
            },
            "max_context_tokens": 32768,
        }
        document["models"] = models
        runtime._save_registry_document(document)
        runtime.update_defaults({"chat_model": "ollama:llama3"})
        runtime.update_defaults({"chat_model": "ollama:qwen2.5:3b-instruct"})

        ok, body = runtime.rollback_defaults()
        self.assertTrue(ok)
        self.assertEqual("ollama:llama3", body["chat_model"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", body["last_chat_model"])

        ok_second, body_second = runtime.rollback_defaults()
        self.assertTrue(ok_second)
        self.assertEqual("ollama:qwen2.5:3b-instruct", body_second["chat_model"])
        self.assertEqual("ollama:llama3", body_second["last_chat_model"])

    def test_defaults_rollback_errors_when_target_missing_or_invalid(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document.setdefault("defaults", {})
        runtime.registry_document["defaults"]["last_chat_model"] = None
        runtime._save_registry_document(runtime.registry_document)

        ok_missing, body_missing = runtime.rollback_defaults()
        self.assertFalse(ok_missing)
        self.assertEqual("no_rollback_available", body_missing["error"])
        self.assertEqual("no_rollback_available", body_missing["error_kind"])

        document = runtime.registry_document
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        defaults["last_chat_model"] = "ollama:missing-chat-model"
        document["defaults"] = defaults
        runtime._save_registry_document(document)

        ok_invalid, body_invalid = runtime.rollback_defaults()
        self.assertFalse(ok_invalid)
        self.assertEqual("rollback_target_invalid", body_invalid["error"])
        self.assertEqual("rollback_target_invalid", body_invalid["error_kind"])

    def test_defaults_accepts_provider_scoped_model_name_and_returns_canonical(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        document = runtime.registry_document
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        models["ollama:qwen2.5:3b-instruct"] = {
            "provider": "ollama",
            "model": "qwen2.5:3b-instruct",
            "capabilities": ["chat"],
            "quality_rank": 3,
            "cost_rank": 1,
            "default_for": ["chat"],
            "enabled": True,
            "pricing": {
                "input_per_million_tokens": None,
                "output_per_million_tokens": None,
            },
            "max_context_tokens": 32768,
        }
        document["models"] = models
        runtime._save_registry_document(document)

        ok, updated = runtime.update_defaults(
            {
                "default_provider": "ollama",
                "default_model": "qwen2.5:3b-instruct",
            }
        )
        self.assertTrue(ok)
        self.assertEqual("ollama", updated["default_provider"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", updated["default_model"])

        current = runtime.get_defaults()
        self.assertEqual("ollama", current["default_provider"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", current["default_model"])

    def test_defaults_migrates_legacy_default_model_to_chat_model(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        document = runtime.registry_document
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        defaults.pop("chat_model", None)
        defaults["default_model"] = "ollama:llama3"
        defaults["default_provider"] = "ollama"
        document["defaults"] = defaults
        runtime.registry_document = document

        current = runtime.get_defaults()
        self.assertEqual("ollama:llama3", current["chat_model"])
        self.assertEqual("ollama:llama3", current["default_model"])
        self.assertEqual("ollama:llama3", current["resolved_default_model"])

    def test_chat_delegates_to_orchestrator_for_execution(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        captured: dict[str, object] = {}

        class _FakeOrchestrator:
            def handle_message(self, text: str, *, user_id: str, chat_context: dict[str, object] | None = None) -> OrchestratorResponse:
                captured["text"] = text
                captured["user_id"] = user_id
                captured["chat_context"] = dict(chat_context or {})
                return OrchestratorResponse(
                    "ok",
                    {
                        "route": "generic_chat",
                        "used_runtime_state": True,
                        "used_llm": True,
                        "used_memory": True,
                        "used_tools": [],
                        "ok": True,
                        "provider": "ollama",
                        "model": "ollama:qwen2.5:3b-instruct",
                        "fallback_used": False,
                        "attempts": [],
                        "duration_ms": 1,
                    },
                )

        fake_orchestrator = _FakeOrchestrator()

        with patch.object(runtime, "orchestrator", return_value=fake_orchestrator), patch(
            "agent.api_server.route_inference",
            side_effect=AssertionError("runtime.chat should delegate to the orchestrator"),
        ), patch(
            "agent.api_server.prepare_runtime_chat_request",
            side_effect=AssertionError("runtime.chat should not run chat preflight directly"),
        ), patch.object(
            runtime._router,
            "chat",
            side_effect=AssertionError("runtime.chat should not call _router.chat directly"),
        ):
            ok_chat, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "memory_context_text": "MEMORY[S-DET]",
                    "session_id": "session-1",
                    "thread_id": "thread-1",
                }
            )

        self.assertTrue(ok_chat)
        self.assertTrue(body["ok"])
        self.assertEqual("hello", captured.get("text"))
        self.assertEqual("api:session-1", captured.get("user_id"))
        chat_context = captured.get("chat_context") if isinstance(captured.get("chat_context"), dict) else {}
        self.assertEqual("thread-1", chat_context.get("thread_id"))
        self.assertEqual("MEMORY[S-DET]", chat_context.get("memory_context_text"))
        self.assertEqual([{"role": "user", "content": "hello"}], chat_context.get("messages"))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("generic_chat", meta.get("route"))
        self.assertTrue(bool(meta.get("used_runtime_state")))
        self.assertTrue(bool(meta.get("used_llm")))
        self.assertTrue(bool(meta.get("used_memory")))
        self.assertFalse(hasattr(runtime, "_maybe_handle_setup_chat"))
        self.assertFalse(hasattr(runtime, "_setup_chat_response"))
        self.assertFalse(hasattr(runtime, "_runtime_state_unavailable_response"))
        source = inspect.getsource(AgentRuntime.chat)
        self.assertNotIn("_router.chat(", source)
        self.assertNotIn("route_inference(", source)
        self.assertNotIn("classify_runtime_chat_route(", source)
        self.assertNotIn("_maybe_handle_setup_chat(", source)
        self.assertNotIn("prepare_runtime_chat_request(", source)
        self.assertIn("self.orchestrator().handle_message(", source)
        self.assertIn("serialize_orchestrator_chat_response(", source)

    def test_chat_normalizes_legacy_message_payload_to_messages(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        captured: dict[str, object] = {}

        class _FakeOrchestrator:
            def handle_message(self, text: str, *, user_id: str, chat_context: dict[str, object] | None = None) -> OrchestratorResponse:
                captured["text"] = text
                captured["chat_context"] = dict(chat_context or {})
                return OrchestratorResponse(
                    "legacy ok",
                    {
                        "route": "generic_chat",
                        "used_runtime_state": True,
                        "used_llm": True,
                        "used_memory": False,
                        "used_tools": [],
                        "ok": True,
                        "provider": "ollama",
                        "model": "ollama:qwen2.5:3b-instruct",
                        "fallback_used": False,
                        "attempts": [],
                        "duration_ms": 1,
                    },
                )

        with patch.object(runtime, "orchestrator", return_value=_FakeOrchestrator()), patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            return_value=None,
        ):
            ok_chat, body = runtime.chat({"message": "tell me a joke"})

        self.assertTrue(ok_chat)
        self.assertTrue(body["ok"])
        self.assertEqual("tell me a joke", captured.get("text"))
        chat_context = captured.get("chat_context") if isinstance(captured.get("chat_context"), dict) else {}
        self.assertEqual(
            [{"role": "user", "content": "tell me a joke"}],
            chat_context.get("messages"),
        )

    def test_chat_model_status_query_works_from_legacy_message_payload(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            ok_chat, body = runtime.chat({"message": "what model are you using?"})

        self.assertTrue(ok_chat)
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("model_status", meta.get("route"))
        self.assertNotIn("messages must be a non-empty list", str(body.get("message") or ""))

    def test_deterministic_chat_routes_do_not_emit_model_selection_meta(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("LLM should not run"),
        ):
            ok_chat, body = runtime.chat({"messages": [{"role": "user", "content": "what model are you using right now?"}]})

        self.assertTrue(ok_chat)
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("model_status", meta.get("route"))
        self.assertFalse(bool(meta.get("used_llm")))
        self.assertIsNone(meta.get("provider"))
        self.assertIsNone(meta.get("model"))

    def test_chat_uses_serializer_helper_for_response_mapping(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _FakeOrchestrator:
            def handle_message(self, text: str, *, user_id: str, chat_context: dict[str, object] | None = None) -> OrchestratorResponse:
                return OrchestratorResponse(
                    "serializer source",
                    {
                        "route": "provider_status",
                        "used_runtime_state": True,
                        "used_llm": False,
                        "used_memory": False,
                        "used_tools": [],
                        "runtime_payload": {
                            "type": "provider_status",
                            "provider": "openrouter",
                            "configured": False,
                        },
                    },
                )

        expected = SerializedChatResponse(
            ok=True,
            body={
                "ok": True,
                "assistant": {"role": "assistant", "content": "serializer result"},
                "message": "serializer result",
                "meta": {
                    "route": "provider_status",
                    "used_runtime_state": True,
                    "used_llm": False,
                    "used_memory": False,
                    "used_tools": [],
                },
                "setup": {
                    "type": "provider_status",
                    "provider": "openrouter",
                    "configured": False,
                },
            },
            route="provider_status",
            route_reason="provider_status",
            generic_fallback_allowed=False,
            generic_fallback_reason=None,
        )

        with patch.object(runtime, "orchestrator", return_value=_FakeOrchestrator()), patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            return_value=None,
        ), patch(
            "agent.api_server.serialize_orchestrator_chat_response",
            return_value=expected,
        ) as serialize_mock:
            ok_chat, body = runtime.chat({"messages": [{"role": "user", "content": "is openrouter configured?"}]})

        self.assertTrue(ok_chat)
        self.assertEqual(expected.body, body)
        serialize_mock.assert_called_once()
        _, kwargs = serialize_mock.call_args
        self.assertEqual("api", kwargs["source_surface"])
        self.assertEqual("api:default", kwargs["user_id"])
        self.assertEqual("api:default:thread", kwargs["thread_id"])
        self.assertIsInstance(kwargs["autopilot_meta"], dict)

    def test_chat_preserves_response_envelope_for_preflight_short_circuit(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        prepared = PreparedChatRequest(
            messages=[{"role": "user", "content": "hello"}],
            last_user_text="hello",
            provider_override="openrouter",
            model_override="openrouter:base-chat",
            require_tools=False,
            selection_reason="premium_over_cap_confirmation_required",
            escalation_reasons=("user_request",),
            premium_selected_model="openrouter:premium-reasoner",
            direct_result={
                "ok": True,
                "text": "Premium escalation is over the cost cap.",
                "provider": "openrouter",
                "model": "openrouter:base-chat",
                "fallback_used": False,
                "attempts": [],
                "duration_ms": 0,
                "error_kind": None,
                "selection_reason": "premium_over_cap_confirmation_required",
            },
            response_selection_policy={
                "mode": "premium_over_cap",
                "baseline_model": "openrouter:base-chat",
                "premium_candidate": "openrouter:premium-reasoner",
                "premium_cost_per_1m": 2.0,
                "premium_cap_per_1m": 1.0,
                "escalation_reasons": ["user_request"],
            },
            log_reason="premium_over_cap_confirmation_required",
            log_fallback_used=False,
        )

        with patch.object(runtime._router, "enabled", return_value=True), patch(
            "agent.api_server.prepare_runtime_chat_request",
            return_value=prepared,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("short-circuit chat should not call route_inference"),
        ), patch.object(
            runtime._router,
            "chat",
            side_effect=AssertionError("runtime.chat should not call _router.chat directly"),
        ):
            ok_chat, body = runtime.chat({"messages": [{"role": "user", "content": "hello"}]})

        self.assertTrue(ok_chat)
        self.assertTrue(body["ok"])
        assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
        self.assertEqual("Premium escalation is over the cost cap.", assistant.get("content"))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("openrouter", meta.get("provider"))
        self.assertEqual("openrouter:base-chat", meta.get("model"))
        self.assertEqual("generic_chat", meta.get("route"))
        self.assertEqual([], meta.get("attempts"))
        self.assertEqual(0, meta.get("duration_ms"))
        self.assertFalse(bool(meta.get("used_llm")))
        policy = meta.get("selection_policy") if isinstance(meta.get("selection_policy"), dict) else {}
        self.assertEqual("premium_over_cap", policy.get("mode"))

    def test_chat_ordinary_message_still_works_through_orchestrator(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        router_result = {
            "ok": True,
            "text": "Hello from the orchestrator path.",
            "provider": "ollama",
            "model": "ollama:llama3",
            "fallback_used": False,
            "attempts": [],
            "duration_ms": 4,
            "error_kind": None,
            "selection_reason": "router_default",
        }

        with patch.object(runtime._router, "enabled", return_value=True), patch(
            "agent.orchestrator.route_inference",
            return_value=router_result,
        ):
            ok_chat, body = runtime.chat({"messages": [{"role": "user", "content": "hello there"}]})

        self.assertTrue(ok_chat)
        assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
        self.assertEqual("Hello from the orchestrator path.", assistant.get("content"))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("generic_chat", meta.get("route"))
        self.assertTrue(bool(meta.get("used_runtime_state")))
        self.assertTrue(bool(meta.get("used_llm")))
        self.assertFalse(bool(meta.get("used_memory")))
        self.assertEqual([], meta.get("used_tools"))
        self.assertEqual("ollama", meta.get("provider"))
        self.assertEqual("ollama:llama3", meta.get("model"))

    def test_defaults_treats_known_provider_prefix_as_full_id(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        document = runtime.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        providers["acme"] = {
            "provider_type": "openai_compat",
            "base_url": "https://acme.example",
            "chat_path": "/v1/chat/completions",
            "api_key_source": None,
            "default_headers": {},
            "default_query_params": {},
            "enabled": True,
            "local": False,
        }
        document["providers"] = providers
        runtime._save_registry_document(document)

        ok, response = runtime.update_defaults(
            {
                "default_provider": "ollama",
                "default_model": "acme:latest",
            }
        )
        self.assertFalse(ok)
        self.assertEqual("default_model not found", response["error"])

    def test_refresh_models_marks_embedding_models_as_embedding_only(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        def _fake_get(url: str, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            if url.endswith("/api/tags"):
                return {"models": [{"name": "llama3.2"}, {"name": "nomic-embed-text"}]}
            return {"data": []}

        runtime._http_get_json = _fake_get  # type: ignore[assignment]

        ok, _response = runtime.refresh_models()
        self.assertTrue(ok)

        embed_model = runtime.registry_document["models"]["ollama:nomic-embed-text"]
        self.assertEqual(["embedding"], embed_model["capabilities"])

    def test_refresh_models_ollama_uses_tags_and_adds_qwen_as_chat(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        def _fake_get(url: str, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            if url.endswith("/v1/models"):
                raise AssertionError("ollama refresh must not use /v1/models")
            if url.endswith("/api/tags"):
                return {"models": [{"name": "qwen2.5:3b-instruct"}]}
            return {}

        runtime._http_get_json = _fake_get  # type: ignore[assignment]
        ok, response = runtime.refresh_models({"provider": "ollama"})
        self.assertTrue(ok)
        self.assertTrue(response["ok"])
        model_payload = runtime.registry_document["models"]["ollama:qwen2.5:3b-instruct"]
        self.assertEqual(["chat"], model_payload["capabilities"])
        self.assertTrue(model_payload["available"])

    def test_refresh_models_quarantines_stale_ollama_entries(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        document = runtime.registry_document
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        models["ollama:stale-model"] = {
            "provider": "ollama",
            "model": "stale-model",
            "capabilities": ["chat"],
            "quality_rank": 2,
            "cost_rank": 0,
            "default_for": ["chat"],
            "enabled": True,
            "available": True,
            "pricing": {
                "input_per_million_tokens": None,
                "output_per_million_tokens": None,
            },
            "max_context_tokens": 8192,
        }
        document["models"] = models
        runtime._save_registry_document(document)

        def _fake_get(url: str, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            if url.endswith("/v1/models"):
                return {"data": [{"id": "llama3.2"}, {"id": "stale-model"}]}
            if url.endswith("/api/tags"):
                return {"models": [{"name": "llama3.2"}]}
            return {}

        runtime._http_get_json = _fake_get  # type: ignore[assignment]
        ok, _response = runtime.refresh_models()
        self.assertTrue(ok)

        available_model = runtime.registry_document["models"]["ollama:llama3.2"]
        stale_model = runtime.registry_document["models"]["ollama:stale-model"]
        self.assertTrue(available_model["available"])
        self.assertFalse(stale_model["available"])

    def test_providers_and_models_include_health_fields(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        providers_payload = runtime.list_providers()
        self.assertTrue(bool(providers_payload["operator_surface"]))
        self.assertTrue(bool(providers_payload["non_canonical_for_assistant"]))
        self.assertIn("runtime_truth.providers_status", str(providers_payload["canonical_truth_source"]))
        self.assertTrue(providers_payload["providers"])
        first_provider = providers_payload["providers"][0]
        self.assertIn("health", first_provider)
        self.assertIn("status", first_provider["health"])

        models_payload = runtime.models()
        self.assertTrue(models_payload["models"])
        first_model = models_payload["models"][0]
        self.assertIn("health", first_model)
        self.assertIn("status", first_model["health"])

    def test_providers_surface_uses_canonical_truth_and_only_adds_operator_detail(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document["providers"] = {
            "openrouter": {
                "provider_type": "openai_compat",
                "base_url": "https://openrouter.ai/api/v1",
                "chat_path": "/chat/completions",
                "enabled": True,
                "local": False,
            }
        }
        runtime.registry_document["models"] = {
            "openrouter:openai/gpt-4o-mini": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
            }
        }

        class _FakeTruth:
            def providers_status(self) -> dict[str, object]:
                return {
                    "providers": [
                        {
                            "provider": "openrouter",
                            "enabled": True,
                            "local": False,
                            "configured": True,
                            "active": False,
                            "effective_active": False,
                            "auth_required": True,
                            "secret_present": True,
                            "health_status": "ok",
                            "health_reason": None,
                            "connection_state": "configured_and_usable",
                            "selection_state": "configured_and_usable",
                            "usable_for_selection": True,
                            "policy_blocked": False,
                            "model_ids": ["openrouter:openai/gpt-4o-mini"],
                        }
                    ]
                }

            def model_readiness_status(self) -> dict[str, object]:
                return {
                    "models": [
                        {
                            "model_id": "openrouter:openai/gpt-4o-mini",
                            "provider_id": "openrouter",
                            "model_name": "openai/gpt-4o-mini",
                            "capabilities": ["chat"],
                            "enabled": True,
                            "available": True,
                            "local": False,
                            "active": False,
                            "installed": False,
                            "configured": True,
                            "usable_now": True,
                            "quality_rank": 5,
                            "cost_rank": 2,
                            "context_window": 128000,
                            "price_in": 0.15,
                            "price_out": 0.6,
                            "availability_state": "usable_now",
                            "availability_reason": "healthy and ready now",
                            "eligibility_state": "usable_now",
                            "eligibility_reason": "healthy and ready now",
                            "provider_connection_state": "configured_and_usable",
                            "provider_selection_state": "configured_and_usable",
                            "lifecycle_state": "ready",
                            "lifecycle_message": "Ready now.",
                            "acquirable": False,
                            "acquisition_state": "ready_now",
                            "acquisition_reason": None,
                            "model_health_status": "ok",
                        }
                    ]
                }

            def _provider_health_row(self, provider_id: str) -> dict[str, object]:
                _ = provider_id
                return {"status": "down", "last_error_kind": "provider_unavailable", "status_code": 503}

            def _model_health_row(self, model_id: str) -> dict[str, object]:
                _ = model_id
                return {"status": "down", "last_error_kind": "model_unavailable", "status_code": 503}

        with patch.object(runtime, "runtime_truth_service", return_value=_FakeTruth()):
            payload = runtime.list_providers()

        self.assertTrue(bool(payload["operator_surface"]))
        self.assertEqual(
            "runtime_truth.providers_status+runtime_truth.model_readiness_status",
            payload["canonical_truth_source"],
        )
        provider = payload["providers"][0]
        self.assertEqual("openrouter", provider["id"])
        self.assertEqual("ok", provider["health"]["status"])
        self.assertEqual("provider_unavailable", provider["health"]["last_error_kind"])
        self.assertEqual("configured_and_usable", provider["health"]["selection_state"])
        model = provider["models"][0]
        self.assertEqual("openrouter:openai/gpt-4o-mini", model["id"])
        self.assertEqual("ok", model["health"]["status"])
        self.assertEqual("usable_now", model["availability_state"])

    def test_health_normalization_marks_disabled_provider_and_models_with_timestamps(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        snapshot = {
            "providers": [
                {
                    "id": "openrouter",
                    "enabled": False,
                    "local": False,
                    "health": {"status": "down"},
                }
            ],
            "models": [
                {
                    "id": "openrouter:openai/gpt-4o-mini",
                    "provider": "openrouter",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "health": {"status": "down"},
                }
            ],
        }
        with patch.object(runtime._router, "doctor_snapshot", return_value=snapshot), patch.object(
            runtime,
            "llm_health_summary",
            return_value={"ok": True, "health": {"drift": {"details": {}}}},
        ):
            payload = runtime.llm_status()

        providers = payload.get("providers") if isinstance(payload.get("providers"), list) else []
        models = payload.get("models") if isinstance(payload.get("models"), list) else []
        provider_row = next(
            (row for row in providers if isinstance(row, dict) and str(row.get("id") or "").strip().lower() == "openrouter"),
            {},
        )
        model_row = next(
            (
                row
                for row in models
                if isinstance(row, dict) and str(row.get("id") or "").strip() == "openrouter:openai/gpt-4o-mini"
            ),
            {},
        )
        provider_health = provider_row.get("health") if isinstance(provider_row.get("health"), dict) else {}
        model_health = model_row.get("health") if isinstance(model_row.get("health"), dict) else {}

        self.assertEqual("down", provider_health.get("status"))
        self.assertEqual("provider_disabled", provider_health.get("last_error_kind"))
        self.assertIsInstance(provider_health.get("last_checked_at"), int)
        self.assertTrue(str(provider_health.get("last_checked_at_iso") or "").strip())

        self.assertFalse(bool(model_row.get("routable", True)))
        self.assertEqual("down", model_health.get("status"))
        self.assertEqual("provider_disabled", model_health.get("last_error_kind"))
        self.assertIsInstance(model_health.get("last_checked_at"), int)
        self.assertTrue(str(model_health.get("last_checked_at_iso") or "").strip())

    def test_provider_disabled_health_stamps_all_provider_models_even_when_enabled_flag_true(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        now_epoch = int(time.time())
        snapshot = {
            "providers": [
                {
                    "id": "openrouter",
                    "enabled": True,
                    "local": False,
                    "health": {
                        "status": "down",
                        "last_error_kind": "provider_disabled",
                        "status_code": 404,
                        "last_checked_at": now_epoch,
                    },
                }
            ],
            "models": [
                {
                    "id": "openrouter:openai/gpt-4o-mini",
                    "provider": "openrouter",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "health": {
                        "status": "down",
                        "last_checked_at": None,
                        "last_error_kind": None,
                        "status_code": None,
                        "last_status_code": None,
                    },
                },
                {
                    "id": "openrouter:anthropic/claude-3.5-sonnet",
                    "provider": "openrouter",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "health": {
                        "status": "down",
                        "last_checked_at": None,
                        "last_error_kind": None,
                        "status_code": None,
                        "last_status_code": None,
                    },
                },
            ],
        }
        with patch.object(runtime._router, "doctor_snapshot", return_value=snapshot), patch.object(
            runtime,
            "llm_health_summary",
            return_value={"ok": True, "health": {"drift": {"details": {}}}},
        ):
            payload = runtime.llm_status()

        models = payload.get("models") if isinstance(payload.get("models"), list) else []
        openrouter_rows = [
            row for row in models if isinstance(row, dict) and str(row.get("provider") or "").strip().lower() == "openrouter"
        ]
        self.assertEqual(2, len(openrouter_rows))
        self.assertTrue(all(not bool(row.get("routable", True)) for row in openrouter_rows))
        self.assertTrue(
            all(
                str(((row.get("health") if isinstance(row.get("health"), dict) else {}).get("last_error_kind") or "")).strip().lower()
                == "provider_disabled"
                for row in openrouter_rows
            )
        )
        self.assertTrue(
            all(
                isinstance((row.get("health") if isinstance(row.get("health"), dict) else {}).get("last_checked_at"), int)
                for row in openrouter_rows
            )
        )
        null_checked_for_provider_disabled = sum(
            1
            for row in openrouter_rows
            if (row.get("health") if isinstance(row.get("health"), dict) else {}).get("last_checked_at") is None
        )
        self.assertEqual(0, null_checked_for_provider_disabled)

    def test_normalize_health_record_fills_missing_fields_for_down_status(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        normalized = runtime._normalize_health_record(  # type: ignore[attr-defined]
            {
                "status": "down",
                "last_checked_at": None,
                "last_ts": None,
                "status_code": None,
                "last_status_code": None,
                "last_error_kind": None,
                "cooldown_until": None,
                "down_since": None,
                "successes": None,
                "failures": None,
                "failure_streak": None,
            },
            now_epoch=1_700_000_000,
        )
        self.assertEqual("down", normalized.get("status"))
        self.assertEqual(1_700_000_000, normalized.get("last_checked_at"))
        self.assertEqual(1_700_000_000, normalized.get("down_since"))
        self.assertEqual(1_700_000_000.0, normalized.get("last_ts"))
        self.assertEqual(0, normalized.get("successes"))
        self.assertEqual(0, normalized.get("failures"))
        self.assertEqual(0, normalized.get("failure_streak"))
        self.assertIn("last_checked_at_iso", normalized)
        self.assertIn("down_since_iso", normalized)

    def test_llm_status_includes_runtime_contract_fields(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        payload = runtime.llm_status()
        self.assertIn("runtime_mode", payload)
        self.assertIn(payload["runtime_mode"], {"READY", "BOOTSTRAP_REQUIRED", "DEGRADED", "FAILED"})
        self.assertIn("runtime_status", payload)
        runtime_status = payload.get("runtime_status") if isinstance(payload.get("runtime_status"), dict) else {}
        self.assertEqual(payload["runtime_mode"], runtime_status.get("runtime_mode"))
        self.assertIn("summary", runtime_status)
        self.assertIn("next_action", runtime_status)
        self.assertTrue(bool(payload.get("compat_only")))
        self.assertTrue(bool(payload.get("non_canonical_for_assistant")))
        self.assertIn("runtime_truth.current_chat_target_status", str(payload.get("canonical_truth_source") or ""))

    def test_llm_status_uses_canonical_snapshot_for_remote_default_health(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        snapshot = {
            "providers": [
                {
                    "id": "openrouter",
                    "enabled": True,
                    "local": False,
                    "available": True,
                    "health": {"status": "ok"},
                }
            ],
            "models": [
                {
                    "id": "openrouter:openai/gpt-4o-mini",
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "capabilities": ["chat", "json", "tools"],
                    "health": {"status": "ok"},
                }
            ],
        }
        with patch.object(
            runtime,
            "get_defaults",
            return_value={
                "default_provider": "openrouter",
                "default_model": "openrouter:openai/gpt-4o-mini",
                "chat_model": "openrouter:openai/gpt-4o-mini",
                "embed_model": None,
                "last_chat_model": None,
                "routing_mode": "prefer_local_lowest_cost_capable",
                "allow_remote_fallback": True,
            },
        ), patch.object(runtime._router, "doctor_snapshot", return_value=snapshot), patch.object(
            runtime,
            "llm_health_summary",
            return_value={
                "ok": True,
                "health": {
                    "drift": {
                        "details": {
                            "resolved_default_model": "openrouter:openai/gpt-4o-mini",
                        }
                    }
                },
            },
        ):
            payload = runtime.llm_status()

        self.assertEqual("openrouter", payload["default_provider"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", payload["resolved_default_model"])
        self.assertEqual("unknown", payload["active_provider_health"]["status"])
        self.assertEqual("unknown", payload["active_model_health"]["status"])
        self.assertEqual("BOOTSTRAP_REQUIRED", payload["runtime_mode"])

    def test_telegram_secret_and_test_endpoints(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        status_before = runtime.telegram_status()
        self.assertTrue(status_before["ok"])
        self.assertFalse(status_before["configured"])

        ok, saved = runtime.set_telegram_secret({"bot_token": "1234:abcd"})
        self.assertTrue(ok)
        self.assertTrue(saved["ok"])
        self.assertEqual("1234:abcd", runtime.secret_store.get_secret("telegram:bot_token"))
        self.assertTrue(runtime.telegram_status()["configured"])

        class _FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                _ = exc_type
                _ = exc
                _ = tb
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "id": 99,
                            "is_bot": True,
                            "first_name": "PA",
                            "username": "personal_agent_bot",
                        },
                    }
                ).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=_FakeResponse()):
            ok, tested = runtime.test_telegram()

        self.assertTrue(ok)
        self.assertTrue(tested["ok"])
        self.assertEqual(99, tested["telegram_user"]["id"])
        self.assertEqual("personal_agent_bot", tested["telegram_user"]["username"])
        self.assertNotIn("bot_token", json.dumps(tested, ensure_ascii=True))

    def test_start_embedded_telegram_sets_running_state_and_prints_events(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, telegram_enabled=True))
        started = threading.Event()

        class _FakeUpdater:
            async def start_polling(self, **_kwargs: object) -> None:
                started.set()

            async def stop(self) -> None:
                return

        class _FakeApp:
            def __init__(self) -> None:
                self.updater = _FakeUpdater()

            async def initialize(self) -> None:
                return

            async def start(self) -> None:
                return

            async def stop(self) -> None:
                return

            async def shutdown(self) -> None:
                return

        output = io.StringIO()
        with redirect_stdout(output):
            started_ok = runtime.start_embedded_telegram(
                token_resolver=lambda: "x",
                app_factory=lambda **_kwargs: _FakeApp(),
            )
            self.assertTrue(started_ok)
            self.assertTrue(started.wait(1.0))
            for _ in range(20):
                if runtime.telegram_status().get("last_event") == "telegram.started":
                    break
                time.sleep(0.05)
            status = runtime.telegram_status()
            self.assertTrue(bool(status.get("embedded_running")))
            runtime.stop_embedded_telegram()
        text = output.getvalue()
        self.assertIn("telegram.embedded: start called", text)
        self.assertIn("telegram.embedded: start result=true", text)
        self.assertIn("telegram.started", text)

    def test_version_endpoint_returns_runtime_metadata(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

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

        handler = _HandlerForTest(runtime, "/version")
        handler.do_GET()

        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertIsInstance(payload["pid"], int)
        self.assertTrue(str(payload.get("version") or ""))
        self.assertTrue(str(payload.get("version_source") or ""))

    def test_legacy_model_scout_http_endpoints_are_removed(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(
                self,
                runtime_obj: AgentRuntime,
                path: str,
                payload: dict[str, object] | None = None,
                *,
                client_address: tuple[str, int] = ("127.0.0.1", 12345),
            ) -> None:
                self.runtime = runtime_obj
                self.path = path
                self.headers = {}
                self.client_address = client_address
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

        for path in (
            "/model_scout/status",
            "/model_scout/suggestions",
            "/model_scout/sources",
            "/model_scout/run",
            "/llm/scout/run",
            "/model_scout/suggestions/local%3Aabc/dismiss",
            "/model_scout/suggestions/local%3Aabc/mark_installed",
        ):
            handler = _HandlerForTest(runtime, path, {} if path.endswith("/run") or "/suggestions/" in path else None)
            if path in {
                "/model_scout/run",
                "/llm/scout/run",
                "/model_scout/suggestions/local%3Aabc/dismiss",
                "/model_scout/suggestions/local%3Aabc/mark_installed",
            }:
                handler.do_POST()
            else:
                handler.do_GET()
            self.assertEqual(404, handler.status_code, path)
            payload = json.loads(handler.body.decode("utf-8"))
            self.assertEqual("not_found", payload.get("error"))

    def test_desktop_model_scout_tab_uses_canonical_llm_model_surfaces(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        app_source = Path(repo_root, "desktop", "src", "App.jsx").read_text(encoding="utf-8")
        tab_source = Path(repo_root, "desktop", "src", "components", "ModelScoutTab.jsx").read_text(encoding="utf-8")
        vite_source = Path(repo_root, "desktop", "vite.config.js").read_text(encoding="utf-8")

        for legacy_path in (
            "/model_scout/status",
            "/model_scout/suggestions",
            "/model_scout/sources",
            "/model_scout/run",
            "/llm/scout/run",
        ):
            self.assertNotIn(legacy_path, app_source)
            self.assertNotIn(legacy_path, tab_source)
            self.assertNotIn(legacy_path, vite_source)

        self.assertIn("/llm/models/check", app_source)
        self.assertIn("/llm/models/lifecycle", app_source)
        self.assertIn('"/llm": apiTarget', vite_source)

    def test_model_watch_routes_exist_and_are_not_not_found(self) -> None:
        os.environ["AGENT_MODEL_WATCH_CATALOG_PATH"] = os.path.join(self.tmpdir.name, "catalog.json")
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                model_watch_state_path=os.path.join(self.tmpdir.name, "model_watch_state.json"),
                model_watch_config_path=os.path.join(self.tmpdir.name, "model_watch_config.json"),
            )
        )
        with open(runtime.config.model_watch_config_path, "w", encoding="utf-8") as handle:
            json.dump({"huggingface_watch_authors": []}, handle, ensure_ascii=True)

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

        latest_handler = _HandlerForTest(runtime, "/model_watch/latest")
        latest_handler.do_GET()
        self.assertEqual(200, latest_handler.status_code)
        latest_payload = json.loads(latest_handler.body.decode("utf-8"))
        self.assertTrue(latest_payload["ok"])
        self.assertFalse(latest_payload["found"])
        self.assertNotEqual("not_found", latest_payload.get("error"))

        hf_status_handler = _HandlerForTest(runtime, "/model_watch/hf/status")
        hf_status_handler.do_GET()
        self.assertEqual(200, hf_status_handler.status_code)
        hf_status_payload = json.loads(hf_status_handler.body.decode("utf-8"))
        self.assertTrue(hf_status_payload["ok"])
        self.assertIn("enabled", hf_status_payload)

        with patch.object(runtime, "run_model_watch_once", return_value=(True, {"ok": True, "fetched_candidates": 12})):
            run_handler = _HandlerForTest(runtime, "/model_watch/run", {})
            run_handler.do_POST()
            self.assertEqual(200, run_handler.status_code)
            run_payload = json.loads(run_handler.body.decode("utf-8"))
            self.assertTrue(run_payload["ok"])
            self.assertEqual(12, run_payload["fetched_candidates"])

        with patch.object(runtime, "model_watch_refresh", return_value=(True, {"ok": True, "model_count": 5})):
            refresh_handler = _HandlerForTest(runtime, "/model_watch/refresh", {})
            refresh_handler.do_POST()
            self.assertEqual(200, refresh_handler.status_code)
            refresh_payload = json.loads(refresh_handler.body.decode("utf-8"))
            self.assertTrue(refresh_payload["ok"])
            self.assertEqual(5, refresh_payload["model_count"])

        with patch.object(
            runtime,
            "model_watch_hf_scan",
            return_value=(True, {"ok": True, "scan": {"ok": True, "discovered_count": 1}, "proposal_created": True}),
        ):
            hf_scan_handler = _HandlerForTest(runtime, "/model_watch/hf/scan", {})
            hf_scan_handler.do_POST()
            self.assertEqual(200, hf_scan_handler.status_code)
            hf_scan_payload = json.loads(hf_scan_handler.body.decode("utf-8"))
            self.assertTrue(hf_scan_payload["ok"])
            self.assertTrue(hf_scan_payload["proposal_created"])

    def test_model_endpoint_returns_current_selection_payload(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
                self.runtime = runtime_obj
                self.path = path
                self.headers = {}
                self.status_code = 0
                self.body = b""

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                self.status_code = status
                self.body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")

            def _send_bytes(
                self,
                status: int,
                body: bytes,
                *,
                content_type: str,
                cache_control: str | None = None,
            ) -> None:
                _ = (content_type, cache_control)
                self.status_code = status
                self.body = body

        handler = _HandlerForTest(runtime, "/model")
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertIn("current", payload)
        self.assertIn("provider", payload["current"])
        self.assertIn("model_id", payload["current"])
        self.assertIn("selection_policy", payload)
        self.assertIn("model_watch", payload)
        self.assertIn("llm_availability", payload)
        llm_availability = payload.get("llm_availability") if isinstance(payload.get("llm_availability"), dict) else {}
        ollama = llm_availability.get("ollama") if isinstance(llm_availability.get("ollama"), dict) else {}
        for key in (
            "configured_base_url",
            "native_base",
            "openai_base",
            "native_ok",
            "openai_compat_ok",
            "last_error_kind",
            "last_status_code",
        ):
            self.assertIn(key, ollama)
        self.assertTrue(bool(payload.get("compat_only")))
        self.assertTrue(bool(payload.get("non_canonical_for_assistant")))
        self.assertIn("runtime_truth.current_chat_target_status", str(payload.get("canonical_assistant_surface") or ""))

    def test_llm_model_alias_returns_identical_payload(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        fixed_payload = {
            "ok": True,
            "current": {"provider": "openrouter", "model_id": "openrouter:gpt-4o-mini"},
            "selection_policy": {},
            "model_watch": {},
            "llm_availability": {},
        }

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
                self.runtime = runtime_obj
                self.path = path
                self.headers = {}
                self.status_code = 0
                self.body = b""

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                self.status_code = status
                self.body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")

            def _send_bytes(
                self,
                status: int,
                body: bytes,
                *,
                content_type: str,
                cache_control: str | None = None,
            ) -> None:
                _ = (content_type, cache_control)
                self.status_code = status
                self.body = body

        with patch.object(runtime, "model_status", return_value=fixed_payload):
            model_handler = _HandlerForTest(runtime, "/model")
            model_handler.do_GET()
            llm_model_handler = _HandlerForTest(runtime, "/llm/model")
            llm_model_handler.do_GET()

        self.assertEqual(200, model_handler.status_code)
        self.assertEqual(200, llm_model_handler.status_code)
        model_payload = json.loads(model_handler.body.decode("utf-8"))
        llm_model_payload = json.loads(llm_model_handler.body.decode("utf-8"))
        self.assertEqual(model_payload, llm_model_payload)

    def test_model_endpoint_does_not_leak_secrets(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        raw_secret = "sk-test-model-endpoint-secret"
        runtime.secret_store.set_secret("provider:openrouter:api_key", raw_secret)

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
                self.runtime = runtime_obj
                self.path = path
                self.headers = {}
                self.status_code = 0
                self.body = b""

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                self.status_code = status
                self.body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")

            def _send_bytes(
                self,
                status: int,
                body: bytes,
                *,
                content_type: str,
                cache_control: str | None = None,
            ) -> None:
                _ = (content_type, cache_control)
                self.status_code = status
                self.body = body

        handler = _HandlerForTest(runtime, "/model")
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        body_text = handler.body.decode("utf-8")
        self.assertNotIn("sk-", body_text)
        self.assertNotIn("OPENROUTER_API_KEY", body_text)
        self.assertNotIn(raw_secret, body_text)

    def test_run_model_watch_once_does_not_raise_nameerror(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch("agent.api_server.run_watch_once_for_config", return_value={"ok": True, "fetched_candidates": 3}), patch(
            "agent.api_server.log_event"
        ) as log_event_mock:
            ok, payload = runtime.run_model_watch_once(trigger="manual")
        self.assertTrue(ok)
        self.assertTrue(payload["ok"])
        self.assertEqual(3, payload["fetched_candidates"])
        self.assertTrue(log_event_mock.called)

    def test_scheduler_loop_survives_scheduled_job_exception(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._scheduler_next_run = {"model_watch": 0.0}
        stop_event = threading.Event()
        calls: list[str] = []
        sleep_calls: list[float] = []

        def _model_watch_side_effect(*, trigger: str = "scheduler") -> tuple[bool, dict[str, object]]:
            calls.append(str(trigger))
            if len(calls) == 1:
                raise RuntimeError("boom")
            return True, {"ok": True, "next_check_after_seconds": 1}

        with patch.object(runtime, "run_model_watch_once", side_effect=_model_watch_side_effect):
            runtime._scheduler_loop(
                sleep_fn=lambda seconds: sleep_calls.append(float(seconds)),
                stop_event=stop_event,
                max_iters=2,
            )

        self.assertEqual(["scheduler"], calls)
        self.assertEqual([1.0, 1.0], sleep_calls)
        self.assertGreater(float(runtime._scheduler_next_run.get("model_watch") or 0.0), 0.0)

    def test_scheduler_loop_does_not_die_when_log_event_nameerror_occurs(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._scheduler_next_run = {"model_watch": 0.0}
        stop_event = threading.Event()

        with patch.object(
            runtime,
            "run_model_watch_once",
            return_value=(False, {"ok": False, "error": "run_failed", "next_check_after_seconds": 1}),
        ), patch("agent.api_server.log_event", side_effect=NameError("log_event is not defined")):
            runtime._scheduler_loop(
                sleep_fn=lambda _seconds: None,
                stop_event=stop_event,
                max_iters=1,
            )

        self.assertIn("model_watch", runtime._scheduler_next_run)

    def test_run_server_starts_embedded_telegram_runner(self) -> None:
        started: list[bool] = []
        closed: list[bool] = []
        listening_marked: list[bool] = []

        class _FakeRuntime:
            listening_url = "http://127.0.0.1:8765"
            pid = 123
            registry_store = type("RegistryStoreStub", (), {"path": "/tmp/registry.json"})()
            version = "test"
            git_commit = "abc123"

            def set_listening(self, _host: str, _port: int) -> None:
                return

            def mark_server_listening(self) -> None:
                listening_marked.append(True)

            def start_embedded_telegram(self) -> bool:
                started.append(True)
                return True

            def close(self) -> None:
                closed.append(True)

        class _FakeServer:
            def __init__(self, _addr: tuple[str, int], _handler: object) -> None:
                return

            def serve_forever(self) -> None:
                raise KeyboardInterrupt()

            def server_close(self) -> None:
                return

        with patch("agent.api_server.build_runtime", return_value=_FakeRuntime()), patch(
            "agent.api_server.ThreadingHTTPServer",
            _FakeServer,
        ):
            from agent.api_server import run_server

            run_server("127.0.0.1", 8765)

        self.assertEqual([True], listening_marked)
        self.assertEqual([True], started)
        self.assertEqual([True], closed)

    def test_ready_responds_quickly_while_warmup_pending_then_ready_after_completion(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        warmup_gate = threading.Event()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        runtime._record_authoritative_provider_success("ollama", "ollama:qwen2.5:7b-instruct")  # type: ignore[attr-defined]

        def _slow_bootstrap() -> None:
            warmup_gate.wait(2.0)

        with runtime._startup_warmup_lock:
            runtime._startup_warmup_remaining = ["memory_bootstrap"]

        with patch.object(runtime, "_initialize_memory_v2_bootstrap", side_effect=_slow_bootstrap):
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

            try:
                runtime._memory_v2_store = object()  # type: ignore[assignment]
                with patch.object(runtime, "_memory_v2_bootstrap_completed", return_value=False), patch(
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
                    runtime.set_listening("127.0.0.1", 8765)
                    runtime.mark_server_listening()

                    start = time.time()
                    handler = _HandlerForTest(runtime, "/ready")
                    handler.do_GET()
                    elapsed = time.time() - start
                    payload = json.loads(handler.body.decode("utf-8"))
                    self.assertLess(elapsed, 0.5)
                    self.assertTrue(payload.get("ok"))
                    self.assertFalse(payload.get("ready"))
                    self.assertEqual("warmup", payload.get("phase"))
                    self.assertIn(payload.get("startup_phase"), {"warming", "listening"})
                    self.assertEqual(["memory_bootstrap"], payload.get("warmup_remaining"))

                    warmup_gate.set()
                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        if runtime.startup_phase == "ready":
                            break
                        time.sleep(0.02)
                    handler_ready = _HandlerForTest(runtime, "/ready")
                    handler_ready.do_GET()
                    payload_ready = json.loads(handler_ready.body.decode("utf-8"))
                    self.assertTrue(payload_ready.get("ready"))
                    self.assertEqual("ready", payload_ready.get("phase"))
                    self.assertEqual("ready", payload_ready.get("startup_phase"))
                    self.assertEqual([], payload_ready.get("warmup_remaining"))
            finally:
                warmup_gate.set()
                runtime.close()

    def test_deferred_warmup_skips_native_and_router_prebind(self) -> None:
        call_order: list[str] = []
        original_native = AgentRuntime._ensure_native_packs_registered
        original_reload = AgentRuntime._reload_router
        original_chat_bootstrap = AgentRuntime._ensure_chat_runtime_bootstrapped

        def _wrapped_native(runtime_obj: AgentRuntime) -> None:
            call_order.append("native_packs")
            return original_native(runtime_obj)

        def _wrapped_reload(runtime_obj: AgentRuntime) -> None:
            call_order.append("router_reload")
            return original_reload(runtime_obj)

        def _wrapped_chat_bootstrap(runtime_obj: AgentRuntime):  # type: ignore[no-untyped-def]
            call_order.append("chat_runtime_bootstrap")
            return original_chat_bootstrap(runtime_obj)

        with patch.object(AgentRuntime, "_ensure_native_packs_registered", _wrapped_native), patch.object(
            AgentRuntime, "_reload_router", _wrapped_reload
        ), patch.object(
            AgentRuntime, "_ensure_chat_runtime_bootstrapped", _wrapped_chat_bootstrap
        ):
            runtime = AgentRuntime(_config(self.registry_path, self.db_path), defer_bootstrap_warmup=True)
            try:
                self.assertEqual([], call_order)
                self.assertEqual(
                    ["native_packs", "router_reload", "chat_runtime_bootstrap", "model_catalog_refresh"],
                    runtime._warmup_remaining_snapshot(),
                )
                runtime.set_listening("127.0.0.1", 8765)
                runtime.mark_server_listening()
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    if runtime.startup_phase in {"ready", "degraded"}:
                        break
                    time.sleep(0.02)
                self.assertGreaterEqual(len(call_order), 3)
                self.assertEqual("native_packs", call_order[0])
                self.assertEqual("router_reload", call_order[1])
                self.assertEqual("chat_runtime_bootstrap", call_order[2])
                self.assertEqual("ready", runtime.startup_phase)
                self.assertEqual([], runtime._warmup_remaining_snapshot())
            finally:
                runtime.close()

    def test_llm_health_autoconfig_and_hygiene_endpoints(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._health_monitor._probe_fn = lambda *_args: {"ok": True}  # type: ignore[attr-defined]
        runtime.update_permissions(
            {
                "mode": "auto",
                "actions": {
                    "llm.autoconfig.apply": True,
                    "llm.hygiene.apply": True,
                    "llm.capabilities.reconcile.apply": True,
                },
            }
        )

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

        health_get = _HandlerForTest(runtime, "/llm/health")
        health_get.do_GET()
        self.assertEqual(200, health_get.status_code)
        health_payload = json.loads(health_get.body.decode("utf-8"))
        self.assertTrue(health_payload["ok"])
        self.assertIn("health", health_payload)

        health_run = _HandlerForTest(runtime, "/llm/health/run", {})
        health_run.do_POST()
        self.assertEqual(200, health_run.status_code)
        health_run_payload = json.loads(health_run.body.decode("utf-8"))
        self.assertTrue(health_run_payload["ok"])
        self.assertIn("health", health_run_payload)

        autoconfig_plan = _HandlerForTest(runtime, "/llm/autoconfig/plan", {"actor": "test"})
        autoconfig_plan.do_POST()
        self.assertEqual(200, autoconfig_plan.status_code)
        autoconfig_plan_payload = json.loads(autoconfig_plan.body.decode("utf-8"))
        self.assertTrue(autoconfig_plan_payload["ok"])
        self.assertIn("plan", autoconfig_plan_payload)

        capabilities_reconcile_plan = _HandlerForTest(runtime, "/llm/capabilities/reconcile/plan", {"actor": "test"})
        capabilities_reconcile_plan.do_POST()
        self.assertEqual(200, capabilities_reconcile_plan.status_code)
        capabilities_reconcile_plan_payload = json.loads(capabilities_reconcile_plan.body.decode("utf-8"))
        self.assertTrue(capabilities_reconcile_plan_payload["ok"])
        self.assertIn("plan", capabilities_reconcile_plan_payload)

        capabilities_reconcile_apply = _HandlerForTest(
            runtime,
            "/llm/capabilities/reconcile/apply",
            {"actor": "test", "confirm": True},
        )
        capabilities_reconcile_apply.do_POST()
        self.assertEqual(200, capabilities_reconcile_apply.status_code)
        capabilities_reconcile_apply_payload = json.loads(capabilities_reconcile_apply.body.decode("utf-8"))
        self.assertTrue(capabilities_reconcile_apply_payload["ok"])
        self.assertIn("applied", capabilities_reconcile_apply_payload)

        autoconfig_apply = _HandlerForTest(
            runtime,
            "/llm/autoconfig/apply",
            {"actor": "test", "confirm": True},
        )
        autoconfig_apply.do_POST()
        self.assertEqual(200, autoconfig_apply.status_code)
        autoconfig_apply_payload = json.loads(autoconfig_apply.body.decode("utf-8"))
        self.assertTrue(autoconfig_apply_payload["ok"])
        self.assertTrue(autoconfig_apply_payload["applied"])

        hygiene_plan = _HandlerForTest(runtime, "/llm/hygiene/plan", {"actor": "test"})
        hygiene_plan.do_POST()
        self.assertEqual(200, hygiene_plan.status_code)
        hygiene_plan_payload = json.loads(hygiene_plan.body.decode("utf-8"))
        self.assertTrue(hygiene_plan_payload["ok"])
        self.assertIn("plan", hygiene_plan_payload)

        hygiene_apply = _HandlerForTest(
            runtime,
            "/llm/hygiene/apply",
            {"actor": "test", "confirm": True},
        )
        hygiene_apply.do_POST()
        self.assertEqual(200, hygiene_apply.status_code)
        hygiene_apply_payload = json.loads(hygiene_apply.body.decode("utf-8"))
        self.assertTrue(hygiene_apply_payload["ok"])
        self.assertIn("applied", hygiene_apply_payload)

        restarted = AgentRuntime(_config(self.registry_path, self.db_path))
        self.assertEqual(
            runtime.get_defaults()["default_model"],
            restarted.get_defaults()["default_model"],
        )

    def test_llm_apply_endpoints_deny_when_permission_not_allowed(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._health_monitor._probe_fn = lambda *_args: {"ok": True}  # type: ignore[attr-defined]

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

        autoconfig_apply = _HandlerForTest(
            runtime,
            "/llm/autoconfig/apply",
            {"actor": "test", "confirm": True},
        )
        autoconfig_apply.do_POST()
        self.assertEqual(400, autoconfig_apply.status_code)
        autoconfig_payload = json.loads(autoconfig_apply.body.decode("utf-8"))
        self.assertFalse(autoconfig_payload["ok"])
        self.assertEqual("action_not_permitted", autoconfig_payload["error"])

        hygiene_apply = _HandlerForTest(
            runtime,
            "/llm/hygiene/apply",
            {"actor": "test", "confirm": True},
        )
        hygiene_apply.do_POST()
        self.assertEqual(400, hygiene_apply.status_code)
        hygiene_payload = json.loads(hygiene_apply.body.decode("utf-8"))
        self.assertFalse(hygiene_payload["ok"])
        self.assertEqual("action_not_permitted", hygiene_payload["error"])

        capabilities_reconcile_apply = _HandlerForTest(
            runtime,
            "/llm/capabilities/reconcile/apply",
            {"actor": "test", "confirm": True},
        )
        capabilities_reconcile_apply.do_POST()
        self.assertEqual(400, capabilities_reconcile_apply.status_code)
        capabilities_reconcile_payload = json.loads(capabilities_reconcile_apply.body.decode("utf-8"))
        self.assertFalse(capabilities_reconcile_payload["ok"])
        self.assertEqual("action_not_permitted", capabilities_reconcile_payload["error"])

    def test_unknown_llm_endpoint_returns_not_found(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

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

        handler = _HandlerForTest(runtime, "/llm/unknown")
        handler.do_GET()
        self.assertEqual(404, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertFalse(payload["ok"])
        self.assertEqual("not_found", payload["error"])

    def test_llm_notifications_endpoints_and_permission_gate(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=False))

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

        with patch.object(runtime, "_send_telegram_message", return_value=None), patch.object(
            runtime, "_resolve_telegram_target", return_value=("token", "chat-1")
        ):
            denied_test = _HandlerForTest(runtime, "/llm/notifications/test", {"actor": "test", "confirm": True})
            denied_test.do_POST()
            self.assertEqual(400, denied_test.status_code)
            denied_payload = json.loads(denied_test.body.decode("utf-8"))
            self.assertFalse(denied_payload["ok"])
            self.assertEqual("action_not_permitted", denied_payload["error"])

            runtime.update_permissions(
                {
                    "mode": "auto",
                    "actions": {
                        "llm.notifications.test": True,
                    },
                }
            )
            allowed_test = _HandlerForTest(runtime, "/llm/notifications/test", {"actor": "test", "confirm": True})
            allowed_test.do_POST()
            self.assertEqual(200, allowed_test.status_code)
            allowed_payload = json.loads(allowed_test.body.decode("utf-8"))
            self.assertTrue(allowed_payload["ok"])
            self.assertEqual("sent", allowed_payload["result"]["outcome"])

        list_handler = _HandlerForTest(runtime, "/llm/notifications?limit=5")
        list_handler.do_GET()
        self.assertEqual(200, list_handler.status_code)
        list_payload = json.loads(list_handler.body.decode("utf-8"))
        self.assertTrue(list_payload["ok"])
        self.assertTrue(isinstance(list_payload["notifications"], list))
        self.assertTrue(len(list_payload["notifications"]) >= 1)
        status_handler = _HandlerForTest(runtime, "/llm/notifications/status")
        status_handler.do_GET()
        self.assertEqual(200, status_handler.status_code)
        status_payload = json.loads(status_handler.body.decode("utf-8"))
        self.assertTrue(status_payload["ok"])
        self.assertTrue(isinstance(status_payload["status"], dict))
        self.assertIn("stored_count", status_payload["status"])
        audit_entries = runtime.get_audit(limit=20)["entries"]
        notification_test_audit = [
            entry for entry in audit_entries if str(entry.get("action") or "") == "llm.notifications.test"
        ]
        self.assertTrue(len(notification_test_audit) >= 2)
        self.assertEqual("deny", notification_test_audit[1]["decision"])
        self.assertEqual("action_not_permitted", notification_test_audit[1]["reason"])
        self.assertEqual("allow", notification_test_audit[0]["decision"])

    def test_llm_notifications_test_allows_loopback_default_without_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=None))
        runtime.set_listening("127.0.0.1", 8765)

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

        with patch.object(runtime, "_send_telegram_message", return_value=None), patch.object(
            runtime, "_resolve_telegram_target", return_value=("token", "chat-1")
        ):
            allowed_test = _HandlerForTest(runtime, "/llm/notifications/test", {"actor": "test", "confirm": True})
            allowed_test.do_POST()
            self.assertEqual(200, allowed_test.status_code)
            payload = json.loads(allowed_test.body.decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual("sent", payload["result"]["outcome"])

        notifications = runtime.llm_notifications(limit=5)["notifications"]
        self.assertTrue(len(notifications) >= 1)
        audit_entries = runtime.get_audit(limit=20)["entries"]
        notification_test_audit = [
            entry for entry in audit_entries if str(entry.get("action") or "") == "llm.notifications.test"
        ]
        self.assertTrue(notification_test_audit)
        self.assertEqual("allow", notification_test_audit[0]["decision"])
        self.assertEqual("local_loopback_default", notification_test_audit[0]["reason"])

    def test_llm_notifications_prune_endpoint_requires_permission(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, llm_notifications_allow_test=False))

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

        runtime._notification_store.append(  # type: ignore[attr-defined]
            ts=1_000,
            message="m",
            dedupe_hash="h",
            delivered_to="local",
            deferred=False,
            outcome="sent",
            reason="sent_local",
            modified_ids=["defaults:default_model"],
            mark_sent=True,
        )
        denied = _HandlerForTest(runtime, "/llm/notifications/prune", {"actor": "test", "confirm": True})
        denied.do_POST()
        self.assertEqual(400, denied.status_code)
        denied_payload = json.loads(denied.body.decode("utf-8"))
        self.assertFalse(denied_payload["ok"])
        self.assertEqual("action_not_permitted", denied_payload["error"])

        runtime.update_permissions(
            {
                "mode": "auto",
                "actions": {
                    "llm.notifications.prune": True,
                },
            }
        )
        allowed = _HandlerForTest(runtime, "/llm/notifications/prune", {"actor": "test", "confirm": True})
        allowed.do_POST()
        self.assertEqual(200, allowed.status_code)
        allowed_payload = json.loads(allowed.body.decode("utf-8"))
        self.assertTrue(allowed_payload["ok"])
        self.assertTrue(isinstance(allowed_payload["result"], dict))
        self.assertIn("stored_count", allowed_payload["result"])

    def test_notify_autopilot_changes_noop_when_no_diff(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        before = copy.deepcopy(runtime.registry_document)
        after = copy.deepcopy(runtime.registry_document)
        result = runtime._notify_autopilot_changes(  # type: ignore[attr-defined]
            before_document=before,
            after_document=after,
            reasons=["noop"],
            modified_ids=[],
            trigger="scheduler",
        )
        self.assertIsNone(result)
        self.assertEqual([], runtime.llm_notifications(limit=5)["notifications"])

    def test_permissions_and_modelops_endpoints(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

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

        get_permissions = _HandlerForTest(runtime, "/permissions")
        get_permissions.do_GET()
        self.assertEqual(200, get_permissions.status_code)
        permissions_payload = json.loads(get_permissions.body.decode("utf-8"))
        self.assertTrue(permissions_payload["ok"])
        self.assertFalse(permissions_payload["permissions"]["actions"]["modelops.pull_ollama_model"])

        put_permissions = _HandlerForTest(
            runtime,
            "/permissions",
            {
                "actions": {
                    "modelops.pull_ollama_model": True,
                },
                "constraints": {
                    "allowed_providers": ["ollama"],
                    "max_download_bytes": 1024,
                },
            },
        )
        put_permissions.do_PUT()
        self.assertEqual(200, put_permissions.status_code)
        put_payload = json.loads(put_permissions.body.decode("utf-8"))
        self.assertTrue(put_payload["ok"])
        self.assertTrue(put_payload["permissions"]["actions"]["modelops.pull_ollama_model"])

        # Deny by default for actions not explicitly enabled.
        plan_handler = _HandlerForTest(
            runtime,
            "/modelops/plan",
            {
                "action": "modelops.install_ollama",
                "params": {},
            },
        )
        plan_handler.do_POST()
        self.assertEqual(200, plan_handler.status_code)
        plan_payload = json.loads(plan_handler.body.decode("utf-8"))
        self.assertTrue(plan_payload["ok"])
        self.assertFalse(plan_payload["decision"]["allow"])

        exec_handler = _HandlerForTest(
            runtime,
            "/modelops/execute",
            {
                "action": "modelops.install_ollama",
                "params": {},
                "confirm": True,
            },
        )
        exec_handler.do_POST()
        self.assertEqual(400, exec_handler.status_code)
        exec_payload = json.loads(exec_handler.body.decode("utf-8"))
        self.assertFalse(exec_payload["ok"])
        self.assertEqual("action_not_permitted", exec_payload["error"])

    def test_modelops_execute_pull_model_uses_canonical_install_executor(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch.object(
            runtime,
            "_modelops_permission_decision",
            return_value={
                "allow": True,
                "reason": "allowed",
                "requires_confirmation": True,
                "mode": "allow",
                "permissions": {},
            },
        ), patch(
            "agent.api_server.build_model_inventory",
            return_value=[],
        ), patch(
            "agent.api_server.execute_install_plan",
            return_value={
                "ok": True,
                "executed": True,
                "model_id": "ollama:qwen2.5:3b-instruct",
                "install_name": "qwen2.5:3b-instruct",
                "trace_id": "modelops-install-1",
                "error_kind": None,
                "message": "Installed and verified ollama:qwen2.5:3b-instruct.",
                "verification": {
                    "found": True,
                    "installed": True,
                    "available": True,
                    "healthy": True,
                    "verification_status": "ok",
                },
                "stdout_tail": "",
                "stderr_tail": "",
            },
        ) as install_mock, patch.object(
            runtime.modelops_executor,
            "execute_plan",
        ) as execute_plan_mock, patch.object(
            runtime,
            "refresh_models",
            return_value=(True, {"ok": True}),
        ):
            ok, body = runtime.modelops_execute(
                {
                    "action": "modelops.pull_ollama_model",
                    "params": {"model": "qwen2.5:3b-instruct"},
                    "confirm": True,
                }
            )

        self.assertTrue(ok)
        self.assertTrue(body["ok"])
        self.assertIn("canonical_install_plan", body["plan"])
        self.assertTrue(body["result"]["ok"])
        self.assertEqual("modelops-install-1", body["result"]["trace_id"])
        install_mock.assert_called_once()
        self.assertTrue(bool(install_mock.call_args.kwargs["approve"]))
        self.assertEqual(
            "qwen2.5:3b-instruct",
            install_mock.call_args.kwargs["plan"]["candidates"][0]["install_name"],
        )
        execute_plan_mock.assert_not_called()

    def test_modelops_execute_pull_model_uses_canonical_install_approval(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch.object(
            runtime,
            "_modelops_permission_decision",
            return_value={
                "allow": True,
                "reason": "allowed",
                "requires_confirmation": True,
                "mode": "allow",
                "permissions": {},
            },
        ), patch(
            "agent.api_server.build_model_inventory",
            return_value=[],
        ), patch(
            "agent.api_server.execute_install_plan",
            return_value={
                "ok": False,
                "executed": False,
                "model_id": "ollama:qwen2.5:3b-instruct",
                "install_name": "qwen2.5:3b-instruct",
                "trace_id": "modelops-install-approval",
                "error_kind": "approval_required",
                "message": "Explicit approval is required before executing this local install.",
                "verification": {},
                "stdout_tail": "",
                "stderr_tail": "",
            },
        ) as install_mock, patch.object(
            runtime.modelops_executor,
            "execute_plan",
        ) as execute_plan_mock:
            ok, body = runtime.modelops_execute(
                {
                    "action": "modelops.pull_ollama_model",
                    "params": {"model": "qwen2.5:3b-instruct"},
                    "confirm": False,
                }
            )

        self.assertFalse(ok)
        self.assertFalse(body["ok"])
        self.assertEqual("approval_required", body["result"]["error_kind"])
        install_mock.assert_called_once()
        self.assertFalse(bool(install_mock.call_args.kwargs["approve"]))
        execute_plan_mock.assert_not_called()

    def test_pull_ollama_model_source_does_not_shell_out_directly(self) -> None:
        source = inspect.getsource(AgentRuntime.pull_ollama_model)
        self.assertNotIn("safe_runner.run", source)
        self.assertNotIn('[\"ollama\", \"pull\"', source)

    def test_fixit_hf_execution_source_routes_through_model_manager(self) -> None:
        source = inspect.getsource(AgentRuntime._execute_llm_fixit_plan)
        self.assertNotIn("hf_snapshot_download(", source)
        self.assertIn("build_model_manager_request_from_hf_plan_rows", source)

    def test_modelops_execute_import_gguf_uses_canonical_model_manager(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        manager = type(
            "_ManagerStub",
            (),
            {
                "execute_request": lambda self, request, **_: {
                    "ok": True,
                    "executed": True,
                    "model_id": "ollama:custom-import",
                    "trace_id": "modelops-import-1",
                    "message": "Imported model into Ollama.",
                    "verification": {"found": True, "installed": True, "available": True, "healthy": True},
                }
            },
        )()
        with patch.object(
            runtime,
            "_modelops_permission_decision",
            return_value={
                "allow": True,
                "reason": "allowed",
                "requires_confirmation": True,
                "mode": "allow",
                "permissions": {},
            },
        ), patch.object(
            runtime,
            "_model_manager",
            return_value=manager,
        ) as manager_mock, patch.object(
            runtime.modelops_executor,
            "execute_plan",
        ) as execute_plan_mock:
            ok, body = runtime.modelops_execute(
                {
                    "action": "modelops.import_gguf_to_ollama",
                    "params": {"model_name": "custom-import", "modelfile_path": "/tmp/ModelFile"},
                    "confirm": True,
                }
            )

        self.assertTrue(ok)
        self.assertTrue(body["ok"])
        self.assertTrue(body["result"]["ok"])
        manager_mock.assert_called_once()
        execute_plan_mock.assert_not_called()

    def test_pull_ollama_model_is_blocked_by_safe_mode_in_one_canonical_guard(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen2.5:3b-instruct",
            )
        )
        with patch("agent.api_server.execute_install_plan") as install_mock:
            ok, body = runtime.pull_ollama_model({"model": "qwen2.5:3b-instruct", "confirm": True})

        self.assertFalse(ok)
        self.assertFalse(body["ok"])
        self.assertEqual("safe_mode_blocked", body["error_kind"])
        self.assertIn("did not start downloading or installing", str(body["message"] or "").lower())
        self.assertIn("current mode does not allow", str(body.get("why") or "").lower())
        self.assertTrue(str(body.get("next_action") or "").strip())
        install_mock.assert_not_called()

    def test_canonical_model_manager_blocks_hf_download_and_import_in_safe_mode(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen2.5:3b-instruct",
            )
        )
        download_mock = unittest.mock.Mock()
        subprocess_mock = unittest.mock.Mock()
        manager = CanonicalModelManager(
            runtime,
            install_planner_fn=lambda **_: {"ok": True},
            install_executor_fn=lambda **_: {"ok": True},
            hf_snapshot_download_fn=download_mock,
            subprocess_run_fn=subprocess_mock,
        )

        hf_result = manager.execute_request(
            {
                "kind": "hf_local_download",
                "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
                "revision": "main",
                "target_dir": self.tmpdir.name,
                "selected_gguf": "qwen2.5-3b-instruct-q4_k_m.gguf",
                "ollama_model_name": "qwen2.5:3b-instruct",
            },
            approve=True,
            trace_id="hf-safe-mode",
            source="test",
        )
        import_result = manager.execute_request(
            {
                "kind": "ollama_import_gguf",
                "model_name": "custom-import",
                "modelfile_path": "/tmp/Modelfile",
            },
            approve=True,
            trace_id="import-safe-mode",
            source="test",
        )

        self.assertFalse(hf_result["ok"])
        self.assertEqual("safe_mode_blocked", hf_result["error_kind"])
        self.assertFalse(import_result["ok"])
        self.assertEqual("safe_mode_blocked", import_result["error_kind"])
        download_mock.assert_not_called()
        subprocess_mock.assert_not_called()

    def test_runtime_truth_surfaces_acquirable_local_model_without_treating_it_as_ready(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document = {
            "schema_version": 2,
            "defaults": {},
            "providers": {
                "ollama": {
                    "base_url": "http://127.0.0.1:11434",
                    "local": True,
                    "enabled": True,
                }
            },
            "models": {
                "ollama:qwen2.5:3b-instruct": {
                    "provider": "ollama",
                    "model": "qwen2.5:3b-instruct",
                    "capabilities": ["chat"],
                    "enabled": True,
                    "available": True,
                    "quality_rank": 4,
                }
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()
        inventory_rows = [
            {
                "id": "ollama:qwen2.5:3b-instruct",
                "provider": "ollama",
                "installed": False,
                "available": False,
                "healthy": False,
                "capabilities": ["chat"],
                "local": True,
                "approved": True,
                "reason": "model_not_installed",
                "quality_rank": 4,
                "cost_rank": 1,
                "health_status": "down",
                "health_failure_kind": "model_not_installed",
                "health_reason": "model_not_installed",
                "model_name": "qwen2.5:3b-instruct",
                "runtime_known": True,
                "routable": False,
            }
        ]

        with patch.object(truth, "_runtime_inventory_rows", return_value=inventory_rows), patch.object(
            truth,
            "_provider_health_row",
            return_value={"status": "ok"},
        ), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "down", "last_error_kind": "model_not_installed"},
        ):
            inventory = truth.model_inventory_status()
            readiness = truth.model_readiness_status()

        inventory_row = inventory["models"][0]
        readiness_row = readiness["models"][0]
        self.assertFalse(bool(inventory_row["installed_local"]))
        self.assertEqual("not_installed", inventory_row["lifecycle_state"])
        self.assertFalse(bool(readiness_row["usable_now"]))
        self.assertTrue(bool(readiness_row["acquirable"]))
        self.assertEqual("acquirable", readiness_row["acquisition_state"])
        self.assertEqual("acquirable", readiness_row["availability_state"])

    def test_runtime_truth_marks_acquirable_local_model_blocked_in_safe_mode(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen2.5:3b-instruct",
            )
        )
        runtime.registry_document = {
            "schema_version": 2,
            "defaults": {},
            "providers": {
                "ollama": {
                    "base_url": "http://127.0.0.1:11434",
                    "local": True,
                    "enabled": True,
                }
            },
            "models": {
                "ollama:qwen2.5:3b-instruct": {
                    "provider": "ollama",
                    "model": "qwen2.5:3b-instruct",
                    "capabilities": ["chat"],
                    "enabled": True,
                    "available": True,
                    "quality_rank": 4,
                }
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()
        inventory_rows = [
            {
                "id": "ollama:qwen2.5:3b-instruct",
                "provider": "ollama",
                "installed": False,
                "available": False,
                "healthy": False,
                "capabilities": ["chat"],
                "local": True,
                "approved": True,
                "reason": "model_not_installed",
                "quality_rank": 4,
                "cost_rank": 1,
                "health_status": "down",
                "health_failure_kind": "model_not_installed",
                "health_reason": "model_not_installed",
                "model_name": "qwen2.5:3b-instruct",
                "runtime_known": True,
                "routable": False,
            }
        ]

        with patch.object(truth, "_runtime_inventory_rows", return_value=inventory_rows), patch.object(
            truth,
            "_provider_health_row",
            return_value={"status": "ok"},
        ), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "down", "last_error_kind": "model_not_installed"},
        ):
            readiness = truth.model_readiness_status()

        readiness_row = readiness["models"][0]
        self.assertTrue(bool(readiness_row["acquirable"]))
        self.assertEqual("blocked_by_policy", readiness_row["acquisition_state"])
        self.assertEqual("install_blocked", readiness_row["availability_state"])

    def test_pull_ollama_model_routes_non_safe_mode_acquisition_through_canonical_manager(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        manager = type(
            "_ManagerStub",
            (),
            {
                "execute_request": lambda self, request, **_: {
                    "ok": True,
                    "executed": True,
                    "message": "Installed ollama:qwen2.5:3b-instruct.",
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "trace_id": "pull-1",
                    "verification": {"found": True, "installed": True, "available": True, "healthy": True},
                    "refresh_ok": True,
                    "refresh_result": {"ok": True},
                }
            },
        )()

        with patch.object(runtime, "_model_manager", return_value=manager) as manager_mock:
            ok, body = runtime.pull_ollama_model({"model": "qwen2.5:3b-instruct", "confirm": True})

        self.assertTrue(ok)
        self.assertTrue(body["ok"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", body["canonical_model"])
        manager_mock.assert_called_once()

    def test_runtime_truth_inventory_and_readiness_use_model_lifecycle_bridge(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document = {
            "schema_version": 2,
            "defaults": {},
            "providers": {
                "ollama": {
                    "base_url": "http://127.0.0.1:11434",
                    "local": True,
                    "enabled": True,
                }
            },
            "models": {
                "ollama:qwen2.5:3b-instruct": {
                    "provider": "ollama",
                    "model": "qwen2.5:3b-instruct",
                    "capabilities": ["chat"],
                    "enabled": True,
                    "available": True,
                    "quality_rank": 2,
                }
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        truth = runtime.runtime_truth_service()
        save_model_manager_state(
            model_manager_state_path_for_runtime(runtime),
            {
                "schema_version": 1,
                "targets": {
                    "ollama:qwen2.5:3b-instruct": {
                        "target_key": "ollama:qwen2.5:3b-instruct",
                        "target_type": "model",
                        "provider_id": "ollama",
                        "model_id": "ollama:qwen2.5:3b-instruct",
                        "state": "downloading",
                        "message": "Installing model.",
                    }
                },
            },
        )

        inventory = truth.model_inventory_status()
        inventory_row = next(
            row
            for row in inventory["models"]
            if row["model_id"] == "ollama:qwen2.5:3b-instruct"
        )
        self.assertEqual("downloading", inventory_row["lifecycle_state"])

        with patch.object(
            truth,
            "_provider_health_row",
            return_value={"status": "ok"},
        ), patch.object(
            truth,
            "_model_health_row",
            return_value={"status": "ok"},
        ):
            readiness = truth.model_readiness_status()

        readiness_row = next(
            row
            for row in readiness["models"]
            if row["model_id"] == "ollama:qwen2.5:3b-instruct"
        )
        self.assertEqual("ready", readiness_row["lifecycle_state"])
        self.assertTrue(bool(readiness_row["usable_now"]))

    def test_llm_models_lifecycle_endpoint_exposes_canonical_manager_state(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        save_model_manager_state(
            model_manager_state_path_for_runtime(runtime),
            {
                "schema_version": 1,
                "targets": {
                    "ollama:llava:7b": {
                        "target_key": "ollama:llava:7b",
                        "target_type": "model",
                        "provider_id": "ollama",
                        "model_id": "ollama:llava:7b",
                        "state": "downloading",
                        "message": "Installing ollama:llava:7b.",
                    },
                    "ollama:deepseek-r1:7b": {
                        "target_key": "ollama:deepseek-r1:7b",
                        "target_type": "model",
                        "provider_id": "ollama",
                        "model_id": "ollama:deepseek-r1:7b",
                        "state": "failed",
                        "message": "Ollama create failed.",
                        "error_kind": "ollama_create_failed",
                    },
                },
            },
        )

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime) -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/lifecycle"
                self.headers = {}
                self.status_code = 0
                self.body = b""

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                self.status_code = status
                self.body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")

            def _send_bytes(
                self,
                status: int,
                body: bytes,
                *,
                content_type: str,
                cache_control: str | None = None,
            ) -> None:
                _ = (content_type, cache_control)
                self.status_code = status
                self.body = body

        handler = _HandlerForTest(runtime)
        handler.do_GET()

        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(1, payload["counts"]["downloading"])
        self.assertEqual(1, payload["counts"]["failed"])
        downloading_target = next(
            row
            for row in payload["downloading_targets"]
            if row["target_key"] == "ollama:llava:7b"
        )
        failed_target = next(
            row
            for row in payload["failed_targets"]
            if row["target_key"] == "ollama:deepseek-r1:7b"
        )
        self.assertEqual("downloading", downloading_target["lifecycle_state"])
        self.assertEqual("failed", failed_target["lifecycle_state"])

    def test_root_route_serves_webui_index_html(self) -> None:
        webui_dist = os.path.join(self.tmpdir.name, "webui", "dist")
        os.makedirs(webui_dist, exist_ok=True)
        with open(os.path.join(webui_dist, "index.html"), "w", encoding="utf-8") as handle:
            handle.write("<!doctype html><html><head><meta name='personal-agent-webui' content='1'></head><body></body></html>")

        os.environ["AGENT_WEBUI_DIST_PATH"] = webui_dist
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

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

        handler = _HandlerForTest(runtime, "/")
        handler.do_GET()

        body_text = handler.body.decode("utf-8", errors="replace").lower()
        self.assertEqual(200, handler.status_code)
        self.assertIn("text/html", handler.content_type)
        self.assertIn("<html", body_text)
        self.assertIn("personal-agent-webui", body_text)
        self.assertIn("personal-agent-version", body_text)
        self.assertIn("__personal_agent_build__", body_text)

    def test_read_json_returns_empty_dict_on_invalid_utf8(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, raw_body: bytes) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": str(len(raw_body))}
                self.rfile = io.BytesIO(raw_body)

        handler = _HandlerForTest(runtime, b"\xff")
        self.assertEqual({}, handler._read_json())

    def test_chat_low_confidence_returns_clarification_contract(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        handler = _HandlerForTest(runtime, {})
        handler.do_POST()

        payload = handler.response_payload
        self.assertEqual(200, handler.status_code)
        self.assertEqual(True, payload.get("ok"))
        self.assertEqual("chat", payload.get("intent"))
        self.assertEqual(0.0, payload.get("confidence"))
        self.assertEqual(False, payload.get("did_work"))
        self.assertEqual("needs_clarification", payload.get("error_kind"))
        self.assertTrue(str(payload.get("message") or "").strip())
        self.assertEqual(payload.get("message"), payload.get("next_question"))
        self.assertEqual([], payload.get("actions"))
        self.assertEqual(["needs_clarification"], payload.get("errors"))
        self.assertTrue(str(payload.get("trace_id") or "").strip())
        envelope = payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        clarification = (envelope or {}).get("clarification")
        self.assertTrue(isinstance(clarification, dict))
        self.assertTrue(str((clarification or {}).get("reason") or "").strip())
        self.assertTrue(isinstance((clarification or {}).get("hints"), list))

    def test_ask_low_confidence_returns_clarification_contract(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/ask"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        handler = _HandlerForTest(runtime, {})
        handler.do_POST()

        payload = handler.response_payload
        self.assertEqual(200, handler.status_code)
        self.assertEqual(True, payload.get("ok"))
        self.assertEqual("ask", payload.get("intent"))
        self.assertEqual("needs_clarification", payload.get("error_kind"))
        self.assertEqual(["needs_clarification"], payload.get("errors"))
        self.assertEqual(payload.get("message"), payload.get("next_question"))

    def test_chat_ambiguous_clarifies_when_llm_available(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        payload = {"messages": [{"role": "user", "content": "fix it"}]}
        handler = _HandlerForTest(runtime, payload)
        with patch.object(
            runtime,
            "llm_availability_state",
            return_value={"available": True, "reason": "ok"},
        ), patch.object(runtime, "chat", side_effect=AssertionError("chat should not be called")):
            handler.do_POST()

        response = handler.response_payload
        self.assertEqual(200, handler.status_code)
        self.assertEqual(True, response.get("ok"))
        self.assertEqual("needs_clarification", response.get("error_kind"))
        message = str(response.get("message") or "")
        self.assertIn("Do you mean:", message)
        self.assertIn("A)", message)
        self.assertIn("B)", message)
        self.assertNotIn("1)", message)

    def test_chat_runtime_status_queries_bypass_intent_chooser(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        utterances = (
            ("what model are you using?", "model_status"),
            ("what models do we have downloaded?", "model_status"),
            ("what local models are available?", "model_status"),
            ("show cloud models", "model_status"),
            ("what cloud models are available", "model_status"),
            ("recommend a local model", "action_tool"),
            ("recommend a coding model", "action_tool"),
            ("recommend a research model", "action_tool"),
            ("should I switch models", "action_tool"),
            ("which model are you using?", "model_status"),
            ("is the agent healthy?", "runtime_status"),
            ("can you tell if everything is working with the agent?", "runtime_status"),
            ("can you read the runtime now?", "runtime_status"),
            ("openrouter health", "provider_status"),
            ("is openrouter configured?", "provider_status"),
            ("agent doctor", "operational_status"),
            ("how much memory am I using?", "operational_status"),
            ("how is my storage?", "operational_status"),
        )

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            for utterance, expected_route in utterances:
                handler = _HandlerForTest(runtime, {"messages": [{"role": "user", "content": utterance}]})
                handler.do_POST()
                response = handler.response_payload
                self.assertEqual(200, handler.status_code)
                self.assertNotEqual("needs_clarification", response.get("error_kind"))
                self.assertNotIn("Which of these is your goal", str(response.get("message") or ""))
                meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"))

    def test_chat_confirmation_followup_bypasses_short_message_clarification(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        truth = runtime.runtime_truth_service()
        preview_payload = {
            "ok": True,
            "action": "shell_install_package",
            "mutated": True,
            "command_name": "apt_install",
            "argv": ["apt-get", "install", "-y", "ripgrep"],
            "cwd": "/home/c/personal-agent",
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timed_out": False,
            "truncated": False,
            "blocked_reason": None,
            "error_kind": None,
            "message": "Install preview ready.",
            "manager": "apt",
            "package": "ripgrep",
            "scope": None,
            "dry_run": False,
            "type": "shell_install_package_preview",
            "source": "runtime_truth.shell",
        }
        executed_payload = {
            "ok": True,
            "action": "shell_install_package",
            "mutated": True,
            "command_name": "apt_install",
            "argv": ["apt-get", "install", "-y", "ripgrep"],
            "cwd": "/home/c/personal-agent",
            "stdout": "Installing ripgrep\n",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "truncated": False,
            "blocked_reason": None,
            "error_kind": None,
            "message": "Command completed.",
            "manager": "apt",
            "package": "ripgrep",
            "scope": None,
            "dry_run": False,
            "type": "shell_install_package",
            "source": "runtime_truth.shell",
        }

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            truth,
            "shell_preview_install_package",
            return_value=preview_payload,
        ) as preview_mock, patch.object(
            truth,
            "shell_install_package",
            return_value=executed_payload,
        ) as install_mock:
            first = _HandlerForTest(
                runtime,
                {
                    "messages": [{"role": "user", "content": "install ripgrep"}],
                    "user_id": "api:confirm-followup",
                    "thread_id": "api:confirm-followup:thread",
                    "source_surface": "api",
                },
            )
            first.do_POST()
            second = _HandlerForTest(
                runtime,
                {
                    "messages": [{"role": "user", "content": "yes"}],
                    "user_id": "api:confirm-followup",
                    "thread_id": "api:confirm-followup:thread",
                    "source_surface": "api",
                },
            )
            second.do_POST()

        self.assertEqual(200, first.status_code)
        self.assertNotEqual("needs_clarification", first.response_payload.get("error_kind"))
        first_meta = first.response_payload.get("meta") if isinstance(first.response_payload.get("meta"), dict) else {}
        first_setup = first.response_payload.get("setup") if isinstance(first.response_payload.get("setup"), dict) else {}
        self.assertEqual("action_tool", first_meta.get("route"))
        self.assertTrue(first_setup.get("requires_confirmation"))
        self.assertEqual(1, preview_mock.call_count)

        self.assertEqual(200, second.status_code)
        self.assertNotEqual("needs_clarification", second.response_payload.get("error_kind"))
        second_meta = second.response_payload.get("meta") if isinstance(second.response_payload.get("meta"), dict) else {}
        self.assertEqual("action_tool", second_meta.get("route"))
        self.assertIn("Installing ripgrep", str(second.response_payload.get("message") or ""))
        self.assertEqual(1, install_mock.call_count)

    def test_chat_confirmed_model_switch_preserves_model_controller_metadata(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        truth = runtime.runtime_truth_service()
        orchestrator = runtime.orchestrator()
        active = {
            "provider": "ollama",
            "model": "ollama:qwen3.5:4b",
        }

        def _current_target() -> dict[str, object]:
            return {
                "provider": active["provider"],
                "model": active["model"],
                "ready": True,
                "health_status": "ok",
                "provider_health_status": "ok",
            }

        def _set_temporary(model_id: str, *, provider_id: str | None = None) -> tuple[bool, dict[str, object]]:
            active["provider"] = str(provider_id or "ollama").strip().lower() or "ollama"
            active["model"] = model_id
            return True, {
                "provider": active["provider"],
                "model_id": model_id,
                "message": f"Temporarily using {model_id} for chat.",
            }

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            orchestrator,
            "_runtime_model_catalog",
            return_value=["ollama:qwen3.5:4b", "ollama:qwen2.5:7b-instruct"],
        ), patch.object(
            truth,
            "current_chat_target_status",
            side_effect=_current_target,
        ), patch.object(
            truth,
            "set_temporary_chat_model_target",
            side_effect=_set_temporary,
        ) as switch_mock:
            first = _HandlerForTest(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch temporarily to ollama:qwen2.5:7b-instruct"}],
                    "user_id": "api:model-switch-confirm",
                    "thread_id": "api:model-switch-confirm:thread",
                    "source_surface": "api",
                },
            )
            first.do_POST()
            second = _HandlerForTest(
                runtime,
                {
                    "messages": [{"role": "user", "content": "yes"}],
                    "user_id": "api:model-switch-confirm",
                    "thread_id": "api:model-switch-confirm:thread",
                    "source_surface": "api",
                },
            )
            second.do_POST()

        self.assertEqual(200, first.status_code)
        first_meta = first.response_payload.get("meta") if isinstance(first.response_payload.get("meta"), dict) else {}
        first_setup = first.response_payload.get("setup") if isinstance(first.response_payload.get("setup"), dict) else {}
        self.assertEqual("model_status", first_meta.get("route"))
        self.assertEqual(["model_controller"], first_meta.get("used_tools"))
        self.assertFalse(bool(first_meta.get("used_llm")))
        self.assertTrue(first_setup.get("requires_confirmation"))

        self.assertEqual(200, second.status_code)
        second_meta = second.response_payload.get("meta") if isinstance(second.response_payload.get("meta"), dict) else {}
        self.assertEqual("model_status", second_meta.get("route"))
        self.assertEqual(["model_controller"], second_meta.get("used_tools"))
        self.assertFalse(bool(second_meta.get("used_llm")))
        self.assertIn("Temporarily using ollama:qwen2.5:7b-instruct for chat.", str(second.response_payload.get("message") or ""))
        self.assertEqual(1, switch_mock.call_count)

    def test_chat_intent_choice_reply_is_consumed_and_cleared(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        runtime.set_intent_choice_prompt(source="api", user_id="api:default")
        first = _HandlerForTest(runtime, {"messages": [{"role": "user", "content": "chat"}]})
        first.do_POST()
        self.assertEqual(200, first.status_code)
        self.assertEqual(True, first.response_payload.get("did_work"))
        self.assertIn("next message", str(first.response_payload.get("message") or "").lower())
        self.assertEqual({}, runtime._intent_choice_state)

    def test_safe_mode_chat_bypasses_legacy_choice_and_fixit_consumers(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")
        runtime.set_clarify_recovery_prompt(source="api", reason="llm_unavailable")
        runtime.set_binary_clarification_prompt(
            source="api",
            user_id="api:legacy",
            question="Choose one.",
            option_a_label="Status",
            option_b_label="Task",
        )
        runtime.set_intent_choice_prompt(source="api", user_id="api:legacy")

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self.client_address = ("127.0.0.1", 12345)
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        payload = {
            "messages": [{"role": "user", "content": "what model are you using?"}],
            "source_surface": "api",
            "user_id": "api:legacy",
            "thread_id": "api:legacy:thread",
        }
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            runtime,
            "consume_clarify_recovery_choice",
            side_effect=AssertionError("legacy recovery should be bypassed"),
        ), patch.object(
            runtime,
            "consume_binary_clarification_choice",
            side_effect=AssertionError("legacy binary choice should be bypassed"),
        ), patch.object(
            runtime,
            "consume_intent_choice",
            side_effect=AssertionError("legacy intent choice should be bypassed"),
        ), patch.object(
            runtime,
            "llm_fixit",
            side_effect=AssertionError("operator fix-it must not be reachable from /chat"),
        ), patch.object(
            runtime,
            "rollback_defaults",
            side_effect=AssertionError("operator rollback must not be reachable from /chat"),
        ):
            handler = _HandlerForTest(runtime, payload)
            handler.do_POST()

        response = handler.response_payload
        self.assertEqual(200, handler.status_code)
        meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
        self.assertEqual("model_status", meta.get("route"))
        self.assertFalse(bool(runtime._clarify_recovery_state.get("active")))
        self.assertEqual({}, runtime._binary_clarification_state)
        self.assertEqual({}, runtime._intent_choice_state)

    def test_chat_adapter_routes_user_messages_through_orchestrator(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")

        class _FakeOrchestrator:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def handle_message(
                self,
                text: str,
                *,
                user_id: str,
                chat_context: dict[str, object] | None = None,
            ) -> OrchestratorResponse:
                self.calls.append(
                    {
                        "text": text,
                        "user_id": user_id,
                        "chat_context": dict(chat_context or {}),
                    }
                )
                return OrchestratorResponse(
                    "Current model: ollama:qwen3.5:4b",
                    {
                        "route": "model_status",
                        "used_llm": False,
                    },
                )

        fake_orchestrator = _FakeOrchestrator()
        payload = {
            "messages": [{"role": "user", "content": "what model are you using?"}],
            "user_id": "api:lockdown",
            "thread_id": "api:lockdown:thread",
            "source_surface": "api",
        }
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            runtime,
            "orchestrator",
            return_value=fake_orchestrator,
        ):
            ok, body = runtime.chat(payload)

        self.assertTrue(ok)
        self.assertEqual(1, len(fake_orchestrator.calls))
        call = fake_orchestrator.calls[0]
        self.assertEqual("what model are you using?", call.get("text"))
        self.assertEqual("api:lockdown", call.get("user_id"))
        chat_context = call.get("chat_context") if isinstance(call.get("chat_context"), dict) else {}
        self.assertEqual("api", chat_context.get("source_surface"))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual("model_status", meta.get("route"))

    def test_chat_runtime_status_queries_bypass_intent_chooser_for_legacy_message_payload(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        utterances = (
            ("what model are you using?", "model_status"),
            ("is openrouter configured?", "provider_status"),
        )

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            for utterance, expected_route in utterances:
                handler = _HandlerForTest(runtime, {"message": utterance})
                handler.do_POST()
                response = handler.response_payload
                self.assertEqual(200, handler.status_code)
                self.assertNotEqual("needs_clarification", response.get("error_kind"))
                self.assertNotIn("Which of these is your goal", str(response.get("message") or ""))
                self.assertNotIn("messages must be a non-empty list", str(response.get("message") or ""))
                meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"))

    def test_chat_typo_tolerant_model_queries_route_deterministically(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        utterances = (
            ("waht model are you using?", "model_status"),
            ("wat models are availble?", "model_status"),
            ("ollma status", "provider_status"),
        )

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            for utterance, expected_route in utterances:
                handler = _HandlerForTest(runtime, {"messages": [{"role": "user", "content": utterance}]})
                handler.do_POST()
                response = handler.response_payload
                self.assertEqual(200, handler.status_code)
                self.assertNotEqual("needs_clarification", response.get("error_kind"))
                self.assertNotIn("continuing the current thread", str(response.get("message") or "").lower())
                meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"))

    def test_chat_model_queries_do_not_trigger_thread_integrity_prompt(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        payload = {
            "messages": [
                {"role": "user", "content": "help me debug a python traceback"},
                {"role": "assistant", "content": "sure paste the traceback"},
                {"role": "user", "content": "what models are available for use?"},
            ],
            "user_id": "api:model-query",
            "thread_id": "api:model-query:thread",
        }
        runtime.set_thread_integrity_prompt(
            source="api",
            user_id="api:model-query",
            pending_text="continue or start new",
            payload_template=payload,
        )

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            handler = _HandlerForTest(runtime, payload)
            handler.do_POST()

        response = handler.response_payload
        self.assertEqual(200, handler.status_code)
        self.assertNotEqual("needs_clarification", response.get("error_kind"))
        self.assertNotIn("continuing the current thread", str(response.get("message") or "").lower())
        meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
        self.assertEqual("model_status", meta.get("route"))
        self.assertNotIn(
            runtime._thread_integrity_key(source="api", user_id="api:model-query"),
            runtime._thread_integrity_state,
        )

    def test_chat_repair_followups_reuse_recent_unhealthy_context_without_chooser(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        truth = runtime.runtime_truth_service()

        def _provider_status(provider_id: str) -> dict[str, object]:
            provider_key = str(provider_id).strip().lower()
            if provider_key == "ollama":
                return {
                    "provider": "ollama",
                    "provider_label": "Ollama",
                    "known": True,
                    "enabled": True,
                    "local": True,
                    "configured": True,
                    "active": True,
                    "secret_present": False,
                    "health_status": "down",
                    "health_reason": "timeout while reaching Ollama",
                    "model_id": "ollama:qwen3.5:4b",
                    "model_ids": ["ollama:qwen3.5:4b"],
                    "current_provider": "ollama",
                    "current_model_id": "ollama:qwen3.5:4b",
                    "effective_provider": "ollama",
                    "effective_model_id": "ollama:qwen3.5:4b",
                    "effective_active": True,
                    "qualification_reason": "Configured default ollama:qwen3.5:4b on ollama is not ready right now.",
                    "degraded_reason": "Configured default ollama:qwen3.5:4b on ollama is not ready right now.",
                }
            return {
                "provider": provider_key,
                "provider_label": provider_key.title(),
                "known": False,
                "enabled": False,
                "local": False,
                "configured": False,
                "active": False,
                "secret_present": False,
                "health_status": "unknown",
                "health_reason": None,
                "model_id": None,
                "model_ids": [],
                "current_provider": "ollama",
                "current_model_id": "ollama:qwen3.5:4b",
                "effective_provider": "ollama",
                "effective_model_id": "ollama:qwen3.5:4b",
                "effective_active": False,
                "qualification_reason": None,
                "degraded_reason": None,
            }

        with patch.object(
            truth,
            "current_chat_target_status",
            return_value={
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "ready": False,
                "health_status": "down",
                "provider_health_status": "down",
            },
        ), patch.object(
            truth,
            "chat_target_truth",
            return_value={
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "configured_ready": False,
                "effective_provider": "ollama",
                "effective_model": "ollama:qwen3.5:4b",
                "effective_ready": False,
                "qualification_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                "degraded_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
            },
        ), patch.object(
            truth,
            "provider_status",
            side_effect=_provider_status,
        ), patch.object(
            runtime,
            "test_provider",
            return_value=(False, {"ok": False, "error": "timeout"}),
        ), patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            return_value=None,
        ):
            first = _HandlerForTest(
                runtime,
                {"messages": [{"role": "user", "content": "what model are you using?"}], "user_id": "api:repair", "thread_id": "api:repair:thread"},
            )
            first.do_POST()
            second = _HandlerForTest(
                runtime,
                {"messages": [{"role": "user", "content": "can you repair it?"}], "user_id": "api:repair", "thread_id": "api:repair:thread"},
            )
            second.do_POST()
            third = _HandlerForTest(
                runtime,
                {"messages": [{"role": "user", "content": "can you fix ollama?"}], "user_id": "api:repair", "thread_id": "api:repair:thread"},
            )
            third.do_POST()

        for handler in (first, second, third):
            self.assertEqual(200, handler.status_code)
            self.assertNotEqual("needs_clarification", handler.response_payload.get("error_kind"))
            message = str(handler.response_payload.get("message") or "")
            self.assertNotIn("Which of these is your goal", message)
            self.assertNotIn("Tell me whether you want chat, ask, or model check/switch.", message)
            self.assertNotIn("continuing the current thread", message.lower())

        second_meta = second.response_payload.get("meta") if isinstance(second.response_payload.get("meta"), dict) else {}
        third_meta = third.response_payload.get("meta") if isinstance(third.response_payload.get("meta"), dict) else {}
        self.assertEqual("setup_flow", second_meta.get("route"))
        self.assertEqual("setup_flow", third_meta.get("route"))
        self.assertIn("ollama is currently down", str(second.response_payload.get("message") or "").lower())
        self.assertIn("reconfigure ollama", str(second.response_payload.get("message") or "").lower())
        self.assertIn("ollama is currently down", str(third.response_payload.get("message") or "").lower())

    def test_chat_repair_help_keeps_grounded_local_recovery_when_model_unhealthy(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        truth = runtime.runtime_truth_service()

        inventory_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                },
            ],
            "usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "other_usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "not_ready_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                }
            ],
        }
        inventory_status_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [dict(row) for row in inventory_payload["models"]],
            "local_installed_models": [
                dict(row)
                for row in inventory_payload["models"]
                if bool(row.get("local", False)) and bool(row.get("available", False))
            ],
        }
        readiness_status_payload = {
            "active_provider": "ollama",
            "active_model": "ollama:qwen3.5:4b",
            "configured_provider": "ollama",
            "configured_model": "ollama:qwen3.5:4b",
            "models": [dict(row) for row in inventory_payload["models"]],
            "ready_now_models": [dict(row) for row in inventory_payload["usable_models"]],
            "other_ready_now_models": [dict(row) for row in inventory_payload["other_usable_models"]],
            "not_ready_models": [dict(row) for row in inventory_payload["not_ready_models"]],
        }

        def _provider_status(provider_id: str) -> dict[str, object]:
            provider_key = str(provider_id).strip().lower()
            if provider_key == "ollama":
                return {
                    "provider": "ollama",
                    "provider_label": "Ollama",
                    "known": True,
                    "enabled": True,
                    "local": True,
                    "configured": True,
                    "active": True,
                    "secret_present": False,
                    "health_status": "ok",
                    "health_reason": None,
                    "model_id": "ollama:qwen3.5:4b",
                    "model_ids": ["ollama:qwen3.5:4b", "ollama:qwen2.5:3b-instruct"],
                    "current_provider": "ollama",
                    "current_model_id": "ollama:qwen3.5:4b",
                    "effective_provider": "ollama",
                    "effective_model_id": "ollama:qwen3.5:4b",
                    "effective_active": True,
                    "qualification_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                    "degraded_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                }
            return {
                "provider": provider_key,
                "provider_label": provider_key.title(),
                "known": False,
                "enabled": False,
                "local": False,
                "configured": False,
                "active": False,
                "secret_present": False,
                "health_status": "unknown",
                "health_reason": None,
                "model_id": None,
                "model_ids": [],
                "current_provider": "ollama",
                "current_model_id": "ollama:qwen3.5:4b",
                "effective_provider": "ollama",
                "effective_model_id": "ollama:qwen3.5:4b",
                "effective_active": False,
                "qualification_reason": None,
                "degraded_reason": None,
            }

        with patch.object(
            truth,
            "current_chat_target_status",
            return_value={
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "ready": False,
                "health_status": "down",
                "provider_health_status": "ok",
            },
        ), patch.object(
            truth,
            "chat_target_truth",
            return_value={
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "configured_ready": False,
                "effective_provider": "ollama",
                "effective_model": "ollama:qwen3.5:4b",
                "effective_ready": False,
                "qualification_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                "degraded_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
            },
        ), patch.object(
            truth,
            "provider_status",
            side_effect=_provider_status,
        ), patch.object(
            truth,
            "model_inventory_status",
            return_value=inventory_status_payload,
        ), patch.object(
            truth,
            "model_readiness_status",
            return_value=readiness_status_payload,
        ), patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True}),
        ), patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            return_value=None,
        ):
            first = _HandlerForTest(
                runtime,
                {"messages": [{"role": "user", "content": "what model are you using?"}], "user_id": "api:repair-help", "thread_id": "api:repair-help:thread"},
            )
            first.do_POST()
            second = _HandlerForTest(
                runtime,
                {"messages": [{"role": "user", "content": "Help me get this working"}], "user_id": "api:repair-help", "thread_id": "api:repair-help:thread"},
            )
            second.do_POST()
            third = _HandlerForTest(
                runtime,
                {"messages": [{"role": "user", "content": "1"}], "user_id": "api:repair-help", "thread_id": "api:repair-help:thread"},
            )
            third.do_POST()
            fourth = _HandlerForTest(
                runtime,
                {"messages": [{"role": "user", "content": "2"}], "user_id": "api:repair-help", "thread_id": "api:repair-help:thread"},
            )
            fourth.do_POST()

        second_message = str(second.response_payload.get("message") or "")
        third_message = str(third.response_payload.get("message") or "")
        fourth_message = str(fourth.response_payload.get("message") or "")
        second_meta = second.response_payload.get("meta") if isinstance(second.response_payload.get("meta"), dict) else {}
        third_meta = third.response_payload.get("meta") if isinstance(third.response_payload.get("meta"), dict) else {}
        fourth_meta = fourth.response_payload.get("meta") if isinstance(fourth.response_payload.get("meta"), dict) else {}

        for message in (second_message, third_message, fourth_message):
            self.assertNotIn("Which of these is your goal", message)
            self.assertNotIn("Tell me whether you want chat, ask, or model check/switch.", message)
            self.assertNotIn("no chat model available right now", message.lower())
            self.assertNotIn("start ollama locally", message.lower())
            self.assertNotIn("install a local chat model", message.lower())

        self.assertEqual("setup_flow", second_meta.get("route"))
        self.assertIn("ollama is reachable", second_message.lower())
        self.assertIn("current chat model ollama:qwen3.5:4b is not healthy right now", second_message.lower())
        self.assertIn("1) recheck ollama:qwen3.5:4b now", second_message.lower())
        self.assertIn("2) switch to ollama:qwen2.5:3b-instruct", second_message.lower())

        self.assertEqual("setup_flow", third_meta.get("route"))
        self.assertIn("ollama is reachable", third_message.lower())
        self.assertIn("not healthy right now", third_message.lower())

        self.assertEqual("setup_flow", fourth_meta.get("route"))
        self.assertIn("switch chat to ollama:qwen2.5:3b-instruct now", fourth_message.lower())

    def test_chat_ambiguous_suggests_when_llm_unavailable_and_intercepts_numeric_choice(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        first = _HandlerForTest(runtime, {"messages": [{"role": "user", "content": "fix it"}]})
        with patch.object(
            runtime,
            "llm_availability_state",
            return_value={"available": False, "reason": "provider_unhealthy"},
        ), patch.object(runtime, "chat", side_effect=AssertionError("chat should not be called")):
            first.do_POST()

        response_first = first.response_payload
        self.assertEqual(200, first.status_code)
        self.assertEqual(True, response_first.get("ok"))
        first_message = str(response_first.get("message") or "")
        self.assertIn("1)", first_message)
        self.assertIn("2)", first_message)
        self.assertIn("3)", first_message)
        self.assertIn("Reply 1, 2, or 3.", first_message)

        second = _HandlerForTest(runtime, {"messages": [{"role": "user", "content": "1"}]})
        with patch.object(runtime, "chat", side_effect=AssertionError("chat should not be called")):
            second.do_POST()
        response_second = second.response_payload
        self.assertEqual(200, second.status_code)
        self.assertEqual(True, response_second.get("ok"))
        self.assertTrue(bool(response_second.get("did_work")))
        self.assertIn("Current provider/model:", str(response_second.get("message") or ""))

    def test_chat_thread_integrity_drift_is_handled_as_normal_request(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        payload = {
            "messages": [
                {"role": "user", "content": "help me debug a python traceback"},
                {"role": "assistant", "content": "sure paste the traceback"},
                {
                    "role": "user",
                    "content": "can you help with this traceback and also write a dinner recipe for chicken and rice tonight?",
                },
            ]
        }
        handler = _HandlerForTest(runtime, payload)
        with patch.object(
            runtime,
            "chat",
            return_value=(
                True,
                {
                    "ok": True,
                    "message": "Handled directly.",
                    "assistant": {"content": "Handled directly."},
                    "meta": {"route": "generic_chat", "used_llm": False},
                },
            ),
        ) as chat_mock:
            handler.do_POST()

        response = handler.response_payload
        self.assertEqual(200, handler.status_code)
        self.assertEqual(True, response.get("ok"))
        self.assertNotEqual("needs_clarification", response.get("error_kind"))
        self.assertNotIn("same thread", str(response.get("message") or "").lower())
        self.assertNotIn("new thread", str(response.get("message") or "").lower())
        chat_mock.assert_called_once()

    def test_chat_success_includes_intent_assessment_envelope(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        handler = _HandlerForTest(
            runtime,
            {"messages": [{"role": "user", "content": "hello there"}]},
        )
        with patch.object(
            runtime,
            "chat",
            return_value=(True, {"ok": True, "assistant": {"role": "assistant", "content": "hi"}}),
        ):
            handler.do_POST()

        self.assertEqual(200, handler.status_code)
        envelope = handler.response_payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        intent_assessment = (envelope or {}).get("intent_assessment")
        self.assertTrue(isinstance(intent_assessment, dict))
        self.assertIn("decision", intent_assessment or {})
        self.assertIn("candidates", intent_assessment or {})

    def test_llm_models_check_endpoint_returns_stable_envelope(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        seen_path = os.path.join(self.tmpdir.name, "modelops_seen_models.json")
        os.environ["AGENT_MODELOPS_SEEN_MODELS_PATH"] = seen_path

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/check"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        fake_models = [
            ModelInfo(
                provider="ollama",
                model_id="qwen2.5:7b-instruct",
                display_name="qwen2.5:7b-instruct",
                context_tokens=32768,
                tags=["chat"],
                created_at=None,
                metadata={},
            ),
            ModelInfo(
                provider="ollama",
                model_id="deepseek-coder:6.7b",
                display_name="deepseek-coder:6.7b",
                context_tokens=16384,
                tags=["code"],
                created_at=None,
                metadata={},
            ),
        ]
        handler = _HandlerForTest(runtime, {"purposes": ["chat", "code"]})
        with patch("agent.api_server.list_models_ollama", return_value=fake_models):
            handler.do_POST()

        self.assertEqual(200, handler.status_code)
        payload = handler.response_payload
        self.assertTrue(bool(payload.get("ok")))
        self.assertEqual("modelops_check", payload.get("intent"))
        self.assertTrue(str(payload.get("message") or "").strip())
        envelope = payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        self.assertIn("recommendations_by_purpose", envelope or {})
        self.assertIn("recommendation_roles", envelope or {})
        recommendations = ((envelope or {}).get("recommendations_by_purpose") or {}).get("chat") if isinstance(envelope, dict) else []
        if recommendations:
            first = recommendations[0] if isinstance(recommendations[0], dict) else {}
            self.assertTrue(bool(first.get("compat_only")))
            self.assertTrue(str(first.get("source_role") or "").strip())
        best_local = ((envelope or {}).get("recommendation_roles") or {}).get("best_local") if isinstance(envelope, dict) else {}
        self.assertTrue(isinstance(best_local, dict))
        self.assertIn("state", best_local or {})
        self.assertIn("current_model", envelope or {})
        self.assertTrue(str(payload.get("trace_id") or "").strip())

    def test_llm_models_check_handles_missing_openrouter_key(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document["providers"] = {
            "openrouter": {
                "enabled": True,
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
            }
        }

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime) -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/check"
                self.headers = {"Content-Length": "0"}
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return {}

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        handler = _HandlerForTest(runtime)
        handler.do_POST()

        self.assertEqual(200, handler.status_code)
        self.assertTrue(bool(handler.response_payload.get("ok")))
        envelope = handler.response_payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        warnings = (envelope or {}).get("warnings")
        self.assertTrue(isinstance(warnings, list))
        self.assertIn("openrouter_not_configured", warnings or [])

    def test_llm_models_switch_endpoint_blocks_remote_target_in_safe_mode(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen3.5:4b",
            )
        )
        os.environ["OPENROUTER_API_KEY"] = "sk-test"
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "quality_rank": 6,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "quality_rank": 10,
                "available": True,
                "max_context_tokens": 131072,
            },
        )
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": True,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
                "openrouter": {"status": "ok", "last_checked_at": 123},
            },
            "models": {
                "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "openrouter:openai/gpt-4o-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime) -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/switch"
                self.headers = {"Content-Length": "0"}
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return {
                    "provider": "openrouter",
                    "model_id": "openai/gpt-4o-mini",
                    "purpose": "chat",
                    "confirm": True,
                }

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        handler = _HandlerForTest(runtime)
        handler.do_POST()

        self.assertEqual(400, handler.status_code)
        self.assertEqual("safe_mode_remote_switch_blocked", handler.response_payload.get("error_kind"))
        self.assertIn("SAFE MODE is local-only right now", str(handler.response_payload.get("message") or ""))
        target_status = runtime.safe_mode_target_status()
        self.assertEqual("ollama:qwen3.5:4b", target_status.get("effective_model"))
        self.assertTrue(bool(target_status.get("effective_local")))
        self.assertFalse(bool(target_status.get("explicit_override_active")))

    def test_llm_models_check_and_recommend_surfaces_controlled_mode_policy(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))
        os.environ["OPENROUTER_API_KEY"] = "sk-test"
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "quality_rank": 6,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "quality_rank": 10,
                "available": True,
                "max_context_tokens": 131072,
            },
        )
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": True,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
                "openrouter": {"status": "ok", "last_checked_at": 123},
            },
            "models": {
                "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "openrouter:openai/gpt-4o-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch("agent.api_server.list_models_ollama", return_value=[]), patch(
            "agent.api_server.list_models_openrouter",
            return_value=[],
        ):
            scout = runtime.runtime_truth_service().model_scout_v2_status(
                task_request={"task_type": "chat", "requirements": ["chat"], "preferred_local": True}
            )
            ok_check, check = runtime.llm_models_check({"purposes": ["chat"]})
            ok_recommend, recommend = runtime.llm_models_recommend(
                {"provider": "ollama", "model_id": "qwen3.5:4b", "purpose": "chat"}
            )

        self.assertTrue(ok_check)
        self.assertTrue(ok_recommend)
        check_policy = ((check.get("envelope") or {}).get("policy") if isinstance(check.get("envelope"), dict) else {})
        recommend_policy = (
            ((recommend.get("envelope") or {}).get("policy") if isinstance(recommend.get("envelope"), dict) else {})
        )
        for policy in (check_policy, recommend_policy):
            self.assertEqual("controlled", policy.get("mode"))
            self.assertTrue(bool(policy.get("allow_remote_switch")))
            self.assertTrue(bool(policy.get("allow_install_pull")))
            self.assertFalse(bool(policy.get("safe_mode")))
        self.assertEqual(
            scout.get("recommendation_roles"),
            ((check.get("envelope") or {}).get("recommendation_roles") if isinstance(check.get("envelope"), dict) else {}),
        )
        self.assertEqual(
            scout.get("recommendation_roles"),
            ((recommend.get("envelope") or {}).get("recommendation_roles") if isinstance(recommend.get("envelope"), dict) else {}),
        )
        self.assertIn(
            "comparison",
            (((check.get("envelope") or {}).get("recommendation_roles") or {}).get("best_local") or {}),
        )
        self.assertIn(
            "advisory_actions",
            (((check.get("envelope") or {}).get("recommendation_roles") or {}).get("best_local") or {}),
        )
        self.assertIn(
            "comparison",
            (((recommend.get("envelope") or {}).get("recommendation_roles") or {}).get("best_local") or {}),
        )
        self.assertIn(
            "advisory_actions",
            (((recommend.get("envelope") or {}).get("recommendation_roles") or {}).get("best_local") or {}),
        )

    def test_llm_models_proposals_endpoint_returns_structured_non_canonical_review_queue(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "openrouter:vendor/coder-pro": {
                    "provider": "openrouter",
                    "model": "vendor/coder-pro",
                    "capabilities": ["chat", "tools"],
                    "task_types": ["coding", "general_chat"],
                    "quality_rank": 8,
                    "cost_rank": 5,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 2.0},
                },
                "openrouter:vendor/old-chat": {
                    "provider": "openrouter",
                    "model": "vendor/old-chat",
                    "capabilities": ["chat"],
                    "quality_rank": 4,
                    "cost_rank": 5,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 1.0},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": None,
                "default_model": None,
                "allow_remote_fallback": True,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        runtime._model_discovery_policy_store.upsert_entry(
            "openrouter:vendor/old-chat",
            status="known_stale",
            notes="Reviewed as stale.",
            reviewed_at="2026-03-30T00:00:00Z",
        )

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, client_host: str = "127.0.0.1") -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/proposals"
                self.headers = {"Content-Length": "0"}
                self.client_address = (client_host, 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return {}

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        handler = _HandlerForTest(runtime)
        handler.do_POST()

        self.assertEqual(200, handler.status_code)
        self.assertTrue(bool(handler.response_payload.get("ok")))
        envelope = handler.response_payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        self.assertTrue(bool((envelope or {}).get("operator_only")))
        self.assertTrue(bool((envelope or {}).get("non_canonical_for_assistant")))
        self.assertEqual("model_discovery_proposals", (envelope or {}).get("separate_contract"))
        proposals = (envelope or {}).get("proposals")
        self.assertTrue(isinstance(proposals, list))
        by_model = {
            row.get("model_id"): row
            for row in proposals or []
            if isinstance(row, dict) and str(row.get("model_id") or "").strip()
        }
        self.assertEqual("candidate_good", (by_model.get("openrouter:vendor/coder-pro") or {}).get("proposal_kind"))
        self.assertEqual("candidate_stale", (by_model.get("openrouter:vendor/old-chat") or {}).get("proposal_kind"))
        self.assertEqual("known_stale", (by_model.get("openrouter:vendor/old-chat") or {}).get("policy_status"))
        self.assertTrue(bool((by_model.get("openrouter:vendor/coder-pro") or {}).get("review_required")))
        self.assertTrue(bool((by_model.get("openrouter:vendor/coder-pro") or {}).get("non_canonical")))
        self.assertEqual("not_adopted", (by_model.get("openrouter:vendor/coder-pro") or {}).get("canonical_status"))
        coding_review = ((by_model.get("openrouter:vendor/coder-pro") or {}).get("review_suggestion")) or {}
        self.assertTrue(bool(coding_review.get("available")))
        self.assertEqual("/llm/models/policy", coding_review.get("write_surface"))
        self.assertEqual("known_good", coding_review.get("suggested_status"))
        self.assertEqual(["coding"], coding_review.get("suggested_role_hints"))
        self.assertEqual(
            "operator_review",
            ((coding_review.get("payload_template") or {}).get("source")),
        )
        stale_review = ((by_model.get("openrouter:vendor/old-chat") or {}).get("review_suggestion")) or {}
        self.assertTrue(bool(stale_review.get("available")))
        self.assertEqual("/llm/models/policy", stale_review.get("write_surface"))
        self.assertEqual("known_stale", stale_review.get("suggested_status"))

    def test_llm_models_proposals_endpoint_filters_by_source_and_kind(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "openrouter:vendor/coder-pro": {
                    "provider": "openrouter",
                    "model": "vendor/coder-pro",
                    "capabilities": ["chat", "tools"],
                    "task_types": ["coding", "general_chat"],
                    "quality_rank": 8,
                    "cost_rank": 5,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 2.0},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": None,
                "default_model": None,
                "allow_remote_fallback": True,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        runtime._model_watch_catalog_path = Path(self.tmpdir.name) / "model_watch_catalog_snapshot.json"
        write_snapshot_atomic(
            runtime._model_watch_catalog_path,
            {
                "provider": "openrouter",
                "source": "openrouter_models",
                "fetched_at": 1774915200,
                "models": [
                    {
                        "id": "openrouter:vendor/cheap-text",
                        "provider_id": "openrouter",
                        "model": "vendor/cheap-text",
                        "context_length": 131072,
                        "modalities": ["text"],
                        "supports_tools": False,
                        "pricing": {
                            "prompt_per_million": 0.1,
                            "completion_per_million": 0.2,
                        },
                    }
                ],
            },
        )

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/proposals"
                self.headers = {"Content-Length": "0"}
                self.client_address = ("127.0.0.1", 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}
                self._payload = dict(payload)

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        external = _HandlerForTest(runtime, {"source": "external_openrouter_snapshot"})
        external.do_POST()
        self.assertEqual(200, external.status_code)
        external_envelope = external.response_payload.get("envelope")
        self.assertTrue(isinstance(external_envelope, dict))
        self.assertEqual("external_openrouter_snapshot", ((external_envelope or {}).get("filters") or {}).get("source"))
        self.assertIn("registry", (external_envelope or {}).get("available_sources") or [])
        self.assertIn("external_openrouter_snapshot", (external_envelope or {}).get("available_sources") or [])
        external_rows = (external_envelope or {}).get("proposals") or []
        self.assertEqual(1, len(external_rows))
        self.assertEqual(
            "external_openrouter_snapshot",
            (external_rows[0] if isinstance(external_rows[0], dict) else {}).get("source"),
        )
        self.assertTrue(bool((external_rows[0] if isinstance(external_rows[0], dict) else {}).get("non_canonical")))

        canonical = _HandlerForTest(runtime, {"source": "registry"})
        canonical.do_POST()
        self.assertEqual(200, canonical.status_code)
        canonical_rows = (((canonical.response_payload.get("envelope") or {}).get("proposals")) or [])
        self.assertEqual(1, len(canonical_rows))
        self.assertEqual(
            "registry",
            (canonical_rows[0] if isinstance(canonical_rows[0], dict) else {}).get("source"),
        )

        kind_filtered = _HandlerForTest(runtime, {"proposal_kind": "candidate_good"})
        kind_filtered.do_POST()
        self.assertEqual(200, kind_filtered.status_code)
        kind_envelope = kind_filtered.response_payload.get("envelope")
        self.assertTrue(isinstance(kind_envelope, dict))
        self.assertEqual("candidate_good", ((kind_envelope or {}).get("filters") or {}).get("proposal_kind"))
        self.assertIn("candidate_good", (kind_envelope or {}).get("available_proposal_kinds") or [])
        self.assertTrue(
            all(
                isinstance(row, dict) and row.get("proposal_kind") == "candidate_good"
                for row in ((kind_envelope or {}).get("proposals") or [])
            )
        )

    def test_llm_models_proposals_endpoint_rejects_invalid_source_and_kind_filters(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/proposals"
                self.headers = {"Content-Length": "0"}
                self.client_address = ("127.0.0.1", 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}
                self._payload = dict(payload)

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        invalid_source = _HandlerForTest(runtime, {"source": "unknown_source"})
        invalid_source.do_POST()
        self.assertEqual(400, invalid_source.status_code)
        self.assertEqual("bad_request", invalid_source.response_payload.get("error_kind"))
        self.assertIn(
            "external_openrouter_snapshot",
            (((invalid_source.response_payload.get("envelope") or {}).get("allowed_sources")) or []),
        )

        invalid_kind = _HandlerForTest(runtime, {"proposal_kind": "unsupported_kind"})
        invalid_kind.do_POST()
        self.assertEqual(400, invalid_kind.status_code)
        self.assertEqual("bad_request", invalid_kind.response_payload.get("error_kind"))
        self.assertIn(
            "candidate_good",
            (((invalid_kind.response_payload.get("envelope") or {}).get("allowed_proposal_kinds")) or []),
        )

    def test_llm_models_policy_endpoint_writes_updates_and_removes_entries(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "openrouter:vendor/coder-pro": {
                    "provider": "openrouter",
                    "model": "vendor/coder-pro",
                    "capabilities": ["chat", "tools"],
                    "task_types": ["coding", "general_chat"],
                    "quality_rank": 8,
                    "cost_rank": 5,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 2.0},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": None,
                "default_model": None,
                "allow_remote_fallback": True,
            },
        }
        runtime._save_registry_document(runtime.registry_document)

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object], client_host: str = "127.0.0.1") -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/policy"
                self.headers = {"Content-Length": "0"}
                self.client_address = (client_host, 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}
                self._payload = dict(payload)

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        create = _HandlerForTest(
            runtime,
            {
                "model_id": "openrouter:vendor/coder-pro",
                "status": "known_good",
                "role_hints": ["coding"],
                "notes": "Reviewed as a strong coding option.",
                "justification": "Operator review",
                "reviewed_at": "2026-03-30T00:00:00Z",
            },
        )
        create.do_POST()
        self.assertEqual(200, create.status_code)
        self.assertEqual("known_good", (((create.response_payload.get("envelope") or {}).get("entry")) or {}).get("status"))
        self.assertEqual(["coding"], (((create.response_payload.get("envelope") or {}).get("entry")) or {}).get("role_hints"))

        update = _HandlerForTest(
            runtime,
            {
                "model_id": "openrouter:vendor/coder-pro",
                "status": "avoid",
                "role_hints": ["research"],
                "notes": "No longer preferred.",
            },
        )
        update.do_POST()
        self.assertEqual(200, update.status_code)
        update_entry = (((update.response_payload.get("envelope") or {}).get("entry")) or {})
        self.assertEqual("avoid", update_entry.get("status"))
        self.assertEqual(["research"], update_entry.get("role_hints"))
        self.assertEqual("No longer preferred.", update_entry.get("notes"))

        proposals = _HandlerForTest(runtime, {}, client_host="127.0.0.1")
        proposals.path = "/llm/models/proposals"
        proposals.do_POST()
        self.assertEqual(200, proposals.status_code)
        proposal_rows = ((proposals.response_payload.get("envelope") or {}).get("proposals")) or []
        proposal = next(
            (
                row
                for row in proposal_rows
                if isinstance(row, dict) and row.get("model_id") == "openrouter:vendor/coder-pro"
            ),
            {},
        )
        self.assertEqual("avoid", proposal.get("policy_status"))
        self.assertEqual("candidate_stale", proposal.get("proposal_kind"))
        proposal_review = (proposal.get("review_suggestion") or {})
        self.assertTrue(bool(proposal_review.get("available")))
        self.assertEqual("/llm/models/policy", proposal_review.get("write_surface"))
        self.assertEqual("avoid", proposal_review.get("suggested_status"))

        remove = _HandlerForTest(
            runtime,
            {
                "model_id": "openrouter:vendor/coder-pro",
                "action": "remove",
            },
        )
        remove.do_POST()
        self.assertEqual(200, remove.status_code)
        self.assertTrue(bool(((remove.response_payload.get("envelope") or {}).get("removed"))))
        policy_entries = ((remove.response_payload.get("envelope") or {}).get("policy_entries")) or []
        self.assertFalse(any(isinstance(row, dict) and row.get("model_id") == "openrouter:vendor/coder-pro" for row in policy_entries))

    def test_llm_models_policy_get_lists_filters_and_matches_written_entries(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))
        runtime.registry_document = {
            "schema_version": 2,
            "providers": {
                "openrouter": {
                    "enabled": True,
                    "local": False,
                    "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                },
            },
            "models": {
                "openrouter:vendor/coder-pro": {
                    "provider": "openrouter",
                    "model": "vendor/coder-pro",
                    "capabilities": ["chat", "tools"],
                    "task_types": ["coding", "general_chat"],
                    "quality_rank": 8,
                    "cost_rank": 5,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 65536,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 2.0},
                },
                "openrouter:vendor/old-chat": {
                    "provider": "openrouter",
                    "model": "vendor/old-chat",
                    "capabilities": ["chat"],
                    "quality_rank": 4,
                    "cost_rank": 5,
                    "enabled": True,
                    "available": True,
                    "max_context_tokens": 32768,
                    "pricing": {"input_per_million_tokens": 1.0, "output_per_million_tokens": 1.0},
                },
            },
            "defaults": {
                "routing_mode": "auto",
                "default_provider": None,
                "default_model": None,
                "allow_remote_fallback": True,
            },
        }
        runtime._save_registry_document(runtime.registry_document)
        runtime._model_discovery_policy_store.upsert_entry(
            "openrouter:vendor/coder-pro",
            status="known_good",
            role_hints=["coding"],
            notes="Reviewed as strong for coding.",
            reviewed_at="2026-03-31T00:00:00Z",
        )
        runtime._model_discovery_policy_store.upsert_entry(
            "openrouter:vendor/old-chat",
            status="avoid",
            notes="Reviewed as stale.",
            reviewed_at="2026-03-30T00:00:00Z",
        )

        class _GetHandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, path: str, client_host: str = "127.0.0.1") -> None:
                self.runtime = runtime_obj
                self.path = path
                self.headers = {"Content-Length": "0"}
                self.client_address = (client_host, 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        list_handler = _GetHandlerForTest(runtime, "/llm/models/policy")
        list_handler.do_GET()
        self.assertEqual(200, list_handler.status_code)
        envelope = list_handler.response_payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        rows = (envelope or {}).get("policy_entries")
        self.assertTrue(isinstance(rows, list))
        self.assertEqual(
            ["openrouter:vendor/coder-pro", "openrouter:vendor/old-chat"],
            [row.get("model_id") for row in rows if isinstance(row, dict)],
        )
        self.assertEqual(2, (envelope or {}).get("count"))

        model_handler = _GetHandlerForTest(runtime, "/llm/models/policy?model_id=openrouter:vendor/coder-pro")
        model_handler.do_GET()
        self.assertEqual(200, model_handler.status_code)
        model_rows = (((model_handler.response_payload.get("envelope") or {}).get("policy_entries")) or [])
        self.assertEqual(1, len(model_rows))
        self.assertEqual("openrouter:vendor/coder-pro", (model_rows[0] if isinstance(model_rows[0], dict) else {}).get("model_id"))

        status_handler = _GetHandlerForTest(runtime, "/llm/models/policy?status=avoid")
        status_handler.do_GET()
        self.assertEqual(200, status_handler.status_code)
        status_rows = (((status_handler.response_payload.get("envelope") or {}).get("policy_entries")) or [])
        self.assertEqual(1, len(status_rows))
        self.assertEqual("avoid", (status_rows[0] if isinstance(status_rows[0], dict) else {}).get("status"))

        proposals = _GetHandlerForTest(runtime, "/llm/models/policy?status=known_good")
        proposals.do_GET()
        self.assertEqual(200, proposals.status_code)
        proposal_ok, proposal_body = runtime.llm_models_proposals({})
        self.assertTrue(proposal_ok)
        proposal_rows = (((proposal_body.get("envelope") or {}).get("proposals")) or [])
        coder_proposal = next(
            (
                row
                for row in proposal_rows
                if isinstance(row, dict) and row.get("model_id") == "openrouter:vendor/coder-pro"
            ),
            {},
        )
        self.assertTrue(bool(coder_proposal.get("non_canonical")))
        self.assertEqual("not_adopted", coder_proposal.get("canonical_status"))

    def test_llm_models_policy_get_rejects_invalid_filters_and_is_loopback_only(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))

        class _GetHandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, path: str, client_host: str) -> None:
                self.runtime = runtime_obj
                self.path = path
                self.headers = {"Content-Length": "0"}
                self.client_address = (client_host, 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        forbidden = _GetHandlerForTest(runtime, "/llm/models/policy", client_host="10.0.0.8")
        forbidden.do_GET()
        self.assertEqual(403, forbidden.status_code)
        self.assertEqual("forbidden", forbidden.response_payload.get("error_kind"))

        invalid = _GetHandlerForTest(runtime, "/llm/models/policy?status=unsupported_status", client_host="127.0.0.1")
        invalid.do_GET()
        self.assertEqual(400, invalid.status_code)
        self.assertEqual("bad_request", invalid.response_payload.get("error_kind"))
        envelope = invalid.response_payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        self.assertIn("known_good", (envelope or {}).get("allowed_statuses") or [])

    def test_llm_models_policy_endpoint_rejects_invalid_values_and_is_loopback_only(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))

        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object], client_host: str) -> None:
                self.runtime = runtime_obj
                self.path = "/llm/models/policy"
                self.headers = {"Content-Length": "0"}
                self.client_address = (client_host, 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}
                self._payload = dict(payload)

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        forbidden = _HandlerForTest(
            runtime,
            {"model_id": "openrouter:vendor/coder-pro", "status": "known_good"},
            client_host="10.0.0.8",
        )
        forbidden.do_POST()
        self.assertEqual(403, forbidden.status_code)
        self.assertEqual("forbidden", forbidden.response_payload.get("error_kind"))

        invalid_status = _HandlerForTest(
            runtime,
            {"model_id": "openrouter:vendor/coder-pro", "status": "unsupported_status"},
            client_host="127.0.0.1",
        )
        invalid_status.do_POST()
        self.assertEqual(400, invalid_status.status_code)
        self.assertEqual("bad_request", invalid_status.response_payload.get("error_kind"))

        invalid_hints = _HandlerForTest(
            runtime,
            {
                "model_id": "openrouter:vendor/coder-pro",
                "status": "known_good",
                "role_hints": ["not_a_real_hint"],
            },
            client_host="127.0.0.1",
        )
        invalid_hints.do_POST()
        self.assertEqual(400, invalid_hints.status_code)
        self.assertEqual("bad_request", invalid_hints.response_payload.get("error_kind"))
        envelope = invalid_hints.response_payload.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        self.assertIn("coding", (envelope or {}).get("allowed_role_hints") or [])

    def test_llm_models_switch_remote_confirm_uses_canonical_confirmed_target_setter(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))

        with patch.object(
            runtime,
            "set_confirmed_chat_model_target",
            return_value=(
                True,
                {
                    "ok": True,
                    "provider": "openrouter",
                    "model_id": "openrouter:openai/gpt-4o-mini",
                    "message": "Now using openrouter:openai/gpt-4o-mini for chat.",
                },
            ),
        ) as confirmed_mock, patch.object(
            runtime,
            "set_default_chat_model",
            side_effect=AssertionError("remote switch should use confirmed target setter"),
        ):
            ok, body = runtime.llm_models_switch(
                {
                    "provider": "openrouter",
                    "model_id": "openai/gpt-4o-mini",
                    "purpose": "chat",
                    "confirm": True,
                }
            )

        self.assertTrue(ok)
        self.assertTrue(bool(body.get("ok")))
        confirmed_mock.assert_called_once_with(
            "openrouter:openai/gpt-4o-mini",
            provider_id="openrouter",
        )

    def test_llm_models_switch_remote_confirm_blocks_unusable_target_in_controlled_mode(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))
        os.environ["OPENROUTER_API_KEY"] = "sk-test"
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "quality_rank": 6,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "quality_rank": 10,
                "available": True,
                "max_context_tokens": 131072,
            },
        )
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
                "openrouter": {"status": "down", "last_checked_at": 123},
            },
            "models": {
                "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "openrouter:openai/gpt-4o-mini": {"provider_id": "openrouter", "status": "down", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        ok, body = runtime.llm_models_switch(
            {
                "provider": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "purpose": "chat",
                "confirm": True,
            }
        )

        self.assertFalse(ok)
        self.assertEqual("switch_target_not_usable", body.get("error_kind"))
        self.assertIn("provider is down", str(body.get("message") or ""))

    def test_llm_models_switch_remote_confirm_allowed_when_remote_fallback_is_disabled(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, safe_mode_enabled=False))
        os.environ["OPENROUTER_API_KEY"] = "sk-test"
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "quality_rank": 6,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "quality_rank": 10,
                "available": True,
                "max_context_tokens": 131072,
            },
        )
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": False,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
                "openrouter": {"status": "ok", "last_checked_at": 123},
            },
            "models": {
                "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "openrouter:openai/gpt-4o-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        ok, body = runtime.llm_models_switch(
            {
                "provider": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "purpose": "chat",
                "confirm": True,
            }
        )

        self.assertTrue(ok)
        self.assertTrue(bool(body.get("ok")))
        self.assertIn("Switched to openrouter:openai/gpt-4o-mini", str(body.get("message") or ""))
        envelope = body.get("envelope") if isinstance(body.get("envelope"), dict) else {}
        execution = envelope.get("execution") if isinstance(envelope.get("execution"), list) else []
        first_step = execution[0] if execution and isinstance(execution[0], dict) else {}
        first_body = first_step.get("body") if isinstance(first_step.get("body"), dict) else {}
        self.assertEqual("openrouter:openai/gpt-4o-mini", first_body.get("model_id"))


if __name__ == "__main__":
    unittest.main()
