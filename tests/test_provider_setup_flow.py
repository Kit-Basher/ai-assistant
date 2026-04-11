from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from agent import cli
from agent.api_server import APIServerHandler, AgentRuntime
from agent.audit_log import AuditLog
from agent.config import Config
from agent.ux.llm_fixit_wizard import LLMFixitWizardStore
from telegram_adapter.bot import _handle_message


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


class _FakeDB:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set_preference(self, key: str, value: str) -> None:
        self.values[str(key)] = str(value)


class _FakeOrchestrator:
    def __init__(self, response_text: str = "orchestrator fallback") -> None:
        self.calls: list[tuple[str, str]] = []
        self.response_text = response_text

    def handle_message(self, text: str, *, user_id: str) -> object:
        self.calls.append((text, user_id))
        return type("Response", (), {"text": self.response_text, "data": None})()


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.message_id = 1
        self.replies: list[dict[str, str | None]] = []

    async def reply_text(self, text: str, parse_mode: str | None = None) -> None:
        self.replies.append({"text": text, "parse_mode": parse_mode})


class _FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _FakeUpdate:
    def __init__(self, chat_id: int, text: str) -> None:
        self.effective_chat = _FakeChat(chat_id)
        self.effective_message = _FakeMessage(text)


class _FakeContext:
    def __init__(self, bot_data: dict[str, object]) -> None:
        self.application = type("App", (), {"bot_data": bot_data})()


class _ExplodingTelegramRuntime:
    def chat(self, _payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
        raise AssertionError("Telegram ordinary chat should proxy to the API-backed runtime")


def _invoke_chat_http(runtime: AgentRuntime, payload: dict[str, object]) -> dict[str, object]:
    class _HandlerForTest(APIServerHandler):
        def __init__(self, runtime_obj: AgentRuntime, request_payload: dict[str, object]) -> None:
            self.runtime = runtime_obj
            self.path = "/chat"
            self.headers = {"Content-Length": "0"}
            self._payload = dict(request_payload)
            self.status_code = 0
            self.response_payload: dict[str, object] = {}

        def _read_json(self) -> dict[str, object]:  # type: ignore[override]
            return dict(self._payload)

        def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
            _ = status
            self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

    handler = _HandlerForTest(runtime, payload)
    handler.do_POST()
    return handler.response_payload


class TestProviderSetupFlow(unittest.TestCase):
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

    def _configure_openrouter_runtime(self, runtime: AgentRuntime, *, make_default: bool = False) -> dict[str, object]:
        with patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True, "provider": "openrouter", "model": "openai/gpt-4o-mini"}),
        ), patch.object(
            runtime,
            "refresh_models",
            return_value=(True, {"ok": True, "models": []}),
        ):
            ok, body = runtime.configure_openrouter("sk-or-v1-testsecret1234567890", {"make_default": make_default})
        self.assertTrue(ok)
        return body

    def test_auto_bootstrap_prefers_working_local_chat_model(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.update_defaults({"default_provider": None, "chat_model": None})
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "quality_rank": 8,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        snapshot = {
            "providers": [
                {"id": "ollama", "enabled": True, "local": True, "health": {"status": "ok"}},
                {"id": "openrouter", "enabled": True, "local": False, "health": {"status": "down"}},
            ],
            "models": [
                {
                    "id": "ollama:qwen2.5:7b-instruct",
                    "provider": "ollama",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "capabilities": ["chat"],
                    "health": {"status": "ok"},
                }
            ],
        }

        with patch.object(runtime, "_canonical_runtime_router_snapshot", return_value=snapshot), patch.object(
            runtime,
            "refresh_models",
            return_value=(True, {"ok": True}),
        ), patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True, "provider": "ollama", "model": "qwen2.5:7b-instruct"}),
        ):
            payload = runtime.llm_status()

        self.assertEqual("ollama", payload["default_provider"])
        self.assertEqual("ollama:qwen2.5:7b-instruct", payload["resolved_default_model"])
        self.assertEqual("ollama:qwen2.5:7b-instruct", runtime.get_defaults()["resolved_default_model"])

    def test_chat_openrouter_setup_requests_key_in_structured_contract(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        for utterance in ("help me set up openrouter", "configure openrouter"):
            with self.subTest(utterance=utterance):
                with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
                    ok, body = runtime.chat(
                        {
                            "messages": [{"role": "user", "content": utterance}],
                            "source_surface": "api",
                        }
                    )

                self.assertTrue(ok)
                self.assertEqual("request_secret", body["setup"]["type"])
                self.assertIn("OpenRouter", body["assistant"]["content"])
                self.assertIn("API key", body["assistant"]["content"])
                self.assertNotIn("install openrouter", body["assistant"]["content"].lower())
                self.assertNotIn("protocol", body["assistant"]["content"].lower())

    def test_http_chat_openrouter_setup_skips_memory_lookup_and_bootstrap(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        with patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            side_effect=AssertionError("openrouter setup prompt should not auto-bootstrap first"),
        ), patch.object(
            runtime,
            "build_memory_context_for_payload",
            side_effect=AssertionError("openrouter setup prompt should not build memory context first"),
        ):
            body = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "configure openrouter"}],
                    "source_surface": "telegram",
                },
            )

        self.assertEqual("setup_flow", body["meta"]["route"])
        self.assertEqual("request_secret", body["setup"]["type"])
        self.assertIn("OpenRouter", body["assistant"]["content"])
        self.assertIn("API key", body["assistant"]["content"])

    def test_http_chat_configure_openrouter_with_stored_key_returns_fast_reuse_prompt(self) -> None:
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
        runtime._record_authoritative_provider_success("ollama", "ollama:qwen3.5:4b")
        ok, _secret = runtime.set_provider_secret("openrouter", {"api_key": "sk-or-v1-testsecret1234567890"})
        self.assertTrue(ok)

        with patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            side_effect=AssertionError("stored-key setup init should not auto-bootstrap first"),
        ), patch.object(
            runtime,
            "build_memory_context_for_payload",
            side_effect=AssertionError("stored-key setup init should not build memory context first"),
        ), patch.object(
            runtime,
            "configure_openrouter",
            side_effect=AssertionError("stored-key setup init should not run provider test immediately"),
        ):
            body = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "configure openrouter"}],
                    "source_surface": "telegram",
                },
            )

        self.assertEqual("setup_flow", body["meta"]["route"])
        self.assertEqual("confirm_reuse_secret", body["setup"]["type"])
        self.assertIn("stored", body["assistant"]["content"].lower())
        self.assertIn("OpenRouter", body["assistant"]["content"])
        self.assertNotIn("timed out", body["assistant"]["content"].lower())

    def test_http_chat_deterministic_runtime_routes_skip_bootstrap_and_memory_lookup(self) -> None:
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

        utterances = (
            ("what model are you using?", "model_status"),
            ("is openrouter configured?", "provider_status"),
            ("runtime", "runtime_status"),
            ("configure ollama", "setup_flow"),
            ("configure openrouter", "setup_flow"),
            ("what managed adapters exist?", "governance_status"),
            ("what is my model selection policy?", "model_policy_status"),
            ("what is my cheap remote cap?", "model_policy_status"),
            ("what execution mode does Telegram use?", "model_policy_status"),
        )

        with patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            side_effect=AssertionError("deterministic runtime routes should not auto-bootstrap first"),
        ), patch.object(
            runtime,
            "build_memory_context_for_payload",
            side_effect=AssertionError("deterministic runtime routes should not build memory context first"),
        ):
            for utterance, expected_route in utterances:
                with self.subTest(utterance=utterance):
                    body = _invoke_chat_http(
                        runtime,
                        {
                            "messages": [{"role": "user", "content": utterance}],
                            "source_surface": "telegram",
                        },
                    )
                    self.assertEqual(expected_route, body["meta"]["route"])
                    if utterance == "what execution mode does Telegram use?":
                        self.assertEqual("model_controller_policy", body["setup"]["type"])
                        self.assertIn("Mode: Controlled Mode.", body["assistant"]["content"])

    def test_http_chat_model_policy_explainability_routes_are_grounded(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
                "max_context_tokens": 32768,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "free-remote",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 7,
                "max_context_tokens": 65536,
                "pricing": {"input_per_million_tokens": 0.0, "output_per_million_tokens": 0.0},
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "cheap-remote",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 9,
                "max_context_tokens": 131072,
                "pricing": {"input_per_million_tokens": 0.3, "output_per_million_tokens": 0.3},
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")
        runtime._record_authoritative_provider_success("ollama", "ollama:qwen3.5:4b")
        ok, _secret = runtime.set_provider_secret("openrouter", {"api_key": "sk-or-v1-testsecret1234567890"})
        self.assertTrue(ok)
        health_summary = {
            "providers": [
                {"id": "ollama", "status": "ok"},
                {"id": "openrouter", "status": "ok"},
            ],
            "models": [
                {"id": "ollama:qwen3.5:4b", "status": "ok"},
                {"id": "openrouter:free-remote", "status": "ok"},
                {"id": "openrouter:cheap-remote", "status": "ok"},
            ],
        }
        utterances = {
            "what is my model selection policy?": "local first",
            "what is my cheap remote cap?": "$0.50 per 1M tokens",
            "why are you using this model?": "ollama:qwen3.5:4b",
            "what model would you switch to right now?": "keep ollama:qwen3.5:4b",
            "what free remote model would you choose?": "openrouter:free-remote",
            "what cheap remote model would you choose?": "$0.50 per 1M tokens",
            "why didn't you switch to openrouter?": "OpenRouter",
        }

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            runtime._health_monitor,
            "summary",
            return_value=health_summary,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("model policy explainability should not use generic chat"),
        ):
            for utterance, expected_text in utterances.items():
                with self.subTest(utterance=utterance):
                    body = _invoke_chat_http(
                        runtime,
                        {
                            "messages": [{"role": "user", "content": utterance}],
                            "source_surface": "api",
                        },
                    )
                    self.assertEqual("model_policy_status", body["meta"]["route"])
                    self.assertFalse(body["meta"]["generic_fallback_used"])
                    self.assertIn(expected_text.lower(), body["assistant"]["content"].lower())

    def test_http_chat_yes_for_pending_openrouter_reuse_skips_memory_lookup_and_bootstrap(self) -> None:
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
        ok, _secret = runtime.set_provider_secret("openrouter", {"api_key": "sk-or-v1-testsecret1234567890"})
        self.assertTrue(ok)
        first = _invoke_chat_http(
            runtime,
            {
                "messages": [{"role": "user", "content": "configure openrouter"}],
                "source_surface": "telegram",
                "user_id": "telegram:12345",
                "thread_id": "telegram:12345:thread",
            },
        )
        self.assertEqual("setup_flow", first["meta"]["route"])
        self.assertEqual("confirm_reuse_secret", first["setup"]["type"])

        with patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            side_effect=AssertionError("pending setup confirmation should not auto-bootstrap first"),
        ), patch.object(
            runtime,
            "build_memory_context_for_payload",
            side_effect=AssertionError("pending setup confirmation should not build memory context first"),
        ), patch.object(
            runtime,
            "test_provider",
            return_value=(
                True,
                {
                    "ok": True,
                    "provider": "openrouter",
                    "model_id": "openrouter:openai/gpt-4o-mini",
                    "message": "OpenRouter is ready. I tested it with openrouter:openai/gpt-4o-mini.",
                },
            ),
        ), patch.object(
            runtime,
            "refresh_models",
            side_effect=AssertionError("pending setup confirmation should defer full model refresh"),
        ):
            second = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "yes"}],
                    "source_surface": "telegram",
                    "user_id": "telegram:12345",
                    "thread_id": "telegram:12345:thread",
                },
            )

        self.assertEqual("setup_flow", second["meta"]["route"])
        self.assertIn("OpenRouter", second["assistant"]["content"])
        self.assertNotIn("timed out", second["assistant"]["content"].lower())
        self.assertTrue(second["setup"]["ok"])
        self.assertEqual("openrouter", second["setup"]["provider"])
        with open(runtime.config.log_path, "r", encoding="utf-8") as handle:
            log_rows = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
        configure_rows = [
            row.get("payload")
            for row in log_rows
            if str(row.get("type") or "") == "runtime.configure_openrouter" and isinstance(row.get("payload"), dict)
        ]
        self.assertTrue(configure_rows)
        latest_row = configure_rows[-1]
        self.assertEqual(True, latest_row.get("refresh_deferred"))
        self.assertIn("timings_ms", latest_row)
        self.assertIn("total_duration_ms", latest_row)

    def test_configure_openrouter_persists_secret_and_default_model(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        body = self._configure_openrouter_runtime(runtime, make_default=True)
        ok = True

        self.assertTrue(ok)
        self.assertEqual("https://openrouter.ai/api/v1", runtime.registry_document["providers"]["openrouter"]["base_url"])
        self.assertEqual("/chat/completions", runtime.registry_document["providers"]["openrouter"]["chat_path"])
        self.assertEqual(
            "sk-or-v1-testsecret1234567890",
            runtime.secret_store.get_secret("provider:openrouter:api_key"),
        )
        self.assertIn("openrouter:openai/gpt-4o-mini", runtime.registry_document["models"])
        self.assertEqual("openrouter", runtime.get_defaults()["default_provider"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", runtime.get_defaults()["resolved_default_model"])
        self.assertTrue(bool(body.get("switched")))

    def test_http_chat_confirmed_switch_in_safe_mode_uses_exact_offered_target(self) -> None:
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
        runtime._record_authoritative_provider_success("ollama", "ollama:qwen3.5:4b")  # type: ignore[attr-defined]
        ok, _secret = runtime.set_provider_secret("openrouter", {"api_key": "sk-or-v1-testsecret1234567890"})
        self.assertTrue(ok)

        first = _invoke_chat_http(
            runtime,
            {
                "messages": [{"role": "user", "content": "configure openrouter"}],
                "source_surface": "api",
                "user_id": "user-safe",
                "thread_id": "thread-safe",
            },
        )
        self.assertEqual("setup_flow", first["meta"]["route"])
        self.assertEqual("confirm_reuse_secret", first["setup"]["type"])

        with patch.object(
            runtime,
            "test_provider",
            return_value=(
                True,
                {
                    "ok": True,
                    "provider": "openrouter",
                    "model": "ai21/jamba-large-1.7",
                    "message": "OpenRouter is ready. I tested it with openrouter:ai21/jamba-large-1.7.",
                },
            ),
        ), patch.object(
            runtime,
            "refresh_models",
            side_effect=AssertionError("safe-mode confirmation should not refresh the catalog"),
        ):
            second = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "yes"}],
                    "source_surface": "api",
                    "user_id": "user-safe",
                    "thread_id": "thread-safe",
                },
            )

        self.assertEqual("setup_flow", second["meta"]["route"])
        self.assertIn("openrouter:ai21/jamba-large-1.7", second["assistant"]["content"])

        third = _invoke_chat_http(
            runtime,
            {
                "messages": [{"role": "user", "content": "yes"}],
                "source_surface": "api",
                "user_id": "user-safe",
                "thread_id": "thread-safe",
            },
        )

        self.assertEqual("setup_flow", third["meta"]["route"])
        self.assertIn("SAFE MODE is local-only right now", third["assistant"]["content"])
        self.assertIn("did not switch chat to remote openrouter:ai21/jamba-large-1.7", third["assistant"]["content"])
        defaults = runtime.get_defaults()
        self.assertEqual("ollama", defaults["default_provider"])
        self.assertEqual("ollama:qwen3.5:4b", defaults["chat_model"])
        self.assertEqual("ollama:qwen3.5:4b", defaults["resolved_default_model"])

    def test_successful_openrouter_switch_updates_canonical_runtime_truth(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        stale_epoch = 1_771_050_000
        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            "openrouter": {
                "status": "down",
                "last_error_kind": "server_error",
                "status_code": 503,
                "last_checked_at": stale_epoch,
                "cooldown_until": None,
                "down_since": stale_epoch,
                "failure_streak": 3,
                "next_probe_at": stale_epoch + 900,
            }
        }
        runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
            "openrouter:openai/gpt-4o-mini": {
                "provider_id": "openrouter",
                "status": "down",
                "last_error_kind": "server_error",
                "status_code": 503,
                "last_checked_at": stale_epoch,
                "cooldown_until": None,
                "down_since": stale_epoch,
                "failure_streak": 3,
                "next_probe_at": stale_epoch + 900,
            }
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
                "max_context_tokens": 32768,
            },
        )
        self._configure_openrouter_runtime(runtime, make_default=True)
        runtime._record_authoritative_provider_success("ollama", "ollama:qwen3.5:4b")  # type: ignore[attr-defined]
        runtime.update_defaults({"allow_remote_fallback": False})

        llm_status = runtime.llm_status()
        with patch.object(
            runtime,
            "telegram_status",
            return_value={
                "ok": True,
                "enabled": True,
                "configured": True,
                "state": "running",
                "effective_state": "enabled_running",
                "next_action": "No action needed.",
            },
        ):
            ready_status = runtime.ready_status()
        provider_truth = runtime.runtime_truth_service().provider_status("openrouter")

        self.assertEqual("openrouter", llm_status["default_provider"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", llm_status["resolved_default_model"])
        self.assertEqual("ok", llm_status["active_provider_health"]["status"])
        self.assertEqual("ok", llm_status["active_model_health"]["status"])
        self.assertGreater(int(llm_status["active_provider_health"]["last_checked_at"]), stale_epoch)
        self.assertGreater(int(llm_status["active_model_health"]["last_checked_at"]), stale_epoch)
        self.assertFalse(bool(llm_status["active_model_visible"]))
        self.assertEqual("READY", llm_status["runtime_mode"])

        self.assertTrue(bool(ready_status["ready"]))
        self.assertEqual("READY", ready_status["runtime_mode"])
        self.assertEqual("openrouter", ready_status["llm"]["provider"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", ready_status["llm"]["model"])
        self.assertEqual("ok", ready_status["llm"]["active_provider_health"]["status"])
        self.assertEqual("ok", ready_status["llm"]["active_model_health"]["status"])

        self.assertTrue(provider_truth["configured"])
        self.assertTrue(provider_truth["active"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", provider_truth["model_id"])
        self.assertEqual("ok", provider_truth["health_status"])

    def test_chat_provider_status_and_policy_candidate_stay_grounded_after_openrouter_switch(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
                "max_context_tokens": 32768,
            },
        )
        self._configure_openrouter_runtime(runtime, make_default=True)
        runtime._record_authoritative_provider_success("ollama", "ollama:qwen3.5:4b")  # type: ignore[attr-defined]

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            ok_provider, provider_body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "is openrouter configured?"}],
                    "source_surface": "api",
                }
            )
            ok_policy, policy_body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "what model would you switch to right now?"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok_provider)
        self.assertEqual("provider_status", provider_body["meta"]["route"])
        self.assertTrue(provider_body["setup"]["configured"])
        self.assertTrue(provider_body["setup"]["active"])
        self.assertEqual("openrouter", provider_body["setup"]["provider"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", provider_body["setup"]["model_id"])
        self.assertIn("OpenRouter", provider_body["assistant"]["content"])
        self.assertNotIn("internal error", provider_body["assistant"]["content"].lower())

        self.assertTrue(ok_policy)
        self.assertEqual("model_policy_status", policy_body["meta"]["route"])
        self.assertFalse(policy_body["meta"]["generic_fallback_used"])
        self.assertIn("ollama:qwen3.5:4b", policy_body["assistant"]["content"])
        self.assertNotIn("I couldn't read that from the runtime state.", policy_body["assistant"]["content"])

    def test_chat_model_policy_candidate_is_grounded_when_openrouter_is_configured_but_not_active(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
                "max_context_tokens": 32768,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")
        runtime._record_authoritative_provider_success("ollama", "ollama:qwen3.5:4b")  # type: ignore[attr-defined]
        self._configure_openrouter_runtime(runtime, make_default=False)

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "what model would you switch to right now?"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok)
        self.assertEqual("model_policy_status", body["meta"]["route"])
        self.assertIn("keep ollama:qwen3.5:4b", body["assistant"]["content"].lower())
        self.assertNotIn("I couldn't read that from the runtime state.", body["assistant"]["content"])

    def test_cli_status_stays_ready_after_successful_openrouter_switch(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        self._configure_openrouter_runtime(runtime, make_default=True)
        with patch.object(
            runtime,
            "telegram_status",
            return_value={
                "ok": True,
                "enabled": True,
                "configured": True,
                "state": "running",
                "effective_state": "enabled_running",
                "next_action": "No action needed.",
            },
        ):
            ready_payload = runtime.ready_status()
        output = io.StringIO()
        with patch("agent.cli._load_ready_status_payload", return_value=(True, ready_payload)), patch(
            "agent.cli.get_telegram_runtime_state",
            return_value={"effective_state": "enabled_running", "next_action": "No action needed."},
        ), redirect_stdout(output):
            code = cli.main(["status"])

        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("runtime_mode: READY", text)
        self.assertIn("openrouter", text.lower())
        self.assertNotIn("LLM provider unavailable", text)

    def test_chat_reports_current_model_after_switch(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok)
        self.assertIn("ollama", body["assistant"]["content"].lower())
        self.assertIn("qwen2.5:7b-instruct", body["assistant"]["content"])

    def test_chat_reports_configured_and_effective_target_truthfully(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "ai21/jamba-large-1.7",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("openrouter:ai21/jamba-large-1.7")
        state = dict(runtime._health_monitor.store.empty_state())  # type: ignore[attr-defined]
        state["providers"] = {
            "ollama": {
                "status": "ok",
                "last_checked_at": 1_700_000_001,
            },
            "openrouter": {
                "status": "down",
                "last_error_kind": "http_503",
                "status_code": 503,
                "last_checked_at": 1_700_000_000,
            }
        }
        state["models"] = {
            "ollama:qwen3.5:4b": {
                "provider_id": "ollama",
                "status": "ok",
                "last_checked_at": 1_700_000_001,
            },
            "openrouter:ai21/jamba-large-1.7": {
                "provider_id": "openrouter",
                "status": "down",
                "last_error_kind": "http_503",
                "status_code": 503,
                "last_checked_at": 1_700_000_000,
            }
        }
        runtime._health_monitor.state = runtime._health_monitor.store.save(state)  # type: ignore[attr-defined]
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "what model are you using right now?"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok)
        self.assertIn("configured to use", body["assistant"]["content"].lower())
        self.assertIn("not responding right now", body["assistant"]["content"].lower())
        self.assertIn("openrouter:ai21/jamba-large-1.7", body["assistant"]["content"].lower())
        truth = runtime.runtime_truth_service().chat_target_truth()
        self.assertEqual("openrouter", truth["configured_provider"])
        self.assertEqual("openrouter:ai21/jamba-large-1.7", truth["configured_model"])
        self.assertEqual("openrouter", truth["effective_provider"])
        self.assertEqual("openrouter:ai21/jamba-large-1.7", truth["effective_model"])

    def test_chat_openrouter_provider_status_query_is_grounded(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "what model do we have set up for openrouter?"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok)
        self.assertEqual("provider_status", body["setup"]["type"])
        self.assertEqual("openrouter", body["setup"]["provider"])
        self.assertFalse(body["setup"]["configured"])
        self.assertEqual("provider_status", body["meta"]["route"])
        self.assertFalse(body["meta"]["generic_fallback_used"])
        self.assertIn("OpenRouter", body["assistant"]["content"])
        self.assertIn("API key", body["assistant"]["content"])
        self.assertNotIn("cannot access internal information", body["assistant"]["content"].lower())

    def test_chat_is_openrouter_configured_returns_structured_status(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._configure_openrouter_runtime(runtime, make_default=False)
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "is openrouter configured?"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok)
        self.assertEqual("provider_status", body["setup"]["type"])
        self.assertFalse(body["setup"]["configured"])
        self.assertEqual("openrouter", body["setup"]["provider"])
        self.assertIsNone(body["setup"]["model_id"])
        self.assertEqual("provider_status", body["meta"]["route"])
        self.assertFalse(body["meta"]["generic_fallback_used"])
        self.assertIn("OpenRouter", body["assistant"]["content"])
        self.assertIn("partly set up", body["assistant"]["content"].lower())
        self.assertIn("chat model ready", body["assistant"]["content"].lower())

    def test_runtime_truth_service_exposes_structured_provider_state(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")

        status = runtime.runtime_truth_service().provider_status("ollama")

        self.assertEqual("ollama", status["provider"])
        self.assertTrue(status["configured"])
        self.assertTrue(status["active"])
        self.assertEqual("ollama:qwen2.5:7b-instruct", status["model_id"])
        self.assertEqual("ollama:qwen2.5:7b-instruct", status["current_model_id"])

    def test_runtime_truth_service_provider_status_does_not_require_router_snapshot(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
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

        with patch.object(runtime, "_canonical_runtime_router_snapshot", side_effect=AssertionError("router snapshot should not be required")):
            status = runtime.runtime_truth_service().provider_status("ollama")

        self.assertEqual("ok", status["health_status"])
        self.assertTrue(status["configured"])

    def test_assistant_status_replies_prefer_live_ollama_truth_over_stale_safe_mode_health(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen2.5:3b-instruct",
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
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {
                    "status": "down",
                    "reason": "timeout while reaching Ollama",
                }
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "status": "ok",
                }
            },
        }

        with patch.object(
            runtime,
            "model_status",
            return_value={
                "ok": True,
                "current": {
                    "provider": "ollama",
                    "model_id": "ollama:qwen3.5:4b",
                    "default_provider": "ollama",
                    "default_model": "ollama:qwen3.5:4b",
                    "resolved_default_model": "ollama:qwen3.5:4b",
                },
                "active_provider_health": {
                    "status": "ok",
                    "last_checked_at": 123,
                },
                "active_model_health": {
                    "status": "ok",
                    "last_checked_at": 123,
                },
                "llm_availability": {
                    "available": True,
                    "reason": "ok",
                    "ollama": {
                        "native_ok": True,
                        "openai_compat_ok": True,
                    },
                },
            },
        ), patch.object(
            runtime,
            "_ollama_tags_models",
            return_value={
                "ok": True,
                "models": ["qwen3.5:4b", "qwen2.5:3b-instruct"],
            },
        ), patch.object(
            runtime,
            "safe_mode_target_status",
            return_value={
                "enabled": True,
                "configured_model": "ollama:qwen2.5:3b-instruct",
                "configured_provider": "ollama",
                "configured_valid": False,
                "explicit_override_model": None,
                "explicit_override_provider": None,
                "explicit_override_active": False,
                "effective_model": "ollama:qwen3.5:4b",
                "effective_provider": "ollama",
                "effective_local": True,
                "reason": "configured_pin_invalid",
                "message": "Safe mode pin ollama:qwen2.5:3b-instruct is unavailable. Falling back to ollama:qwen3.5:4b.",
            },
        ), patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            truth = runtime.runtime_truth_service()
            current = truth.current_chat_target_status()
            provider_status = truth.provider_status("ollama")
            target_truth = truth.chat_target_truth()
            model_reply = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:truth",
                    "thread_id": "api:truth:thread",
                },
            )
            provider_reply = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "ollama status"}],
                    "source_surface": "api",
                    "user_id": "api:truth",
                    "thread_id": "api:truth:thread",
                },
            )

        model_text = str((model_reply.get("assistant") or {}).get("content") or model_reply.get("message") or "")
        provider_text = str((provider_reply.get("assistant") or {}).get("content") or provider_reply.get("message") or "")
        model_meta = model_reply.get("meta") if isinstance(model_reply.get("meta"), dict) else {}
        provider_meta = provider_reply.get("meta") if isinstance(provider_reply.get("meta"), dict) else {}

        self.assertTrue(bool(current.get("ready", False)))
        self.assertEqual("ok", current.get("provider_health_status"))
        self.assertEqual("ok", provider_status.get("health_status"))
        self.assertIs(True, target_truth.get("safe_mode_pin_exists_live"))
        self.assertIn("healthy and ready", str(target_truth.get("qualification_reason") or "").lower())
        self.assertNotIn("safe mode pin", str(target_truth.get("qualification_reason") or "").lower())
        self.assertEqual("model_status", model_meta.get("route"))
        self.assertIn("ollama:qwen3.5:4b", model_text.lower())
        self.assertNotIn("not responding right now", model_text.lower())
        self.assertNotIn("safe mode pin", model_text.lower())
        self.assertEqual("provider_status", provider_meta.get("route"))
        self.assertIn("ollama is reachable", provider_text.lower())
        self.assertNotIn("not responding right now", provider_text.lower())

    def test_assistant_model_status_keeps_live_current_target_truth_when_safe_mode_pin_exists(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen2.5:3b-instruct",
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
        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            "ollama": {"status": "ok", "last_checked_at": 123}
        }
        runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
            "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "down", "last_checked_at": 123}
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch.object(
            runtime,
            "_ollama_tags_models",
            return_value={
                "ok": True,
                "models": ["qwen3.5:4b", "qwen2.5:3b-instruct"],
            },
        ), patch.object(
            runtime,
            "get_defaults",
            return_value={
                "default_provider": "ollama",
                "chat_model": "ollama:qwen3.5:4b",
                "default_model": "ollama:qwen3.5:4b",
                "resolved_default_model": "ollama:qwen3.5:4b",
                "embed_model": None,
                "last_chat_model": None,
                "routing_mode": "auto",
                "allow_remote_fallback": False,
            },
        ), patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            target_truth = runtime.runtime_truth_service().chat_target_truth()
            current = runtime.runtime_truth_service().current_chat_target_status()
            payload = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:safe-mode-current",
                    "thread_id": "api:safe-mode-current:thread",
                },
            )

        text = str((payload.get("assistant") or {}).get("content") or payload.get("message") or "")
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

        self.assertIs(True, target_truth.get("safe_mode_pin_exists_live"))
        self.assertEqual("ollama", current.get("provider"))
        self.assertEqual("ollama:qwen3.5:4b", current.get("model"))
        self.assertEqual("ok", current.get("provider_health_status"))
        self.assertEqual("down", current.get("health_status"))
        self.assertFalse(bool(current.get("ready", False)))
        self.assertEqual("model_status", meta.get("route"))
        self.assertIn("ollama:qwen3.5:4b", text.lower())
        self.assertIn("not healthy right now", text.lower())
        self.assertNotIn("safe mode pin", text.lower())
        self.assertNotIn("unavailable", text.lower())
        self.assertNotIn("not responding right now", text.lower())

    def test_assistant_model_status_uses_runtime_truth_without_llm_status_or_model_status(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="qwen2.5:3b-instruct",
            )
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:3b-instruct")

        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            "ollama": {"status": "ok", "last_checked_at": 123}
        }
        runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
            "ollama:qwen2.5:3b-instruct": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123}
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch.object(
            runtime,
            "_ollama_tags_models",
            return_value={
                "ok": True,
                "models": ["qwen2.5:3b-instruct", "qwen3.5:4b"],
            },
        ), patch.object(runtime, "model_status", side_effect=AssertionError("assistant path must not use model_status")), patch.object(
            runtime,
            "llm_status",
            side_effect=AssertionError("assistant path must not use llm_status"),
        ), patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            truth = runtime.runtime_truth_service()
            current = truth.current_chat_target_status()
            payload = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using now?"}],
                    "source_surface": "api",
                    "user_id": "api:model-live",
                    "thread_id": "api:model-live:thread",
                },
            )

        text = str((payload.get("assistant") or {}).get("content") or payload.get("message") or "")
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

        self.assertTrue(bool(current.get("ready", False)))
        self.assertEqual("ok", current.get("provider_health_status"))
        self.assertEqual("ok", current.get("health_status"))
        self.assertEqual("model_status", meta.get("route"))
        self.assertIn("currently using ollama:qwen2.5:3b-instruct", text.lower())
        self.assertNotIn("not ready right now", text.lower())
        self.assertNotIn("not responding right now", text.lower())

    def test_setup_explanation_uses_canonical_live_model_truth(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                safe_mode_enabled=True,
                safe_mode_chat_model="ollama:qwen2.5:3b-instruct",
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
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen3.5:4b")

        with patch.object(
            runtime,
            "model_status",
            return_value={
                "ok": True,
                "current": {
                    "provider": "ollama",
                    "model_id": "ollama:qwen3.5:4b",
                    "default_provider": "ollama",
                    "default_model": "ollama:qwen3.5:4b",
                    "resolved_default_model": "ollama:qwen3.5:4b",
                },
                "llm_availability": {
                    "available": True,
                    "reason": "ok",
                    "providers": {
                        "configured": ["ollama"],
                        "local_up": ["ollama"],
                        "local_down": [],
                        "local_unknown": [],
                    },
                    "ollama": {
                        "configured_base_url": "http://127.0.0.1:11434",
                        "native_base": "http://127.0.0.1:11434",
                        "openai_base": "http://127.0.0.1:11434/v1",
                        "native_ok": True,
                        "openai_compat_ok": True,
                        "last_error_kind": None,
                        "last_status_code": None,
                    },
                    "openrouter": {
                        "known": False,
                        "status": "unknown",
                        "last_error_kind": None,
                        "status_code": None,
                        "cooldown_until": None,
                    },
                },
            },
        ), patch.object(
            runtime,
            "safe_mode_target_status",
            return_value={
                "enabled": True,
                "configured_model": "ollama:qwen2.5:3b-instruct",
                "configured_provider": "ollama",
                "configured_valid": False,
                "explicit_override_model": None,
                "explicit_override_provider": None,
                "explicit_override_active": False,
                "effective_model": "ollama:qwen3.5:4b",
                "effective_provider": "ollama",
                "effective_local": True,
                "reason": "configured_pin_invalid",
                "message": "Safe mode pin ollama:qwen2.5:3b-instruct is unavailable. Falling back to ollama:qwen3.5:4b.",
            },
        ), patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            payload = _invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "Check setup and explain what's wrong"}],
                    "source_surface": "api",
                    "user_id": "api:setup-explain",
                    "thread_id": "api:setup-explain:thread",
                },
            )

        text = str((payload.get("assistant") or {}).get("content") or payload.get("message") or "")
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

        self.assertEqual("setup_flow", meta.get("route"))
        self.assertIn("setup looks okay right now", text.lower())
        self.assertIn("ollama:qwen3.5:4b", text.lower())
        self.assertIn("ollama is reachable", text.lower())
        self.assertNotIn("no chat model available right now", text.lower())
        self.assertNotIn("start ollama locally", text.lower())
        self.assertNotIn("install a local chat model", text.lower())
        self.assertNotIn("safe mode pin", text.lower())

    def test_runtime_truth_service_exposes_structured_runtime_status(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")

        status = runtime.runtime_truth_service().runtime_status()

        self.assertEqual("ready", status["scope"])
        self.assertIn("runtime_mode", status)
        self.assertIn("summary", status)
        self.assertEqual("ollama", status["provider"])
        self.assertEqual("ollama:qwen2.5:7b-instruct", status["model"])

    def test_runtime_truth_service_runtime_status_does_not_call_llm_status(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")

        with patch.object(runtime, "llm_status", side_effect=AssertionError("llm_status should not be required for runtime_status")):
            status = runtime.runtime_truth_service().runtime_status()

        self.assertEqual("ready", status["scope"])
        self.assertEqual("ollama", status["provider"])

    def test_chat_use_openrouter_reuses_stored_key_and_switches(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._configure_openrouter_runtime(runtime, make_default=False)
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True, "provider": "openrouter", "model": "openai/gpt-4o-mini"}),
        ), patch.object(
            runtime,
            "refresh_models",
            return_value=(True, {"ok": True, "models": []}),
        ):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "use openrouter"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok)
        self.assertEqual("setup_complete", body["setup"]["type"])
        self.assertEqual("openrouter", body["setup"]["provider"])
        self.assertEqual("setup_flow", body["meta"]["route"])
        self.assertFalse(body["meta"]["generic_fallback_used"])
        self.assertIn("OpenRouter", body["assistant"]["content"])
        self.assertEqual("openrouter", runtime.get_defaults()["default_provider"])
        self.assertEqual("openrouter:openai/gpt-4o-mini", runtime.get_defaults()["resolved_default_model"])

    def test_chat_providers_status_is_structured(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "what providers are configured"}],
                    "source_surface": "api",
                }
            )

        self.assertTrue(ok)
        self.assertEqual("providers_status", body["setup"]["type"])
        self.assertEqual("provider_status", body["meta"]["route"])
        self.assertFalse(body["meta"]["generic_fallback_used"])
        self.assertIn("providers", body["setup"])
        self.assertIn("ollama:qwen2.5:7b-instruct", body["assistant"]["content"])

    def test_chat_product_status_queries_do_not_call_generic_inference(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        for utterance in (
            "what model do we have set up for openrouter?",
            "is openrouter configured?",
        ):
            with self.subTest(utterance=utterance):
                with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch(
                    "agent.orchestrator.route_inference",
                    side_effect=AssertionError("generic inference should not run"),
                ):
                    ok, body = runtime.chat(
                        {
                            "messages": [{"role": "user", "content": utterance}],
                            "source_surface": "api",
                        }
                    )
                self.assertTrue(ok)
                self.assertEqual("provider_status", body["meta"]["route"])
                self.assertFalse(body["meta"]["generic_fallback_used"])

    def test_telegram_setup_uses_same_grounded_openrouter_prompt(self) -> None:
        backend_runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        orchestrator = _FakeOrchestrator("I don't have access to external systems.")
        context = _FakeContext(
            {
                "runtime": _ExplodingTelegramRuntime(),
                "orchestrator": orchestrator,
                "db": _FakeDB(),
                "log_path": os.path.join(self.tmpdir.name, "agent.log"),
                "audit_log": AuditLog(path=os.path.join(self.tmpdir.name, "audit.jsonl")),
                "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                "llm_fixit_store": LLMFixitWizardStore(path=os.path.join(self.tmpdir.name, "fixit.json")),
            }
        )
        update = _FakeUpdate(12345, "help me set up openrouter")

        with patch.object(backend_runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch(
            "telegram_adapter.bot._post_local_api_chat_json_async",
            side_effect=lambda payload: _invoke_chat_http(backend_runtime, payload),
        ):
            asyncio.run(_handle_message(update, context))

        reply = str(update.effective_message.replies[-1]["text"] or "")
        self.assertIn("OpenRouter", reply)
        self.assertIn("API key", reply)
        self.assertNotIn("install openrouter", reply.lower())
        self.assertEqual([], orchestrator.calls)

    def test_telegram_configure_openrouter_returns_prompt_without_timeout_reply(self) -> None:
        backend_runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        orchestrator = _FakeOrchestrator("unused")
        context = _FakeContext(
            {
                "runtime": _ExplodingTelegramRuntime(),
                "orchestrator": orchestrator,
                "db": _FakeDB(),
                "log_path": os.path.join(self.tmpdir.name, "agent.log"),
                "audit_log": AuditLog(path=os.path.join(self.tmpdir.name, "audit.jsonl")),
                "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                "llm_fixit_store": LLMFixitWizardStore(path=os.path.join(self.tmpdir.name, "fixit.json")),
            }
        )
        update = _FakeUpdate(12345, "configure openrouter")

        with patch(
            "telegram_adapter.bot._post_local_api_chat_json_async",
            side_effect=lambda payload: _invoke_chat_http(backend_runtime, payload),
        ):
            asyncio.run(_handle_message(update, context))

        reply = str(update.effective_message.replies[-1]["text"] or "")
        self.assertIn("OpenRouter", reply)
        self.assertIn("API key", reply)
        self.assertNotIn("timed out", reply.lower())
        self.assertEqual([], orchestrator.calls)

    def test_telegram_configure_openrouter_with_stored_key_returns_reuse_prompt_without_timeout(self) -> None:
        backend_runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        backend_runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        backend_runtime.set_default_chat_model("ollama:qwen3.5:4b")
        ok, _secret = backend_runtime.set_provider_secret("openrouter", {"api_key": "sk-or-v1-testsecret1234567890"})
        self.assertTrue(ok)
        orchestrator = _FakeOrchestrator("unused")
        context = _FakeContext(
            {
                "runtime": _ExplodingTelegramRuntime(),
                "orchestrator": orchestrator,
                "db": _FakeDB(),
                "log_path": os.path.join(self.tmpdir.name, "agent.log"),
                "audit_log": AuditLog(path=os.path.join(self.tmpdir.name, "audit.jsonl")),
                "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                "llm_fixit_store": LLMFixitWizardStore(path=os.path.join(self.tmpdir.name, "fixit.json")),
            }
        )
        update = _FakeUpdate(12345, "configure openrouter")

        with patch.object(
            backend_runtime,
            "configure_openrouter",
            side_effect=AssertionError("telegram setup init should not run provider test immediately"),
        ), patch(
            "telegram_adapter.bot._post_local_api_chat_json_async",
            side_effect=lambda payload: _invoke_chat_http(backend_runtime, payload),
        ):
            asyncio.run(_handle_message(update, context))

        reply = str(update.effective_message.replies[-1]["text"] or "")
        self.assertIn("OpenRouter", reply)
        self.assertIn("stored", reply.lower())
        self.assertNotIn("timed out", reply.lower())
        self.assertEqual([], orchestrator.calls)

    def test_telegram_configure_openrouter_with_stored_key_then_yes_completes_without_timeout(self) -> None:
        backend_runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        backend_runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        backend_runtime.set_default_chat_model("ollama:qwen3.5:4b")
        ok, _secret = backend_runtime.set_provider_secret("openrouter", {"api_key": "sk-or-v1-testsecret1234567890"})
        self.assertTrue(ok)
        orchestrator = _FakeOrchestrator("unused")
        context = _FakeContext(
            {
                "runtime": _ExplodingTelegramRuntime(),
                "orchestrator": orchestrator,
                "db": _FakeDB(),
                "log_path": os.path.join(self.tmpdir.name, "agent.log"),
                "audit_log": AuditLog(path=os.path.join(self.tmpdir.name, "audit.jsonl")),
                "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                "llm_fixit_store": LLMFixitWizardStore(path=os.path.join(self.tmpdir.name, "fixit.json")),
            }
        )
        first = _FakeUpdate(12345, "configure openrouter")
        second = _FakeUpdate(12345, "yes")
        with patch.object(
            backend_runtime,
            "test_provider",
            return_value=(
                True,
                {
                    "ok": True,
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                },
            ),
        ), patch.object(
            backend_runtime,
            "refresh_models",
            side_effect=AssertionError("telegram setup confirmation should defer full model refresh"),
        ), patch(
            "telegram_adapter.bot._post_local_api_chat_json_async",
            side_effect=lambda payload: _invoke_chat_http(backend_runtime, payload),
        ):
            asyncio.run(_handle_message(first, context))
            asyncio.run(_handle_message(second, context))

        first_reply = str(first.effective_message.replies[-1]["text"] or "")
        second_reply = str(second.effective_message.replies[-1]["text"] or "")
        self.assertIn("stored", first_reply.lower())
        self.assertIn("OpenRouter", second_reply)
        self.assertNotIn("timed out", second_reply.lower())
        self.assertEqual([], orchestrator.calls)

    def test_telegram_openrouter_status_uses_same_grounded_interceptor(self) -> None:
        backend_runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        backend_runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        backend_runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        orchestrator = _FakeOrchestrator("I don't have real-time access to external systems.")
        context = _FakeContext(
            {
                "runtime": _ExplodingTelegramRuntime(),
                "orchestrator": orchestrator,
                "db": _FakeDB(),
                "log_path": os.path.join(self.tmpdir.name, "agent.log"),
                "audit_log": AuditLog(path=os.path.join(self.tmpdir.name, "audit.jsonl")),
                "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                "llm_fixit_store": LLMFixitWizardStore(path=os.path.join(self.tmpdir.name, "fixit.json")),
            }
        )
        update = _FakeUpdate(12345, "what model do we have set up for openrouter?")

        with patch.object(backend_runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch(
            "telegram_adapter.bot._post_local_api_chat_json_async",
            side_effect=lambda payload: _invoke_chat_http(backend_runtime, payload),
        ):
            asyncio.run(_handle_message(update, context))

        reply = str(update.effective_message.replies[-1]["text"] or "")
        self.assertIn("OpenRouter", reply)
        self.assertIn("API key", reply)
        self.assertNotIn("cannot access internal information", reply.lower())
        self.assertEqual([], orchestrator.calls)

    def test_telegram_exact_openrouter_status_queries_do_not_hit_generic_chat(self) -> None:
        backend_runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        orchestrator = _FakeOrchestrator("I don't have access to external systems.")
        context = _FakeContext(
            {
                "runtime": _ExplodingTelegramRuntime(),
                "orchestrator": orchestrator,
                "db": _FakeDB(),
                "log_path": os.path.join(self.tmpdir.name, "agent.log"),
                "audit_log": AuditLog(path=os.path.join(self.tmpdir.name, "audit.jsonl")),
                "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                "llm_fixit_store": LLMFixitWizardStore(path=os.path.join(self.tmpdir.name, "fixit.json")),
            }
        )
        for utterance in (
            "what model do we have set up for openrouter?",
            "is openrouter configured?",
        ):
            with self.subTest(utterance=utterance):
                update = _FakeUpdate(12345, utterance)
                with patch.object(backend_runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch(
                    "telegram_adapter.bot._post_local_api_chat_json_async",
                    side_effect=lambda payload: _invoke_chat_http(backend_runtime, payload),
                ):
                    asyncio.run(_handle_message(update, context))
                reply = str(update.effective_message.replies[-1]["text"] or "")
                self.assertIn("OpenRouter", reply)
                self.assertNotIn("real-time access", reply.lower())
                self.assertNotIn("external systems", reply.lower())
        self.assertEqual([], orchestrator.calls)

    def test_telegram_live_runtime_queries_are_grounded_from_api_backed_runtime_truth(self) -> None:
        backend_runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        backend_runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        backend_runtime.set_default_chat_model("ollama:qwen3.5:4b")
        backend_runtime._record_authoritative_provider_success("ollama", "ollama:qwen3.5:4b")  # type: ignore[attr-defined]
        log_path = os.path.join(self.tmpdir.name, "agent.log")
        orchestrator = _FakeOrchestrator("should not be used")
        context = _FakeContext(
            {
                "runtime": _ExplodingTelegramRuntime(),
                "orchestrator": orchestrator,
                "db": _FakeDB(),
                "log_path": log_path,
                "audit_log": AuditLog(path=os.path.join(self.tmpdir.name, "audit.jsonl")),
                "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                "llm_fixit_store": LLMFixitWizardStore(path=os.path.join(self.tmpdir.name, "fixit.json")),
                "runtime_version": backend_runtime.version,
                "runtime_git_commit": backend_runtime.git_commit,
            }
        )
        utterances = (
            ("hello, is the openrouter setup?", "provider_status"),
            ("openrouter health", "provider_status"),
            ("what model are you using?", "model_status"),
            ("which model are you using?", "model_status"),
            ("check what model is currently enabled please", "model_status"),
            ("is openrouter configured?", "provider_status"),
            ("can you tell if everything is working with the agent?", "runtime_status"),
            ("is the agent healthy?", "runtime_status"),
            ("is the agent healthy right now?", "runtime_status"),
            ("what execution mode does Telegram use?", "governance_status"),
        )

        with patch.object(backend_runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            backend_runtime,
            "telegram_status",
            return_value={
                "ok": True,
                "enabled": True,
                "configured": True,
                "state": "running",
                "effective_state": "enabled_running",
                "next_action": "No action needed.",
            },
        ), patch(
            "telegram_adapter.bot._post_local_api_chat_json_async",
            side_effect=lambda payload: _invoke_chat_http(backend_runtime, payload),
        ):
            for utterance, _route in utterances:
                update = _FakeUpdate(12345, utterance)
                asyncio.run(_handle_message(update, context))
                reply = str(update.effective_message.replies[-1]["text"] or "")
                if utterance in {"what model are you using?", "which model are you using?", "check what model is currently enabled please"}:
                    self.assertIn("ollama:qwen3.5:4b", reply)
                    self.assertIn("Ollama", reply)
                elif utterance in {
                    "can you tell if everything is working with the agent?",
                    "is the agent healthy?",
                    "is the agent healthy right now?",
                }:
                    self.assertIn("ready", reply.lower())
                    self.assertIn("ollama:qwen3.5:4b", reply)
                    self.assertNotIn("scheduler", reply.lower())
                    self.assertNotIn("daily brief", reply.lower())
                    self.assertNotIn("db", reply.lower())
                elif utterance == "what execution mode does Telegram use?":
                    self.assertIn("Mode: Controlled Mode.", reply)
                else:
                    self.assertIn("OpenRouter", reply)
                    self.assertNotIn("I couldn't read that from the runtime state.", reply)
                self.assertNotIn("Which of these is your goal", reply)

        self.assertEqual([], orchestrator.calls)
        with open(log_path, "r", encoding="utf-8") as handle:
            log_rows = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
        telegram_rows = [
            row.get("payload")
            for row in log_rows
            if str(row.get("type")) == "telegram_message" and isinstance(row.get("payload"), dict)
        ]
        self.assertEqual(
            [
                "provider_status",
                "provider_status",
                "model_status",
                "model_status",
                "model_status",
                "provider_status",
                "runtime_status",
                "runtime_status",
                "runtime_status",
                "model_policy_status",
            ],
            [str(row.get("selected_route") or "") for row in telegram_rows[-10:]],
        )
        self.assertTrue(all(str(row.get("handler_name") or "") == "api_chat_proxy" for row in telegram_rows[-10:]))
        governance_row = telegram_rows[-1]
        self.assertIn("proxy_elapsed_ms", governance_row)
        self.assertIn("proxy_timeout_seconds", governance_row)


if __name__ == "__main__":
    unittest.main()
