from __future__ import annotations

import copy
import io
import json
import os
import threading
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.llm.types import LLMError, Response, Usage
from agent.modelops.discovery import ModelInfo


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

    def test_chat_uses_chat_model_not_embed_model(self) -> None:
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
        models["ollama:nomic-embed-text:latest"] = {
            "provider": "ollama",
            "model": "nomic-embed-text:latest",
            "capabilities": ["embedding"],
            "enabled": True,
            "available": True,
            "quality_rank": 1,
            "cost_rank": 1,
            "default_for": ["embedding"],
            "pricing": {
                "input_per_million_tokens": None,
                "output_per_million_tokens": None,
            },
            "max_context_tokens": 8192,
        }
        document["models"] = models
        runtime._save_registry_document(document)

        ok_defaults, _defaults = runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen2.5:3b-instruct",
                "embed_model": "ollama:nomic-embed-text:latest",
            }
        )
        self.assertTrue(ok_defaults)

        captured: dict[str, object] = {}

        def _fake_chat(messages, **kwargs):  # type: ignore[no-untyped-def]
            _ = messages
            captured.update(kwargs)
            return {
                "ok": True,
                "text": "ok",
                "provider": "ollama",
                "model": str(kwargs.get("model_override") or ""),
                "fallback_used": False,
                "attempts": [],
                "duration_ms": 1,
                "error_class": None,
            }

        with patch.object(runtime._router, "chat", side_effect=_fake_chat):  # type: ignore[attr-defined]
            ok_chat, body = runtime.chat({"messages": [{"role": "user", "content": "hello"}]})

        self.assertTrue(ok_chat)
        self.assertTrue(body["ok"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", str(captured.get("model_override") or ""))

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
        self.assertTrue(providers_payload["providers"])
        first_provider = providers_payload["providers"][0]
        self.assertIn("health", first_provider)
        self.assertIn("status", first_provider["health"])

        models_payload = runtime.models()
        self.assertTrue(models_payload["models"])
        first_model = models_payload["models"][0]
        self.assertIn("health", first_model)
        self.assertIn("status", first_model["health"])

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
        provider_row = providers[0] if providers else {}
        model_row = models[0] if models else {}
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

    def test_model_scout_endpoints(self) -> None:
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

        status_handler = _HandlerForTest(runtime, "/model_scout/status")
        status_handler.do_GET()
        self.assertEqual(200, status_handler.status_code)
        status_payload = json.loads(status_handler.body.decode("utf-8"))
        self.assertTrue(status_payload["ok"])
        self.assertIn("status", status_payload)

        suggestions_handler = _HandlerForTest(runtime, "/model_scout/suggestions")
        suggestions_handler.do_GET()
        self.assertEqual(200, suggestions_handler.status_code)
        suggestions_payload = json.loads(suggestions_handler.body.decode("utf-8"))
        self.assertTrue(suggestions_payload["ok"])
        self.assertIn("suggestions", suggestions_payload)

        with patch.object(runtime, "run_model_scout", return_value=(True, {"ok": True, "suggestions": []})):
            run_handler = _HandlerForTest(runtime, "/model_scout/run", {})
            run_handler.do_POST()
            self.assertEqual(200, run_handler.status_code)
            run_payload = json.loads(run_handler.body.decode("utf-8"))
            self.assertTrue(run_payload["ok"])

            llm_run_handler = _HandlerForTest(runtime, "/llm/scout/run", {})
            llm_run_handler.do_POST()
            self.assertEqual(200, llm_run_handler.status_code)
            llm_run_payload = json.loads(llm_run_handler.body.decode("utf-8"))
            self.assertTrue(llm_run_payload["ok"])

        with patch.object(
            runtime,
            "dismiss_model_scout_suggestion",
            return_value=(True, {"ok": True, "status": "dismissed", "id": "local:abc"}),
        ):
            dismiss_handler = _HandlerForTest(runtime, "/model_scout/suggestions/local%3Aabc/dismiss", {})
            dismiss_handler.do_POST()
            self.assertEqual(200, dismiss_handler.status_code)
            dismiss_payload = json.loads(dismiss_handler.body.decode("utf-8"))
            self.assertTrue(dismiss_payload["ok"])
            self.assertEqual("dismissed", dismiss_payload["status"])

        with patch.object(
            runtime,
            "mark_model_scout_installed",
            return_value=(True, {"ok": True, "status": "installed", "id": "local:abc"}),
        ):
            install_handler = _HandlerForTest(runtime, "/model_scout/suggestions/local%3Aabc/mark_installed", {})
            install_handler.do_POST()
            self.assertEqual(200, install_handler.status_code)
            install_payload = json.loads(install_handler.body.decode("utf-8"))
            self.assertTrue(install_payload["ok"])
            self.assertEqual("installed", install_payload["status"])

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
                self.assertIn(payload.get("phase"), {"warming", "listening"})
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
                self.assertEqual([], payload_ready.get("warmup_remaining"))
            finally:
                warmup_gate.set()
                runtime.close()

    def test_deferred_warmup_skips_native_and_router_prebind(self) -> None:
        call_order: list[str] = []
        original_native = AgentRuntime._ensure_native_packs_registered
        original_reload = AgentRuntime._reload_router

        def _wrapped_native(runtime_obj: AgentRuntime) -> None:
            call_order.append("native_packs")
            return original_native(runtime_obj)

        def _wrapped_reload(runtime_obj: AgentRuntime) -> None:
            call_order.append("router_reload")
            return original_reload(runtime_obj)

        with patch.object(AgentRuntime, "_ensure_native_packs_registered", _wrapped_native), patch.object(
            AgentRuntime, "_reload_router", _wrapped_reload
        ):
            runtime = AgentRuntime(_config(self.registry_path, self.db_path), defer_bootstrap_warmup=True)
            try:
                self.assertEqual([], call_order)
                self.assertEqual(
                    ["native_packs", "router_reload", "model_catalog_refresh"],
                    runtime._warmup_remaining_snapshot(),
                )
                runtime.set_listening("127.0.0.1", 8765)
                runtime.mark_server_listening()
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    if runtime.startup_phase in {"ready", "degraded"}:
                        break
                    time.sleep(0.02)
                self.assertGreaterEqual(len(call_order), 2)
                self.assertEqual("native_packs", call_order[0])
                self.assertEqual("router_reload", call_order[1])
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

        sources_get = _HandlerForTest(runtime, "/model_scout/sources")
        sources_get.do_GET()
        self.assertEqual(200, sources_get.status_code)
        sources_payload = json.loads(sources_get.body.decode("utf-8"))
        self.assertTrue(sources_payload["ok"])
        self.assertIn("sources", sources_payload)

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

    def test_chat_thread_integrity_drift_returns_clarification_contract(self) -> None:
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
        with patch.object(runtime, "chat", side_effect=AssertionError("chat should not be called")):
            handler.do_POST()

        response = handler.response_payload
        self.assertEqual(200, handler.status_code)
        self.assertEqual(True, response.get("ok"))
        self.assertEqual("needs_clarification", response.get("error_kind"))
        envelope = response.get("envelope")
        self.assertTrue(isinstance(envelope, dict))
        self.assertTrue(isinstance((envelope or {}).get("clarification"), dict))
        thread_integrity = (envelope or {}).get("thread_integrity")
        self.assertTrue(isinstance(thread_integrity, dict))
        self.assertIn(
            str((thread_integrity or {}).get("reason") or ""),
            {"multi_intent", "topic_shift"},
        )

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


if __name__ == "__main__":
    unittest.main()
