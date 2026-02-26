from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.llm.types import LLMError, Response, Usage


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
        self.assertEqual("upstream_down", response["error_kind"])
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
        self.assertFalse(updated["allow_remote_fallback"])

        current = runtime.get_defaults()
        self.assertEqual("prefer_local_lowest_cost_capable", current["routing_mode"])
        self.assertEqual("ollama", current["default_provider"])
        self.assertEqual("ollama:llama3", current["default_model"])
        self.assertFalse(current["allow_remote_fallback"])
        self.assertEqual("prefer_local_lowest_cost_capable", runtime._router.policy.mode)

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


if __name__ == "__main__":
    unittest.main()
