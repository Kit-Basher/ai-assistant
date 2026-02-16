from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config


def _config(registry_path: str, db_path: str) -> Config:
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
        llm_registry_path=registry_path,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
    )


class TestAPIServerRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

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

        original_chat = runtime._router.chat  # type: ignore[attr-defined]
        runtime._router.chat = lambda *args, **kwargs: {  # type: ignore[assignment]
            "ok": True,
            "provider": "acme",
            "model": "chat-model",
            "duration_ms": 1,
            "text": "PONG",
            "fallback_used": False,
            "attempts": [],
            "error_class": None,
        }
        try:
            ok, tested = runtime.test_provider("acme", {"model": "acme:chat"})
        finally:
            runtime._router.chat = original_chat  # type: ignore[assignment]

        self.assertTrue(ok)
        self.assertTrue(tested["ok"])
        self.assertEqual("acme", tested["provider"])

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

        runtime._http_get_json = lambda *_args, **_kwargs: {  # type: ignore[assignment]
            "data": [
                {"id": "llama3.2"},
                {"id": "nomic-embed-text"},
            ]
        }

        ok, _response = runtime.refresh_models()
        self.assertTrue(ok)

        embed_model = runtime.registry_document["models"]["ollama:nomic-embed-text"]
        self.assertEqual(["embedding"], embed_model["capabilities"])

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


if __name__ == "__main__":
    unittest.main()
