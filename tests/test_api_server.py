from __future__ import annotations

import json
import os
import tempfile
import unittest

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
