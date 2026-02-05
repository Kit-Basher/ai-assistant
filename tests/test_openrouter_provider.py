import json
import os
import unittest
from unittest.mock import patch

from agent.llm.providers.openrouter_provider import OpenRouterProvider
from agent.llm.router import LLMNarrationRouter


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestOpenRouterProvider(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_openrouter_unavailable_without_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "OPENROUTER_MODEL": "openai/gpt-4o-mini",
            },
            clear=False,
        ):
            router = LLMNarrationRouter()
            self.assertFalse(router._available_openrouter())

    def test_openrouter_available_with_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "key",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "OPENROUTER_MODEL": "openai/gpt-4o-mini",
            },
            clear=False,
        ):
            router = LLMNarrationRouter()
            self.assertTrue(router._available_openrouter())

    def test_headers_and_payload(self) -> None:
        provider = OpenRouterProvider(
            api_key="sk-test",
            model="openai/gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
            site_url="https://example.com",
            app_name="PersonalAgent",
        )

        captured = {}

        def _fake_urlopen(request, timeout=15):
            header_sources = [getattr(request, "headers", {}), getattr(request, "unredirected_hdrs", {})]
            normalized = {}
            for source in header_sources:
                for key, value in source.items():
                    normalized[str(key).lower()] = value
            captured["headers"] = normalized
            captured["data"] = request.data
            return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        with patch("urllib.request.urlopen", _fake_urlopen):
            result = provider.summarize(
                "storage_report",
                {"temperature": 0.4, "max_tokens": 120, "model": "openai/gpt-4o-mini"},
                timeout_s=5,
            )

        self.assertEqual(result.text, "ok")
        headers = captured["headers"]
        self.assertEqual(headers.get("authorization"), "Bearer sk-test")
        self.assertEqual(headers.get("http-referer"), "https://example.com")
        self.assertEqual(headers.get("x-title"), "PersonalAgent")

        body = json.loads(captured["data"].decode("utf-8"))
        self.assertEqual(body["model"], "openai/gpt-4o-mini")
        self.assertEqual(body["temperature"], 0.4)
        self.assertEqual(body["max_tokens"], 120)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["role"], "user")

    def test_router_prefers_local_unless_upgrade(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OLLAMA_HOST": "http://127.0.0.1:11434",
                "OLLAMA_MODEL": "llama3",
                "OPENAI_API_KEY": "key",
                "OPENAI_MODEL": "gpt-4.1-mini",
                "OPENROUTER_API_KEY": "key",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "OPENROUTER_MODEL": "openai/gpt-4o-mini",
            },
            clear=False,
        ):
            router = LLMNarrationRouter()
            self.assertEqual(router._provider_order({}), ["ollama", "openai"])
            self.assertEqual(
                router._provider_order({"upgrade": True}),
                ["openrouter", "openai", "ollama"],
            )


if __name__ == "__main__":
    unittest.main()
