from __future__ import annotations

from dataclasses import replace
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError

from agent.api_server import AgentRuntime
from agent.search.safe_web_search import SafeWebSearchClient, SafeWebSearchConfig, redact_search_query
from agent.setup_chat_flow import classify_runtime_chat_route
from tests.test_api_packs_endpoints import _HandlerForTest, _config
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeOpener:
    def __init__(self, payload: dict[str, object] | None = None, *, exc: Exception | None = None) -> None:
        self.payload = payload or {"results": []}
        self.exc = exc
        self.opened_urls: list[str] = []
        self.timeouts: list[float] = []

    def open(self, request, timeout: float = 5.0):  # noqa: ANN001
        self.opened_urls.append(getattr(request, "full_url", str(request)))
        self.timeouts.append(timeout)
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.payload)


class TestSafeWebSearch(unittest.TestCase):
    def test_disabled_search_refuses_cleanly(self) -> None:
        client = SafeWebSearchClient(SafeWebSearchConfig(enabled=False))

        result = client.search("current news")

        self.assertFalse(result.ok)
        self.assertEqual("search_disabled", result.error_kind)
        self.assertIn("disabled", result.message.lower())
        self.assertFalse(result.safety["page_fetching"])

    def test_missing_endpoint_refuses_cleanly(self) -> None:
        client = SafeWebSearchClient(SafeWebSearchConfig(enabled=True, searxng_base_url=None))

        result = client.search("current news")

        self.assertFalse(result.ok)
        self.assertEqual("endpoint_missing", result.error_kind)
        self.assertIn("SEARXNG_BASE_URL", result.message)

    def test_mocked_searxng_json_results_normalize_as_untrusted_metadata(self) -> None:
        opener = _FakeOpener(
            {
                "results": [
                    {
                        "title": "Example result",
                        "url": "https://example.com/page",
                        "content": "A short snippet.",
                        "engine": "test-engine",
                    },
                    {
                        "title": "Bad URL skipped",
                        "url": "javascript:alert(1)",
                        "content": "unsafe",
                    },
                ]
            }
        )
        client = SafeWebSearchClient(
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888", max_results=5),
            opener=opener,
        )

        result = client.search("search term")

        self.assertTrue(result.ok)
        self.assertEqual(1, len(result.results))
        self.assertEqual("Example result", result.results[0].title)
        self.assertEqual("https://example.com/page", result.results[0].url)
        self.assertTrue(result.results[0].untrusted)
        self.assertIn("/search?", opener.opened_urls[0])
        self.assertEqual(1, len(opener.opened_urls), "safe search should only contact the SearXNG JSON endpoint")

    def test_timeout_or_transport_error_returns_safe_message(self) -> None:
        opener = _FakeOpener(exc=URLError("offline"))
        client = SafeWebSearchClient(
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=opener,
        )

        result = client.search("anything")

        self.assertFalse(result.ok)
        self.assertEqual("search_error", result.error_kind)
        self.assertIn("failed safely", result.message.lower())

    def test_redacts_sensitive_query_fragments(self) -> None:
        redacted = redact_search_query("/home/c/Takeout/history.json token=abcdef1234567890abcdef123456")

        self.assertIn("[REDACTED_PATH]", redacted)
        self.assertIn("token=[REDACTED]", redacted)
        self.assertNotIn("/home/c/Takeout", redacted)


class TestSafeWebSearchRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.skills_path = str(Path(__file__).resolve().parents[1] / "skills")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self, *, search_enabled: bool = True) -> AgentRuntime:
        config = replace(
            _config(self.registry_path, self.db_path, self.skills_path),
            search_enabled=search_enabled,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
            search_timeout_seconds=1.0,
            search_max_results=3,
        )
        return AgentRuntime(config)

    def _install_fake_search_client(self, runtime: AgentRuntime, title: str = "Metadata result") -> _FakeOpener:
        opener = _FakeOpener(
            {
                "results": [
                    {
                        "title": title,
                        "url": "https://example.com/result",
                        "content": "Metadata-only snippet.",
                        "engine": "test-engine",
                    }
                ]
            }
        )
        runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=opener,
        )
        return opener

    def _chat(self, runtime: AgentRuntime, message: str, *, session_id: str = "safe-search-test") -> tuple[dict, str, dict]:
        handler = _MemoryHandlerForTest(
            runtime,
            "/chat",
            {
                "messages": [{"role": "user", "content": message}],
                "session_id": session_id,
                "thread_id": f"{session_id}-thread",
                "source_surface": "webui",
                "purpose": "chat",
                "task_type": "chat",
            },
        )
        handler.do_POST()
        body = json.loads(handler.body.decode("utf-8"))
        text = _assistant_text(body)
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertEqual(200, handler.status_code)
        return body, text, meta

    def test_search_status_and_query_endpoints(self) -> None:
        runtime = self._runtime()
        runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=_FakeOpener({"results": [{"title": "One", "url": "https://example.com", "content": "Snippet"}]}),
        )

        status_handler = _HandlerForTest(runtime, "/search/status")
        status_handler.do_GET()
        status = json.loads(status_handler.body.decode("utf-8"))
        self.assertTrue(status["available"])
        self.assertIsNone(status["reason"])
        self.assertTrue(status["safety"]["metadata_only"])

        query_handler = _HandlerForTest(runtime, "/search/query", {"query": "example", "max_results": 1})
        query_handler.do_POST()
        payload = json.loads(query_handler.body.decode("utf-8"))
        self.assertEqual(200, query_handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["results"][0]["untrusted"])
        client = runtime._safe_web_search_client  # noqa: SLF001
        self.assertIsInstance(client, SafeWebSearchClient)
        opener = client._opener  # noqa: SLF001
        self.assertEqual(2, len(opener.opened_urls))
        self.assertTrue(all(url.startswith("http://127.0.0.1:8888/search?") for url in opener.opened_urls))

    def test_search_status_uses_same_json_search_path_as_query(self) -> None:
        opener = _FakeOpener({"results": [{"title": "One", "url": "https://example.com", "content": "Snippet"}]})
        client = SafeWebSearchClient(
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888", timeout_seconds=4.0),
            opener=opener,
        )

        status = client.status()
        query = client.search("personal agent test", max_results=1)

        self.assertTrue(status["available"])
        self.assertIsNone(status["reason"])
        self.assertTrue(query.ok)
        self.assertEqual(2, len(opener.opened_urls))
        self.assertTrue(all(url.startswith("http://127.0.0.1:8888/search?") for url in opener.opened_urls))
        self.assertEqual([4.0, 4.0], opener.timeouts)

    def test_chat_route_uses_native_search_without_pack_acquisition(self) -> None:
        runtime = self._runtime()
        opener = _FakeOpener(
            {"results": [{"title": "Native search result", "url": "https://example.com/r", "content": "Metadata only."}]}
        )
        runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=opener,
        )
        handler = _MemoryHandlerForTest(
            runtime,
            "/chat",
            {
                "messages": [{"role": "user", "content": "search the web for native search tests"}],
                "session_id": "safe-search",
                "thread_id": "safe-search-thread",
                "source_surface": "webui",
                "purpose": "chat",
                "task_type": "chat",
            },
        )

        handler.do_POST()
        body = json.loads(handler.body.decode("utf-8"))
        text = _assistant_text(body)
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual(200, handler.status_code)
        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("Native search result", text)
        self.assertIn("untrusted", text.lower())
        self.assertIn("did not open pages", text.lower())
        self.assertNotIn("pack", " ".join(str(item) for item in meta.get("used_tools", [])))
        self.assertEqual(1, len(opener.opened_urls))

    def test_post_setup_online_entity_followup_uses_native_search(self) -> None:
        runtime = self._runtime(search_enabled=False)
        setup_handler = _MemoryHandlerForTest(
            runtime,
            "/chat",
            {
                "messages": [{"role": "user", "content": "can you search for something for me online"}],
                "session_id": "safe-search-handoff",
                "thread_id": "safe-search-handoff-thread",
                "source_surface": "webui",
                "purpose": "chat",
                "task_type": "chat",
            },
        )
        setup_handler.do_POST()
        setup_body = json.loads(setup_handler.body.decode("utf-8"))
        setup_meta = setup_body.get("meta") if isinstance(setup_body.get("meta"), dict) else {}
        setup_text = _assistant_text(setup_body)

        self.assertEqual(200, setup_handler.status_code)
        self.assertIn("managed_local_service_setup_preview", setup_meta.get("used_tools", []))
        self.assertIn("Say yes to continue", setup_text)

        runtime.config = replace(
            runtime.config,
            search_enabled=True,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888",
        )
        opener = _FakeOpener(
            {
                "results": [
                    {
                        "title": "Kwite - YouTube",
                        "url": "https://www.youtube.com/@Kwite",
                        "content": "Kwite is a YouTube channel with commentary videos.",
                        "engine": "test-engine",
                    }
                ]
            }
        )
        runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
            SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888"),
            opener=opener,
        )
        followup_handler = _MemoryHandlerForTest(
            runtime,
            "/chat",
            {
                "messages": [{"role": "user", "content": "can you tell me about the youtube channel Kwite?"}],
                "session_id": "safe-search-handoff",
                "thread_id": "safe-search-handoff-thread",
                "source_surface": "webui",
                "purpose": "chat",
                "task_type": "chat",
            },
        )

        followup_handler.do_POST()
        body = json.loads(followup_handler.body.decode("utf-8"))
        text = _assistant_text(body)
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual(200, followup_handler.status_code)
        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("metadata-only web results", text.lower())
        self.assertIn("Kwite - YouTube", text)
        self.assertIn("untrusted", text.lower())
        self.assertIn("did not open pages", text.lower())
        self.assertNotIn("127.0.0.1:8888", text)
        self.assertGreaterEqual(len(opener.opened_urls), 1)
        self.assertIn("Kwite", opener.opened_urls[-1])

    def test_search_fallback_routes_online_and_project_entities(self) -> None:
        examples = (
            ("can you tell me about the youtube channel Kwite?", "Kwite result"),
            ("what is dots.tts?", "dots.tts result"),
            ("is pi.dev useful for my assistant project?", "pi.dev result"),
        )
        for message, title in examples:
            with self.subTest(message=message):
                runtime = self._runtime()
                opener = self._install_fake_search_client(runtime, title)

                _body, text, meta = self._chat(runtime, message, session_id=f"search-fallback-{title}")

                self.assertEqual("action_tool", meta.get("route"))
                self.assertIn("safe_web_search", meta.get("used_tools", []))
                self.assertIn(title, text)
                self.assertIn("metadata-only web results", text.lower())
                self.assertIn("untrusted", text.lower())
                self.assertIn("did not open pages", text.lower())
                self.assertEqual(1, len(opener.opened_urls))

    def test_search_fallback_does_not_force_search_for_timeless_or_text_transform(self) -> None:
        examples = (
            "explain why the sky is blue in two sentences",
            "rewrite this paragraph: the quick brown fox jumps over the lazy dog",
        )
        for message in examples:
            with self.subTest(message=message):
                decision = classify_runtime_chat_route(message)

                self.assertNotEqual("safe_web_search", decision.get("kind"))

    def test_search_fallback_honors_do_not_search(self) -> None:
        runtime = self._runtime()
        opener = self._install_fake_search_client(runtime, "Kwite result")

        _body, text, meta = self._chat(runtime, "do not search, what do you know about Kwite?", session_id="no-search-kwite")

        self.assertEqual("action_tool", meta.get("route"))
        self.assertNotIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("will not search", text.lower())
        self.assertIn("limited or outdated", text.lower())
        self.assertEqual(0, len(opener.opened_urls))

    def test_search_fallback_disabled_offers_plan_mode_setup(self) -> None:
        runtime = self._runtime(search_enabled=False)

        _body, text, meta = self._chat(runtime, "what is dots.tts?", session_id="disabled-search-dots")

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("managed_local_service_setup_preview", meta.get("used_tools", []))
        self.assertIn("Web search is not set up", text)
        self.assertIn("Plan Mode confirmation", text)
