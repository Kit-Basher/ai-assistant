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
from agent.policy import build_mutator_plan
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

    @staticmethod
    def _managed_services_status(*, enabled: bool = False, configured: bool = False, reachable: bool = False) -> dict[str, object]:
        return {
            "ok": True,
            "read_only": True,
            "podman_available": True,
            "podman_rootless": True,
            "docker_available": False,
            "services": [
                {
                    "service_id": "searxng",
                    "display_name": "SearXNG",
                    "enabled": enabled,
                    "configured": configured,
                    "reachable": reachable,
                    "url": "http://127.0.0.1:8888",
                    "podman_available": True,
                    "podman_rootless": True,
                }
            ],
        }

    @staticmethod
    def _managed_setup_plan(_payload: dict[str, object] | None = None) -> dict[str, object]:
        mutation_plan = build_mutator_plan(
            action_type="managed_local_service.setup_apply",
            resources={
                "created": ["container:personal-agent-searxng"],
                "changed": ["runtime_search_config", "memory/local_services/searxng/settings.yml"],
                "deleted": [],
            },
            rollback_scope="remove only the owned personal-agent-searxng container created by this action and restore previous runtime search config",
            rollback_supported=True,
            confirmation_token="confirm-test-search",
            expires_at=4102444800,
            plan_id="search-setup-test",
        )
        return {
            "ok": True,
            "requires_confirmation": True,
            "plan": {
                "service_id": "searxng",
                "setup_mode": "managed_container",
                "provider": "searxng",
                "selected_engine": "podman",
                "preferred_engine": "podman",
                "image": "docker.io/searxng/searxng:latest",
                "container_name": "personal-agent-searxng",
                "loopback_bind": "127.0.0.1:8888:8080",
                "bind_address": "127.0.0.1",
                "port": 8888,
                "health_url": "http://127.0.0.1:8888",
                "rollback_scope": str(mutation_plan["rollback_scope"]),
                "mutation_plan": mutation_plan,
                "plan_id": "search-setup-test",
                "confirmation_token": "confirm-test-search",
                "expires_at": 4102444800,
            },
        }

    def _install_managed_search_adapter(
        self,
        runtime: AgentRuntime,
        *,
        search_status: dict[str, object],
        services_status: dict[str, object],
    ) -> None:
        runtime.search_status = lambda: dict(search_status)  # type: ignore[method-assign]
        runtime.managed_services_status = lambda: dict(services_status)  # type: ignore[method-assign]
        runtime.search_setup_plan = self._managed_setup_plan  # type: ignore[method-assign]

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
        self._install_managed_search_adapter(
            runtime,
            search_status={
                "ok": True,
                "enabled": False,
                "provider": "searxng",
                "available": False,
                "endpoint_configured": False,
                "base_url": None,
                "reason": "search_disabled",
            },
            services_status=self._managed_services_status(enabled=False, configured=False, reachable=False),
        )
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

    def test_explicit_lookup_tts_model_uses_search_not_voice_or_linux_pack(self) -> None:
        runtime = self._runtime()
        opener = self._install_fake_search_client(runtime, "dot.tts project result")

        _body, text, meta = self._chat(
            runtime,
            "can you look up dot.tts im wondering if it would be a good model to use for a project",
            session_id="lookup-dot-tts",
        )

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("metadata-only web results", text.lower())
        self.assertIn("dot.tts project result", text)
        self.assertNotIn("Linux Troubleshooting Workflow", text)
        self.assertNotIn("voice output", text.lower())
        self.assertNotIn("127.0.0.1:8888", text)
        self.assertEqual(1, len(opener.opened_urls))
        self.assertIn("dot.tts", opener.opened_urls[-1])

    def test_messy_public_entity_inputs_use_search_not_capability_or_pack_text(self) -> None:
        examples = (
            ("what is dots.tts", "dots.tts result"),
            ("dots tts any good?", "dots tts result"),
            ("pi.dev?", "pi.dev result"),
            ("is nex-agi adaptive thinking useful for memory?", "nex-agi result"),
        )
        for message, title in examples:
            with self.subTest(message=message):
                runtime = self._runtime()
                opener = self._install_fake_search_client(runtime, title)

                _body, text, meta = self._chat(runtime, message, session_id=f"messy-search-{title}")

                self.assertEqual("action_tool", meta.get("route"))
                self.assertIn("safe_web_search", meta.get("used_tools", []))
                self.assertIn("metadata-only web results", text.lower())
                self.assertIn(title, text)
                self.assertNotIn("Linux Troubleshooting Workflow", text)
                self.assertNotIn("voice output", text.lower())
                self.assertEqual(1, len(opener.opened_urls))

    def test_structured_semantic_intents_cover_daily_driver_boundaries(self) -> None:
        cases = (
            ("what is dots.tts", "web_search"),
            ("dots tts any good?", "web_search"),
            ("what is it?", "ask_clarifying_question"),
            ("do not search, what is dots.tts?", "answer_directly"),
            ("can you install dots.tts?", "package_or_system_mutation_preview"),
            ("what is photosynthesis?", "answer_directly"),
            ("rewrite this: what is dots.tts", "answer_directly"),
            ("is telegram working?", "status_check"),
        )
        for message, semantic_intent in cases:
            with self.subTest(message=message):
                decision = classify_runtime_chat_route(message)

                self.assertEqual(semantic_intent, decision.get("semantic_intent"))
                self.assertIsInstance(decision.get("semantic"), dict)
                self.assertEqual(semantic_intent, decision["semantic"].get("intent"))

    def test_messy_search_inputs_ask_one_followup_when_too_ambiguous(self) -> None:
        examples = (
            "that tts thing people are talking about",
            "what is it?",
        )
        for message in examples:
            with self.subTest(message=message):
                runtime = self._runtime()
                opener = self._install_fake_search_client(runtime, "unused")

                _body, text, meta = self._chat(runtime, message, session_id=f"ambiguous-search-{message[:8]}")

                self.assertEqual("assistant_clarification", meta.get("route"))
                self.assertNotIn("safe_web_search", meta.get("used_tools", []))
                self.assertIn("What exact public project, model, tool, site, or topic should I search for?", text)
                self.assertEqual(0, len(opener.opened_urls))
                self.assertNotIn("Linux Troubleshooting Workflow", text)
                self.assertNotIn("voice output", text.lower())

    def test_search_fallback_does_not_force_search_for_timeless_or_text_transform(self) -> None:
        examples = (
            "explain why the sky is blue in two sentences",
            "what is photosynthesis?",
            "rewrite this paragraph: the quick brown fox jumps over the lazy dog",
            "rewrite this: what is dots.tts",
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

    def test_do_not_search_internet_native_question_does_not_search(self) -> None:
        runtime = self._runtime()
        opener = self._install_fake_search_client(runtime, "dots.tts result")

        _body, text, meta = self._chat(runtime, "do not search, what is dots.tts?", session_id="no-search-dots")

        self.assertEqual("action_tool", meta.get("route"))
        self.assertNotIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("will not search", text.lower())
        self.assertIn("limited or outdated", text.lower())
        self.assertEqual(0, len(opener.opened_urls))

    def test_install_internet_native_name_does_not_search_or_trigger_voice_pack(self) -> None:
        runtime = self._runtime()
        opener = self._install_fake_search_client(runtime, "unused")

        _body, text, meta = self._chat(runtime, "can you install dots.tts?", session_id="install-dots-tts")

        self.assertEqual("action_tool", meta.get("route"))
        self.assertNotIn("safe_web_search", meta.get("used_tools", []))
        self.assertEqual(0, len(opener.opened_urls))
        self.assertNotIn("Linux Troubleshooting Workflow", text)
        self.assertNotIn("voice output", text.lower())

    def test_search_fallback_disabled_offers_plan_mode_setup(self) -> None:
        runtime = self._runtime(search_enabled=False)
        self._install_managed_search_adapter(
            runtime,
            search_status={
                "ok": True,
                "enabled": False,
                "provider": "searxng",
                "available": False,
                "endpoint_configured": False,
                "base_url": None,
                "reason": "search_disabled",
            },
            services_status=self._managed_services_status(enabled=False, configured=False, reachable=False),
        )

        _body, text, meta = self._chat(runtime, "what is dots.tts?", session_id="disabled-search-dots")

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("managed_local_service_setup_preview", meta.get("used_tools", []))
        self.assertIn("Search is not currently working", text)
        self.assertIn("Plan Mode confirmation", text)

    def test_search_status_disabled_says_not_working_and_offers_managed_start(self) -> None:
        runtime = self._runtime(search_enabled=False)
        self._install_managed_search_adapter(
            runtime,
            search_status={
                "ok": True,
                "enabled": False,
                "provider": "searxng",
                "available": False,
                "endpoint_configured": False,
                "base_url": None,
                "reason": "search_disabled",
            },
            services_status=self._managed_services_status(enabled=False, configured=False, reachable=False),
        )

        _body, text, meta = self._chat(runtime, "is search working?", session_id="search-working-disabled")

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("managed_local_service_setup_preview", meta.get("used_tools", []))
        self.assertIn("Search is not currently working", text)
        self.assertIn("Plan Mode confirmation", text)
        self.assertNotIn("Assistant web search is available", text)
        self.assertNotIn("visit http://127.0.0.1:8888", text.lower())

    def test_unreachable_direct_page_offers_managed_recovery(self) -> None:
        runtime = self._runtime(search_enabled=False)
        self._install_managed_search_adapter(
            runtime,
            search_status={
                "ok": True,
                "enabled": True,
                "provider": "searxng",
                "available": False,
                "endpoint_configured": True,
                "base_url": "http://127.0.0.1:8888",
                "reason": "endpoint_unreachable",
            },
            services_status=self._managed_services_status(enabled=True, configured=True, reachable=False),
        )

        _body, text, meta = self._chat(
            runtime,
            "hmm weird, i went to http://127.0.0.1:8888/ and it worked but now it says it cant be reached",
            session_id="search-page-unreachable",
        )

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("managed_local_service_setup_preview", meta.get("used_tools", []))
        self.assertIn("Search is not currently working", text)
        self.assertIn("direct local SearXNG page will refuse connection", text)
        self.assertNotIn("podman", text.lower())

    def test_restart_it_after_search_failure_uses_managed_plan_mode_not_manual_commands(self) -> None:
        runtime = self._runtime(search_enabled=False)
        self._install_managed_search_adapter(
            runtime,
            search_status={
                "ok": True,
                "enabled": True,
                "provider": "searxng",
                "available": False,
                "endpoint_configured": True,
                "base_url": "http://127.0.0.1:8888",
                "reason": "endpoint_unreachable",
            },
            services_status=self._managed_services_status(enabled=True, configured=True, reachable=False),
        )
        self._chat(
            runtime,
            "hmm weird, i went to http://127.0.0.1:8888/ and it worked but now it says it cant be reached",
            session_id="search-restart-followup",
        )

        _body, text, meta = self._chat(runtime, "can you restart it for me?", session_id="search-restart-followup")

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("managed_local_service_setup_preview", meta.get("used_tools", []))
        self.assertIn("I can start or repair the managed search service for you", text)
        self.assertIn("Plan Mode confirmation", text)
        self.assertNotIn("podman run", text.lower())
        self.assertNotIn("docker run", text.lower())

    def test_search_status_available_mentions_assistant_and_direct_page(self) -> None:
        runtime = self._runtime()
        self._install_managed_search_adapter(
            runtime,
            search_status={
                "ok": True,
                "enabled": True,
                "provider": "searxng",
                "available": True,
                "endpoint_configured": True,
                "base_url": "http://127.0.0.1:8888",
                "reason": None,
            },
            services_status=self._managed_services_status(enabled=True, configured=True, reachable=True),
        )

        _body, text, meta = self._chat(runtime, "is search working?", session_id="search-working-available")

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("Assistant web search is available", text)
        self.assertIn("Direct local search page: http://127.0.0.1:8888", text)
        self.assertIn("Use assistant search for metadata-only summaries", text)
