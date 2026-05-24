from __future__ import annotations

from dataclasses import replace
import json
import os
import tempfile
import unittest
from pathlib import Path

from agent.api_server import AgentRuntime
from agent.search.safe_web_search import SafeWebSearchClient, SafeWebSearchConfig
from agent.search.search_setup_ux import build_search_setup_ux, render_search_setup_ux
from agent.setup_chat_flow import classify_runtime_chat_route
from tests.test_api_packs_endpoints import _config
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text


class TestSearchSetupUX(unittest.TestCase):
    def test_disabled_status_renders_setup_hint(self) -> None:
        text = render_search_setup_ux(
            {
                "enabled": False,
                "available": False,
                "provider": "searxng",
                "endpoint_configured": False,
                "reason": "search_disabled",
            }
        )

        self.assertIn("Web search is disabled", text)
        self.assertIn("SEARCH_ENABLED=1", text)
        self.assertIn("SEARXNG_BASE_URL", text)
        self.assertIn("metadata only", text)
        self.assertIn("untrusted", text)

    def test_endpoint_missing_renders_setup_hint(self) -> None:
        ux = build_search_setup_ux(
            {
                "enabled": True,
                "available": False,
                "provider": "searxng",
                "endpoint_configured": False,
                "reason": "endpoint_missing",
            }
        )

        self.assertFalse(ux.available)
        self.assertEqual("SEARXNG_BASE_URL", ux.missing_requirement)
        self.assertIn("SearXNG endpoint is missing", ux.message)

    def test_configured_status_renders_metadata_only_reminder(self) -> None:
        text = render_search_setup_ux(
            {
                "enabled": True,
                "available": True,
                "provider": "searxng",
                "endpoint_configured": True,
                "reason": None,
            }
        )

        self.assertIn("configured and available", text)
        self.assertIn("metadata only", text)
        self.assertIn("untrusted", text)
        self.assertIn("does not fetch pages", text)

    def test_unsupported_provider_renders_safe_failure(self) -> None:
        result = SafeWebSearchClient(SafeWebSearchConfig(enabled=True, provider="other")).search("news")
        hint = result.setup_hint or {}

        self.assertFalse(result.ok)
        self.assertEqual("unsupported_provider", result.error_kind)
        self.assertEqual("SEARCH_PROVIDER=searxng", hint.get("missing_requirement"))
        self.assertIn("only supports SearXNG", hint.get("message", ""))

    def test_status_prompts_route_to_search_setup_ux(self) -> None:
        prompts = [
            "is web search enabled?",
            "is search configured?",
            "how do I set up web search?",
            "how do I set up SearXNG?",
            "why can't you search the internet?",
            "can you search online?",
            "what is your search status?",
        ]

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                decision = classify_runtime_chat_route(prompt)
                self.assertEqual("action_tool", decision.get("route"))
                self.assertEqual("safe_web_search_status", decision.get("kind"))

    def test_docs_mention_no_page_fetch_download_or_import(self) -> None:
        text = Path("docs/operator/SAFE_WEB_SEARCH.md").read_text(encoding="utf-8")

        self.assertIn("does not fetch pages", text)
        self.assertIn("does not download files", text)
        self.assertIn("does not install or import external packs", text)


class TestSearchSetupChatUX(unittest.TestCase):
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

    def _runtime(self, *, search_enabled: bool, endpoint: str | None) -> AgentRuntime:
        config = replace(
            _config(self.registry_path, self.db_path, self.skills_path),
            search_enabled=search_enabled,
            search_provider="searxng",
            searxng_base_url=endpoint,
            search_timeout_seconds=1.0,
            search_max_results=3,
        )
        return AgentRuntime(config)

    def _chat(self, runtime: AgentRuntime, prompt: str) -> tuple[dict[str, object], str]:
        handler = _MemoryHandlerForTest(
            runtime,
            "/chat",
            {
                "messages": [{"role": "user", "content": prompt}],
                "session_id": "search-setup",
                "thread_id": "search-setup-thread",
                "source_surface": "webui",
                "purpose": "chat",
                "task_type": "chat",
            },
        )
        handler.do_POST()
        body = json.loads(handler.body.decode("utf-8"))
        return body, _assistant_text(body)

    def test_disabled_chat_status_prompt_is_not_generic(self) -> None:
        body, text = self._chat(self._runtime(search_enabled=False, endpoint=None), "is web search enabled?")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("Web search is optional and not set up", text)
        self.assertIn("Safe web search uses local SearXNG", text)
        self.assertRegex(text, r"Docker|Podman")
        self.assertNotIn("search works", text.lower())

    def test_endpoint_missing_chat_status_prompt_names_endpoint(self) -> None:
        _body, text = self._chat(self._runtime(search_enabled=True, endpoint=None), "what is your search status?")

        self.assertIn("Safe web search uses local SearXNG", text)
        self.assertIn("not set up", text)
        self.assertNotIn("configured and available", text)

    def test_configured_chat_status_prompt_reminds_untrusted_metadata_only(self) -> None:
        _body, text = self._chat(
            self._runtime(search_enabled=True, endpoint="http://127.0.0.1:8080"),
            "can you search online?",
        )

        self.assertIn("configured and available", text)
        self.assertIn("metadata only", text)
        self.assertIn("untrusted", text)

    def test_search_query_failure_includes_setup_hint(self) -> None:
        _body, text = self._chat(
            self._runtime(search_enabled=True, endpoint=None),
            "search the web for SearXNG setup",
        )

        self.assertIn("Web search is optional and not set up", text)
        self.assertIn("Safe web search uses local SearXNG", text)
        self.assertIn("will not open pages", text)

