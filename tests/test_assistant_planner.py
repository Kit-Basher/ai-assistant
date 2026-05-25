from __future__ import annotations

from dataclasses import replace
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent.api_server import AgentRuntime
from agent.assistant_planner import AssistantPlan, validate_assistant_plan
from agent.search.safe_web_search import SafeWebSearchClient, SafeWebSearchConfig
from tests.test_api_packs_endpoints import _config
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text
from tests.test_safe_web_search import _FakeOpener


class _StaticPlanner:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def plan(self, *, user_text: str, llm_client: Any, context: dict[str, Any] | None = None) -> AssistantPlan:
        self.calls.append(user_text)
        return validate_assistant_plan(self.payload)


class TestAssistantPlannerSchema(unittest.TestCase):
    def test_valid_web_search_query_plan(self) -> None:
        plan = validate_assistant_plan(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "web_search", "action": "query", "goal": "find release notes"},
                "confidence": 0.91,
                "user_facing_summary": "User wants web search.",
            }
        )

        self.assertTrue(plan.valid)
        self.assertTrue(plan.usable)
        self.assertEqual("web_search", plan.agent_request.capability)
        self.assertEqual("query", plan.agent_request.action)

    def test_unknown_capability_rejected(self) -> None:
        plan = validate_assistant_plan(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "docker", "action": "query", "goal": "run a container"},
                "confidence": 0.9,
            }
        )

        self.assertFalse(plan.valid)
        self.assertIn("planner_unknown_capability", plan.errors)

    def test_invalid_action_rejected(self) -> None:
        plan = validate_assistant_plan(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "web_search", "action": "docker_run", "goal": "start service"},
                "confidence": 0.9,
            }
        )

        self.assertFalse(plan.valid)
        self.assertIn("planner_unknown_action", plan.errors)

    def test_answer_directly_cannot_smuggle_agent_request(self) -> None:
        plan = validate_assistant_plan(
            {
                "intent": "answer_directly",
                "agent_request": {"capability": "web_search", "action": "query", "goal": "news"},
                "confidence": 0.8,
            }
        )

        self.assertFalse(plan.valid)
        self.assertIn("planner_agent_request_not_allowed_for_intent", plan.errors)


class TestAssistantPlannerOrchestrator(unittest.TestCase):
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

    def _runtime(self, planner_payload: dict[str, Any], *, search_enabled: bool = False) -> tuple[AgentRuntime, _StaticPlanner]:
        config = replace(
            _config(self.registry_path, self.db_path, self.skills_path),
            search_enabled=search_enabled,
            search_provider="searxng",
            searxng_base_url="http://127.0.0.1:8888" if search_enabled else None,
            search_timeout_seconds=0.5,
            search_max_results=2,
        )
        runtime = AgentRuntime(config)
        planner = _StaticPlanner(planner_payload)
        runtime._assistant_planner = planner  # noqa: SLF001
        if search_enabled:
            runtime._safe_web_search_client = SafeWebSearchClient(  # noqa: SLF001
                SafeWebSearchConfig(enabled=True, searxng_base_url="http://127.0.0.1:8888", max_results=2),
                opener=_FakeOpener({"results": [{"title": "Planner result", "url": "https://example.com", "content": "Snippet"}]}),
            )
        return runtime, planner

    def _chat(self, runtime: AgentRuntime, prompt: str) -> tuple[dict[str, Any], str]:
        handler = _MemoryHandlerForTest(
            runtime,
            "/chat",
            {
                "messages": [{"role": "user", "content": prompt}],
                "session_id": "planner",
                "thread_id": "planner-thread",
                "source_surface": "webui",
                "purpose": "chat",
                "task_type": "chat",
            },
        )
        handler.do_POST()
        body = json.loads(handler.body.decode("utf-8"))
        return body, _assistant_text(body)

    def test_planner_routes_varied_web_search_query_to_safe_search(self) -> None:
        runtime, planner = self._runtime(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "web_search", "action": "query", "goal": "planner search topic"},
                "confidence": 0.9,
                "user_facing_summary": "Search metadata.",
            },
            search_enabled=True,
        )
        body, text = self._chat(runtime, "look outside my local notes for this")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual(["look outside my local notes for this"], planner.calls)
        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("Planner result", text)
        self.assertIn("untrusted", text.lower())

    def test_planner_routes_web_search_setup_without_phrase_match(self) -> None:
        runtime, _planner = self._runtime(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "web_search", "action": "setup", "goal": "enable internet lookup"},
                "confidence": 0.88,
                "user_facing_summary": "Set up web search.",
            }
        )
        body, text = self._chat(runtime, "i need outside lookup sometime")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertIn("safe_web_search", meta.get("used_tools", []))
        self.assertIn("Web search", text)
        self.assertNotIn("I ran", text)

    def test_planner_routes_external_skills_request(self) -> None:
        runtime, _planner = self._runtime(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "external_skills", "action": "query", "goal": "install a skill that lets you browse"},
                "confidence": 0.9,
                "user_facing_summary": "User wants a browser skill.",
            }
        )
        body, text = self._chat(runtime, "could you learn webpage reading somehow")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertIn("pack_acquisition", meta.get("used_tools", []))
        self.assertIn("not installed or usable", text.lower())

    def test_planner_routes_telegram_setup(self) -> None:
        runtime, _planner = self._runtime(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "telegram", "action": "setup", "goal": "connect Telegram"},
                "confidence": 0.8,
                "user_facing_summary": "Telegram setup.",
            }
        )
        body, text = self._chat(runtime, "make phone chat work")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertIn("telegram", text.lower())

    def test_planner_routes_chat_model_status(self) -> None:
        runtime, _planner = self._runtime(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "chat_model", "action": "status", "goal": "current model"},
                "confidence": 0.8,
                "user_facing_summary": "Model status.",
            }
        )
        body, text = self._chat(runtime, "which brain is active")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual("model_status", meta.get("route"))
        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertIn("model", text.lower())

    def test_clarify_plan_returns_clarification(self) -> None:
        runtime, _planner = self._runtime(
            {
                "intent": "clarify",
                "agent_request": {"capability": "none", "action": "none", "goal": ""},
                "confidence": 0.6,
                "user_facing_summary": "Which capability do you want me to set up?",
            }
        )
        body, text = self._chat(runtime, "make it better")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual("assistant_clarification", meta.get("route"))
        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertIn("Which capability", text)

    def test_invalid_planner_action_is_rejected_before_tools(self) -> None:
        runtime, _planner = self._runtime(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "web_search", "action": "docker_run", "goal": "run SearXNG"},
                "confidence": 0.9,
            }
        )
        body, text = self._chat(runtime, "start whatever container you need")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual("assistant_clarification", meta.get("route"))
        self.assertEqual("planner_validation_failed", body.get("error_kind"))
        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertNotIn("docker", " ".join(str(item) for item in meta.get("used_tools", [])).lower())
        self.assertIn("safely", text.lower())

    def test_unknown_capability_is_rejected_before_tools(self) -> None:
        runtime, _planner = self._runtime(
            {
                "intent": "ask_agent",
                "agent_request": {"capability": "docker", "action": "query", "goal": "run arbitrary docker"},
                "confidence": 0.9,
            }
        )
        body, _text = self._chat(runtime, "docker can fix this right")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual("assistant_clarification", meta.get("route"))
        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertNotIn("managed_local_service_setup", meta.get("used_tools", []))

    def test_answer_directly_uses_model_unavailable_fallback(self) -> None:
        runtime, _planner = self._runtime(
            {
                "intent": "answer_directly",
                "agent_request": {"capability": "none", "action": "none", "goal": ""},
                "confidence": 0.8,
                "user_facing_summary": "Normal chat.",
            }
        )
        runtime.assistant_chat_available = lambda: False  # force model-unavailable fallback after planning
        body, text = self._chat(runtime, "write a tiny note")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}

        self.assertEqual("generic_chat", meta.get("route"))
        self.assertIn("assistant_planner", meta.get("used_tools", []))
        self.assertFalse(bool(meta.get("used_llm")))
        self.assertTrue(text.strip())


if __name__ == "__main__":
    unittest.main()
