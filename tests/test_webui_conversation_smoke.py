from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any
from unittest import mock

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.intent.assessment import IntentAssessment, IntentCandidate
from agent.public_chat import build_no_llm_public_message
from agent.orchestrator import OrchestratorResponse


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


def _request_json(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any], str]:
    headers = {"Accept": "application/json, text/html"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip().startswith("{") else {}
            if not isinstance(parsed, dict):
                parsed = {}
            return int(getattr(response, "status", 200)), parsed, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw) if raw.strip().startswith("{") else {}
        if not isinstance(parsed, dict):
            parsed = {}
        return int(getattr(exc, "code", 500)), parsed, raw


def _request_text(base_url: str, path: str) -> tuple[int, str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "text/html,application/json"})
    with urllib.request.urlopen(request, timeout=5.0) as response:
        return int(getattr(response, "status", 200)), response.read().decode("utf-8", errors="replace")


def _assistant_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    return str(assistant.get("content") or payload.get("message") or payload.get("error") or "").strip()


def _assistant_warnings(text: str) -> list[str]:
    lowered = str(text or "").lower()
    warnings: list[str] = []
    if not str(text or "").strip():
        warnings.append("empty assistant reply")
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        warnings.append("raw json reply")
    if any(
        token in lowered
        for token in (
            "trace_id:",
            "route_reason:",
            "selection_policy",
            "runtime_payload",
            "runtime_state_failure_reason",
            "setup_type:",
            "operator_only:",
            "source_surface:",
            "thread_id:",
            "user_id:",
            "local_observations",
        )
    ):
        warnings.append("internal text leak")
    if len(str(text).split()) < 3:
        warnings.append("too short to read as assistant output")
    return warnings


class _MemoryHandlerForTest(APIServerHandler):
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
        return dict(self._payload)


class _ConversationOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def handle_message(self, text: str, *, user_id: str, chat_context: dict[str, Any] | None = None) -> OrchestratorResponse:
        call = {
            "text": text,
            "user_id": user_id,
            "chat_context": dict(chat_context or {}),
        }
        self.calls.append(call)
        if len(self.calls) == 1:
            reply = "Sure. I can help with that."
        else:
            reply = "Start with the smallest next step."
        return OrchestratorResponse(
            reply,
            {
                "ok": True,
                "route": "generic_chat",
                "route_reason": "generic_chat",
                "used_runtime_state": False,
                "used_llm": True,
                "used_memory": False,
                "used_tools": [],
                "fallback_used": False,
                "generic_fallback_used": True,
                "generic_fallback_allowed": True,
                "provider": "ollama",
                "model": "ollama:qwen2.5:7b-instruct",
            },
        )


class TestWebuiConversationSmoke(unittest.TestCase):
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

    def _runtime(self) -> AgentRuntime:
        return AgentRuntime(_config(self.registry_path, self.db_path))

    def _serve(self, runtime: AgentRuntime) -> tuple[ThreadingHTTPServer, str, threading.Thread]:
        class _HandlerForTest(APIServerHandler):
            pass

        _HandlerForTest.runtime = runtime
        server = ThreadingHTTPServer(("127.0.0.1", 0), _HandlerForTest)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_address[1]}", thread

    def _memory_request_json(
        self,
        runtime: AgentRuntime,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any], str]:
        handler = _MemoryHandlerForTest(runtime, path, payload)
        if method.upper() == "GET":
            handler.do_GET()
        elif method.upper() == "POST":
            handler.do_POST()
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unsupported method: {method}")
        raw = handler.body.decode("utf-8", errors="replace")
        parsed = json.loads(raw) if raw.strip().startswith("{") else {}
        if not isinstance(parsed, dict):
            parsed = {}
        return int(handler.status_code), parsed, raw

    def _memory_request_text(self, runtime: AgentRuntime, path: str) -> tuple[int, str]:
        handler = _MemoryHandlerForTest(runtime, path)
        handler.do_GET()
        return int(handler.status_code), handler.body.decode("utf-8", errors="replace")

    def test_webui_frontdoor_supports_a_real_two_turn_assistant_conversation(self) -> None:
        runtime = self._runtime()
        orchestrator = _ConversationOrchestrator()
        ready_payload = {
            "ok": True,
            "ready": True,
            "phase": "ready",
            "startup_phase": "ready",
            "runtime_mode": "READY",
            "summary": "Ready to chat.",
        }

        with (
            mock.patch.object(runtime, "ready_status", return_value=ready_payload),
            mock.patch.object(runtime, "chat_route_decision", return_value={"route": "generic_chat"}),
            mock.patch.object(runtime, "should_use_assistant_frontdoor", return_value=False),
            mock.patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None),
            mock.patch.object(runtime, "orchestrator", return_value=orchestrator),
            mock.patch.object(runtime, "consume_clarify_recovery_choice", return_value=(False, {})),
            mock.patch.object(runtime, "consume_binary_clarification_choice", return_value=(False, {})),
            mock.patch.object(runtime, "consume_intent_choice", return_value=(False, {})),
            mock.patch.object(runtime, "consume_thread_integrity_choice", return_value=(False, {})),
            mock.patch(
                "agent.api_server.detect_low_confidence",
                return_value=SimpleNamespace(is_low_confidence=False, reason="none", debug={"norm": "hello"}),
            ),
            mock.patch(
                "agent.api_server.classify_ambiguity",
                return_value=SimpleNamespace(ambiguous=False, reason="none"),
            ),
            mock.patch(
                "agent.api_server.assess_intent_deterministic",
                return_value=IntentAssessment(
                    decision="proceed",
                    confidence=1.0,
                    candidates=[IntentCandidate(intent="chat", score=1.0, reason="smoke", details={})],
                    next_question=None,
                    debug={"source": "smoke"},
                ),
            ),
        ):
            server = None
            thread = None
            base_url = None
            use_network = True
            try:
                server, base_url, thread = self._serve(runtime)
            except PermissionError:
                use_network = False
            try:
                request_json = _request_json if use_network else self._memory_request_json
                request_text = _request_text if use_network else self._memory_request_text
                request_base = str(base_url or "")

                root_status, root_html = request_text(request_base, "/") if use_network else request_text(runtime, "/")
                self.assertEqual(200, root_status)
                self.assertIn("id=\"root\"", root_html)
                self.assertIn("personal-agent-webui", root_html)

                ready_status, ready_body, _ready_raw = request_json(request_base, "GET", "/ready") if use_network else request_json(runtime, "GET", "/ready")
                self.assertEqual(200, ready_status)
                self.assertTrue(bool(ready_body.get("ready")))
                self.assertEqual("ready", str(ready_body.get("phase") or "").strip().lower())
                self.assertIn("Ready", str(ready_body.get("summary") or ""))

                chat_payload = {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "session_id": "webui-smoke-session",
                    "thread_id": "webui-smoke-thread",
                    "user_id": "webui-smoke-user",
                    "source_surface": "webui",
                    "purpose": "chat",
                    "task_type": "chat",
                }
                first_status, first_body, first_raw = (
                    request_json(request_base, "POST", "/chat", chat_payload)
                    if use_network
                    else request_json(runtime, "POST", "/chat", chat_payload)
                )
                self.assertEqual(200, first_status)
                self.assertTrue(bool(first_body.get("ok")))
                first_text = _assistant_text(first_body)
                self.assertEqual(first_text, str(first_body.get("message") or "").strip())
                self.assertEqual([], _assistant_warnings(first_text))
                first_meta = first_body.get("meta") if isinstance(first_body.get("meta"), dict) else {}
                self.assertEqual("generic_chat", first_meta.get("route"))
                self.assertTrue(bool(first_meta.get("used_llm")))
                self.assertIn("Sure. I can help with that.", first_text)
                self.assertNotIn("trace_id", first_raw.lower())

                followup_payload = {
                    "messages": [{"role": "user", "content": "What should I do first?"}],
                    "session_id": "webui-smoke-session",
                    "thread_id": "webui-smoke-thread",
                    "user_id": "webui-smoke-user",
                    "source_surface": "webui",
                    "purpose": "chat",
                    "task_type": "chat",
                }
                second_status, second_body, second_raw = (
                    request_json(request_base, "POST", "/chat", followup_payload)
                    if use_network
                    else request_json(runtime, "POST", "/chat", followup_payload)
                )
                self.assertEqual(200, second_status)
                self.assertTrue(bool(second_body.get("ok")))
                second_text = _assistant_text(second_body)
                self.assertEqual(second_text, str(second_body.get("message") or "").strip())
                self.assertEqual([], _assistant_warnings(second_text))
                second_meta = second_body.get("meta") if isinstance(second_body.get("meta"), dict) else {}
                self.assertEqual("generic_chat", second_meta.get("route"))
                self.assertTrue(bool(second_meta.get("used_llm")))
                self.assertIn("Start with the smallest next step.", second_text)
                self.assertNotEqual(first_text, second_text)
                self.assertNotIn("runtime_payload", second_raw.lower())

                self.assertEqual(2, len(orchestrator.calls))
                first_call = orchestrator.calls[0]
                second_call = orchestrator.calls[1]
                self.assertEqual("webui-smoke-user", first_call["user_id"])
                self.assertEqual("webui-smoke-user", second_call["user_id"])
                self.assertEqual("webui-smoke-thread", str(first_call["chat_context"].get("thread_id") or ""))
                self.assertEqual("webui-smoke-thread", str(second_call["chat_context"].get("thread_id") or ""))
                self.assertEqual("webui", str(first_call["chat_context"].get("source_surface") or ""))
                self.assertEqual("webui", str(second_call["chat_context"].get("source_surface") or ""))
                self.assertEqual("Hello", str(first_call["chat_context"].get("messages")[-1]["content"]))
                self.assertEqual("What should I do first?", str(second_call["chat_context"].get("messages")[-1]["content"]))
            finally:
                if server is not None and thread is not None:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2.0)

    def test_webui_frontdoor_shows_canonical_no_llm_guidance_on_a_clean_machine(self) -> None:
        runtime = self._runtime()
        orchestrator = runtime.orchestrator()
        ready_payload = {
            "ok": True,
            "ready": False,
            "phase": "degraded",
            "startup_phase": "starting",
            "runtime_mode": "BOOTSTRAP_REQUIRED",
            "summary": "Setup needed. No chat model is ready yet.",
            "runtime_status": {
                "runtime_mode": "BOOTSTRAP_REQUIRED",
                "summary": "Setup needed. No chat model is ready yet.",
                "next_action": "Run: python -m agent setup",
            },
            "onboarding": {
                "state": "LLM_MISSING",
                "summary": build_no_llm_public_message(),
                "next_action": "Run: python -m agent setup",
            },
        }

        with (
            mock.patch.object(runtime, "ready_status", return_value=ready_payload),
            mock.patch.object(runtime, "chat_route_decision", return_value={"route": "generic_chat"}),
            mock.patch.object(runtime, "should_use_assistant_frontdoor", return_value=False),
            mock.patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None),
            mock.patch.object(orchestrator, "_llm_chat_available", return_value=False),
            mock.patch.object(runtime, "consume_clarify_recovery_choice", return_value=(False, {})),
            mock.patch.object(runtime, "consume_binary_clarification_choice", return_value=(False, {})),
            mock.patch.object(runtime, "consume_intent_choice", return_value=(False, {})),
            mock.patch.object(runtime, "consume_thread_integrity_choice", return_value=(False, {})),
            mock.patch(
                "agent.api_server.detect_low_confidence",
                return_value=SimpleNamespace(is_low_confidence=False, reason="none", debug={"norm": "hello"}),
            ),
            mock.patch(
                "agent.api_server.classify_ambiguity",
                return_value=SimpleNamespace(ambiguous=False, reason="none"),
            ),
            mock.patch(
                "agent.api_server.assess_intent_deterministic",
                return_value=IntentAssessment(
                    decision="proceed",
                    confidence=1.0,
                    candidates=[IntentCandidate(intent="chat", score=1.0, reason="smoke", details={})],
                    next_question=None,
                    debug={"source": "smoke"},
                ),
            ),
        ):
            server = None
            thread = None
            base_url = None
            use_network = True
            try:
                server, base_url, thread = self._serve(runtime)
            except PermissionError:
                use_network = False
            try:
                request_json = _request_json if use_network else self._memory_request_json
                request_base = str(base_url or "")

                ready_status, ready_body, _ready_raw = (
                    request_json(request_base, "GET", "/ready")
                    if use_network
                    else request_json(runtime, "GET", "/ready")
                )
                self.assertEqual(200, ready_status)
                self.assertFalse(bool(ready_body.get("ready")))
                self.assertIn("No chat model", str(ready_body.get("summary") or ""))
                runtime_status = ready_body.get("runtime_status") if isinstance(ready_body.get("runtime_status"), dict) else {}
                self.assertEqual("Run: python -m agent setup", str(runtime_status.get("next_action") or ""))

                chat_payload = {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "session_id": "webui-no-llm-session",
                    "thread_id": "webui-no-llm-thread",
                    "user_id": "webui-no-llm-user",
                    "source_surface": "webui",
                    "purpose": "chat",
                    "task_type": "chat",
                }
                first_status, first_body, first_raw = (
                    request_json(request_base, "POST", "/chat", chat_payload)
                    if use_network
                    else request_json(runtime, "POST", "/chat", chat_payload)
                )
                self.assertEqual(200, first_status)
                self.assertTrue(bool(first_body.get("ok")))
                first_text = _assistant_text(first_body)
                self.assertEqual(build_no_llm_public_message(), first_text)
                self.assertEqual(first_text, str(first_body.get("message") or "").strip())
                self.assertEqual([], _assistant_warnings(first_text))
                first_meta = first_body.get("meta") if isinstance(first_body.get("meta"), dict) else {}
                self.assertEqual("generic_chat", first_meta.get("route"))
                self.assertFalse(bool(first_meta.get("used_llm")))
                self.assertNotIn("trace_id", first_raw.lower())

                followup_payload = {
                    "messages": [{"role": "user", "content": "What should I do first?"}],
                    "session_id": "webui-no-llm-session",
                    "thread_id": "webui-no-llm-thread",
                    "user_id": "webui-no-llm-user",
                    "source_surface": "webui",
                    "purpose": "chat",
                    "task_type": "chat",
                }
                second_status, second_body, second_raw = (
                    request_json(request_base, "POST", "/chat", followup_payload)
                    if use_network
                    else request_json(runtime, "POST", "/chat", followup_payload)
                )
                self.assertEqual(200, second_status)
                self.assertTrue(bool(second_body.get("ok")))
                second_text = _assistant_text(second_body)
                self.assertEqual(build_no_llm_public_message(), second_text)
                self.assertEqual(second_text, str(second_body.get("message") or "").strip())
                self.assertEqual([], _assistant_warnings(second_text))
                second_meta = second_body.get("meta") if isinstance(second_body.get("meta"), dict) else {}
                self.assertEqual("generic_chat", second_meta.get("route"))
                self.assertFalse(bool(second_meta.get("used_llm")))
                self.assertEqual(first_text, second_text)
                self.assertNotIn("runtime_payload", second_raw.lower())
            finally:
                if server is not None and thread is not None:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
