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


def _assistant_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    return str(assistant.get("content") or payload.get("message") or payload.get("error") or "").strip()


def _assistant_leak_warnings(text: str) -> list[str]:
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
            "reason:",
        )
    ):
        warnings.append("internal text leak")
    if lowered.startswith("thinking…") or lowered.startswith("sorry — the agent encountered an error"):
        warnings.append("transport placeholder")
    if len(str(text).split()) < 4:
        warnings.append("too short to read as assistant output")
    return warnings


class _MemoryHandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, Any] | None = None) -> None:
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


class TestAssistantBehaviorReleaseGate(unittest.TestCase):
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

    def _assert_public_assistant_reply(
        self,
        body: dict[str, Any],
        *,
        no_llm_expected: bool = False,
    ) -> str:
        self.assertTrue(bool(body.get("ok")))
        text = _assistant_text(body)
        self.assertEqual(text, str(body.get("message") or "").strip())
        self.assertEqual(text, str((body.get("assistant") or {}).get("content") or "").strip())
        self.assertEqual([], _assistant_leak_warnings(text))
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        self.assertNotIn("route_reason", meta)
        self.assertNotIn("selection_policy", meta)
        self.assertNotIn("runtime_state_failure_reason", meta)
        self.assertNotIn("setup_type", meta)
        self.assertNotIn("thread_id", meta)
        self.assertNotIn("user_id", meta)
        if no_llm_expected:
            self.assertEqual(build_no_llm_public_message(), text)
        else:
            self.assertNotEqual(build_no_llm_public_message(), text)
        return text

    def test_release_gate_covers_core_assistant_behaviors(self) -> None:
        runtime = self._runtime()
        orchestrator = runtime.orchestrator()
        ready_payload = {
            "ok": True,
            "ready": True,
            "phase": "ready",
            "startup_phase": "ready",
            "runtime_mode": "READY",
            "summary": "Ready to chat.",
        }

        def _fake_route_inference(**kwargs: Any) -> dict[str, Any]:
            user_text = str(kwargs.get("user_text") or "").strip().lower()
            if "hello" in user_text:
                reply = "Hello. I can help with that."
            elif "help me plan my day" in user_text or "switch to openrouter" in user_text:
                reply = "I can help with one request at a time. Pick one thing."
            elif any(token in user_text for token in ("asdf", "qwer", "zzzz")):
                reply = "I’m not sure what you mean. Try asking a short question about one topic."
            else:
                reply = "Tell me one thing you want to do and I’ll help."
            return {
                "ok": True,
                "text": reply,
                "provider": "ollama",
                "model": "ollama:qwen2.5:7b-instruct",
                "duration_ms": 1,
                "attempts": [],
                "fallback_used": False,
            }

        with (
            mock.patch.object(runtime, "ready_status", return_value=ready_payload),
            mock.patch.object(runtime, "chat_route_decision", return_value={"route": "generic_chat"}),
            mock.patch.object(runtime, "should_use_assistant_frontdoor", return_value=False),
            mock.patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None),
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
            mock.patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference),
            mock.patch.object(orchestrator, "_llm_chat_available", return_value=True),
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
                cases = [
                    ("greeting", "hello"),
                    ("vague", "help me"),
                    ("nonsense", "asdf qwer zzzz"),
                    ("mixed", "help me plan my day and also switch to openrouter"),
                ]
                for case_name, prompt in cases:
                    with self.subTest(case=case_name):
                        payload = {
                            "messages": [{"role": "user", "content": prompt}],
                            "source_surface": "api",
                            "user_id": f"gate:{case_name}",
                            "thread_id": f"gate:{case_name}:thread",
                        }
                        status, body, raw = (
                            request_json(request_base, "POST", "/chat", payload)
                            if use_network
                            else request_json(runtime, "POST", "/chat", payload)
                        )
                        self.assertEqual(200, status)
                        text = self._assert_public_assistant_reply(body)
                        self.assertNotIn("trace_id", raw.lower())
                        self.assertGreaterEqual(len(text.split()), 4)

                with mock.patch.object(orchestrator, "_llm_chat_available", return_value=False):
                    no_llm_payload = {
                        "messages": [{"role": "user", "content": "hello"}],
                        "source_surface": "api",
                        "user_id": "gate:no-llm",
                        "thread_id": "gate:no-llm:thread",
                    }
                    status, body, raw = (
                        request_json(request_base, "POST", "/chat", no_llm_payload)
                        if use_network
                        else request_json(runtime, "POST", "/chat", no_llm_payload)
                    )
                self.assertEqual(200, status)
                text = self._assert_public_assistant_reply(body, no_llm_expected=True)
                self.assertNotIn("trace_id", raw.lower())
                self.assertEqual(build_no_llm_public_message(), text)
            finally:
                if server is not None and thread is not None:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
