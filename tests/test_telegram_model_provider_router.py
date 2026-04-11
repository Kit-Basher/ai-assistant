from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

from agent.audit_log import AuditLog
from agent.public_chat import build_no_llm_public_message
from agent.ux.llm_fixit_wizard import LLMFixitWizardStore
from telegram_adapter.bot import (
    _handle_message,
    _handle_model,
)


class _FakeResponse:
    def __init__(self, text: str, data: dict[str, object] | None = None) -> None:
        self.text = text
        self.data = data


class _FakeOrchestrator:
    def __init__(self, reply_text: str = "chat fallback", data: dict[str, object] | None = None) -> None:
        self.reply_text = reply_text
        self.data = data
        self.calls: list[tuple[str, str]] = []

    def handle_message(self, text: str, *, user_id: str) -> _FakeResponse:
        self.calls.append((text, user_id))
        return _FakeResponse(self.reply_text, self.data)


class _FakeDB:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set_preference(self, key: str, value: str) -> None:
        self.values[str(key)] = str(value)


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.message_id = 1
        self.replies: list[dict[str, str | None]] = []

    async def reply_text(self, text: str, parse_mode: str | None = None) -> None:
        self.replies.append({"text": text, "parse_mode": parse_mode})


class _FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _FakeUpdate:
    def __init__(self, chat_id: int, text: str) -> None:
        self.effective_chat = _FakeChat(chat_id)
        self.effective_message = _FakeMessage(text)


class _FakeContext:
    def __init__(self, bot_data: dict[str, object]) -> None:
        self.application = type("App", (), {"bot_data": bot_data})()


class _FakeRuntime:
    def __init__(self) -> None:
        self.model_watch_calls = 0
        self._llm_available = True
        self._watch_payload: dict[str, object] = {"ok": True, "proposal": None}
        self._status_payload: dict[str, object] = {
            "ok": True,
            "default_provider": "ollama",
            "default_model": "ollama:qwen2.5:3b-instruct",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "providers": [
                {"id": "ollama", "enabled": True, "health": {"status": "ok"}},
                {"id": "openrouter", "enabled": True, "health": {"status": "ok"}},
            ],
            "models": [
                {
                    "id": "ollama:qwen2.5:3b-instruct",
                    "provider": "ollama",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                },
                {
                    "id": "ollama:qwen2.5:7b-instruct",
                    "provider": "ollama",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                },
                {
                    "id": "openrouter:acme/fast",
                    "provider": "openrouter",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                },
            ],
        }

    def llm_status(self) -> dict[str, object]:
        return dict(self._status_payload)

    def ready_status(self) -> dict[str, object]:
        model = str(
            self._status_payload.get("resolved_default_model")
            or self._status_payload.get("default_model")
            or ""
        ).strip()
        provider = str(self._status_payload.get("default_provider") or "").strip().lower()
        provider_health = (
            self._status_payload.get("active_provider_health")
            if isinstance(self._status_payload.get("active_provider_health"), dict)
            else {"status": "ok" if model else "down"}
        )
        model_health = (
            self._status_payload.get("active_model_health")
            if isinstance(self._status_payload.get("active_model_health"), dict)
            else {"status": "ok" if model else "down"}
        )
        ready = bool(model and str(provider_health.get("status") or "").strip().lower() == "ok")
        runtime_mode = "READY" if ready else "BOOTSTRAP_REQUIRED"
        summary = (
            f"Ready. Using {provider} / {model}."
            if ready
            else "Setup needed. No chat model is ready yet."
        )
        return {
            "ok": True,
            "ready": ready,
            "phase": "ready" if ready else "degraded",
            "runtime_mode": runtime_mode,
            "runtime_status": {
                "runtime_mode": runtime_mode,
                "summary": summary,
                "next_action": None if ready else "Run: python -m agent setup",
            },
            "telegram": {"state": "running", "configured": True},
            "onboarding": {
                "state": "READY" if ready else "LLM_MISSING",
                "summary": "Setup complete. The agent is ready." if ready else build_no_llm_public_message(),
                "next_action": "No action needed." if ready else "Run: python -m agent setup",
            },
        }

    def llm_availability_state(self) -> dict[str, object]:
        if self._llm_available:
            return {"available": True, "reason": "ok"}
        return {"available": False, "reason": "provider_unhealthy"}

    def model_watch_latest(self) -> dict[str, object]:
        return {"ok": True, "found": False, "reason": "No model catalog snapshot available; run refresh"}

    def run_model_watch_once(self, *, trigger: str = "manual") -> tuple[bool, dict[str, object]]:
        self.model_watch_calls += 1
        payload = dict(self._watch_payload)
        payload["trigger"] = trigger
        return True, payload


def _read_audit_rows(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines() if line.strip()]


def _read_log_rows(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines() if line.strip()]


class _ExplodingRuntime:
    def chat(self, _payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
        raise AssertionError("Telegram ordinary chat should not call a local runtime")


class _FakeChatApiBackend:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, object]] = []

    def __call__(self, payload: dict[str, object]) -> dict[str, object]:
        self.chat_calls.append(dict(payload))
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        text = str((messages[-1] or {}).get("content") if messages else "").strip().lower()
        reply_text = "generic reply"
        route = "generic_chat"
        used_llm = True
        used_runtime_state = False

        if text == "hello, is the openrouter setup?":
            reply_text = "OpenRouter is configured and ready to use."
            route = "provider_status"
            used_llm = False
            used_runtime_state = True
        elif text == "is the agent healthy right now?":
            reply_text = "The agent is healthy and ready right now."
            route = "runtime_status"
            used_llm = False
            used_runtime_state = True
        elif text == "what model are you using?":
            reply_text = "I’m using ollama / ollama:qwen3.5:4b."
            route = "model_status"
            used_llm = False
            used_runtime_state = True
        elif text == "check what model is currently enabled please":
            reply_text = "The current chat model is ollama:qwen3.5:4b on Ollama."
            route = "model_status"
            used_llm = False
            used_runtime_state = True
        elif text == "is the agent healthy?":
            reply_text = "The agent is healthy and ready right now."
            route = "runtime_status"
            used_llm = False
            used_runtime_state = True
        elif text == "is openrouter configured?":
            reply_text = "Yes. OpenRouter is configured."
            route = "provider_status"
            used_llm = False
            used_runtime_state = True
        elif text in {
            "setup openrouter",
            "setup ollama",
            "configure ollama",
            "use ollama",
            "switch to ollama",
            "repair openrouter",
            "openrouter unavailable",
            "why isn't this working?",
        }:
            reply_text = "I can help repair provider setup. Tell me which provider you want to fix."
            route = "setup_flow"
            used_llm = False
            used_runtime_state = True

        return {
            "ok": True,
            "assistant": {"content": reply_text},
            "message": reply_text,
            "meta": {
                "route": route,
                "used_llm": used_llm,
                "used_memory": False,
                "used_runtime_state": used_runtime_state,
                "used_tools": [],
                "generic_fallback_used": False,
                "generic_fallback_reason": None,
            },
        }


class TestTelegramModelProviderRouter(unittest.TestCase):
    def test_exact_runtime_queries_use_canonical_telegram_router(self) -> None:
        phrases = (
            ("hello, is the openrouter setup?", "provider_status", "OpenRouter is configured and ready to use."),
            ("is the agent healthy right now?", "runtime_status", "The agent is healthy and ready right now."),
            ("what model are you using?", "model_status", "I’m using ollama / ollama:qwen3.5:4b."),
            ("check what model is currently enabled please", "model_status", "The current chat model is ollama:qwen3.5:4b on Ollama."),
            ("is the agent healthy?", "runtime_status", "The agent is healthy and ready right now."),
            ("is openrouter configured?", "provider_status", "Yes. OpenRouter is configured."),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            api_backend = _FakeChatApiBackend()
            orchestrator = _FakeOrchestrator()
            log_path = f"{tmpdir}/agent.log"
            context = _FakeContext(
                {
                    "runtime": _ExplodingRuntime(),
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": log_path,
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )

            with patch("telegram_adapter.bot._post_local_api_chat_json_async", side_effect=api_backend):
                for phrase, expected_route, expected_text in phrases:
                    update = _FakeUpdate(12345, phrase)
                    asyncio.run(_handle_message(update, context))
                    reply = str(update.effective_message.replies[-1]["text"] or "")
                    self.assertEqual(expected_text, reply)
                    self.assertNotIn("brief:", reply.lower())
                    self.assertNotIn("response:", reply.lower())

            self.assertEqual([], orchestrator.calls)
            self.assertEqual([item[0] for item in phrases], [str(((call.get("messages") or [{}])[-1]).get("content") or "") for call in api_backend.chat_calls])

            log_rows = _read_log_rows(log_path)
            telegram_rows = [
                row.get("payload")
                for row in log_rows
                if str(row.get("type")) == "telegram_message" and isinstance(row.get("payload"), dict)
            ]
            self.assertEqual(len(phrases), len(telegram_rows))
            selected_routes = [str(row.get("selected_route") or "") for row in telegram_rows]
            self.assertEqual(
                ["provider_status", "runtime_status", "model_status", "model_status", "runtime_status", "provider_status"],
                selected_routes,
            )
            for row in telegram_rows:
                self.assertEqual("api_chat_proxy", row.get("handler_name"))
                self.assertTrue(bool(row.get("used_runtime_state")))
                self.assertFalse(bool(row.get("legacy_compatibility")))

    def test_rotate_token_text_routes_to_setup_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "rotate token")
            asyncio.run(_handle_message(update, context))

            self.assertTrue(update.effective_message.replies)
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("python -m agent.secrets set telegram:bot_token", reply)
            self.assertEqual([], orchestrator.calls)

    def test_where_were_we_routes_to_memory_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "where were we")
            asyncio.run(_handle_message(update, context))

            self.assertEqual([("/memory", "12345")], orchestrator.calls)
            self.assertTrue(update.effective_message.replies)
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertEqual("chat fallback", reply)
            self.assertNotIn("Reply 1, 2, or 3.", reply)

    def test_memory_text_routes_to_memory_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            runtime._llm_available = False
            runtime._status_payload = {
                "ok": True,
                "default_provider": "ollama",
                "default_model": None,
                "resolved_default_model": None,
                "allow_remote_fallback": False,
                "active_provider_health": {"status": "down"},
                "active_model_health": {"status": "down"},
                "providers": [{"id": "ollama", "enabled": True, "health": {"status": "down"}}],
                "models": [],
            }
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "memory")
            asyncio.run(_handle_message(update, context))
            self.assertEqual([("/memory", "12345")], orchestrator.calls)
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertEqual("chat fallback", reply)
            self.assertNotIn("Setup state:", reply)

    def test_resume_text_routes_to_memory_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            runtime._llm_available = False
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "resume")
            asyncio.run(_handle_message(update, context))
            self.assertEqual([("/memory", "12345")], orchestrator.calls)
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertEqual("chat fallback", reply)

    def test_breif_text_routes_to_brief_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            runtime._llm_available = False
            runtime._status_payload = {
                "ok": True,
                "default_provider": "ollama",
                "default_model": None,
                "resolved_default_model": None,
                "allow_remote_fallback": False,
                "active_provider_health": {"status": "down"},
                "active_model_health": {"status": "down"},
                "providers": [{"id": "ollama", "enabled": True, "health": {"status": "down"}}],
                "models": [],
            }
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "breif")
            asyncio.run(_handle_message(update, context))
            self.assertEqual([("/brief", "12345")], orchestrator.calls)
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertEqual("chat fallback", reply)
            self.assertNotIn("Setup state:", reply)

    def test_setup_text_uses_bootstrap_guidance_when_chat_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            runtime._status_payload = {
                "ok": True,
                "default_provider": "ollama",
                "default_model": None,
                "resolved_default_model": None,
                "allow_remote_fallback": False,
                "active_provider_health": {"status": "down"},
                "active_model_health": {"status": "down"},
                "providers": [{"id": "ollama", "enabled": True, "health": {"status": "down"}}],
                "models": [],
            }
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "setup")
            asyncio.run(_handle_message(update, context))

            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn(build_no_llm_public_message(), reply)
            self.assertNotIn("No chat model available right now.", reply)
            self.assertNotIn("Reason:", reply)
            self.assertNotIn("Reply 1, 2, or 3.", reply)
            self.assertNotIn("LLM unavailable right now.", reply)
            self.assertEqual([], orchestrator.calls)

    def test_setup_text_returns_complete_summary_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "setup")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Setup is complete", reply)
            self.assertNotIn("status is unavailable", reply.lower())
            self.assertEqual([], orchestrator.calls)

    def test_setup_text_reports_not_started_when_runtime_state_empty(self) -> None:
        class _NotStartedRuntime(_FakeRuntime):
            def ready_status(self) -> dict[str, object]:
                return {
                    "ok": True,
                    "ready": False,
                    "phase": "starting",
                    "runtime_mode": "BOOTSTRAP_REQUIRED",
                    "runtime_status": {
                        "runtime_mode": "BOOTSTRAP_REQUIRED",
                        "summary": "Setup needed. No chat model is ready yet.",
                        "next_action": "Run: python -m agent setup",
                    },
                    "onboarding": {
                        "state": "NOT_STARTED",
                        "summary": "Setup has not started.",
                        "next_action": "Run: python -m agent setup",
                    },
                    "telegram": {"state": "running", "configured": True},
                }

            def llm_status(self) -> dict[str, object]:
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _NotStartedRuntime()
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "setup")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Setup state: not started", reply)
            self.assertIn("Next:", reply)
            self.assertNotIn("status is unavailable", reply.lower())
            self.assertEqual([], orchestrator.calls)

    def test_setup_uses_runtime_ready_payload_when_llm_status_is_empty(self) -> None:
        class _ReadyRuntime(_FakeRuntime):
            def ready_status(self) -> dict[str, object]:
                return {
                    "ok": True,
                    "ready": True,
                    "phase": "ready",
                    "runtime_mode": "READY",
                    "runtime_status": {
                        "runtime_mode": "READY",
                        "summary": "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
                        "next_action": None,
                    },
                    "onboarding": {
                        "state": "READY",
                        "summary": "Setup complete. The agent is ready.",
                        "next_action": "No action needed.",
                    },
                    "telegram": {"state": "running", "configured": True},
                }

            def llm_status(self) -> dict[str, object]:
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _ReadyRuntime()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": _FakeOrchestrator(),
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "setup")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Setup is complete", reply)
            self.assertNotIn("not started", reply.lower())

    def test_status_uses_runtime_ready_payload_when_llm_status_is_empty(self) -> None:
        class _ReadyRuntime(_FakeRuntime):
            def ready_status(self) -> dict[str, object]:
                return {
                    "ok": True,
                    "ready": True,
                    "phase": "ready",
                    "runtime_mode": "READY",
                    "runtime_status": {
                        "runtime_mode": "READY",
                        "summary": "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
                        "next_action": None,
                    },
                    "telegram": {
                        "state": "running",
                        "configured": True,
                    },
                    "api": {"version": "0.2.0", "git_commit": "abcdef123456", "uptime_seconds": 123},
                }

            def llm_status(self) -> dict[str, object]:
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _ReadyRuntime()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": _FakeOrchestrator(),
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "status")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("runtime_mode: READY", reply)
            self.assertIn("telegram: running", reply)
            self.assertNotIn("BOOTSTRAP_REQUIRED", reply)
            self.assertNotIn("No chat model is ready yet", reply)

    def test_help_prioritizes_setup_when_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            runtime._status_payload = {
                "ok": True,
                "default_provider": "ollama",
                "default_model": None,
                "resolved_default_model": None,
                "allow_remote_fallback": False,
                "active_provider_health": {"status": "down"},
                "active_model_health": {"status": "down"},
                "providers": [{"id": "ollama", "enabled": True, "health": {"status": "down"}}],
                "models": [],
            }
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "help")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Setup state:", reply)
            self.assertIn("Next:", reply)
            self.assertEqual([], orchestrator.calls)

    def test_help_shows_normal_commands_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "help")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Available commands:", reply)
            self.assertIn("doctor", reply.lower())
            self.assertIn("setup", reply.lower())
            self.assertIn("memory", reply.lower())
            self.assertNotIn("Reply 1, 2, or 3.", reply)
            self.assertEqual([], orchestrator.calls)

    def test_setup_language_routes_through_canonical_runtime_chat(self) -> None:
        phrases = (
            "why isn't this working?",
            "setup openrouter",
            "setup ollama",
            "configure ollama",
            "use ollama",
            "switch to ollama",
            "repair openrouter",
            "openrouter unavailable",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            api_backend = _FakeChatApiBackend()
            orchestrator = _FakeOrchestrator()
            log_path = f"{tmpdir}/agent.log"
            context = _FakeContext(
                {
                    "runtime": _ExplodingRuntime(),
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": log_path,
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )

            with patch("telegram_adapter.bot._post_local_api_chat_json_async", side_effect=api_backend):
                for phrase in phrases:
                    update = _FakeUpdate(12345, phrase)
                    asyncio.run(_handle_message(update, context))
                    reply = str(update.effective_message.replies[-1]["text"] or "")
                    self.assertIn("repair provider setup", reply.lower())
                    self.assertNotIn("ollama pull", reply)
                    self.assertNotIn("ollama list", reply)

            self.assertEqual([], orchestrator.calls)
            log_rows = _read_log_rows(log_path)
            telegram_rows = [
                row.get("payload")
                for row in log_rows
                if str(row.get("type")) == "telegram_message" and isinstance(row.get("payload"), dict)
            ]
            self.assertEqual(["setup_flow"] * len(phrases), [str(row.get("selected_route") or "") for row in telegram_rows])
            self.assertTrue(all(str(row.get("handler_name") or "") == "api_chat_proxy" for row in telegram_rows))

    def test_numeric_without_active_wizard_returns_no_active_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            audit_path = f"{tmpdir}/audit.jsonl"
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=audit_path),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "1")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("No active choice right now", reply)
            self.assertEqual([], orchestrator.calls)
            rows = _read_audit_rows(audit_path)
            handled_row = [row for row in rows if row.get("action") == "telegram.message.handled"][-1]
            params = handled_row.get("params_redacted") if isinstance(handled_row.get("params_redacted"), dict) else {}
            self.assertEqual("numeric_no_wizard", params.get("route"))

    def test_generic_text_uses_api_backed_generic_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            api_backend = _FakeChatApiBackend()
            orchestrator = _FakeOrchestrator(
                reply_text="chat fallback",
                data={
                    "route": "generic_chat",
                    "used_llm": True,
                    "used_memory": True,
                    "used_runtime_state": False,
                    "used_tools": [],
                },
            )
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "ignored"}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                    "runtime": _ExplodingRuntime(),
                }
            )
            update = _FakeUpdate(12345, "fix it")
            with patch("telegram_adapter.bot._post_local_api_chat_json_async", side_effect=api_backend):
                asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertEqual("generic reply", reply)
            self.assertEqual([], orchestrator.calls)

    def test_transport_source_guard_has_no_setup_wizard_or_direct_pull_guidance(self) -> None:
        with open("telegram_adapter/bot.py", "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("TelegramModelProviderWizardStore", source)
        self.assertNotIn("provider.setup.openrouter", source)
        self.assertNotIn("provider.setup.ollama", source)
        self.assertNotIn("ollama pull qwen2.5:3b-instruct", source)
        self.assertNotIn("ollama list", source)
        self.assertNotIn("set_provider_secret(", source)
        self.assertNotIn("test_provider(", source)
        self.assertNotIn("update_defaults(", source)

    def test_model_command_returns_model_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            update = _FakeUpdate(12345, "/model")
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": _FakeOrchestrator(),
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                }
            )
            asyncio.run(_handle_model(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Current provider/model:", reply)
            self.assertIn("Configured providers:", reply)


if __name__ == "__main__":
    unittest.main()
