from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from agent.audit_log import AuditLog
from agent.ux.llm_fixit_wizard import LLMFixitWizardStore
from telegram_adapter.bot import (
    _handle_message,
    _handle_model,
    classify_model_provider_intent,
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.data = None


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def handle_message(self, text: str, *, user_id: str) -> _FakeResponse:
        self.calls.append((text, user_id))
        return _FakeResponse("chat fallback")


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
            f"Agent is ready. Using {provider} / {model}."
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
                "summary": "Setup complete. The agent is ready." if ready else "No chat model available right now.",
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


class TestTelegramModelProviderRouter(unittest.TestCase):
    def test_classify_model_provider_intent(self) -> None:
        self.assertEqual("model_watch.run_now", classify_model_provider_intent("check for new models"))
        self.assertEqual("provider.status", classify_model_provider_intent("what model are we using"))
        self.assertEqual("none", classify_model_provider_intent("setup ollama"))
        self.assertEqual("none", classify_model_provider_intent("setup openrouter"))
        self.assertEqual("none", classify_model_provider_intent("repair openrouter"))
        self.assertEqual("none", classify_model_provider_intent("openrouter unavailable"))
        self.assertEqual("none", classify_model_provider_intent("setup"))
        self.assertEqual("none", classify_model_provider_intent("write me a poem"))

    def test_prerouter_model_status_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            audit_log = AuditLog(path=f"{tmpdir}/audit.jsonl")
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": audit_log,
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            update = _FakeUpdate(12345, "what model are we using")
            asyncio.run(_handle_message(update, context))

            self.assertTrue(update.effective_message.replies)
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Current provider/model:", reply)
            self.assertIn("Configured providers:", reply)
            self.assertEqual([], orchestrator.calls)

    def test_prerouter_check_models_runs_model_watch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            runtime._watch_payload = {"ok": True, "proposal": None}
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
            update = _FakeUpdate(12345, "check for new/better models")
            asyncio.run(_handle_message(update, context))

            self.assertEqual(1, runtime.model_watch_calls)
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("No new better models in configured providers.", reply)
            self.assertEqual([], orchestrator.calls)

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
            self.assertIn("No chat model available right now.", reply)
            self.assertIn("Next:", reply)
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
                        "summary": "Agent is ready. Using ollama / ollama:qwen2.5:3b-instruct.",
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
                        "summary": "Agent is ready. Using ollama / ollama:qwen2.5:3b-instruct.",
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

    def test_why_not_working_routes_to_setup_guidance(self) -> None:
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
            update = _FakeUpdate(12345, "why isn't this working?")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Setup state:", reply)
            self.assertIn("Next:", reply)
            self.assertEqual([], orchestrator.calls)

    def test_provider_specific_setup_phrases_route_to_canonical_setup(self) -> None:
        phrases = (
            "setup openrouter",
            "setup ollama",
            "repair openrouter",
            "openrouter unavailable",
        )
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

            for phrase in phrases:
                update = _FakeUpdate(12345, phrase)
                asyncio.run(_handle_message(update, context))
                reply = str(update.effective_message.replies[-1]["text"] or "")
                self.assertNotIn("OpenRouter setup", reply)
                self.assertNotIn("ollama pull", reply)
                self.assertNotIn("ollama list", reply)
                self.assertIn("Setup", reply)
            self.assertEqual([], orchestrator.calls)

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

    def test_unknown_text_clarifies_when_llm_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            runtime._llm_available = True
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
            update = _FakeUpdate(12345, "fix it")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("Do you mean:", reply)
            self.assertIn("A)", reply)
            self.assertIn("B)", reply)
            self.assertEqual([], orchestrator.calls)

    def test_unknown_text_suggests_and_numeric_choice_is_intercepted_when_llm_unavailable(self) -> None:
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
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "fixit prompt"}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                }
            )
            first = _FakeUpdate(12345, "fix it")
            asyncio.run(_handle_message(first, context))
            first_reply = str(first.effective_message.replies[-1]["text"] or "")
            self.assertIn("Setup state:", first_reply)
            self.assertIn("Next:", first_reply)
            self.assertNotIn("Reply 1, 2, or 3.", first_reply)
            second = _FakeUpdate(12345, "1")
            asyncio.run(_handle_message(second, context))
            second_reply = str(second.effective_message.replies[-1]["text"] or "")
            self.assertIn("No active choice right now", second_reply)
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
