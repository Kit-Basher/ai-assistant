from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from agent.audit_log import AuditLog
from agent.ux.llm_fixit_wizard import LLMFixitWizardStore
from telegram_adapter.bot import (
    TelegramModelProviderWizardStore,
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
        self.secret_calls: list[tuple[str, dict[str, object]]] = []
        self.test_calls: list[tuple[str, dict[str, object]]] = []
        self.defaults_calls: list[dict[str, object]] = []
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

    def set_provider_secret(self, provider_id: str, payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
        self.secret_calls.append((provider_id, dict(payload)))
        return True, {"ok": True}

    def test_provider(self, provider_id: str, payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
        self.test_calls.append((provider_id, dict(payload)))
        return True, {"ok": True}

    def update_defaults(self, payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
        self.defaults_calls.append(dict(payload))
        return True, {"ok": True}


def _read_audit_rows(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines() if line.strip()]


class TestTelegramModelProviderRouter(unittest.TestCase):
    def test_classify_model_provider_intent(self) -> None:
        self.assertEqual("model_watch.run_now", classify_model_provider_intent("check for new models"))
        self.assertEqual("provider.status", classify_model_provider_intent("what model are we using"))
        self.assertEqual("provider.setup.ollama", classify_model_provider_intent("setup ollama"))
        self.assertEqual("provider.setup.openrouter", classify_model_provider_intent("setup openrouter"))
        self.assertEqual("provider.setup.openrouter", classify_model_provider_intent("repair openrouter"))
        self.assertEqual("provider.setup.openrouter", classify_model_provider_intent("openrouter unavailable"))
        self.assertEqual("provider.help", classify_model_provider_intent("setup"))
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
                    "model_provider_wizard_store": TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json"),
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
                    "model_provider_wizard_store": TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json"),
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
                    "model_provider_wizard_store": TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json"),
                }
            )
            update = _FakeUpdate(12345, "rotate token")
            asyncio.run(_handle_message(update, context))

            self.assertTrue(update.effective_message.replies)
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("python -m agent.secrets set telegram:bot_token", reply)
            self.assertEqual([], orchestrator.calls)

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
                    "model_provider_wizard_store": TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json"),
                }
            )
            update = _FakeUpdate(12345, "setup")
            asyncio.run(_handle_message(update, context))

            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("No chat model available right now.", reply)
            self.assertIn("Start Ollama", reply)
            self.assertEqual([], orchestrator.calls)

    def test_setup_openrouter_numeric_and_key_paste(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = f"{tmpdir}/audit.jsonl"
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            wizard_store = TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json")
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=audit_path),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                    "model_provider_wizard_store": wizard_store,
                }
            )
            chat_id = 12345

            asyncio.run(_handle_message(_FakeUpdate(chat_id, "setup openrouter"), context))
            first = str(context.application.bot_data["model_provider_wizard_store"].state.get("step"))  # type: ignore[index]
            self.assertEqual("awaiting_openrouter_has_key", first)

            asyncio.run(_handle_message(_FakeUpdate(chat_id, "1"), context))
            second = str(context.application.bot_data["model_provider_wizard_store"].state.get("step"))  # type: ignore[index]
            self.assertEqual("awaiting_openrouter_key", second)

            secret_text = "sk-or-super-secret"
            asyncio.run(_handle_message(_FakeUpdate(chat_id, secret_text), context))
            third = str(context.application.bot_data["model_provider_wizard_store"].state.get("step"))  # type: ignore[index]
            self.assertEqual("awaiting_openrouter_default", third)
            self.assertEqual([("openrouter", {"api_key": secret_text})], runtime.secret_calls)
            self.assertEqual([("openrouter", {})], runtime.test_calls)

            asyncio.run(_handle_message(_FakeUpdate(chat_id, "1"), context))
            self.assertFalse(bool(context.application.bot_data["model_provider_wizard_store"].state.get("active")))  # type: ignore[index]
            self.assertTrue(runtime.defaults_calls)
            self.assertEqual("openrouter", runtime.defaults_calls[-1].get("default_provider"))
            self.assertEqual([], orchestrator.calls)

            rows = _read_audit_rows(audit_path)
            self.assertTrue(rows)
            serialized = json.dumps(rows, ensure_ascii=True).lower()
            self.assertNotIn(secret_text.lower(), serialized)

    def test_setup_ollama_numeric_choice_sets_default(self) -> None:
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
                    "model_provider_wizard_store": TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json"),
                }
            )
            chat_id = 12345

            asyncio.run(_handle_message(_FakeUpdate(chat_id, "setup ollama"), context))
            asyncio.run(_handle_message(_FakeUpdate(chat_id, "2"), context))
            asyncio.run(_handle_message(_FakeUpdate(chat_id, "1"), context))
            self.assertTrue(runtime.defaults_calls)
            latest = runtime.defaults_calls[-1]
            self.assertEqual("ollama", latest.get("default_provider"))
            self.assertIn("default_model", latest)
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
                    "model_provider_wizard_store": TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json"),
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

    def test_repair_openrouter_routes_to_setup_wizard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            wizard_store = TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json")
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                    "model_provider_wizard_store": wizard_store,
                }
            )
            update = _FakeUpdate(12345, "Repair openrouter")
            asyncio.run(_handle_message(update, context))
            reply = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("OpenRouter setup", reply)
            self.assertEqual("awaiting_openrouter_has_key", str(wizard_store.state.get("step") or ""))
            self.assertEqual([], orchestrator.calls)

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
                    "model_provider_wizard_store": TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json"),
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
            orchestrator = _FakeOrchestrator()
            wizard_store = TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json")
            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=f"{tmpdir}/audit.jsonl"),
                    "llm_fixit_fn": lambda _payload: (True, {"ok": True, "message": "fixit prompt"}),
                    "llm_fixit_store": LLMFixitWizardStore(path=f"{tmpdir}/fixit.json"),
                    "model_provider_wizard_store": wizard_store,
                }
            )
            first = _FakeUpdate(12345, "fix it")
            asyncio.run(_handle_message(first, context))
            first_reply = str(first.effective_message.replies[-1]["text"] or "")
            self.assertIn("1)", first_reply)
            self.assertIn("2)", first_reply)
            self.assertIn("3)", first_reply)
            self.assertEqual(
                "awaiting_recovery_choice",
                str(context.application.bot_data["model_provider_wizard_store"].state.get("step")),  # type: ignore[index]
            )
            second = _FakeUpdate(12345, "1")
            asyncio.run(_handle_message(second, context))
            second_reply = str(second.effective_message.replies[-1]["text"] or "")
            self.assertIn("Current provider/model:", second_reply)
            self.assertFalse(bool(context.application.bot_data["model_provider_wizard_store"].state.get("active")))  # type: ignore[index]
            self.assertEqual([], orchestrator.calls)

    def test_fixit_prompt_response_persists_wizard_state_and_audits_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _FakeRuntime()
            orchestrator = _FakeOrchestrator()
            audit_path = f"{tmpdir}/audit.jsonl"
            model_store = TelegramModelProviderWizardStore(path=f"{tmpdir}/wizard.json")
            llm_fixit_store = LLMFixitWizardStore(path=f"{tmpdir}/fixit.json")
            model_store.save(
                {
                    "active": True,
                    "flow": "recovery",
                    "step": "awaiting_recovery_choice",
                    "question": "Reply 1, 2, or 3.",
                    "choices": [
                        {"id": "status", "label": "Show current model/provider status", "recommended": True},
                        {"id": "fixit", "label": "Run LLM fix-it wizard", "recommended": False},
                        {"id": "brief", "label": "Show a brief system summary", "recommended": False},
                    ],
                    "chat_id": "12345",
                    "issue_code": "llm.recovery",
                    "pending": {"reason": "provider_unhealthy"},
                }
            )

            def _llm_fixit(_payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
                return True, {
                    "ok": True,
                    "status": "needs_user_choice",
                    "issue_code": "openrouter_down",
                    "message": "OpenRouter is currently unavailable.\n1) Use local-only\n2) Repair OpenRouter\n3) Details",
                    "choices": [
                        {"id": "local_only", "label": "Use local-only", "recommended": True},
                        {"id": "repair_openrouter", "label": "Repair OpenRouter", "recommended": False},
                        {"id": "details", "label": "Details", "recommended": False},
                    ],
                }

            context = _FakeContext(
                {
                    "runtime": runtime,
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                    "audit_log": AuditLog(path=audit_path),
                    "llm_fixit_fn": _llm_fixit,
                    "llm_fixit_store": llm_fixit_store,
                    "model_provider_wizard_store": model_store,
                }
            )
            update = _FakeUpdate(12345, "2")
            asyncio.run(_handle_message(update, context))

            persisted = llm_fixit_store.load()
            self.assertTrue(bool(persisted.get("active")))
            self.assertEqual("awaiting_choice", str(persisted.get("step") or ""))
            choices = persisted.get("choices") if isinstance(persisted.get("choices"), list) else []
            self.assertEqual(3, len(choices))
            rows = _read_audit_rows(audit_path)
            prompt_rows = [row for row in rows if row.get("action") == "telegram.fixit.prompt_shown"]
            self.assertTrue(prompt_rows)

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
