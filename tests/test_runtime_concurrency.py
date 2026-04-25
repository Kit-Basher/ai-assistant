from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.orchestrator import OrchestratorResponse
from agent.memory_runtime import MemoryRuntime
from telegram_adapter.bot import _handle_message


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

    async def reply_text(self, text: str, parse_mode: str | None = None, **_kwargs) -> None:  # type: ignore[no-untyped-def]
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


class _ConcurrentPrefDB:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self._lock = threading.Lock()
        self._active = 0

    def _enter(self) -> None:
        with self._lock:
            if self._active:
                raise sqlite3.InterfaceError("simulated concurrent sqlite access")
            self._active += 1
        time.sleep(0.02)

    def _exit(self) -> None:
        with self._lock:
            self._active -= 1

    def get_user_pref(self, key: str) -> str | None:
        self._enter()
        try:
            return self.values.get(str(key))
        finally:
            self._exit()

    def get_user_pref_entry(self, key: str) -> dict[str, object] | None:
        self._enter()
        try:
            value = self.values.get(str(key))
            if value is None:
                return None
            return {
                "key": str(key),
                "value": value,
                "updated_at": 0,
                "revision": 1,
            }
        finally:
            self._exit()

    def set_user_pref(self, key: str, value: str) -> None:
        self._enter()
        try:
            self.values[str(key)] = str(value)
        finally:
            self._exit()

    def set_user_pref_if_revision(self, key: str, value: str, expected_revision: int) -> dict[str, object]:
        self._enter()
        try:
            self.values[str(key)] = str(value)
            return {
                "ok": True,
                "applied": True,
                "revision": int(expected_revision) + 1,
                "entry": {
                    "key": str(key),
                    "value": str(value),
                    "updated_at": 0,
                    "revision": int(expected_revision) + 1,
                },
            }
        finally:
            self._exit()


class TestRuntimeConcurrency(unittest.TestCase):
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

    @staticmethod
    def _telegram_disabled_optional() -> dict[str, object]:
        return {
            "enabled": False,
            "token_configured": False,
            "token_source": "missing",
            "ready_state": "disabled_optional",
            "effective_state": "disabled_optional",
            "config_source": "default",
            "config_source_path": None,
            "service_installed": True,
            "service_active": False,
            "service_enabled": False,
            "lock_present": False,
            "lock_live": False,
            "lock_stale": False,
            "lock_path": None,
            "lock_pid": None,
            "next_action": "Run: python -m agent telegram_enable",
        }

    @staticmethod
    def _seed_ok_health(runtime: AgentRuntime, *, provider_id: str, model_ids: list[str]) -> None:
        provider_key = str(provider_id).strip().lower()
        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            provider_key: {
                "status": "ok",
                "last_checked_at": 123,
            }
        }
        runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
            model_id: {
                "provider_id": provider_key,
                "status": "ok",
                "last_checked_at": 123,
            }
            for model_id in model_ids
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

    def test_provider_switch_during_chat_request_keeps_one_model_snapshot(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
            },
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 6,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(
            runtime,
            provider_id="ollama",
            model_ids=["ollama:qwen2.5:7b-instruct", "ollama:qwen3.5:4b"],
        )

        truth = runtime.runtime_truth_service()
        original_current_chat_target_status = truth.current_chat_target_status
        entered = threading.Event()
        release = threading.Event()
        response_holder: dict[str, object] = {}

        def _delayed_current_chat_target_status() -> dict[str, object]:
            snapshot = dict(original_current_chat_target_status())
            entered.set()
            self.assertTrue(release.wait(1.0))
            return snapshot

        with patch.object(truth, "current_chat_target_status", side_effect=_delayed_current_chat_target_status):
            def _run_chat() -> None:
                ok, body = runtime.chat(
                    {
                        "messages": [{"role": "user", "content": "what model are you using?"}],
                        "source_surface": "api",
                    }
                )
                response_holder["ok"] = ok
                response_holder["body"] = body

            worker = threading.Thread(target=_run_chat)
            worker.start()
            self.assertTrue(entered.wait(1.0))
            switched_ok, _ = runtime.set_default_chat_model("ollama:qwen3.5:4b")
            self.assertTrue(switched_ok)
            release.set()
            worker.join(timeout=1.0)

        body = response_holder.get("body")
        self.assertIsInstance(body, dict)
        assert isinstance(body, dict)
        self.assertTrue(bool(response_holder.get("ok")))
        self.assertEqual("model_status", body["meta"]["route"])
        self.assertIsNone(body["meta"]["model"])
        self.assertIn("ollama:qwen2.5:7b-instruct", body["assistant"]["content"])
        defaults = runtime.get_defaults()
        self.assertEqual("ollama:qwen3.5:4b", defaults["resolved_default_model"])

    def test_ready_during_warmup_transition_returns_coherent_snapshot(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(
            runtime,
            provider_id="ollama",
            model_ids=["ollama:qwen2.5:7b-instruct"],
        )
        runtime.startup_phase = "warming"
        with runtime._startup_warmup_lock:
            runtime._startup_warmup_remaining = ["router_reload"]

        entered = threading.Event()
        release = threading.Event()
        payload_holder: dict[str, object] = {}
        telegram_payload = {
            "ok": True,
            "enabled": False,
            "configured": False,
            "token_source": "none",
            "state": "disabled_optional",
            "effective_state": "disabled_optional",
            "config_source": "default",
            "config_source_path": None,
            "service_installed": False,
            "service_active": False,
            "service_enabled": False,
            "lock_present": False,
            "lock_live": False,
            "lock_stale": False,
            "lock_path": None,
            "lock_pid": None,
            "next_action": "No action needed.",
        }

        def _delayed_model_watch_status() -> dict[str, object]:
            entered.set()
            self.assertTrue(release.wait(1.0))
            return {"enabled": False, "last_run_ts": None, "last_error": None}

        with patch.object(runtime, "telegram_status", return_value=telegram_payload), patch.object(
            runtime,
            "_model_watch_hf_status_snapshot",
            side_effect=_delayed_model_watch_status,
        ):
            def _run_ready() -> None:
                payload_holder["payload"] = runtime.ready_status()

            worker = threading.Thread(target=_run_ready)
            worker.start()
            self.assertTrue(entered.wait(1.0))
            runtime.startup_phase = "ready"
            with runtime._startup_warmup_lock:
                runtime._startup_warmup_remaining = []
            release.set()
            worker.join(timeout=1.0)

        payload = payload_holder.get("payload")
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual("warmup", payload["phase"])
        self.assertEqual("warming", payload["startup_phase"])
        self.assertFalse(bool(payload["ready"]))
        self.assertEqual("DEGRADED", payload["runtime_mode"])
        self.assertEqual("ollama", payload["llm"]["provider"])
        self.assertEqual("ollama:qwen2.5:7b-instruct", payload["llm"]["model"])

    def test_concurrent_api_and_telegram_chat_both_route_through_orchestrator(self) -> None:
        runtime = self._runtime()
        orchestrator = runtime.orchestrator()
        recorded_calls: list[dict[str, object]] = []
        recorded_lock = threading.Lock()
        api_result: dict[str, object] = {}

        def _fake_handle_message(text: str, user_id: str, *, chat_context: dict[str, object] | None = None) -> OrchestratorResponse:
            with recorded_lock:
                recorded_calls.append(
                    {
                        "text": text,
                        "user_id": user_id,
                        "source_surface": str((chat_context or {}).get("source_surface") or "unknown"),
                    }
                )
            time.sleep(0.05)
            return OrchestratorResponse(
                "orchestrated reply",
                {
                    "route": "generic_chat",
                    "used_llm": False,
                    "used_memory": False,
                    "used_runtime_state": False,
                    "used_tools": [],
                    "ok": True,
                },
            )

        async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
            ok, body = runtime.chat(payload)
            self.assertTrue(ok)
            return body

        update = _FakeUpdate(707, "hello from telegram")
        context = _FakeContext(
            {
                "orchestrator": orchestrator,
                "db": _FakeDB(),
                "log_path": os.path.join(self.tmpdir.name, "agent.log"),
                "fetch_local_api_chat_json": _fake_post,
            }
        )

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            orchestrator,
            "handle_message",
            side_effect=_fake_handle_message,
        ):
            def _api_call() -> None:
                ok, body = runtime.chat(
                    {
                        "messages": [{"role": "user", "content": "hello from api"}],
                        "source_surface": "api",
                    }
                )
                api_result["ok"] = ok
                api_result["body"] = body

            api_thread = threading.Thread(target=_api_call)
            api_thread.start()
            asyncio.run(_handle_message(update, context))
            api_thread.join(timeout=1.0)

        self.assertEqual(2, len(recorded_calls))
        self.assertEqual({"api", "telegram"}, {str(row["source_surface"]) for row in recorded_calls})
        self.assertTrue(bool(api_result.get("ok")))
        self.assertEqual("orchestrated reply", api_result["body"]["assistant"]["content"])
        self.assertEqual("orchestrated reply", str(update.effective_message.replies[-1]["text"] or ""))

    def test_concurrent_plain_chat_requests_serialize_memory_runtime_access(self) -> None:
        runtime = self._runtime()
        orchestrator = runtime.orchestrator()
        orchestrator._memory_runtime = MemoryRuntime(_ConcurrentPrefDB())  # type: ignore[assignment]
        barrier = threading.Barrier(2)
        results: list[tuple[bool, dict[str, object]]] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()
        router_result = {
            "ok": True,
            "text": "Hello from the orchestrator path.",
            "provider": "ollama",
            "model": "ollama:llama3",
            "fallback_used": False,
            "attempts": [],
            "duration_ms": 4,
            "error_kind": None,
            "selection_reason": "router_default",
        }

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            runtime._router,
            "enabled",
            return_value=True,
        ), patch.object(
            orchestrator,
            "_llm_chat_available",
            return_value=True,
        ), patch(
            "agent.orchestrator.route_inference",
            return_value=router_result,
        ):
            def _run_chat(index: int) -> None:
                try:
                    barrier.wait(timeout=1.0)
                    result = runtime.chat(
                        {
                            "messages": [{"role": "user", "content": f"hello from thread {index}"}],
                            "source_surface": "api",
                            "session_id": f"session-{index}",
                            "thread_id": f"thread-{index}",
                        }
                    )
                    with results_lock:
                        results.append(result)
                except BaseException as exc:  # pragma: no cover - failure capture
                    with results_lock:
                        errors.append(exc)

            threads = [threading.Thread(target=_run_chat, args=(idx,)) for idx in (1, 2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)

        self.assertEqual([], errors)
        self.assertEqual(2, len(results))
        for ok, body in results:
            self.assertTrue(ok)
            assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
            self.assertEqual("Hello from the orchestrator path.", assistant.get("content"))

    def test_concurrent_plain_chat_requests_serialize_orchestrator_handle_message(self) -> None:
        runtime = self._runtime()
        orchestrator = runtime.orchestrator()
        barrier = threading.Barrier(2)
        results: list[tuple[bool, dict[str, object]]] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()
        active_lock = threading.Lock()
        active_calls = 0

        def _fake_handle_message(
            text: str,
            user_id: str,
            *,
            chat_context: dict[str, object] | None = None,
        ) -> OrchestratorResponse:
            del text, user_id, chat_context
            nonlocal active_calls
            with active_lock:
                if active_calls:
                    raise sqlite3.OperationalError("simulated overlapping orchestrator access")
                active_calls += 1
            try:
                time.sleep(0.05)
                return OrchestratorResponse(
                    "serialized reply",
                    {
                        "route": "generic_chat",
                        "used_llm": False,
                        "used_memory": False,
                        "used_runtime_state": False,
                        "used_tools": [],
                        "ok": True,
                    },
                )
            finally:
                with active_lock:
                    active_calls -= 1

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            orchestrator,
            "handle_message",
            side_effect=_fake_handle_message,
        ):
            def _run_chat(index: int) -> None:
                try:
                    barrier.wait(timeout=1.0)
                    result = runtime.chat(
                        {
                            "messages": [{"role": "user", "content": f"hello from thread {index}"}],
                            "source_surface": "api",
                            "session_id": f"session-{index}",
                            "thread_id": f"thread-{index}",
                        }
                    )
                    with results_lock:
                        results.append(result)
                except BaseException as exc:  # pragma: no cover - failure capture
                    with results_lock:
                        errors.append(exc)

            threads = [threading.Thread(target=_run_chat, args=(idx,)) for idx in (1, 2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)

        self.assertEqual([], errors)
        self.assertEqual(2, len(results))
        for ok, body in results:
            self.assertTrue(ok)
            assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
            self.assertEqual("serialized reply", assistant.get("content"))

    def test_ready_waits_for_chat_runtime_bootstrap_before_reporting_ready(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path), defer_bootstrap_warmup=True)
        runtime._reload_router()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(
            runtime,
            provider_id="ollama",
            model_ids=["ollama:qwen2.5:7b-instruct"],
        )

        entered = threading.Event()
        release = threading.Event()
        original_bootstrap = runtime._ensure_chat_runtime_bootstrapped

        def _delayed_chat_runtime_bootstrap() -> object:
            entered.set()
            self.assertTrue(release.wait(1.0))
            return original_bootstrap()

        try:
            with patch.object(
                runtime,
                "_ensure_chat_runtime_bootstrapped",
                side_effect=_delayed_chat_runtime_bootstrap,
            ), patch(
                "agent.api_server.get_telegram_runtime_state",
                return_value=self._telegram_disabled_optional(),
            ):
                runtime.set_listening("127.0.0.1", 8765)
                runtime.mark_server_listening()
                self.assertTrue(entered.wait(1.0))
                warming_payload = runtime.ready_status()
                self.assertFalse(bool(warming_payload["ready"]))
                self.assertEqual("warmup", warming_payload["phase"])
                self.assertIn("chat_runtime_bootstrap", list(warming_payload["warmup_remaining"]))
                self.assertFalse(runtime._chat_runtime_bootstrap_completed)
                release.set()
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    if runtime.startup_phase == "ready":
                        break
                    time.sleep(0.02)
                ready_payload = runtime.ready_status()
                self.assertTrue(bool(ready_payload["ready"]))
                self.assertEqual("ready", ready_payload["startup_phase"])
                self.assertTrue(runtime._chat_runtime_bootstrap_completed)
                self.assertEqual([], list(ready_payload["warmup_remaining"]))
        finally:
            release.set()
            runtime.close()

    def test_first_chat_after_deferred_startup_ready_succeeds(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path), defer_bootstrap_warmup=True)
        runtime._reload_router()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(
            runtime,
            provider_id="ollama",
            model_ids=["ollama:qwen2.5:7b-instruct"],
        )
        router_result = {
            "ok": True,
            "text": "Hello from the orchestrator path.",
            "provider": "ollama",
            "model": "ollama:qwen2.5:7b-instruct",
            "fallback_used": False,
            "attempts": [],
            "duration_ms": 4,
            "error_kind": None,
            "selection_reason": "router_default",
        }

        try:
            runtime.set_listening("127.0.0.1", 8765)
            runtime.mark_server_listening()
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if runtime.startup_phase == "ready":
                    break
                time.sleep(0.02)
            self.assertEqual("ready", runtime.startup_phase)
            self.assertTrue(runtime._chat_runtime_bootstrap_completed)
            self.assertIsNotNone(runtime._orchestrator)
            with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
                runtime._router,
                "enabled",
                return_value=True,
            ), patch(
                "agent.orchestrator.route_inference",
                return_value=router_result,
            ):
                ok, body = runtime.chat(
                    {
                        "messages": [{"role": "user", "content": "Answer with exactly one word: ping"}],
                        "source_surface": "api",
                    }
                )
            self.assertTrue(ok)
            assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
            self.assertEqual("Hello from the orchestrator path.", assistant.get("content"))
        finally:
            runtime.close()

    def test_first_chat_and_ready_probe_do_not_race_after_deferred_startup(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path), defer_bootstrap_warmup=True)
        runtime._reload_router()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(
            runtime,
            provider_id="ollama",
            model_ids=["ollama:qwen2.5:7b-instruct"],
        )
        barrier = threading.Barrier(2)
        results: list[object] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()
        router_result = {
            "ok": True,
            "text": "Hello from the orchestrator path.",
            "provider": "ollama",
            "model": "ollama:qwen2.5:7b-instruct",
            "fallback_used": False,
            "attempts": [],
            "duration_ms": 4,
            "error_kind": None,
            "selection_reason": "router_default",
        }

        try:
            runtime.set_listening("127.0.0.1", 8765)
            runtime.mark_server_listening()
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if runtime.startup_phase == "ready":
                    break
                time.sleep(0.02)
            self.assertEqual("ready", runtime.startup_phase)
            self.assertTrue(runtime._chat_runtime_bootstrap_completed)
            with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
                runtime._router,
                "enabled",
                return_value=True,
            ), patch(
                "agent.orchestrator.route_inference",
                return_value=router_result,
            ):
                def _run_chat() -> None:
                    try:
                        barrier.wait(timeout=1.0)
                        result = runtime.chat(
                            {
                                "messages": [{"role": "user", "content": "hello from startup chat"}],
                                "source_surface": "api",
                                "session_id": "startup-session-chat",
                                "thread_id": "startup-thread-chat",
                            }
                        )
                        with results_lock:
                            results.append(result)
                    except BaseException as exc:  # pragma: no cover - failure capture
                        with results_lock:
                            errors.append(exc)

                def _run_ready_probe() -> None:
                    try:
                        barrier.wait(timeout=1.0)
                        payload = runtime.ready_status()
                        with results_lock:
                            results.append(payload)
                    except BaseException as exc:  # pragma: no cover - failure capture
                        with results_lock:
                            errors.append(exc)

                threads = [
                    threading.Thread(target=_run_chat),
                    threading.Thread(target=_run_ready_probe),
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=2.0)
        finally:
            runtime.close()

        self.assertEqual([], errors)
        self.assertEqual(2, len(results))
        chat_results = [row for row in results if isinstance(row, tuple)]
        ready_results = [row for row in results if isinstance(row, dict)]
        self.assertEqual(1, len(chat_results))
        self.assertEqual(1, len(ready_results))
        ok, body = chat_results[0]
        self.assertTrue(ok)
        assert isinstance(body, dict)
        assistant = body.get("assistant") if isinstance(body.get("assistant"), dict) else {}
        self.assertEqual("Hello from the orchestrator path.", assistant.get("content"))
        ready_payload = ready_results[0]
        self.assertTrue(bool(ready_payload.get("ok")))
        self.assertEqual("ready", ready_payload.get("startup_phase"))

    def test_health_update_during_model_policy_selection_uses_one_candidate_snapshot(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 8,
                "max_context_tokens": 32768,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "free-chat",
                "capabilities": ["chat"],
                "available": True,
                "quality_rank": 7,
                "pricing": {
                    "input_per_million_tokens": 0.0,
                    "output_per_million_tokens": 0.0,
                },
            },
        )
        runtime.set_provider_secret("openrouter", {"api_key": "sk-or-test"})
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(
            runtime,
            provider_id="ollama",
            model_ids=["ollama:qwen2.5:7b-instruct"],
        )
        runtime._health_monitor.state["providers"]["openrouter"] = {  # type: ignore[attr-defined]
            "status": "ok",
            "last_checked_at": 123,
        }
        runtime._health_monitor.state["models"]["openrouter:free-chat"] = {  # type: ignore[attr-defined]
            "provider_id": "openrouter",
            "status": "ok",
            "last_checked_at": 123,
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        truth = runtime.runtime_truth_service()
        original_snapshot = truth._router_snapshot
        entered = threading.Event()
        release = threading.Event()
        payload_holder: dict[str, object] = {}
        snapshot_cache: dict[str, dict[str, object]] = {}

        def _delayed_snapshot() -> dict[str, object]:
            cached = snapshot_cache.get("value")
            if isinstance(cached, dict):
                return dict(cached)
            snapshot = dict(original_snapshot())
            snapshot_cache["value"] = dict(snapshot)
            entered.set()
            self.assertTrue(release.wait(1.0))
            return snapshot

        with patch.object(truth, "_router_snapshot", side_effect=_delayed_snapshot):
            def _run_selection() -> None:
                payload_holder["payload"] = truth.model_policy_candidate()

            worker = threading.Thread(target=_run_selection)
            worker.start()
            self.assertTrue(entered.wait(1.0))
            runtime._health_monitor.state["providers"]["ollama"] = {  # type: ignore[attr-defined]
                "status": "down",
                "last_checked_at": 456,
            }
            runtime._health_monitor.state["models"]["ollama:qwen2.5:7b-instruct"] = {  # type: ignore[attr-defined]
                "provider_id": "ollama",
                "status": "down",
                "last_checked_at": 456,
            }
            runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]
            release.set()
            worker.join(timeout=1.0)

        payload = payload_holder.get("payload")
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        self.assertIn(str(candidate.get("model_id") or ""), {"ollama:qwen2.5:7b-instruct", "openrouter:free-chat"})
        selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
        self.assertGreaterEqual(len(selection.get("ordered_candidates") or []), 1)
        self.assertTrue(str(selection.get("decision_reason") or "").strip())
