from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config


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


def _status_payload_no_local_chat(
    *,
    allow_remote_fallback: bool = False,
    chat_model: str = "ollama:llama3",
    last_chat_model: str | None = None,
) -> dict[str, object]:
    return {
        "ok": True,
        "default_provider": "ollama",
        "chat_model": chat_model,
        "default_model": chat_model,
        "resolved_default_model": chat_model,
        "last_chat_model": last_chat_model,
        "allow_remote_fallback": bool(allow_remote_fallback),
        "safe_mode": {
            "paused": False,
            "reason": "not_paused",
            "next_retry": None,
            "cooldown_until": None,
            "last_transition_at": None,
        },
        "providers": [
            {
                "id": "ollama",
                "local": True,
                "enabled": True,
                "health": {
                    "status": "ok",
                    "last_error_kind": None,
                    "status_code": None,
                    "failure_streak": 0,
                    "cooldown_until": None,
                },
            },
            {
                "id": "openrouter",
                "local": False,
                "enabled": True,
                "health": {
                    "status": "down",
                    "last_error_kind": "provider_unavailable",
                    "status_code": 502,
                    "failure_streak": 12,
                    "cooldown_until": None,
                },
            },
        ],
        "models": [
            {
                "id": "ollama:llama3",
                "provider": "ollama",
                "enabled": True,
                "available": True,
                "routable": False,
                "capabilities": ["chat"],
                "health": {
                    "status": "down",
                    "last_error_kind": "provider_unavailable",
                    "status_code": 502,
                    "failure_streak": 12,
                    "cooldown_until": None,
                },
            }
        ],
    }


def _doctor_snapshot_local_and_remote() -> dict[str, object]:
    return {
        "providers": [
            {
                "id": "ollama",
                "local": True,
                "enabled": True,
                "health": {
                    "status": "ok",
                    "last_checked_at": 1_700_000_000,
                },
            },
            {
                "id": "openrouter",
                "local": False,
                "enabled": True,
                "health": {
                    "status": "unknown",
                    "last_checked_at": None,
                },
            },
        ],
        "models": [
            {
                "id": "ollama:llama3",
                "provider": "ollama",
                "enabled": True,
                "available": True,
                "routable": True,
                "capabilities": ["chat"],
                "health": {
                    "status": "ok",
                },
            },
            {
                "id": "openrouter:model-a",
                "provider": "openrouter",
                "enabled": True,
                "available": False,
                "routable": False,
                "capabilities": ["chat"],
                "health": {
                    "status": "unknown",
                    "last_checked_at": None,
                },
            },
            {
                "id": "openrouter:model-b",
                "provider": "openrouter",
                "enabled": True,
                "available": False,
                "routable": False,
                "capabilities": ["chat"],
                "health": {
                    "status": "unknown",
                    "last_checked_at": None,
                },
            },
        ],
    }


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.client_address = ("127.0.0.1", 12345)
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload or {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
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
        return self._payload


class TestLLMFixitEndpointFlow(unittest.TestCase):
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

    def test_fixit_choice_confirm_flow(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime._health_monitor.state["providers"] = {
            "openrouter": {
                "status": "down",
                "last_error_kind": "provider_unavailable",
                "status_code": 502,
                "last_checked_at": 100,
                "cooldown_until": 0,
                "down_since": 100,
                "failure_streak": 12,
                "next_probe_at": 160,
            },
            "ollama": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        first = _HandlerForTest(runtime, "/llm/fixit", {})
        first.do_POST()
        self.assertEqual(200, first.status_code)
        first_payload = json.loads(first.body.decode("utf-8"))
        self.assertTrue(first_payload["ok"])
        self.assertEqual("needs_user_choice", first_payload["status"])
        self.assertEqual("openrouter_down", first_payload["issue_code"])
        self.assertLessEqual(len(first_payload["choices"]), 3)

        second = _HandlerForTest(runtime, "/llm/fixit", {"answer": "1"})
        second.do_POST()
        self.assertEqual(200, second.status_code)
        second_payload = json.loads(second.body.decode("utf-8"))
        self.assertEqual("needs_confirmation", second_payload["status"])
        self.assertEqual(1, second_payload["confirm_code"])
        self.assertEqual(2, second_payload["cancel_code"])
        self.assertNotIn("confirm_token", second_payload)

        with patch.object(
            runtime,
            "_execute_llm_fixit_plan",
            return_value=(True, {"ok": True, "executed_steps": [{"id": "01"}], "blocked_steps": [], "failed_steps": []}),
        ):
            third = _HandlerForTest(
                runtime,
                "/llm/fixit",
                {"confirm": True},
            )
            third.do_POST()

        self.assertEqual(200, third.status_code)
        third_payload = json.loads(third.body.decode("utf-8"))
        self.assertTrue(third_payload["ok"])
        self.assertTrue(third_payload["did_work"])
        self.assertEqual("llm_fixit", third_payload["intent"])
        self.assertFalse(bool(runtime._llm_fixit_store.state.get("active")))  # type: ignore[attr-defined]

    def test_fixit_confirm_expired_returns_needs_clarification(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime._llm_fixit_store.save(  # type: ignore[attr-defined]
            {
                "active": True,
                "issue_hash": "x",
                "issue_code": "openrouter_down",
                "step": "awaiting_confirm",
                "question": "Apply this fix-it plan now?",
                "choices": [],
                "pending_plan": [
                    {
                        "id": "01_provider.set_enabled",
                        "kind": "safe_action",
                        "action": "provider.set_enabled",
                        "reason": "Disable OpenRouter while it is failing.",
                        "params": {"provider": "openrouter", "enabled": False},
                        "safe_to_execute": True,
                    }
                ],
                "pending_confirm_token": "token",
                "pending_created_ts": 100,
                "pending_expires_ts": 101,
                "pending_issue_code": "openrouter_down",
                "last_prompt_ts": 100,
            }
        )
        with patch("agent.api_server.time.time", return_value=200):
            handler = _HandlerForTest(runtime, "/llm/fixit", {"confirm": True})
            handler.do_POST()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual("needs_clarification", payload["error_kind"])
        self.assertIn("expired", payload["message"])

    def test_fixit_cancel_clears_pending_plan(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime._llm_fixit_store.save(  # type: ignore[attr-defined]
            {
                "active": True,
                "issue_hash": "x",
                "issue_code": "openrouter_down",
                "step": "awaiting_confirm",
                "question": "Apply this fix-it plan now?",
                "choices": [],
                "pending_plan": [
                    {
                        "id": "01_provider.set_enabled",
                        "kind": "safe_action",
                        "action": "provider.set_enabled",
                        "reason": "Disable OpenRouter while it is failing.",
                        "params": {"provider": "openrouter", "enabled": False},
                        "safe_to_execute": True,
                    }
                ],
                "pending_confirm_token": "token",
                "pending_created_ts": 100,
                "pending_expires_ts": 400,
                "pending_issue_code": "openrouter_down",
                "last_prompt_ts": 100,
            }
        )
        handler = _HandlerForTest(runtime, "/llm/fixit", {"answer": "cancel"})
        handler.do_POST()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual("cancelled", payload["status"])
        self.assertEqual("Cancelled.", payload["message"])
        self.assertFalse(bool(runtime._llm_fixit_store.state.get("active")))  # type: ignore[attr-defined]

    def test_openrouter_repair_persists_last_test_and_updates_choices(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.set_provider_secret("openrouter", {"api_key": "sk-openrouter-test"})
        runtime._health_monitor.state["providers"] = {
            "openrouter": {
                "status": "down",
                "last_error_kind": "provider_unavailable",
                "status_code": 502,
                "last_checked_at": 100,
                "cooldown_until": 0,
                "down_since": 100,
                "failure_streak": 12,
                "next_probe_at": 160,
            },
            "ollama": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        first = _HandlerForTest(runtime, "/llm/fixit", {})
        first.do_POST()
        first_payload = json.loads(first.body.decode("utf-8"))
        self.assertEqual("openrouter_down", first_payload["issue_code"])
        self.assertEqual("repair_openrouter", first_payload["choices"][1]["id"])

        second = _HandlerForTest(runtime, "/llm/fixit", {"answer": "2"})
        second.do_POST()
        second_payload = json.loads(second.body.decode("utf-8"))
        self.assertEqual("needs_confirmation", second_payload["status"])

        with patch.object(
            runtime,
            "test_provider",
            return_value=(
                False,
                {
                    "ok": False,
                    "provider": "openrouter",
                    "error": "auth_error",
                    "error_kind": "auth_error",
                    "status_code": 401,
                    "message": "Authentication failed for provider.",
                },
            ),
        ):
            third = _HandlerForTest(runtime, "/llm/fixit", {"confirm": True})
            third.do_POST()
        third_payload = json.loads(third.body.decode("utf-8"))
        self.assertEqual("needs_user_choice", third_payload["status"])
        self.assertEqual("openrouter_down", third_payload["issue_code"])
        self.assertEqual("update_openrouter_key", third_payload["choices"][1]["id"])
        state_last_test = runtime._llm_fixit_store.state.get("openrouter_last_test")  # type: ignore[attr-defined]
        self.assertIsInstance(state_last_test, dict)
        self.assertEqual(401, state_last_test.get("status_code"))
        self.assertEqual("auth_error", state_last_test.get("error_kind"))

    def test_fixit_local_only_apply_converges_and_does_not_loop(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime._health_monitor.state["providers"] = {
            "openrouter": {
                "status": "down",
                "last_error_kind": "provider_unavailable",
                "status_code": 502,
                "last_checked_at": 100,
                "cooldown_until": 0,
                "down_since": 100,
                "failure_streak": 12,
                "next_probe_at": 160,
            },
            "ollama": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
        }
        runtime._health_monitor.state["models"] = {
            "ollama:llama3": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
            "openrouter:openai/gpt-4o-mini": {
                "status": "down",
                "last_error_kind": "provider_unavailable",
                "status_code": 502,
                "last_checked_at": 100,
                "cooldown_until": 0,
                "down_since": 100,
                "failure_streak": 12,
                "next_probe_at": 160,
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        first = _HandlerForTest(runtime, "/llm/fixit", {})
        first.do_POST()
        first_payload = json.loads(first.body.decode("utf-8"))
        self.assertEqual(200, first.status_code)
        self.assertEqual("needs_user_choice", first_payload["status"])
        self.assertEqual("openrouter_down", first_payload["issue_code"])

        second = _HandlerForTest(runtime, "/llm/fixit", {"answer": "1"})
        second.do_POST()
        second_payload = json.loads(second.body.decode("utf-8"))
        self.assertEqual(200, second.status_code)
        self.assertEqual("needs_confirmation", second_payload["status"])

        third = _HandlerForTest(runtime, "/llm/fixit", {"confirm": True})
        third.do_POST()
        third_payload = json.loads(third.body.decode("utf-8"))
        self.assertEqual(200, third.status_code)
        self.assertTrue(third_payload["ok"])
        self.assertTrue(third_payload["did_work"])
        self.assertFalse(bool(runtime.get_defaults()["allow_remote_fallback"]))

        fourth = _HandlerForTest(runtime, "/llm/fixit", {})
        fourth.do_POST()
        fourth_payload = json.loads(fourth.body.decode("utf-8"))
        self.assertEqual(200, fourth.status_code)
        self.assertEqual("ok", fourth_payload["status"])
        self.assertEqual("ok", fourth_payload["issue_code"])

    def test_fixit_accepts_openrouter_api_key_payload(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime._health_monitor.state["providers"] = {
            "openrouter": {
                "status": "down",
                "last_error_kind": "provider_unavailable",
                "status_code": 502,
                "last_checked_at": 100,
                "cooldown_until": 0,
                "down_since": 100,
                "failure_streak": 12,
                "next_probe_at": 160,
            }
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]
        with patch.object(
            runtime,
            "test_provider",
            return_value=(
                False,
                {
                    "ok": False,
                    "provider": "openrouter",
                    "error": "payment_required",
                    "error_kind": "payment_required",
                    "status_code": 402,
                    "message": "Provider test hit a credits/limit issue.",
                },
            ),
        ):
            handler = _HandlerForTest(runtime, "/llm/fixit", {"openrouter_api_key": "sk-or-new"})
            handler.do_POST()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertEqual("needs_user_choice", payload["status"])
        self.assertEqual("switch_provider", payload["choices"][1]["id"])
        self.assertEqual(
            "sk-or-new",
            runtime.secret_store.get_secret("provider:openrouter:api_key"),
        )

    def test_ollama_pull_endpoint_happy_path_and_allowlist(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        with patch(
            "agent.api_server.build_model_inventory",
            return_value=[],
        ), patch(
            "agent.api_server.execute_install_plan",
            return_value={
                "ok": True,
                "executed": True,
                "model_id": "ollama:qwen2.5:3b-instruct",
                "install_name": "qwen2.5:3b-instruct",
                "trace_id": "install-1",
                "error_kind": None,
                "message": "Installed and verified ollama:qwen2.5:3b-instruct.",
                "verification": {
                    "found": True,
                    "installed": True,
                    "available": True,
                    "healthy": True,
                    "verification_status": "ok",
                },
                "stdout_tail": "",
                "stderr_tail": "",
            },
        ) as install_mock, patch.object(
            runtime.modelops_executor.safe_runner,
            "run",
        ) as direct_pull_mock, patch.object(
            runtime,
            "refresh_models",
            return_value=(True, {"ok": True}),
        ):
            handler = _HandlerForTest(runtime, "/providers/ollama/pull", {"model": "qwen2.5:3b-instruct", "confirm": True})
            handler.do_POST()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual("qwen2.5:3b-instruct", payload["model"])
        self.assertFalse(bool(payload["already_present"]))
        self.assertEqual("install-1", payload["trace_id"])
        install_mock.assert_called_once()
        self.assertTrue(bool(install_mock.call_args.kwargs["approve"]))
        self.assertEqual(
            "qwen2.5:3b-instruct",
            install_mock.call_args.kwargs["plan"]["candidates"][0]["install_name"],
        )
        direct_pull_mock.assert_not_called()

        disallowed = _HandlerForTest(runtime, "/providers/ollama/pull", {"model": "llama3:8b"})
        disallowed.do_POST()
        self.assertEqual(400, disallowed.status_code)
        disallowed_payload = json.loads(disallowed.body.decode("utf-8"))
        self.assertFalse(disallowed_payload["ok"])
        self.assertEqual("model_not_allowed", disallowed_payload["error_kind"])

        non_loopback = _HandlerForTest(runtime, "/providers/ollama/pull", {"model": "qwen2.5:3b-instruct"})
        non_loopback.client_address = ("192.168.1.2", 4567)
        non_loopback.do_POST()
        self.assertEqual(403, non_loopback.status_code)
        non_loopback_payload = json.loads(non_loopback.body.decode("utf-8"))
        self.assertEqual("forbidden", non_loopback_payload["error_kind"])

    def test_ollama_pull_endpoint_uses_canonical_result_for_noop_and_timeout(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)

        with patch(
            "agent.api_server.build_model_inventory",
            return_value=[],
        ), patch(
            "agent.api_server.execute_install_plan",
            return_value={
                "ok": True,
                "executed": False,
                "model_id": "ollama:qwen2.5:3b-instruct",
                "install_name": "qwen2.5:3b-instruct",
                "trace_id": "install-noop",
                "error_kind": None,
                "message": "Model already installed and healthy.",
                "verification": {
                    "found": True,
                    "installed": True,
                    "available": True,
                    "healthy": True,
                    "verification_status": "ok",
                },
                "stdout_tail": "",
                "stderr_tail": "",
            },
        ), patch.object(
            runtime.modelops_executor.safe_runner,
            "run",
        ) as direct_pull_mock, patch.object(
            runtime,
            "refresh_models",
            return_value=(True, {"ok": True}),
        ):
            handler = _HandlerForTest(runtime, "/providers/ollama/pull", {"model": "ollama:qwen2.5:3b-instruct", "confirm": True})
            handler.do_POST()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertTrue(bool(payload["already_present"]))
        direct_pull_mock.assert_not_called()

        with patch(
            "agent.api_server.build_model_inventory",
            return_value=[],
        ), patch(
            "agent.api_server.execute_install_plan",
            return_value={
                "ok": False,
                "executed": True,
                "model_id": "ollama:qwen2.5:3b-instruct",
                "install_name": "qwen2.5:3b-instruct",
                "trace_id": "install-timeout",
                "error_kind": "timed_out",
                "message": "Ollama pull timed out.",
                "verification": {},
                "stdout_tail": "",
                "stderr_tail": "",
            },
        ):
            timeout_handler = _HandlerForTest(runtime, "/providers/ollama/pull", {"model": "qwen2.5:3b-instruct", "confirm": True})
            timeout_handler.do_POST()
        self.assertEqual(400, timeout_handler.status_code)
        timeout_payload = json.loads(timeout_handler.body.decode("utf-8"))
        self.assertFalse(timeout_payload["ok"])
        self.assertEqual("timed_out", timeout_payload["error_kind"])

    def test_ollama_pull_endpoint_requires_explicit_approval(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)

        with patch(
            "agent.api_server.build_model_inventory",
            return_value=[],
        ), patch(
            "agent.api_server.execute_install_plan",
            return_value={
                "ok": False,
                "executed": False,
                "model_id": "ollama:qwen2.5:3b-instruct",
                "install_name": "qwen2.5:3b-instruct",
                "trace_id": "install-approval",
                "error_kind": "approval_required",
                "message": "Explicit approval is required before executing this local install.",
                "verification": {},
                "stdout_tail": "",
                "stderr_tail": "",
            },
        ) as install_mock, patch.object(
            runtime.modelops_executor.safe_runner,
            "run",
        ) as direct_pull_mock:
            handler = _HandlerForTest(runtime, "/providers/ollama/pull", {"model": "qwen2.5:3b-instruct"})
            handler.do_POST()
        self.assertEqual(400, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertFalse(payload["ok"])
        self.assertEqual("approval_required", payload["error_kind"])
        self.assertFalse(bool(install_mock.call_args.kwargs["approve"]))
        direct_pull_mock.assert_not_called()

    def test_fixit_offers_install_local_models_and_applies_install_plan(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_defaults({"allow_remote_fallback": False})

        with patch.object(runtime, "llm_status", return_value=_status_payload_no_local_chat(allow_remote_fallback=False)):
            first = _HandlerForTest(runtime, "/llm/fixit", {})
            first.do_POST()
        self.assertEqual(200, first.status_code)
        first_payload = json.loads(first.body.decode("utf-8"))
        self.assertEqual("needs_user_choice", first_payload["status"])
        self.assertEqual("no_routable_model", first_payload["issue_code"])
        choice_ids = [str(row.get("id") or "") for row in first_payload.get("choices", [])]
        self.assertEqual(
            ["install_local_small", "install_local_medium", "details"],
            choice_ids,
        )

        with patch.object(runtime, "llm_status", return_value=_status_payload_no_local_chat(allow_remote_fallback=False)):
            second = _HandlerForTest(runtime, "/llm/fixit", {"answer": "1"})
            second.do_POST()
        self.assertEqual(200, second.status_code)
        second_payload = json.loads(second.body.decode("utf-8"))
        self.assertEqual("needs_confirmation", second_payload["status"])
        self.assertIn("qwen2.5:3b-instruct", second_payload["message"])

        runtime.registry_document.setdefault("models", {})
        runtime.registry_document["models"]["ollama:qwen2.5:3b-instruct"] = {
            "provider": "ollama",
            "model": "qwen2.5:3b-instruct",
            "capabilities": ["chat"],
            "quality_rank": 2,
            "cost_rank": 0,
            "default_for": ["chat"],
            "enabled": True,
            "available": True,
            "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            "max_context_tokens": None,
        }

        with patch.object(runtime, "llm_status", return_value=_status_payload_no_local_chat(allow_remote_fallback=False)), patch.object(
            runtime,
            "pull_ollama_model",
            return_value=(
                True,
                {
                    "ok": True,
                    "model": "qwen2.5:3b-instruct",
                    "canonical_model": "ollama:qwen2.5:3b-instruct",
                    "already_present": True,
                    "duration_ms": 1,
                    "message": "already installed",
                },
            ),
        ) as pull_mock, patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True, "provider": "ollama", "model": "qwen2.5:3b-instruct"}),
        ) as test_provider_mock:
            third = _HandlerForTest(runtime, "/llm/fixit", {"confirm": True})
            third.do_POST()

        self.assertEqual(200, third.status_code)
        third_payload = json.loads(third.body.decode("utf-8"))
        self.assertTrue(third_payload["ok"])
        self.assertIn("Installed and configured", third_payload["message"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", third_payload["chat_model"])
        pull_mock.assert_called_once_with({"model": "qwen2.5:3b-instruct", "confirm": True})
        test_provider_mock.assert_called_once()
        called_payload = test_provider_mock.call_args.args[1]
        self.assertEqual("ollama:qwen2.5:3b-instruct", called_payload.get("model"))
        self.assertNotIn("OpenRouter", first_payload["message"])

    def test_defaults_rollback_endpoint_swaps_chat_and_last_chat_model(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.registry_document.setdefault("models", {})
        runtime.registry_document["models"]["ollama:qwen2.5:3b-instruct"] = {
            "provider": "ollama",
            "model": "qwen2.5:3b-instruct",
            "capabilities": ["chat"],
            "quality_rank": 2,
            "cost_rank": 0,
            "default_for": ["chat"],
            "enabled": True,
            "available": True,
            "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            "max_context_tokens": None,
        }
        runtime._save_registry_document(runtime.registry_document)
        runtime.update_defaults({"chat_model": "ollama:llama3"})
        runtime.update_defaults({"chat_model": "ollama:qwen2.5:3b-instruct"})

        handler = _HandlerForTest(runtime, "/defaults/rollback", {})
        handler.do_POST()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual("ollama:llama3", payload["chat_model"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", payload["last_chat_model"])

    def test_fixit_undo_last_chat_model_choice_executes_rollback(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.registry_document.setdefault("models", {})
        runtime.registry_document["models"]["ollama:qwen2.5:3b-instruct"] = {
            "provider": "ollama",
            "model": "qwen2.5:3b-instruct",
            "capabilities": ["chat"],
            "quality_rank": 2,
            "cost_rank": 0,
            "default_for": ["chat"],
            "enabled": True,
            "available": True,
            "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            "max_context_tokens": None,
        }
        runtime._save_registry_document(runtime.registry_document)
        runtime.update_defaults({"chat_model": "ollama:llama3"})
        runtime.update_defaults({"chat_model": "ollama:qwen2.5:3b-instruct"})

        with patch.object(
            runtime,
            "llm_status",
            return_value=_status_payload_no_local_chat(
                allow_remote_fallback=False,
                chat_model="ollama:qwen2.5:3b-instruct",
                last_chat_model="ollama:llama3",
            ),
        ):
            first = _HandlerForTest(runtime, "/llm/fixit", {})
            first.do_POST()
        first_payload = json.loads(first.body.decode("utf-8"))
        self.assertEqual("needs_user_choice", first_payload["status"])
        self.assertEqual("rollback_chat_model", first_payload["choices"][2]["id"])

        with patch.object(
            runtime,
            "llm_status",
            return_value=_status_payload_no_local_chat(
                allow_remote_fallback=False,
                chat_model="ollama:qwen2.5:3b-instruct",
                last_chat_model="ollama:llama3",
            ),
        ):
            second = _HandlerForTest(runtime, "/llm/fixit", {"answer": "3"})
            second.do_POST()
        second_payload = json.loads(second.body.decode("utf-8"))
        self.assertEqual("needs_confirmation", second_payload["status"])
        self.assertIn("roll back to ollama:llama3", second_payload["message"].lower())

        with patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True, "provider": "ollama", "model": "ollama:llama3"}),
        ):
            third = _HandlerForTest(runtime, "/llm/fixit", {"confirm": True})
            third.do_POST()
        third_payload = json.loads(third.body.decode("utf-8"))
        self.assertEqual(200, third.status_code)
        self.assertTrue(third_payload["ok"])
        self.assertIn("Rolled back chat model", third_payload["message"])
        self.assertEqual("ollama:llama3", third_payload["chat_model"])

    def test_llm_status_endpoint_returns_defaults_and_health(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        handler = _HandlerForTest(runtime, "/llm/status")
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertIn("default_provider", payload)
        self.assertIn("default_model", payload)
        self.assertIn("active_provider_health", payload)
        self.assertIn("active_model_health", payload)
        self.assertIn("safe_mode", payload)

    def test_llm_status_filters_remote_models_when_local_only(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_defaults({"allow_remote_fallback": False})
        snapshot = _doctor_snapshot_local_and_remote()
        with patch.object(runtime, "llm_health_summary", return_value={"ok": True, "health": {"drift": {"details": {}}}}), patch.object(
            runtime._router,
            "doctor_snapshot",
            return_value=snapshot,
        ):
            handler = _HandlerForTest(runtime, "/llm/status")
            handler.do_GET()

        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertFalse(bool(payload["allow_remote_fallback"]))
        providers = {str(row.get("provider") or "").strip().lower() for row in payload.get("models", []) if isinstance(row, dict)}
        self.assertEqual({"ollama"}, providers)
        self.assertEqual(2, int((payload.get("hidden_models_by_provider") or {}).get("openrouter") or 0))
        self.assertEqual(3, int(payload.get("total_models_count") or 0))
        self.assertEqual(1, int(payload.get("visible_models_count") or 0))
        self.assertEqual(1, int((payload.get("visible_counts") or {}).get("total") or 0))

    def test_llm_status_keeps_remote_models_when_remote_fallback_enabled(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.update_defaults({"allow_remote_fallback": True})
        snapshot = _doctor_snapshot_local_and_remote()
        with patch.object(runtime, "llm_health_summary", return_value={"ok": True, "health": {"drift": {"details": {}}}}), patch.object(
            runtime._router,
            "doctor_snapshot",
            return_value=snapshot,
        ):
            handler = _HandlerForTest(runtime, "/llm/status")
            handler.do_GET()

        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertTrue(bool(payload["allow_remote_fallback"]))
        providers = [str(row.get("provider") or "").strip().lower() for row in payload.get("models", []) if isinstance(row, dict)]
        self.assertIn("openrouter", providers)
        self.assertEqual({}, payload.get("hidden_models_by_provider"))
        self.assertEqual(3, int(payload.get("total_models_count") or 0))
        self.assertEqual(3, int(payload.get("visible_models_count") or 0))

    def test_fixit_rollback_choice_without_target_returns_clear_message(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime._llm_fixit_store.save(
            {
                "active": True,
                "issue_hash": "issue-no-rollback",
                "issue_code": "no_routable_model",
                "step": "awaiting_choice",
                "question": "Which option should I take?",
                "choices": [
                    {"id": "install_local_small", "label": "Install small local model", "recommended": True},
                    {"id": "install_local_medium", "label": "Install medium local model", "recommended": False},
                    {"id": "rollback_chat_model", "label": "Undo last chat model change", "recommended": False},
                ],
                "pending_plan": [],
                "pending_confirm_token": None,
                "pending_created_ts": None,
                "pending_expires_ts": None,
                "pending_issue_code": None,
                "last_prompt_ts": 1_700_000_001,
                "openrouter_last_test": None,
            }
        )
        with patch.object(
            runtime,
            "llm_status",
            return_value=_status_payload_no_local_chat(
                allow_remote_fallback=False,
                chat_model="ollama:qwen2.5:3b-instruct",
                last_chat_model=None,
            ),
        ):
            handler = _HandlerForTest(runtime, "/llm/fixit", {"answer": "3"})
            handler.do_POST()

        self.assertEqual(200, handler.status_code)
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual("needs_clarification", payload["status"])
        self.assertEqual("No rollback available yet.", payload["message"])


if __name__ == "__main__":
    unittest.main()
