from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import load_config
from agent.runtime_lifecycle import RuntimeLifecyclePhase, derive_runtime_lifecycle_phase


def _config(registry_path: str, db_path: str, **overrides: object):
    cfg = load_config(require_telegram_token=False)
    base = replace(
        cfg,
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        llm_registry_path=registry_path,
        llm_automation_enabled=False,
        telegram_enabled=False,
    )
    return replace(base, **overrides)


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""

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


class TestRuntimeLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self, **config_overrides: object) -> AgentRuntime:
        return AgentRuntime(_config(self.registry_path, self.db_path, **config_overrides))

    @staticmethod
    def _seed_ok_health(runtime: AgentRuntime, *, provider_id: str, model_id: str) -> None:
        provider_key = str(provider_id).strip().lower()
        runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
            provider_key: {"status": "ok", "last_checked_at": 123}
        }
        runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
            model_id: {"provider_id": provider_key, "status": "ok", "last_checked_at": 123}
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

    @staticmethod
    def _telegram_payload() -> dict[str, object]:
        return {
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

    def test_runtime_phase_detection_during_startup(self) -> None:
        self.assertEqual(
            RuntimeLifecyclePhase.BOOT,
            derive_runtime_lifecycle_phase(
                startup_phase="starting",
                warmup_remaining=[],
                startup_warmup_started=False,
                runtime_ready=False,
                runtime_mode="DEGRADED",
            ),
        )
        self.assertEqual(
            RuntimeLifecyclePhase.WARMUP,
            derive_runtime_lifecycle_phase(
                startup_phase="warming",
                warmup_remaining=["router_reload"],
                startup_warmup_started=True,
                runtime_ready=False,
                runtime_mode="DEGRADED",
            ),
        )
        self.assertEqual(
            RuntimeLifecyclePhase.READY,
            derive_runtime_lifecycle_phase(
                startup_phase="ready",
                warmup_remaining=[],
                startup_warmup_started=True,
                runtime_ready=True,
                runtime_mode="READY",
                active_provider_health={"status": "ok"},
                active_model_health={"status": "ok"},
            ),
        )
        self.assertEqual(
            RuntimeLifecyclePhase.DEGRADED,
            derive_runtime_lifecycle_phase(
                startup_phase="ready",
                warmup_remaining=[],
                startup_warmup_started=True,
                runtime_ready=False,
                runtime_mode="DEGRADED",
                active_provider_health={"status": "down"},
                active_model_health={"status": "down"},
            ),
        )

    def test_safe_mode_frontdoor_containment_can_stay_active_when_chat_target_is_not_ready(self) -> None:
        runtime = self._runtime(
            safe_mode_enabled=True,
            safe_mode_chat_model="ollama:qwen3.5:4b",
        )
        truth = runtime.runtime_truth_service()

        with patch.object(
            truth,
            "current_chat_target_status",
            return_value={
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "ready": False,
                "health_status": "down",
                "provider_health_status": "ok",
            },
        ), patch.object(
            truth,
            "chat_target_truth",
            return_value={
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "configured_ready": False,
                "effective_provider": "ollama",
                "effective_model": "ollama:qwen3.5:4b",
                "effective_ready": False,
            },
        ), patch.object(
            truth,
            "model_inventory_status",
            return_value={
                "active_provider": "ollama",
                "active_model": "ollama:qwen3.5:4b",
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "usable_now": False,
                    }
                ],
            },
        ), patch.object(
            truth,
            "model_readiness_status",
            return_value={
                "active_provider": "ollama",
                "active_model": "ollama:qwen3.5:4b",
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "usable_now": False,
                    }
                ],
                "ready_now_models": [],
                "other_ready_now_models": [],
                "not_ready_models": [
                    {
                        "model_id": "ollama:qwen3.5:4b",
                        "provider_id": "ollama",
                        "local": True,
                        "available": True,
                        "usable_now": False,
                    }
                ],
            },
        ):
            self.assertTrue(runtime.assistant_frontdoor_active())
            self.assertFalse(runtime.assistant_chat_available())
        self.assertEqual(
            RuntimeLifecyclePhase.RECOVERING,
            derive_runtime_lifecycle_phase(
                startup_phase="ready",
                warmup_remaining=[],
                startup_warmup_started=True,
                runtime_ready=True,
                runtime_mode="READY",
                active_provider_health={"status": "ok"},
                active_model_health={"status": "ok"},
                previous_phase=RuntimeLifecyclePhase.DEGRADED,
            ),
        )

    def test_ready_includes_runtime_lifecycle_phase(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(runtime, provider_id="ollama", model_id="ollama:qwen2.5:7b-instruct")
        runtime.startup_phase = "ready"
        with patch.object(runtime, "telegram_status", return_value=self._telegram_payload()):
            payload = runtime.ready_status()
        self.assertEqual("ready", payload["phase"])
        self.assertEqual("ready", payload["startup_phase"])
        self.assertTrue(bool(payload["ready"]))
        self.assertEqual("ollama:qwen2.5:7b-instruct", payload["llm"]["model"])

    def test_runtime_endpoint_returns_expected_snapshot_structure(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(runtime, provider_id="ollama", model_id="ollama:qwen2.5:7b-instruct")
        runtime.startup_phase = "ready"
        with patch.object(runtime, "telegram_status", return_value=self._telegram_payload()):
            handler = _HandlerForTest(runtime, "/runtime")
            handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))
        self.assertEqual(200, handler.status_code)
        self.assertTrue(bool(payload["ok"]))
        self.assertEqual("ready", payload["phase"])
        self.assertEqual("ollama:qwen2.5:7b-instruct", payload["default_chat_model"])
        self.assertIn("runtime_status", payload)
        self.assertEqual(payload["runtime_mode"], payload["runtime_status"]["runtime_mode"])
        self.assertIn("control_mode", payload)
        self.assertIn("blocked", payload)
        self.assertIn("providers", payload)
        self.assertIn("router", payload)
        self.assertIn("health_summary", payload)
        self.assertIn("telegram", payload)
        self.assertIn("ollama", payload["providers"])

    def test_safe_mode_valid_pin_is_reported_as_effective_target(self) -> None:
        runtime = self._runtime(
            safe_mode_enabled=True,
            safe_mode_chat_model="ollama:qwen3.5:4b",
            ollama_model="qwen2.5:7b-instruct",
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.update_defaults({"default_provider": "ollama", "chat_model": "ollama:qwen2.5:7b-instruct"})
        self._seed_ok_health(runtime, provider_id="ollama", model_id="ollama:qwen3.5:4b")
        runtime.startup_phase = "ready"
        with patch.object(runtime, "telegram_status", return_value=self._telegram_payload()):
            ready_payload = runtime.ready_status()
            snapshot = runtime.runtime_snapshot()
        self.assertEqual("ollama:qwen3.5:4b", ready_payload["llm"]["model"])
        self.assertEqual("ollama:qwen3.5:4b", snapshot["default_chat_model"])
        self.assertEqual("ollama:qwen3.5:4b", ready_payload["safe_mode_target"]["effective_model"])
        self.assertTrue(bool(ready_payload["safe_mode_target"]["configured_valid"]))
        self.assertEqual("configured_pin", ready_payload["safe_mode_target"]["reason"])

    def test_safe_mode_invalid_pin_is_reported_clearly(self) -> None:
        runtime = self._runtime(
            safe_mode_enabled=True,
            safe_mode_chat_model="ollama:missing-model",
            ollama_model="qwen3.5:4b",
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        self._seed_ok_health(runtime, provider_id="ollama", model_id="ollama:qwen3.5:4b")
        runtime.startup_phase = "ready"
        with patch.object(runtime, "telegram_status", return_value=self._telegram_payload()):
            ready_payload = runtime.ready_status()
            snapshot = runtime.runtime_snapshot()
        self.assertEqual("ollama:qwen3.5:4b", ready_payload["llm"]["model"])
        self.assertEqual("ollama:qwen3.5:4b", snapshot["default_chat_model"])
        self.assertFalse(bool(ready_payload["safe_mode_target"]["configured_valid"]))
        self.assertEqual("configured_pin_invalid", ready_payload["safe_mode_target"]["reason"])
        self.assertIn("Safe mode pin", ready_payload["safe_mode_target"]["message"])
        self.assertIn("falling back", ready_payload["message"].lower())

    def test_safe_mode_installed_pin_is_not_reported_unavailable(self) -> None:
        runtime = self._runtime(
            safe_mode_enabled=True,
            safe_mode_chat_model="ollama:qwen2.5:3b-instruct",
            ollama_model="qwen3.5:4b",
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        self._seed_ok_health(runtime, provider_id="ollama", model_id="ollama:qwen3.5:4b")
        runtime.startup_phase = "ready"

        def _fake_get(url: str, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            if url.endswith("/api/tags"):
                return {"models": [{"name": "qwen3.5:4b"}, {"name": "qwen2.5:3b-instruct"}]}
            return {}

        runtime._http_get_json = _fake_get  # type: ignore[assignment]

        with patch.object(runtime, "telegram_status", return_value=self._telegram_payload()):
            ready_payload = runtime.ready_status()

        safe_mode_target = ready_payload["safe_mode_target"]
        self.assertTrue(bool(safe_mode_target["configured_valid"]))
        self.assertEqual("configured_pin_not_ready", safe_mode_target["reason"])
        self.assertIn("installed", str(safe_mode_target["message"]).lower())
        self.assertNotIn("unavailable", str(safe_mode_target["message"]).lower())
        self.assertEqual("ollama:qwen3.5:4b", safe_mode_target["effective_model"])

    def test_safe_mode_bare_pin_resolves_against_live_runtime_inventory(self) -> None:
        runtime = self._runtime(
            safe_mode_enabled=True,
            safe_mode_chat_model="qwen2.5:3b-instruct",
            ollama_model="qwen3.5:4b",
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.startup_phase = "ready"
        live_models = {
            "ollama:qwen2.5:3b-instruct": {
                "provider": "ollama",
                "enabled": True,
                "available": True,
                "capabilities": ["chat"],
            },
            "ollama:qwen3.5:4b": {
                "provider": "ollama",
                "enabled": True,
                "available": True,
                "capabilities": ["chat"],
            },
        }
        live_providers = {
            "ollama": {
                "enabled": True,
                "local": True,
            }
        }

        with patch.object(
            runtime,
            "_runtime_snapshot_target_documents",
            return_value=(live_models, live_providers, {"ollama"}),
        ):
            target = runtime.safe_mode_target_status()
            defaults = runtime.get_defaults()

        self.assertTrue(bool(target["configured_valid"]))
        self.assertEqual("configured_pin", target["reason"])
        self.assertEqual("ollama", target["effective_provider"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", target["effective_model"])
        self.assertEqual("ollama", defaults["default_provider"])
        self.assertEqual("ollama:qwen2.5:3b-instruct", defaults["resolved_default_model"])

    def test_runtime_endpoint_remains_stable_during_concurrent_health_update(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.add_provider(
            {
                "id": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_source": {"type": "secret", "name": "provider:openrouter:api_key"},
            }
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.secret_store.set_secret("provider:openrouter:api_key", "sk-test")
        runtime.set_default_chat_model("ollama:qwen2.5:7b-instruct")
        self._seed_ok_health(runtime, provider_id="ollama", model_id="ollama:qwen2.5:7b-instruct")
        runtime.startup_phase = "ready"

        entered = threading.Event()
        release = threading.Event()
        payload_holder: dict[str, object] = {}
        original_provider_api_key = runtime._provider_api_key

        delayed_once = {"value": False}

        def _delayed_provider_api_key(provider_payload: dict[str, object], *, key_override: str | None = None) -> str | None:
            if not delayed_once["value"]:
                delayed_once["value"] = True
                entered.set()
                self.assertTrue(release.wait(1.0))
            return original_provider_api_key(provider_payload, key_override=key_override)

        with patch.object(runtime, "telegram_status", return_value=self._telegram_payload()), patch.object(
            runtime,
            "_provider_api_key",
            side_effect=_delayed_provider_api_key,
        ):
            def _run_snapshot() -> None:
                payload_holder["payload"] = runtime.runtime_snapshot()

            worker = threading.Thread(target=_run_snapshot)
            worker.start()
            self.assertTrue(entered.wait(1.0))
            runtime._health_monitor.state["providers"] = {  # type: ignore[attr-defined]
                "ollama": {"status": "down", "last_checked_at": 456},
                "openrouter": {"status": "ok", "last_checked_at": 456},
            }
            runtime._health_monitor.state["models"] = {  # type: ignore[attr-defined]
                "ollama:qwen2.5:7b-instruct": {"provider_id": "ollama", "status": "down", "last_checked_at": 456},
                "openrouter:openai/gpt-4o-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 456},
            }
            release.set()
            worker.join(timeout=1.0)

        payload = payload_holder.get("payload")
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual("ready", payload["phase"])
        self.assertIn(payload["providers"]["ollama"], {"ok", "down"})
        self.assertIn("router", payload)
        self.assertIn("health_summary", payload)


if __name__ == "__main__":
    unittest.main()
