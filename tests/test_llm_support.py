from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.model_watch import normalize_model_watch_state


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
        model_watch_state_path=os.path.join(os.path.dirname(db_path), "model_watch_state.json"),
        autopilot_notify_store_path=os.path.join(os.path.dirname(db_path), "llm_notifications.json"),
    )
    return base.__class__(**{**base.__dict__, **overrides})


class _HandlerForTest(APIServerHandler):
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
        return self._payload


class TestLLMSupport(unittest.TestCase):
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

    def test_support_bundle_redacts_secrets_and_is_deterministic(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_provider_secret("openrouter", {"api_key": "sk-openrouterSECRET12345678901234567890"})
        runtime.update_provider(
            "openrouter",
            {
                "base_url": "https://user:pass@openrouter.ai/api/v1?token=super-secret",
                "default_headers": {"Authorization": "Bearer sk-verysecret12345678901234567890"},
                "default_query_params": {"api_key": "sk-hidden09876543210987654321"},
            },
        )

        first = runtime.llm_support_bundle()
        second = runtime.llm_support_bundle()
        self.assertEqual(first, second)
        self.assertTrue(first["ok"])
        self.assertIn("bundle", first)

        bundle = first["bundle"]
        self.assertIn("policies", bundle)
        self.assertIn("notify_test", bundle["policies"])
        self.assertIn("notify_send", bundle["policies"])
        rendered = json.dumps(bundle, ensure_ascii=True, sort_keys=True)
        self.assertNotIn("sk-openrouterSECRET12345678901234567890", rendered)
        self.assertNotIn("sk-verysecret12345678901234567890", rendered)
        self.assertNotIn("sk-hidden09876543210987654321", rendered)
        self.assertNotIn("user:pass@", rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_support_diagnose_provider_and_model(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        provider_ok, provider_payload = runtime.llm_support_diagnose("openrouter")
        self.assertTrue(provider_ok)
        self.assertTrue(provider_payload["ok"])
        self.assertEqual("provider", provider_payload["kind"])
        self.assertIn("missing_auth", provider_payload["diagnosis"]["root_causes"])
        self.assertTrue(provider_payload["diagnosis"]["recommended_actions"])

        model_id = sorted(runtime.registry_document.get("models", {}).keys())[0]
        updated_document = copy.deepcopy(runtime.registry_document)
        updated_document["models"][model_id]["enabled"] = False
        saved, error = runtime._persist_registry_document(updated_document)
        self.assertTrue(saved)
        self.assertIsNone(error)

        model_ok, model_payload = runtime.llm_support_diagnose(model_id)
        self.assertTrue(model_ok)
        self.assertTrue(model_payload["ok"])
        self.assertEqual("model", model_payload["kind"])
        self.assertIn("model_disabled", model_payload["diagnosis"]["root_causes"])
        self.assertTrue(model_payload["diagnosis"]["recommended_actions"])

    def test_support_remediation_plan_is_plan_only_and_no_side_effects(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        before_hash = runtime._registry_hash(runtime.registry_document)
        before_ledger = runtime._action_ledger.recent(limit=200)  # type: ignore[attr-defined]

        ok, payload = runtime.llm_support_remediate_plan({"target": "defaults", "intent": "fix_routing"})
        self.assertTrue(ok)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["plan"]["plan_only"])
        self.assertTrue(payload["plan"]["steps"])

        after_hash = runtime._registry_hash(runtime.registry_document)
        after_ledger = runtime._action_ledger.recent(limit=200)  # type: ignore[attr-defined]
        self.assertEqual(before_hash, after_hash)
        self.assertEqual(before_ledger, after_ledger)

    def test_support_remediation_payment_required_disables_openrouter_and_prefers_local(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        updated_document = copy.deepcopy(runtime.registry_document)
        updated_document["defaults"]["default_provider"] = "openrouter"
        updated_document["defaults"]["default_model"] = "openrouter:openai/gpt-4o-mini"
        saved, error = runtime._persist_registry_document(updated_document)
        self.assertTrue(saved)
        self.assertIsNone(error)
        runtime._health_monitor.state = {  # type: ignore[attr-defined]
            "providers": {
                "openrouter": {"status": "down", "last_error_kind": "payment_required", "status_code": 402},
                "ollama": {"status": "ok"},
            },
            "models": {
                "ollama:llama3": {"status": "ok", "last_error_kind": None},
            },
        }

        ok, payload = runtime.llm_support_remediate_plan(
            {"target": "defaults", "intent": "fix_routing", "last_error_kind": "payment_required", "status_code": 402}
        )
        self.assertTrue(ok)
        steps = payload["plan"]["steps"]
        disable_openrouter = [
            row
            for row in steps
            if row.get("action") == "modelops.enable_disable_provider_or_model"
            and (row.get("params") or {}).get("id") == "openrouter"
            and (row.get("params") or {}).get("enabled") is False
        ]
        self.assertTrue(disable_openrouter)
        set_local_default = [
            row
            for row in steps
            if row.get("action") == "modelops.set_default_model"
            and (row.get("params") or {}).get("default_provider") == "ollama"
        ]
        self.assertTrue(set_local_default)

    def test_support_remediation_no_local_installed_uses_model_watch_pull_candidate(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._health_monitor.state = {  # type: ignore[attr-defined]
            "providers": {
                "ollama": {"status": "ok"},
            },
            "models": {
                "ollama:llama3": {"status": "down", "last_error_kind": "model_not_installed"},
            },
        }
        runtime._model_watch_store.save(  # type: ignore[attr-defined]
            normalize_model_watch_state(
                {
                    "seen": {},
                    "recommendations": [],
                    "active_batch_id": "batch_local",
                    "batches": [
                        {
                            "batch_id": "batch_local",
                            "created_at": 1_700_000_000,
                            "status": "new",
                            "deferred_until": None,
                            "candidate_ids": ["ollama:qwen2.5:3b-instruct"],
                            "top_pick_id": "ollama:qwen2.5:3b-instruct",
                            "confidence": 0.9,
                            "show_top_only": True,
                            "candidates": [
                                {
                                    "id": "ollama:qwen2.5:3b-instruct",
                                    "provider": "ollama",
                                    "model": "qwen2.5:3b-instruct",
                                    "score": 88.4,
                                    "reason": "fits local constraints",
                                    "tradeoffs": [],
                                    "subscores": {
                                        "task_fit": 25.0,
                                        "local_feasibility": 25.0,
                                        "cost_efficiency": 15.0,
                                        "quality_proxy": 18.4,
                                        "switch_gain": 5.0,
                                    },
                                }
                            ],
                        }
                    ],
                }
            )
        )

        ok, payload = runtime.llm_support_remediate_plan({"target": "defaults", "intent": "fix_routing"})
        self.assertTrue(ok)
        pull_steps = [
            row
            for row in payload["plan"]["steps"]
            if row.get("action") == "modelops.pull_ollama_model"
        ]
        self.assertTrue(pull_steps)
        self.assertEqual("qwen2.5:3b-instruct", (pull_steps[0].get("params") or {}).get("model"))

    def test_support_remediation_ollama_down_adds_user_action_guidance(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._health_monitor.state = {  # type: ignore[attr-defined]
            "providers": {
                "ollama": {"status": "down", "last_error_kind": "provider_unavailable"},
            },
            "models": {},
        }

        ok, payload = runtime.llm_support_remediate_plan({"target": "defaults", "intent": "fix_routing"})
        self.assertTrue(ok)
        user_action_steps = [
            row
            for row in payload["plan"]["steps"]
            if row.get("kind") == "user_action" and row.get("action") == "user.start_verify_ollama"
        ]
        self.assertTrue(user_action_steps)
        self.assertIn("http://127.0.0.1:11434", str(user_action_steps[0].get("instructions") or ""))

    def test_support_endpoints_are_wired(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))

        bundle_handler = _HandlerForTest(runtime, "/llm/support/bundle")
        bundle_handler.do_GET()
        self.assertEqual(200, bundle_handler.status_code)
        bundle_payload = json.loads(bundle_handler.body.decode("utf-8"))
        self.assertTrue(bundle_payload["ok"])
        self.assertIn("bundle", bundle_payload)

        missing_handler = _HandlerForTest(runtime, "/llm/support/diagnose")
        missing_handler.do_GET()
        self.assertEqual(400, missing_handler.status_code)

        diagnose_handler = _HandlerForTest(runtime, "/llm/support/diagnose?id=openrouter")
        diagnose_handler.do_GET()
        self.assertEqual(200, diagnose_handler.status_code)
        diagnose_payload = json.loads(diagnose_handler.body.decode("utf-8"))
        self.assertTrue(diagnose_payload["ok"])
        self.assertEqual("provider", diagnose_payload["kind"])

        remediate_handler = _HandlerForTest(
            runtime,
            "/llm/support/remediate/plan",
            {"target": "defaults", "intent": "fix_routing"},
        )
        remediate_handler.do_POST()
        self.assertEqual(200, remediate_handler.status_code)
        remediate_payload = json.loads(remediate_handler.body.decode("utf-8"))
        self.assertTrue(remediate_payload["ok"])
        self.assertTrue(remediate_payload["plan"]["plan_only"])

        remediate_execute_handler = _HandlerForTest(
            runtime,
            "/llm/support/remediate/execute",
            {"target": "defaults", "intent": "fix_routing", "confirm": True},
        )
        remediate_execute_handler.do_POST()
        self.assertEqual(200, remediate_execute_handler.status_code)
        remediate_execute_payload = json.loads(remediate_execute_handler.body.decode("utf-8"))
        self.assertTrue(remediate_execute_payload["ok"])
        self.assertIn("executed_steps", remediate_execute_payload)
        self.assertIn("blocked_steps", remediate_execute_payload)


if __name__ == "__main__":
    unittest.main()
