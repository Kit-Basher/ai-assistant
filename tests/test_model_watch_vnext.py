from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.model_watch import buzz_scan, map_buzz_leads_to_catalog
from telegram_adapter.bot import maybe_handle_operator_recovery_reply


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
        provider_catalog_state_path=os.path.join(os.path.dirname(db_path), "provider_catalog_state.json"),
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _registry_for_diff() -> dict[str, object]:
    return {
        "defaults": {
            "routing_mode": "auto",
            "default_provider": "openrouter",
            "default_model": "openrouter:tiny-chat",
            "allow_remote_fallback": True,
        },
        "providers": {
            "openrouter": {"enabled": True, "local": False, "available": True},
            "ollama": {"enabled": True, "local": True, "available": True},
        },
        "models": {
            "openrouter:tiny-chat": {
                "provider": "openrouter",
                "model": "tiny-chat",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "routable": True,
                "pricing": {
                    "input_per_million_tokens": 9.0,
                    "output_per_million_tokens": 18.0,
                },
                "max_context_tokens": 4096,
            },
            "openrouter:better-chat": {
                "provider": "openrouter",
                "model": "better-chat",
                "capabilities": ["chat", "json", "tools"],
                "enabled": True,
                "available": True,
                "routable": True,
                "pricing": {
                    "input_per_million_tokens": 0.1,
                    "output_per_million_tokens": 0.2,
                },
                "max_context_tokens": 131072,
            },
        },
    }


def _catalog_delta(*model_ids: str) -> object:
    return type(
        "_Delta",
        (),
        {
            "provider_model_count": len(model_ids),
            "new_models": tuple({"model_id": model_id} for model_id in model_ids),
            "changed_models": tuple(),
            "providers_considered": ("openrouter",),
            "last_run_ts": 1_700_000_123,
        },
    )()


class TestModelWatchVNext(unittest.TestCase):
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

    def test_catalog_diff_creates_fixit_proposal_when_threshold_met(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document = _registry_for_diff()
        runtime.set_default_chat_model("openrouter:tiny-chat")
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-testsecret123"
        health_summary = {
            "providers": [{"id": "openrouter", "status": "ok"}, {"id": "ollama", "status": "ok"}],
            "models": [
                {"id": "openrouter:tiny-chat", "status": "ok"},
                {"id": "openrouter:better-chat", "status": "ok"},
            ],
        }

        with patch("agent.api_server.run_watch_once_for_config", return_value={"ok": True, "fetched_candidates": 2}), patch(
            "agent.api_server.scan_provider_catalogs",
            return_value=_catalog_delta("openrouter:better-chat"),
        ), patch.object(
            runtime._health_monitor,
            "summary",
            return_value=health_summary,
        ), patch.object(
            runtime,
            "telegram_status",
            return_value={"state": "running"},
        ), patch.object(
            runtime,
            "_resolve_telegram_target",
            return_value=("token", "123456789"),
        ), patch.object(runtime, "_send_telegram_message", return_value=None) as send_mock:
            ok, body = runtime.run_model_watch_once(trigger="manual")

        self.assertTrue(ok)
        self.assertTrue(bool(body.get("ok")))
        self.assertTrue(bool(body.get("proposal_created")))
        proposal = body.get("proposal") if isinstance(body.get("proposal"), dict) else {}
        self.assertEqual("model_watch.proposal", proposal.get("issue_code"))
        self.assertGreater(float(proposal.get("score_delta") or 0.0), 0.08)
        self.assertIn("quality_delta", proposal)
        self.assertIn("expected_cost_delta", proposal)
        self.assertIn("Quality delta:", str(proposal.get("details") or ""))
        self.assertIn("Expected cost delta per 1M tokens:", str(proposal.get("details") or ""))
        delta = body.get("catalog_delta") if isinstance(body.get("catalog_delta"), dict) else {}
        self.assertGreaterEqual(len(delta.get("new_models") or []), 1)
        self.assertFalse(bool(body.get("proposal_notification_emitted")))
        self.assertEqual("suppressed_by_control_plane_policy", str(body.get("proposal_notification_reason") or ""))
        send_mock.assert_not_called()

        wizard_state = runtime._llm_fixit_store.state  # type: ignore[attr-defined]
        self.assertTrue(bool(wizard_state.get("active")))
        self.assertEqual("model_watch.proposal", wizard_state.get("issue_code"))

        message = maybe_handle_operator_recovery_reply(
            operator_recovery_fn=lambda payload: runtime.operator_recovery(payload),
            recovery_store=runtime.operator_recovery_store(),
            audit_log=None,
            chat_id="123456789",
            text="1",
            log_path=runtime.config.log_path,
        )
        self.assertIsNotNone(message)
        self.assertIn("Reply YES", str(message))
        self.assertEqual("awaiting_confirm", str(runtime._llm_fixit_store.state.get("step")))  # type: ignore[attr-defined]

    def test_catalog_diff_no_proposal_when_below_threshold(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, model_watch_min_improvement=3.0))
        runtime.registry_document = _registry_for_diff()
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-testsecret123"
        health_summary = {
            "providers": [{"id": "openrouter", "status": "ok"}, {"id": "ollama", "status": "ok"}],
            "models": [
                {"id": "openrouter:tiny-chat", "status": "ok"},
                {"id": "openrouter:better-chat", "status": "ok"},
            ],
        }

        with patch("agent.api_server.run_watch_once_for_config", return_value={"ok": True, "fetched_candidates": 2}), patch(
            "agent.api_server.scan_provider_catalogs",
            return_value=_catalog_delta("openrouter:better-chat"),
        ), patch.object(
            runtime._health_monitor,
            "summary",
            return_value=health_summary,
        ), patch.object(
            runtime,
            "telegram_status",
            return_value={"state": "running"},
        ), patch.object(
            runtime,
            "_resolve_telegram_target",
            return_value=("token", "123456789"),
        ), patch.object(runtime, "_send_telegram_message", return_value=None) as send_mock:
            ok, body = runtime.run_model_watch_once(trigger="manual")

        self.assertTrue(ok)
        self.assertFalse(bool(body.get("proposal_created")))
        send_mock.assert_not_called()
        entries = runtime.get_audit(limit=30).get("entries", [])
        no_change = [row for row in entries if row.get("action") == "llm.model_watch.no_change"]
        self.assertTrue(no_change)

    def test_catalog_diff_respects_default_policy_cost_cap(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                default_policy={
                    "cost_cap_per_1m": 0.05,
                    "allowlist": [],
                    "quality_weight": 1.0,
                    "price_weight": 0.04,
                    "latency_weight": 0.25,
                    "instability_weight": 0.5,
                },
            )
        )
        runtime.registry_document = _registry_for_diff()
        runtime.set_default_chat_model("openrouter:tiny-chat")
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-testsecret123"
        health_summary = {
            "providers": [{"id": "openrouter", "status": "ok"}, {"id": "ollama", "status": "ok"}],
            "models": [
                {"id": "openrouter:tiny-chat", "status": "ok"},
                {"id": "openrouter:better-chat", "status": "ok"},
            ],
        }

        with patch("agent.api_server.run_watch_once_for_config", return_value={"ok": True, "fetched_candidates": 2}), patch(
            "agent.api_server.scan_provider_catalogs",
            return_value=_catalog_delta("openrouter:better-chat"),
        ), patch.object(
            runtime._health_monitor,
            "summary",
            return_value=health_summary,
        ), patch.object(
            runtime,
            "telegram_status",
            return_value={"state": "running"},
        ), patch.object(
            runtime,
            "_resolve_telegram_target",
            return_value=("token", "123456789"),
        ), patch.object(runtime, "_send_telegram_message", return_value=None) as send_mock:
            ok, body = runtime.run_model_watch_once(trigger="manual")

        self.assertTrue(ok)
        self.assertFalse(bool(body.get("proposal_created")))
        eval_payload = body.get("proposal_evaluation") if isinstance(body.get("proposal_evaluation"), dict) else {}
        self.assertEqual("no_allowed_delta_candidate", eval_payload.get("reason"))
        self.assertIn("cost_cap_exceeded", eval_payload.get("policy_rejections") or [])
        send_mock.assert_not_called()

    def test_model_watch_proposal_respects_live_health_and_auth(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document = {
            "defaults": {
                "routing_mode": "auto",
                "default_provider": "ollama",
                "default_model": "ollama:qwen3.5:4b",
                "allow_remote_fallback": True,
            },
            "providers": {
                "ollama": {"enabled": True, "local": True, "available": True},
                "openrouter": {"enabled": True, "local": False, "available": True},
            },
            "models": {
                "ollama:qwen3.5:4b": {
                    "provider": "ollama",
                    "model": "qwen3.5:4b",
                    "capabilities": ["chat"],
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "max_context_tokens": 8192,
                },
                "openrouter:better-chat": {
                    "provider": "openrouter",
                    "model": "better-chat",
                    "capabilities": ["chat", "json"],
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "max_context_tokens": 131072,
                    "pricing": {
                        "input_per_million_tokens": 0.1,
                        "output_per_million_tokens": 0.2,
                    },
                },
            },
        }
        runtime.set_default_chat_model("ollama:qwen3.5:4b")
        with patch.object(
            runtime._health_monitor,
            "summary",
            return_value={
                "providers": [{"id": "ollama", "status": "ok"}, {"id": "openrouter", "status": "down"}],
                "models": [
                    {"id": "ollama:qwen3.5:4b", "status": "ok"},
                    {"id": "openrouter:better-chat", "status": "down"},
                ],
            },
        ):
            proposal = runtime._build_model_watch_proposal(delta_rows=[{"model_id": "openrouter:better-chat"}])

        self.assertIsNone(proposal)
        evaluation = runtime._model_watch_last_proposal_evaluation
        self.assertFalse(bool(proposal))
        self.assertTrue(evaluation.get("reason") in {"no_allowed_delta_candidate", "improvement_below_threshold"})
        self.assertTrue(evaluation.get("policy_rejections") or evaluation.get("rejected_candidates"))

    def test_buzz_scan_is_deterministic_with_allowlist(self) -> None:
        def _fetch(url: str) -> object:
            if "openrouter.ai" in url:
                return {"data": [{"id": "acme/model-x"}, {"id": "acme/model-y"}]}
            if "huggingface.co" in url:
                return [{"repoData": {"id": "acme/model-x"}}, {"repoData": {"id": "other/model-z"}}]
            if "11434" in url:
                return {"models": [{"name": "qwen2.5:3b"}]}
            raise AssertionError(f"unexpected_url:{url}")

        allowlist = ("openrouter_models", "huggingface_trending", "ollama_catalog")
        leads_a = buzz_scan(fetch_json=_fetch, sources_allowlist=allowlist)
        leads_b = buzz_scan(fetch_json=_fetch, sources_allowlist=allowlist)
        self.assertEqual(leads_a, leads_b)
        self.assertEqual("acme/model-x", leads_a[0]["name"])
        self.assertEqual(0.9, float(leads_a[0]["confidence"]))
        names = [str(row.get("name") or "") for row in leads_a]
        self.assertNotIn("other/model-z", names)

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.registry_document = {
            "providers": {"openrouter": {"enabled": True}, "ollama": {"enabled": True}},
            "models": {
                "openrouter:acme/model-x": {"provider": "openrouter", "model": "acme/model-x"},
                "ollama:qwen2.5:3b": {"provider": "ollama", "model": "qwen2.5:3b"},
            },
            "defaults": {},
        }
        mapped = map_buzz_leads_to_catalog(leads=leads_a, runtime=runtime)
        model_x = next((row for row in mapped if str(row.get("name") or "") == "acme/model-x"), None)
        self.assertIsNotNone(model_x)
        self.assertTrue(bool(model_x.get("available")))
        self.assertIn("openrouter:acme/model-x", model_x.get("mapped_model_ids") or [])


if __name__ == "__main__":
    unittest.main()
