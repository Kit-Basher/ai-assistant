from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.orchestrator import Orchestrator, OrchestratorResponse, classify_authoritative_domain
from memory.db import MemoryDB


def _config(registry_path: str, db_path: str) -> Config:
    return Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
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
    )


class _StubLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "ok": True,
            "text": "Likely CPU-limited workload from local observations.",
            "provider": "test",
            "model": "test-model",
            "fallback_used": False,
            "attempts": [],
            "duration_ms": 1,
        }


class TestAuthoritativeDomainGate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.log_path = os.path.join(self.tmpdir.name, "agent.log")
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db = MemoryDB(self.db_path)
        self.db.init_schema(schema_path)
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        self.db.close()
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_classify_authoritative_domain_keywords(self) -> None:
        self.assertEqual({"system.performance"}, classify_authoritative_domain("am i throttling while gaming?"))
        self.assertEqual({"system.health"}, classify_authoritative_domain("show failed units and journal errors"))
        self.assertEqual({"system.storage"}, classify_authoritative_domain("what is eating space on my drive"))
        self.assertEqual(set(), classify_authoritative_domain("draft a short email update"))

    def test_ask_authoritative_enforcement_calls_metrics_and_injects_observations(self) -> None:
        llm = _StubLLM()
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
            enable_writes=False,
        )
        payload = {
            "ok": True,
            "source": "fresh",
            "snapshot": {
                "ts": 1700000000,
                "cpu": {"usage_pct": 91.0},
                "gpu": {"usage_pct": 12.0, "temperature_c": 84.0},
            },
            "events": [],
            "stored": {"snapshot_id": 17, "event_ids": []},
        }
        with patch.object(
            orchestrator,
            "_sys_metrics_snapshot",
            return_value=OrchestratorResponse(json.dumps(payload, ensure_ascii=True)),
        ) as metrics_call:
            response = orchestrator.handle_message("/ask am I throttling?", "user-1")

        self.assertEqual(1, metrics_call.call_count)
        self.assertTrue(llm.calls)
        last_call = llm.calls[-1]
        llm_messages = last_call["messages"]
        assert isinstance(llm_messages, list)
        self.assertIn("LOCAL_OBSERVATIONS", str(llm_messages[1]["content"]))
        self.assertIn("LOCAL_OBSERVATIONS", response.text)

    def test_ask_authoritative_tool_failure_returns_not_sure_with_one_question(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            enable_writes=False,
        )
        with patch.object(orchestrator, "_sys_metrics_snapshot", side_effect=RuntimeError("collector unavailable")):
            response = orchestrator.handle_message("/ask am i throttling?", "user-1")

        self.assertIn("I’m not sure.", response.text)
        self.assertEqual(1, response.text.count("?"))
        self.assertIn("sys_metrics_snapshot", response.text)

    def test_api_chat_auto_enforces_authoritative_domains_unless_require_tools_provided(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        observations = {
            "domains": ["system.performance"],
            "grounding": {
                "collected_at_ts": 1700000000,
                "observation_refs": {
                    "system.performance": {"tool": "sys_metrics_snapshot", "snapshot_id": 17, "ts": 1700000000}
                },
            },
            "observations": {"system.performance": {"ok": True}},
        }
        router_result = {
            "ok": True,
            "text": "Grounded response",
            "provider": "test",
            "model": "test-model",
            "fallback_used": False,
            "attempts": [],
            "duration_ms": 1,
        }

        with (
            patch.object(runtime, "_collect_authoritative_observations", return_value=observations) as collect_call,
            patch.object(runtime._router, "chat", return_value=router_result) as router_chat,
        ):
            ok, _body = runtime.chat({"messages": [{"role": "user", "content": "my pc is lagging badly"}]})

        self.assertTrue(ok)
        self.assertEqual(1, collect_call.call_count)
        self.assertEqual(1, router_chat.call_count)
        call_args = router_chat.call_args
        routed_messages = call_args.args[0]
        self.assertEqual("system", routed_messages[0]["role"])
        self.assertIn("LOCAL_OBSERVATIONS", routed_messages[0]["content"])
        self.assertTrue(call_args.kwargs["require_tools"])

        with (
            patch.object(runtime, "_collect_authoritative_observations", return_value=observations) as collect_call,
            patch.object(runtime._router, "chat", return_value=router_result) as router_chat,
        ):
            ok, _body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": "my pc is lagging badly"}],
                    "require_tools": False,
                }
            )

        self.assertTrue(ok)
        self.assertEqual(0, collect_call.call_count)
        call_args = router_chat.call_args
        routed_messages = call_args.args[0]
        self.assertEqual("user", routed_messages[0]["role"])
        self.assertFalse(call_args.kwargs["require_tools"])


if __name__ == "__main__":
    unittest.main()
