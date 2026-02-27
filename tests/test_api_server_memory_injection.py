from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.memory_v2.types import MemoryItem, MemoryLevel


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
        memory_v2_enabled=True,
    )
    return base.__class__(**{**base.__dict__, **overrides})


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object]) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {"Content-Type": "application/json"}
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload

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


class TestAPIServerMemoryInjection(unittest.TestCase):
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

    def test_chat_includes_memory_envelope_selection_when_enabled(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        self.assertIsNotNone(runtime._memory_v2_store)
        runtime._memory_v2_store.upsert_memory_item(
            MemoryItem(
                id="S-DET",
                level=MemoryLevel.SEMANTIC,
                text="Project uses determinism-first and never-raise API boundary",
                created_at=1_700_000_000,
                updated_at=1_700_000_100,
                tags={"project": "personal-agent"},
                source_kind="doc",
                source_ref="PROJECT_STATUS.md",
                pinned=True,
            )
        )

        captured_payload: dict[str, object] = {}

        def _fake_chat(payload: dict[str, object]):
            captured_payload.clear()
            captured_payload.update(payload)
            return True, {
                "ok": True,
                "assistant": {"role": "assistant", "content": "ok"},
                "meta": {"provider": "test", "model": "test"},
            }

        runtime.chat = _fake_chat  # type: ignore[assignment]

        handler = _HandlerForTest(
            runtime,
            "/chat",
            {
                "messages": [
                    {"role": "user", "content": "determinism first please"},
                ],
            },
        )
        handler.do_POST()

        self.assertEqual(200, handler.status_code)
        response = json.loads(handler.body.decode("utf-8"))
        self.assertTrue(response.get("ok"))
        envelope = response.get("envelope") if isinstance(response.get("envelope"), dict) else {}
        memory = envelope.get("memory") if isinstance(envelope.get("memory"), dict) else {}
        selected_ids = memory.get("selected_ids") if isinstance(memory.get("selected_ids"), list) else []
        self.assertIn("S-DET", selected_ids)
        self.assertIn("memory_context_text", captured_payload)
        self.assertIn("MEMORY[S-DET]", str(captured_payload.get("memory_context_text") or ""))


if __name__ == "__main__":
    unittest.main()
