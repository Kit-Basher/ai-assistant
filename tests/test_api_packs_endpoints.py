from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config


def _config(registry_path: str, db_path: str, skills_path: str) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=skills_path,
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
    return base


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


class TestAPIPacksEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.skills_path = os.path.join(self.tmpdir.name, "skills")
        os.makedirs(self.skills_path, exist_ok=True)
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        self.runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.skills_path))

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def test_install_approve_and_list_pack_endpoints(self) -> None:
        pack_dir = os.path.join(self.tmpdir.name, "pack_one")
        os.makedirs(pack_dir, exist_ok=True)
        with open(os.path.join(pack_dir, "pack.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pack_id": "pack_one",
                    "version": "0.1.0",
                    "title": "Pack One",
                    "description": "test pack",
                    "entrypoints": ["skills.pack_one:handler"],
                    "trust": "trusted",
                    "permissions": {"ifaces": ["pack_one.run"]},
                },
                handle,
                ensure_ascii=True,
            )

        install_handler = _HandlerForTest(
            self.runtime,
            "/packs/install",
            {"source": pack_dir, "enable": True},
        )
        install_handler.do_POST()
        install_payload = json.loads(install_handler.body.decode("utf-8"))
        self.assertEqual(200, install_handler.status_code)
        self.assertTrue(install_payload["ok"])
        self.assertEqual("pack_one", install_payload["pack"]["pack_id"])
        self.assertTrue(install_payload["requires_approval"])

        clarify_handler = _HandlerForTest(
            self.runtime,
            "/packs/approve",
            {"pack_id": "pack_one", "approve": False},
        )
        clarify_handler.do_POST()
        clarify_payload = json.loads(clarify_handler.body.decode("utf-8"))
        self.assertEqual(200, clarify_handler.status_code)
        self.assertTrue(clarify_payload["ok"])
        self.assertEqual("needs_clarification", clarify_payload["error_kind"])

        approve_handler = _HandlerForTest(
            self.runtime,
            "/packs/approve",
            {"pack_id": "pack_one", "approve": True, "enable": True},
        )
        approve_handler.do_POST()
        approve_payload = json.loads(approve_handler.body.decode("utf-8"))
        self.assertEqual(200, approve_handler.status_code)
        self.assertTrue(approve_payload["ok"])
        self.assertTrue(str(approve_payload["pack"]["approved_permissions_hash"]))

        list_handler = _HandlerForTest(self.runtime, "/packs")
        list_handler.do_GET()
        list_payload = json.loads(list_handler.body.decode("utf-8"))
        self.assertEqual(200, list_handler.status_code)
        self.assertTrue(list_payload["ok"])
        pack_ids = [row["pack_id"] for row in list_payload["packs"]]
        self.assertIn("pack_one", pack_ids)


if __name__ == "__main__":
    unittest.main()
