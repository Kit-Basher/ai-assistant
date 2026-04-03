from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.orchestrator import Orchestrator
from agent.skills_loader import SkillLoader
from memory.db import MemoryDB


def _config(registry_path: str, db_path: str, skills_path: str) -> Config:
    return Config(
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


def _write_skill(
    *,
    skills_root: str,
    skill_dir_name: str,
    execution: dict[str, object] | None = None,
    handler_body: str | None = None,
) -> None:
    skill_dir = os.path.join(skills_root, skill_dir_name)
    os.makedirs(skill_dir, exist_ok=True)
    with open(os.path.join(skill_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "name": skill_dir_name,
                "description": "governance test skill",
                "version": "0.1.0",
                "permissions": [],
                "execution": execution or {},
                "functions": [
                    {
                        "name": "run",
                        "args_schema": {"type": "object", "properties": {}},
                    }
                ],
            },
            handle,
            ensure_ascii=True,
        )
    with open(os.path.join(skill_dir, "handler.py"), "w", encoding="utf-8") as handle:
        handle.write(handler_body or "def run(ctx):\n    return {'text': 'ran'}\n")


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.body = b""

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self.status_code = status
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")


class _ChatHandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, payload: dict[str, object]) -> None:
        self.runtime = runtime_obj
        self.path = "/chat"
        self.headers = {"Content-Length": "0"}
        self._payload = dict(payload)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


class TestSkillExecutionGovernance(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.skills_root = os.path.join(self.tmpdir.name, "skills")
        os.makedirs(self.skills_root, exist_ok=True)
        self.db = MemoryDB(self.db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        self.db.close()
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_root,
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )

    def _runtime(self) -> AgentRuntime:
        return AgentRuntime(_config(self.registry_path, self.db_path, self.skills_root))

    def test_default_skill_is_allowed_as_in_process(self) -> None:
        _write_skill(skills_root=self.skills_root, skill_dir_name="default_skill")
        orchestrator = self._orchestrator()

        status = orchestrator.skill_governance_status()
        row = next(item for item in status["skills"] if item["skill_id"] == "default_skill")

        self.assertEqual("in_process", row["requested_execution_mode"])
        self.assertTrue(row["allowed"])
        self.assertFalse(row["requires_user_approval"])

    def test_managed_background_task_request_is_gated_without_approval(self) -> None:
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="scheduled_sync",
            execution={
                "mode": "managed_background_task",
                "capabilities": ["background_task", "network_access"],
                "persistence_requested": True,
            },
        )
        orchestrator = self._orchestrator()

        response = orchestrator._call_skill("user1", "scheduled_sync", "run", {}, [])
        status = orchestrator.skill_governance_status()
        row = next(item for item in status["skills"] if item["skill_id"] == "scheduled_sync")

        self.assertIn("managed background task execution", response.text.lower())
        self.assertFalse(row["allowed"])
        self.assertTrue(row["requires_user_approval"])

    def test_managed_adapter_request_is_gated_without_approval(self) -> None:
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="discord_gateway",
            execution={
                "mode": "managed_adapter",
                "capabilities": ["managed_adapter", "network_access", "notifications"],
                "persistence_requested": True,
            },
        )
        orchestrator = self._orchestrator()

        response = orchestrator._call_skill("user1", "discord_gateway", "run", {}, [])
        status = orchestrator.skill_governance_status()
        row = next(item for item in status["skills"] if item["skill_id"] == "discord_gateway")

        self.assertIn("managed adapter execution", response.text.lower())
        self.assertFalse(row["allowed"])
        self.assertTrue(row["requires_user_approval"])

    def test_unmanaged_service_or_daemon_pattern_is_blocked_by_loader(self) -> None:
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="rogue_daemon",
            handler_body=(
                "import subprocess\n"
                "def run(ctx):\n"
                "    subprocess.Popen(['sleep', '60'], start_new_session=True)\n"
                "    return {'text': 'bad'}\n"
            ),
        )
        loader = SkillLoader(self.skills_root)
        skills = loader.load_all()
        orchestrator = self._orchestrator()

        self.assertNotIn("rogue_daemon", skills)
        self.assertEqual("rogue_daemon", loader.blocked_skills[0]["skill_id"])
        self.assertIn("detached_process_spawn", loader.blocked_skills[0]["source_issues"])
        response = orchestrator._call_skill("user1", "rogue_daemon", "run", {}, [])
        self.assertIn("blocked by execution governance", response.text.lower())

    def test_managed_adapter_registry_entry_lifecycle(self) -> None:
        runtime = self._runtime()

        first_ok, first = runtime.register_managed_adapter(
            adapter_id="discord",
            adapter_type="discord_gateway",
            source_skill="discord_gateway",
            source_package="discord_pack",
            approved=False,
            enabled=False,
            startup_policy="manual",
            health_status="pending",
            last_error=None,
            owner="runtime",
            requested_by="tester",
            reason="Discord gateway integration requested.",
        )
        second_ok, second = runtime.register_managed_adapter(
            adapter_id="discord",
            adapter_type="discord_gateway",
            source_skill="discord_gateway",
            source_package="discord_pack",
            approved=True,
            enabled=True,
            startup_policy="on_boot",
            health_status="running",
            last_error=None,
            owner="runtime",
            requested_by="tester",
            reason="Approved Discord gateway integration.",
        )
        status = runtime.managed_adapters_status()
        row = next(item for item in status["managed_adapters"] if item["adapter_id"] == "discord")

        self.assertTrue(first_ok)
        self.assertFalse(first["approved"])
        self.assertTrue(second_ok)
        self.assertTrue(second["approved"])
        self.assertTrue(row["enabled"])
        self.assertEqual("running", row["health_status"])

    def test_background_task_registry_entry_lifecycle(self) -> None:
        runtime = self._runtime()

        first_ok, first = runtime.register_background_task(
            task_id="catalog_refresh",
            source_skill="catalog_refresh",
            source_package="catalog_pack",
            schedule="hourly",
            trigger_type="interval",
            approved=False,
            enabled=False,
            health_status="pending",
            last_run_at=None,
            last_error=None,
            resource_limits={"cpu": "low"},
            owner="runtime",
            requested_by="tester",
            reason="Periodic catalog refresh requested.",
        )
        second_ok, second = runtime.register_background_task(
            task_id="catalog_refresh",
            source_skill="catalog_refresh",
            source_package="catalog_pack",
            schedule="hourly",
            trigger_type="interval",
            approved=True,
            enabled=True,
            health_status="ok",
            last_run_at=123456,
            last_error=None,
            resource_limits={"cpu": "low"},
            owner="runtime",
            requested_by="tester",
            reason="Approved periodic catalog refresh.",
        )
        status = runtime.background_tasks_status()
        row = next(item for item in status["background_tasks"] if item["task_id"] == "catalog_refresh")

        self.assertTrue(first_ok)
        self.assertFalse(first["approved"])
        self.assertTrue(second_ok)
        self.assertTrue(second["enabled"])
        self.assertEqual(123456, row["last_run_at"])

    def test_runtime_truth_service_exposes_governance_lists(self) -> None:
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="scheduled_sync",
            execution={
                "mode": "managed_background_task",
                "capabilities": ["background_task"],
                "persistence_requested": True,
            },
        )
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="rogue_daemon",
            handler_body=(
                "import subprocess\n"
                "def run(ctx):\n"
                "    subprocess.Popen(['sleep', '60'], start_new_session=True)\n"
                "    return {'text': 'bad'}\n"
            ),
        )
        runtime = self._runtime()
        truth = runtime.runtime_truth_service()

        adapters = truth.list_managed_adapters()
        tasks = truth.list_background_tasks()
        blocked = truth.list_governance_blocks()
        pending = truth.list_pending_governance_requests()
        skill_status = truth.get_skill_governance_status("scheduled_sync")

        self.assertIn("telegram", adapters["active_adapter_ids"])
        self.assertIn(
            "runtime_scheduler",
            {
                str(row.get("task_id") or "").strip()
                for row in tasks["background_tasks"]
                if isinstance(row, dict)
            },
        )
        self.assertEqual("rogue_daemon", blocked["blocked_skills"][0]["skill_id"])
        self.assertEqual("scheduled_sync", pending["pending_skills"][0]["skill_id"])
        self.assertTrue(skill_status["found"])
        self.assertEqual("managed_background_task", skill_status["skill"]["requested_execution_mode"])

    def test_chat_governance_queries_are_grounded(self) -> None:
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="scheduled_sync",
            execution={
                "mode": "managed_background_task",
                "capabilities": ["background_task"],
                "persistence_requested": True,
            },
        )
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="rogue_daemon",
            handler_body=(
                "import subprocess\n"
                "def run(ctx):\n"
                "    subprocess.Popen(['sleep', '60'], start_new_session=True)\n"
                "    return {'text': 'bad'}\n"
            ),
        )
        runtime = self._runtime()
        utterances = (
            ("what managed adapters exist?", "governance_managed_adapters"),
            ("what background tasks are active?", "governance_background_tasks"),
            ("what got blocked by skill governance?", "governance_blocks"),
            ("is any skill waiting for approval?", "governance_pending"),
            ("what execution mode does skill scheduled_sync use?", "governance_execution_mode"),
            ("what execution mode does Telegram use?", "governance_execution_mode"),
            ("execution mode for scheduled_sync", "governance_execution_mode"),
            ("what execution mode does this skill use?", "governance_skill_status"),
            ("why does Telegram exist?", "governance_adapter_detail"),
        )

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("generic inference should not run"),
        ):
            for utterance, expected_type in utterances:
                ok, body = runtime.chat(
                    {
                        "messages": [{"role": "user", "content": utterance}],
                        "source_surface": "api",
                    }
                )
                self.assertTrue(ok, msg=utterance)
                self.assertEqual("governance_status", body["meta"]["route"], msg=utterance)
                self.assertFalse(body["meta"]["generic_fallback_used"], msg=utterance)
                self.assertEqual(expected_type, body["setup"]["type"], msg=utterance)
                if utterance == "what execution mode does Telegram use?":
                    self.assertIn("managed_adapter mode", body["assistant"]["content"])
                if utterance == "what execution mode does skill scheduled_sync use?":
                    self.assertIn("managed_background_task mode", body["assistant"]["content"])
                if utterance == "execution mode for scheduled_sync":
                    self.assertIn("managed_background_task mode", body["assistant"]["content"])
                if utterance == "what execution mode does this skill use?":
                    self.assertIn("Tell me the skill name", body["assistant"]["content"])

    def test_http_chat_governance_queries_bypass_chooser(self) -> None:
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="scheduled_sync",
            execution={
                "mode": "managed_background_task",
                "capabilities": ["background_task"],
                "persistence_requested": True,
            },
        )
        runtime = self._runtime()
        utterances = (
            ("what managed adapters exist?", "governance_status"),
            ("what background tasks are active?", "governance_status"),
            ("what got blocked by skill governance?", "governance_status"),
            ("is any skill waiting for approval?", "governance_status"),
            ("what execution mode does skill scheduled_sync use?", "governance_status"),
            ("what execution mode does Telegram use?", "governance_status"),
            ("execution mode for scheduled_sync", "governance_status"),
            ("what execution mode does this skill use?", "governance_status"),
        )

        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None):
            for utterance, expected_route in utterances:
                handler = _ChatHandlerForTest(runtime, {"messages": [{"role": "user", "content": utterance}]})
                handler.do_POST()
                response = handler.response_payload
                self.assertEqual(200, handler.status_code, msg=utterance)
                self.assertNotEqual("needs_clarification", response.get("error_kind"), msg=utterance)
                self.assertNotIn("Which of these is your goal", str(response.get("message") or ""), msg=utterance)
                meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"), msg=utterance)

    def test_runtime_status_endpoints_include_governed_components(self) -> None:
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="scheduled_sync",
            execution={
                "mode": "managed_background_task",
                "capabilities": ["background_task"],
                "persistence_requested": True,
            },
        )
        runtime = self._runtime()

        handler = _HandlerForTest(runtime, "/skill-governance/status")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertIn("managed_adapters", payload)
        self.assertIn("background_tasks", payload)
        self.assertIn("telegram", {row["adapter_id"] for row in payload["managed_adapters"]})
        self.assertIn("runtime_scheduler", {row["task_id"] for row in payload["background_tasks"]})
        skill_row = next(item for item in payload["skills"] if item["skill_id"] == "scheduled_sync")
        self.assertFalse(skill_row["allowed"])
        self.assertTrue(skill_row["requires_user_approval"])


if __name__ == "__main__":
    unittest.main()
