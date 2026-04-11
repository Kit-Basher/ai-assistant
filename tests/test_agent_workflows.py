from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.memory_runtime import MemoryRuntime
from agent.working_memory import append_turn
from memory.db import MemoryDB

from tests.test_behavioral_eval_battery import _BehavioralRuntimeTruth, _first_line


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
        memory_v2_enabled=False,
        semantic_memory_enabled=False,
    )
    return base.__class__(**{**base.__dict__, **overrides})


class TestAgentWorkflows(unittest.TestCase):
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

    def _runtime(self) -> tuple[AgentRuntime, _BehavioralRuntimeTruth]:
        truth = _BehavioralRuntimeTruth()
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._runtime_truth_service = truth  # type: ignore[assignment]
        return runtime, truth

    def _chat(
        self,
        runtime: AgentRuntime,
        text: str,
        *,
        user_id: str = "user1",
        expect_transport_ok: bool | None = True,
        expect_body_ok: bool | None = True,
    ) -> dict[str, object]:
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")), patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            return_value=None,
        ):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": text}],
                    "source_surface": "api",
                    "user_id": user_id,
                }
            )
        if expect_transport_ok is True:
            self.assertTrue(ok, msg=str(body))
        elif expect_transport_ok is False:
            self.assertFalse(ok, msg=str(body))
        if expect_body_ok is True:
            self.assertTrue(bool(body.get("ok")), msg=str(body))
        elif expect_body_ok is False:
            self.assertFalse(bool(body.get("ok")), msg=str(body))
        return body

    def test_discovery_to_approval_install_verify_and_switch_flow(self) -> None:
        runtime, truth = self._runtime()

        discovery = self._chat(runtime, "there is a brand new tiny Gemma 4 model, can you look into it?")
        discovery_text = str((discovery.get("assistant") or {}).get("content") or "")
        self.assertTrue(discovery_text.startswith("For Gemma, the closest family matches look like openrouter:vendor/tiny-gemma"))
        self.assertIn("Likely family match: openrouter:vendor/tiny-gemma.", discovery_text)
        self.assertIn("Practical local fit: ollama:qwen2.5:7b-instruct.", discovery_text)
        self.assertIn("Sources checked: external_snapshots, huggingface, ollama, openrouter (4 queried).", discovery_text)
        self.assertIn("Source errors: huggingface: hf timeout.", discovery_text)
        self.assertIn("model_discovery_manager", discovery.get("meta", {}).get("used_tools", []))

        install_preview = self._chat(runtime, "install ollama:qwen2.5:7b-instruct")
        install_preview_text = str((install_preview.get("assistant") or {}).get("content") or "")
        self.assertIn("Say yes to continue, or no to cancel.", install_preview_text)
        self.assertIn("canonical model manager", install_preview_text)

        install_confirm = self._chat(runtime, "yes")
        install_confirm_text = str((install_confirm.get("assistant") or {}).get("content") or "")
        self.assertIn("Started acquiring ollama:qwen2.5:7b-instruct", install_confirm_text)
        self.assertIn("ollama:qwen2.5:7b-instruct", truth.installed_model_ids)

        verify = self._chat(runtime, "what models are ready now?")
        verify_text = str((verify.get("assistant") or {}).get("content") or "")
        self.assertIn("Right now chat is using ollama:qwen2.5:7b-instruct", verify_text)
        self.assertIn("only model that looks ready now", verify_text)
        self.assertIn("ollama:qwen2.5:7b-instruct", verify_text)

        switch_preview = self._chat(runtime, "switch to ollama:qwen2.5:7b-instruct")
        switch_preview_text = str((switch_preview.get("assistant") or {}).get("content") or "")
        self.assertIn("Say yes to continue, or no to cancel.", switch_preview_text)

        switch_confirm = self._chat(runtime, "yes")
        switch_confirm_text = str((switch_confirm.get("assistant") or {}).get("content") or "")
        self.assertIn("Now using ollama:qwen2.5:7b-instruct for chat.", switch_confirm_text)
        self.assertEqual("ollama:qwen2.5:7b-instruct", truth.current_model)

    def test_provider_setup_then_discovery_then_usage(self) -> None:
        runtime, truth = self._runtime()
        truth.openrouter_secret_present = True

        setup_preview = self._chat(runtime, "configure openrouter")
        setup_preview_text = str((setup_preview.get("assistant") or {}).get("content") or "")
        self.assertIn("already have an OpenRouter API key stored", setup_preview_text)

        setup_confirm = self._chat(runtime, "yes")
        setup_confirm_text = str((setup_confirm.get("assistant") or {}).get("content") or "")
        self.assertIn("OpenRouter is ready", setup_confirm_text)
        self.assertEqual("openrouter", truth.current_provider)
        self.assertEqual("openrouter:openai/gpt-4o-mini", truth.current_model)

        discovery = self._chat(runtime, "look for information on a new model from hugging face")
        discovery_text = str((discovery.get("assistant") or {}).get("content") or "")
        self.assertIn("openrouter:vendor/tiny-gemma", discovery_text)
        self.assertIn("model_discovery_manager", discovery.get("meta", {}).get("used_tools", []))

        usage = self._chat(runtime, "is openrouter configured?")
        usage_text = str((usage.get("assistant") or {}).get("content") or "")
        self.assertIn("OpenRouter", usage_text)
        self.assertIn("chat is currently using openrouter", usage_text.lower())
        self.assertNotIn("not set up yet", usage_text.lower())

    def test_memory_write_reload_conflict_recovery(self) -> None:
        runtime_a, _ = self._runtime()
        self._chat(runtime_a, "remember that I prefer concise replies")

        runtime_b, _ = self._runtime()
        reload_response = self._chat(runtime_b, "what do you remember about me?")
        reload_text = str((reload_response.get("assistant") or {}).get("content") or "")
        self.assertIn("useful memory", reload_text.lower())
        self.assertNotIn("{", _first_line(reload_text))

        memory_runtime_a = MemoryRuntime(runtime_a.orchestrator().db)
        memory_runtime_b = MemoryRuntime(runtime_b.orchestrator().db)
        state_a, issue_a = memory_runtime_a.load_working_memory_state("user1")
        state_b, issue_b = memory_runtime_b.load_working_memory_state("user1")
        self.assertIsNone(issue_a)
        self.assertIsNone(issue_b)
        append_turn(state_a, role="user", text="runtime a saved this first")
        self.assertTrue(memory_runtime_a.save_working_memory_state("user1", state_a))
        append_turn(state_b, role="user", text="runtime b stale write")
        self.assertFalse(memory_runtime_b.save_working_memory_state("user1", state_b))
        memory_runtime_c = MemoryRuntime(runtime_b.orchestrator().db)
        state_c, issue_c = memory_runtime_c.load_working_memory_state("user1")
        self.assertIsNone(issue_c)
        append_turn(state_c, role="assistant", text="runtime c recovered save")
        self.assertTrue(memory_runtime_c.save_working_memory_state("user1", state_c))

        snapshot = MemoryRuntime(runtime_b.orchestrator().db).inspect_all_state()
        users = snapshot.get("users") if isinstance(snapshot.get("users"), list) else []
        user_row = next(
            (
                row
                for row in users
                if isinstance(row, dict) and str(row.get("user_id") or "").strip() == "user1"
            ),
            {},
        )
        persistence = user_row.get("persistence") if isinstance(user_row.get("persistence"), dict) else {}
        self.assertEqual("stale_write_conflict", (persistence.get("last_conflict") or {}).get("reason"))
        self.assertFalse(bool(persistence.get("active_conflict")))

        runtime_c, _ = self._runtime()
        recovery_response = self._chat(runtime_c, "what do you remember about me?")
        recovery_text = str((recovery_response.get("assistant") or {}).get("content") or "")
        self.assertIn("useful memory", recovery_text.lower())
        self.assertNotIn("{", _first_line(recovery_text))

    def test_controlled_mode_blocked_action_then_approval_and_success(self) -> None:
        runtime, truth = self._runtime()

        blocked = self._chat(
            runtime,
            "switch to openrouter:openai/gpt-4o-mini",
            expect_transport_ok=None,
            expect_body_ok=None,
        )
        blocked_text = str((blocked.get("assistant") or {}).get("content") or "")
        self.assertIn("openrouter", blocked_text.lower())
        self.assertIn("say yes to continue, or no to cancel.", blocked_text.lower())

        ok_mode, mode_body = runtime.llm_control_mode_set({"mode": "controlled", "confirm": True, "actor": "test"})
        self.assertTrue(ok_mode)
        truth.controller_mode = "controlled"

        preview = self._chat(runtime, "switch to openrouter:openai/gpt-4o-mini")
        preview_text = str((preview.get("assistant") or {}).get("content") or "")
        self.assertIn("Say yes to continue, or no to cancel.", preview_text)

        success = self._chat(runtime, "yes")
        success_text = str((success.get("assistant") or {}).get("content") or "")
        self.assertIn("Now using openrouter:openai/gpt-4o-mini for chat.", success_text)
        self.assertEqual("openrouter", truth.current_provider)
        self.assertEqual("openrouter:openai/gpt-4o-mini", truth.current_model)


if __name__ == "__main__":
    unittest.main()
