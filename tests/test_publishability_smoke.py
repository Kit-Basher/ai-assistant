from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.model_watch_catalog import write_snapshot_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(
    registry_path: str,
    db_path: str,
    *,
    home_root: Path,
    safe_mode_enabled: bool = True,
) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=str(REPO_ROOT / "skills"),
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen3.5:4b",
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
        model_watch_enabled=False,
        autopilot_notify_enabled=False,
        llm_notifications_allow_send=False,
        perception_enabled=True,
        perception_roots=(str(home_root),),
        safe_mode_enabled=safe_mode_enabled,
        safe_mode_chat_model="ollama:qwen3.5:4b",
    )
    return base


def _seed_runtime(runtime: AgentRuntime) -> None:
    local_models = (
        {
            "model": "qwen3.5:4b",
            "capabilities": ["chat"],
            "quality_rank": 6,
            "available": True,
            "max_context_tokens": 32768,
        },
        {
            "model": "qwen2.5:7b-instruct",
            "capabilities": ["chat"],
            "quality_rank": 9,
            "available": True,
            "max_context_tokens": 32768,
        },
        {
            "model": "deepseek-r1:7b",
            "capabilities": ["chat"],
            "quality_rank": 7,
            "available": True,
            "max_context_tokens": 32768,
        },
    )
    remote_models = (
        {
            "model": "openrouter/auto",
            "capabilities": ["chat", "image"],
            "available": True,
            "max_context_tokens": 2_000_000,
            "pricing": {"input_per_million_tokens": 0.0, "output_per_million_tokens": 0.0},
        },
        {
            "model": "openai/gpt-4o-mini",
            "capabilities": ["chat"],
            "task_types": ["general_chat"],
            "quality_rank": 6,
            "cost_rank": 2,
            "available": True,
            "max_context_tokens": 128000,
            "pricing": {"input_per_million_tokens": 0.15, "output_per_million_tokens": 0.60},
        },
        {
            "model": "openai/gpt-4.1-mini",
            "capabilities": ["chat"],
            "task_types": ["coding", "general_chat"],
            "quality_rank": 8,
            "cost_rank": 6,
            "available": True,
            "max_context_tokens": 1_047_576,
            "pricing": {"input_per_million_tokens": 0.40, "output_per_million_tokens": 1.60},
        },
        {
            "model": "openai/gpt-4.1",
            "capabilities": ["chat", "vision"],
            "task_types": ["coding", "general_chat", "reasoning"],
            "quality_rank": 9,
            "cost_rank": 8,
            "available": True,
            "max_context_tokens": 1_047_576,
            "pricing": {"input_per_million_tokens": 2.00, "output_per_million_tokens": 8.00},
        },
    )
    for payload in local_models:
        runtime.add_provider_model("ollama", dict(payload))
    for payload in remote_models:
        runtime.add_provider_model("openrouter", dict(payload))
    runtime.set_provider_secret("openrouter", {"api_key": "sk-test"})
    runtime.update_defaults(
        {
            "default_provider": "ollama",
            "chat_model": "ollama:qwen3.5:4b",
            "allow_remote_fallback": True,
        }
    )
    runtime._health_monitor.state = {
        "providers": {
            "ollama": {"status": "ok", "last_checked_at": 123},
            "openrouter": {"status": "ok", "last_checked_at": 123},
        },
        "models": {
            "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
            "ollama:qwen2.5:7b-instruct": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
            "ollama:deepseek-r1:7b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
            "openrouter:openrouter/auto": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            "openrouter:openai/gpt-4o-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            "openrouter:openai/gpt-4.1-mini": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
            "openrouter:openai/gpt-4.1": {"provider_id": "openrouter", "status": "ok", "last_checked_at": 123},
        },
    }
    runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]


class _JSONHandler(APIServerHandler):
    def __init__(
        self,
        runtime_obj: AgentRuntime,
        *,
        path: str,
        payload: dict[str, object] | None = None,
        client_host: str = "127.0.0.1",
    ) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {"Content-Length": "0"}
        self.client_address = (client_host, 12345)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}
        self._payload = dict(payload or {})

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


class TestPublishabilitySmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self._env_backup = dict(os.environ)
        self.home_root = Path(self.tmpdir.name) / "home"
        self.home_root.mkdir(parents=True, exist_ok=True)
        ssh_dir = self.home_root / ".ssh"
        ssh_dir.mkdir(parents=True, exist_ok=True)
        (ssh_dir / "config").write_text("Host smoke\n  User demo\n", encoding="utf-8")
        os.environ["HOME"] = str(self.home_root)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        self.repo_smoke_file = REPO_ROOT / "publishability_smoke_note.txt"
        self.repo_smoke_dir = REPO_ROOT / "publishability-smoke-dir"
        if self.repo_smoke_dir.exists():
            self.repo_smoke_dir.rmdir()
        if self.repo_smoke_file.exists():
            self.repo_smoke_file.unlink()
        self.repo_smoke_file.write_text(
            "publishability smoke token\nTODO: publishability smoke coverage\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        if self.repo_smoke_file.exists():
            self.repo_smoke_file.unlink()
        if self.repo_smoke_dir.exists():
            self.repo_smoke_dir.rmdir()
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _make_runtime(self, *, safe_mode_enabled: bool = True) -> AgentRuntime:
        registry_path = os.path.join(self.tmpdir.name, f"registry-{time.time_ns()}.json")
        db_path = os.path.join(self.tmpdir.name, f"agent-{time.time_ns()}.db")
        runtime = AgentRuntime(
            _config(
                registry_path,
                db_path,
                home_root=self.home_root,
                safe_mode_enabled=safe_mode_enabled,
            )
        )
        _seed_runtime(runtime)
        return runtime

    def _chat(
        self,
        runtime: AgentRuntime,
        prompt: str,
        *,
        user_id: str,
        thread_id: str,
        expected_status: int = 200,
    ) -> tuple[dict[str, object], float]:
        handler = _JSONHandler(
            runtime,
            path="/chat",
            payload={
                "messages": [{"role": "user", "content": prompt}],
                "user_id": user_id,
                "thread_id": thread_id,
                "source_surface": "api",
            },
        )
        started = time.perf_counter()
        handler.do_POST()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.assertEqual(expected_status, handler.status_code, prompt)
        return handler.response_payload, elapsed_ms

    def _post(
        self,
        runtime: AgentRuntime,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        handler = _JSONHandler(runtime, path=path, payload=payload)
        handler.do_POST()
        self.assertEqual(200, handler.status_code, path)
        return handler.response_payload

    def _get(self, runtime: AgentRuntime, path: str) -> dict[str, object]:
        handler = _JSONHandler(runtime, path=path)
        handler.do_GET()
        self.assertEqual(200, handler.status_code, path)
        return handler.response_payload

    def _chat_meta(self, payload: dict[str, object]) -> dict[str, object]:
        meta = payload.get("meta")
        self.assertTrue(isinstance(meta, dict))
        return dict(meta or {})

    def test_publishability_mode_runtime_and_recommendation_flows(self) -> None:
        runtime = self._make_runtime(safe_mode_enabled=True)

        mode_prompt, _ = self._chat(
            runtime,
            "what mode are we in",
            user_id="smoke-mode",
            thread_id="smoke-mode-thread",
        )
        mode_meta = self._chat_meta(mode_prompt)
        self.assertEqual("model_policy_status", mode_meta.get("route"))
        self.assertFalse(bool(mode_meta.get("used_llm")))

        initial_status = self._get(runtime, "/llm/control_mode")
        self.assertEqual("safe", initial_status.get("mode"))

        current_model, current_ms = self._chat(
            runtime,
            "what model am I using",
            user_id="smoke-runtime",
            thread_id="smoke-runtime-thread",
        )
        current_meta = self._chat_meta(current_model)
        self.assertEqual("model_status", current_meta.get("route"))
        self.assertFalse(bool(current_meta.get("used_llm")))
        self.assertIn("ollama:qwen3.5:4b", str(current_model.get("message") or "").lower())
        self.assertLess(current_ms, 5000.0)

        local_models, _ = self._chat(
            runtime,
            "what local models are available",
            user_id="smoke-runtime",
            thread_id="smoke-runtime-thread",
        )
        local_message = str(local_models.get("message") or "").lower()
        local_meta = self._chat_meta(local_models)
        self.assertEqual("model_status", local_meta.get("route"))
        self.assertFalse(bool(local_meta.get("used_llm")))
        self.assertIn("local installed chat models", local_message)
        self.assertIn("ollama:qwen2.5:7b-instruct", local_message)

        cloud_models, _ = self._chat(
            runtime,
            "show cloud models",
            user_id="smoke-runtime",
            thread_id="smoke-runtime-thread",
        )
        cloud_message = str(cloud_models.get("message") or "").lower()
        cloud_meta = self._chat_meta(cloud_models)
        self.assertEqual("model_status", cloud_meta.get("route"))
        self.assertFalse(bool(cloud_meta.get("used_llm")))
        self.assertIn("cloud models", cloud_message)
        self.assertIn("openrouter:", cloud_message)
        self.assertNotIn("ollama:", cloud_message)

        local_rec, _ = self._chat(
            runtime,
            "recommend a local model",
            user_id="smoke-recommend",
            thread_id="smoke-recommend-thread",
        )
        local_rec_meta = self._chat_meta(local_rec)
        self.assertEqual("action_tool", local_rec_meta.get("route"))
        self.assertEqual(["model_scout"], local_rec_meta.get("used_tools"))
        self.assertFalse(bool(local_rec_meta.get("used_llm")))
        self.assertIn("Best local option:", str(local_rec.get("message") or ""))

        controlled = self._post(runtime, "/llm/control_mode", {"mode": "controlled", "confirm": True})
        self.assertEqual("controlled", ((controlled.get("policy") or {}).get("mode")))

        coding_rec, coding_ms = self._chat(
            runtime,
            "recommend a coding model",
            user_id="smoke-recommend",
            thread_id="smoke-recommend-thread",
        )
        coding_meta = self._chat_meta(coding_rec)
        self.assertEqual("action_tool", coding_meta.get("route"))
        self.assertEqual(["model_scout"], coding_meta.get("used_tools"))
        self.assertFalse(bool(coding_meta.get("used_llm")))
        self.assertIn("Best coding option", str(coding_rec.get("message") or ""))
        self.assertLess(coding_ms, 5000.0)

        research_rec, _ = self._chat(
            runtime,
            "recommend a research model",
            user_id="smoke-recommend",
            thread_id="smoke-recommend-thread",
        )
        research_meta = self._chat_meta(research_rec)
        self.assertEqual("action_tool", research_meta.get("route"))
        self.assertEqual(["model_scout"], research_meta.get("used_tools"))
        self.assertFalse(bool(research_meta.get("used_llm")))
        self.assertIn("Best research option", str(research_rec.get("message") or ""))

        cheap_cloud, _ = self._chat(
            runtime,
            "what cheap cloud model should I use",
            user_id="smoke-recommend",
            thread_id="smoke-recommend-thread",
        )
        cheap_meta = self._chat_meta(cheap_cloud)
        self.assertEqual("action_tool", cheap_meta.get("route"))
        self.assertEqual(["model_scout"], cheap_meta.get("used_tools"))
        self.assertFalse(bool(cheap_meta.get("used_llm")))
        self.assertIn("Cheap cloud recommendation:", str(cheap_cloud.get("message") or ""))

        baseline = self._post(runtime, "/llm/control_mode", {"mode": "baseline", "confirm": True})
        baseline_policy = baseline.get("policy") if isinstance(baseline.get("policy"), dict) else {}
        self.assertEqual("safe", baseline_policy.get("mode"))
        self.assertFalse(bool(baseline_policy.get("override_active")))

    def test_publishability_filesystem_and_shell_flows(self) -> None:
        runtime = self._make_runtime(safe_mode_enabled=True)

        listing, _ = self._chat(
            runtime,
            "show me what's in this folder",
            user_id="smoke-files",
            thread_id="smoke-files-thread",
        )
        listing_meta = self._chat_meta(listing)
        self.assertEqual("action_tool", listing_meta.get("route"))
        self.assertEqual(["filesystem"], listing_meta.get("used_tools"))
        self.assertFalse(bool(listing_meta.get("used_llm")))
        listing_message = str(listing.get("message") or "")
        self.assertIn("/home/c/personal-agent contains", listing_message)
        self.assertIn("agent/", listing_message)

        read_text, _ = self._chat(
            runtime,
            "read publishability_smoke_note.txt",
            user_id="smoke-files",
            thread_id="smoke-files-thread",
        )
        read_meta = self._chat_meta(read_text)
        self.assertEqual("action_tool", read_meta.get("route"))
        self.assertEqual(["filesystem"], read_meta.get("used_tools"))
        self.assertFalse(bool(read_meta.get("used_llm")))
        self.assertIn("publishability smoke token", str(read_text.get("message") or "").lower())

        filename_search, _ = self._chat(
            runtime,
            "find files named publishability_smoke_note.txt",
            user_id="smoke-files",
            thread_id="smoke-files-thread",
        )
        filename_meta = self._chat_meta(filename_search)
        self.assertEqual("action_tool", filename_meta.get("route"))
        self.assertEqual(["filesystem"], filename_meta.get("used_tools"))
        self.assertFalse(bool(filename_meta.get("used_llm")))
        self.assertIn("publishability_smoke_note.txt", str(filename_search.get("message") or ""))

        text_search, _ = self._chat(
            runtime,
            "search this repo for publishability smoke token",
            user_id="smoke-files",
            thread_id="smoke-files-thread",
        )
        text_meta = self._chat_meta(text_search)
        self.assertEqual("action_tool", text_meta.get("route"))
        self.assertEqual(["filesystem"], text_meta.get("used_tools"))
        self.assertFalse(bool(text_meta.get("used_llm")))
        self.assertIn("publishability smoke token", str(text_search.get("message") or "").lower())

        blocked_read, _ = self._chat(
            runtime,
            "read ~/.ssh/config",
            user_id="smoke-files",
            thread_id="smoke-files-thread",
            expected_status=400,
        )
        blocked_meta = self._chat_meta(blocked_read)
        self.assertEqual("action_tool", blocked_meta.get("route"))
        self.assertEqual(["filesystem"], blocked_meta.get("used_tools"))
        self.assertFalse(bool(blocked_meta.get("used_llm")))
        self.assertEqual("sensitive_path_blocked", blocked_meta.get("error"))

        blocked_search, _ = self._chat(
            runtime,
            "search ~/.ssh for Host",
            user_id="smoke-files",
            thread_id="smoke-files-thread",
            expected_status=400,
        )
        blocked_search_meta = self._chat_meta(blocked_search)
        self.assertEqual("action_tool", blocked_search_meta.get("route"))
        self.assertEqual(["filesystem"], blocked_search_meta.get("used_tools"))
        self.assertFalse(bool(blocked_search_meta.get("used_llm")))
        self.assertEqual("sensitive_path_blocked", blocked_search_meta.get("error"))

        python_version, _ = self._chat(
            runtime,
            "what version of python do i have",
            user_id="smoke-shell",
            thread_id="smoke-shell-thread",
        )
        python_meta = self._chat_meta(python_version)
        self.assertEqual("action_tool", python_meta.get("route"))
        self.assertEqual(["shell"], python_meta.get("used_tools"))
        self.assertFalse(bool(python_meta.get("used_llm")))
        self.assertIn("python", str(python_version.get("message") or "").lower())

        blocked_shell, _ = self._chat(
            runtime,
            "run ls; whoami",
            user_id="smoke-shell",
            thread_id="smoke-shell-thread",
            expected_status=400,
        )
        blocked_shell_meta = self._chat_meta(blocked_shell)
        self.assertEqual("action_tool", blocked_shell_meta.get("route"))
        self.assertEqual(["shell"], blocked_shell_meta.get("used_tools"))
        self.assertFalse(bool(blocked_shell_meta.get("used_llm")))
        self.assertEqual("shell_interpolation_blocked", blocked_shell_meta.get("error"))

    def test_publishability_mutating_preview_confirm_flows(self) -> None:
        runtime = self._make_runtime(safe_mode_enabled=True)

        switch_preview, _ = self._chat(
            runtime,
            "switch temporarily to ollama:qwen2.5:7b-instruct",
            user_id="smoke-mutate",
            thread_id="smoke-mutate-thread",
        )
        switch_preview_meta = self._chat_meta(switch_preview)
        switch_preview_setup = switch_preview.get("setup") if isinstance(switch_preview.get("setup"), dict) else {}
        self.assertEqual("model_status", switch_preview_meta.get("route"))
        self.assertEqual(["model_controller"], switch_preview_meta.get("used_tools"))
        self.assertFalse(bool(switch_preview_meta.get("used_llm")))
        self.assertTrue(bool(switch_preview_setup.get("requires_confirmation")))

        switch_confirm, _ = self._chat(
            runtime,
            "yes",
            user_id="smoke-mutate",
            thread_id="smoke-mutate-thread",
        )
        switch_confirm_meta = self._chat_meta(switch_confirm)
        self.assertEqual("model_status", switch_confirm_meta.get("route"))
        self.assertEqual(["model_controller"], switch_confirm_meta.get("used_tools"))
        self.assertFalse(bool(switch_confirm_meta.get("used_llm")))
        self.assertIn("Temporarily using ollama:qwen2.5:7b-instruct for chat.", str(switch_confirm.get("message") or ""))

        mkdir_preview, _ = self._chat(
            runtime,
            "create a folder called publishability-smoke-dir in this repo",
            user_id="smoke-mutate",
            thread_id="smoke-mutate-thread",
        )
        mkdir_preview_meta = self._chat_meta(mkdir_preview)
        mkdir_preview_setup = mkdir_preview.get("setup") if isinstance(mkdir_preview.get("setup"), dict) else {}
        self.assertEqual("action_tool", mkdir_preview_meta.get("route"))
        self.assertEqual(["shell"], mkdir_preview_meta.get("used_tools"))
        self.assertFalse(bool(mkdir_preview_meta.get("used_llm")))
        self.assertTrue(bool(mkdir_preview_setup.get("requires_confirmation")))
        self.assertFalse(self.repo_smoke_dir.exists())

        mkdir_confirm, _ = self._chat(
            runtime,
            "yes",
            user_id="smoke-mutate",
            thread_id="smoke-mutate-thread",
        )
        mkdir_confirm_meta = self._chat_meta(mkdir_confirm)
        self.assertEqual("action_tool", mkdir_confirm_meta.get("route"))
        self.assertEqual(["shell"], mkdir_confirm_meta.get("used_tools"))
        self.assertFalse(bool(mkdir_confirm_meta.get("used_llm")))
        self.assertTrue(self.repo_smoke_dir.exists())
        self.assertIn("Created directory", str(mkdir_confirm.get("message") or ""))

        switch_back_preview, _ = self._chat(
            runtime,
            "switch back",
            user_id="smoke-mutate",
            thread_id="smoke-mutate-thread",
        )
        switch_back_setup = switch_back_preview.get("setup") if isinstance(switch_back_preview.get("setup"), dict) else {}
        self.assertTrue(bool(switch_back_setup.get("requires_confirmation")))

        switch_back_confirm, _ = self._chat(
            runtime,
            "yes",
            user_id="smoke-mutate",
            thread_id="smoke-mutate-thread",
        )
        switch_back_meta = self._chat_meta(switch_back_confirm)
        self.assertEqual(["model_controller"], switch_back_meta.get("used_tools"))
        self.assertFalse(bool(switch_back_meta.get("used_llm")))
        self.assertIn("Now using ollama:qwen3.5:4b for chat.", str(switch_back_confirm.get("message") or ""))

    def test_publishability_discovery_and_policy_flows(self) -> None:
        runtime = self._make_runtime(safe_mode_enabled=False)
        runtime._model_watch_catalog_path = Path(self.tmpdir.name) / "model_watch_catalog_snapshot.json"
        write_snapshot_atomic(
            runtime._model_watch_catalog_path,
            {
                "provider": "openrouter",
                "source": "openrouter_models",
                "fetched_at": 1774915200,
                "models": [
                    {
                        "id": "openrouter:vendor/cheap-text",
                        "provider_id": "openrouter",
                        "model": "vendor/cheap-text",
                        "context_length": 131072,
                        "modalities": ["text"],
                        "supports_tools": False,
                        "pricing": {
                            "prompt_per_million": 0.1,
                            "completion_per_million": 0.2,
                        },
                    }
                ],
            },
        )

        listed = self._post(runtime, "/llm/models/proposals", {})
        envelope = listed.get("envelope") if isinstance(listed.get("envelope"), dict) else {}
        proposals = envelope.get("proposals") if isinstance(envelope.get("proposals"), list) else []
        self.assertTrue(proposals)
        first = next((row for row in proposals if isinstance(row, dict)), {})
        self.assertTrue(bool(first.get("non_canonical")))
        self.assertEqual("not_adopted", first.get("canonical_status"))

        filtered_source = self._post(runtime, "/llm/models/proposals", {"source": "external_openrouter_snapshot"})
        filtered_source_envelope = filtered_source.get("envelope") if isinstance(filtered_source.get("envelope"), dict) else {}
        source_rows = filtered_source_envelope.get("proposals") if isinstance(filtered_source_envelope.get("proposals"), list) else []
        self.assertTrue(source_rows)
        self.assertTrue(all(isinstance(row, dict) and row.get("source") == "external_openrouter_snapshot" for row in source_rows))

        filtered_kind = self._post(runtime, "/llm/models/proposals", {"proposal_kind": "candidate_good"})
        filtered_kind_envelope = filtered_kind.get("envelope") if isinstance(filtered_kind.get("envelope"), dict) else {}
        kind_rows = filtered_kind_envelope.get("proposals") if isinstance(filtered_kind_envelope.get("proposals"), list) else []
        self.assertTrue(kind_rows)
        self.assertTrue(all(isinstance(row, dict) and row.get("proposal_kind") == "candidate_good" for row in kind_rows))

        policy_write = self._post(
            runtime,
            "/llm/models/policy",
            {
                "model_id": "openrouter:openai/gpt-4.1-mini",
                "status": "known_good",
                "role_hints": ["coding"],
                "notes": "Publishability smoke review.",
                "justification": "Smoke suite operator write.",
                "reviewed_at": "2026-04-01T00:00:00Z",
            },
        )
        policy_entry = ((policy_write.get("envelope") or {}).get("entry")) or {}
        self.assertEqual("known_good", policy_entry.get("status"))
        self.assertEqual(["coding"], policy_entry.get("role_hints"))

        policy_list = self._get(runtime, "/llm/models/policy?status=known_good")
        policy_envelope = policy_list.get("envelope") if isinstance(policy_list.get("envelope"), dict) else {}
        policy_rows = policy_envelope.get("policy_entries") if isinstance(policy_envelope.get("policy_entries"), list) else []
        self.assertTrue(any(isinstance(row, dict) and row.get("model_id") == "openrouter:openai/gpt-4.1-mini" for row in policy_rows))

        proposals_after = self._post(runtime, "/llm/models/proposals", {})
        proposals_after_rows = (((proposals_after.get("envelope") or {}).get("proposals")) or [])
        reviewed = next(
            (
                row
                for row in proposals_after_rows
                if isinstance(row, dict) and row.get("model_id") == "openrouter:openai/gpt-4.1-mini"
            ),
            {},
        )
        self.assertEqual("known_good", reviewed.get("policy_status"))
        self.assertTrue(bool(reviewed.get("non_canonical")))
        self.assertEqual("not_adopted", reviewed.get("canonical_status"))
