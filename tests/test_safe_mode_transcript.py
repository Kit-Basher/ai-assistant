from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime, compute_autopilot_bootstrap_apply_policy, compute_notification_send_policy, compute_self_heal_apply_policy
from agent.config import Config
from agent.llm.model_manager import model_manager_state_path_for_runtime, save_model_manager_state
from agent.orchestrator import OrchestratorResponse
from agent.telegram_bridge import handle_telegram_text


def _config(registry_path: str, db_path: str, **overrides: object) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=os.path.join(os.getcwd(), "skills"),
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
        model_scout_enabled=False,
        model_watch_enabled=False,
        autopilot_notify_enabled=False,
        llm_notifications_allow_send=False,
        safe_mode_enabled=True,
        safe_mode_chat_model="ollama:qwen3.5:4b",
    )
    return base.__class__(**{**base.__dict__, **overrides})


class TestSafeModeTranscript(unittest.TestCase):
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

    def _runtime(self, **config_overrides: object) -> AgentRuntime:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, **config_overrides))
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen3.5:4b",
                "capabilities": ["chat"],
                "quality_rank": 6,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:7b-instruct",
                "capabilities": ["chat"],
                "quality_rank": 9,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime.update_defaults(
            {
                "default_provider": "ollama",
                "chat_model": "ollama:qwen2.5:7b-instruct",
                "allow_remote_fallback": True,
            }
        )
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "ok", "last_checked_at": 123},
            },
            "models": {
                "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
                "ollama:qwen2.5:7b-instruct": {"provider_id": "ollama", "status": "ok", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]
        return runtime

    def _invoke_chat_http(
        self,
        runtime: AgentRuntime,
        payload: dict[str, object],
    ) -> dict[str, object]:
        class _HandlerForTest(APIServerHandler):
            def __init__(self, runtime_obj: AgentRuntime, request_payload: dict[str, object]) -> None:
                self.runtime = runtime_obj
                self.path = "/chat"
                self.headers = {"Content-Length": "0"}
                self._payload = dict(request_payload)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                return dict(self._payload)

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        handler = _HandlerForTest(runtime, payload)
        handler.do_POST()
        return handler.response_payload

    def _invoke_telegram_text(
        self,
        runtime: AgentRuntime,
        *,
        text: str,
        chat_id: str = "42",
    ) -> dict[str, object]:
        return handle_telegram_text(
            text=text,
            chat_id=chat_id,
            trace_id="trace-test",
            runtime=runtime,
            orchestrator=runtime.orchestrator(),
            fetch_local_api_chat_json=lambda payload: self._invoke_chat_http(runtime, dict(payload)),
        )

    def _install_machine_skill_fakes(self, runtime: AgentRuntime) -> dict[str, int]:
        orchestrator = runtime.orchestrator()
        calls = {"hardware": 0, "resource": 0, "storage": 0}

        def hardware_handler(context, user_id=None):  # type: ignore[no-untyped-def]
            _ = context
            _ = user_id
            calls["hardware"] += 1
            return {
                "status": "ok",
                "text": "I can see an AMD Ryzen 9 7900X CPU, an NVIDIA RTX 4080 GPU, 64 GiB of RAM, and two main storage mounts.",
                "payload": {
                    "cpu_model": "AMD Ryzen 9 7900X",
                    "gpu": {"available": True, "gpus": [{"name": "NVIDIA RTX 4080"}]},
                    "memory": {"total_bytes": 64 * 1024 * 1024 * 1024},
                    "disk": [{"mountpoint": "/", "device": "/dev/nvme0n1p2"}, {"mountpoint": "/data", "device": "/dev/sdb1"}],
                    "os": "Linux x86_64",
                },
                "cards_payload": {
                    "cards": [
                        {"title": "Hardware inventory", "lines": ["CPU: AMD Ryzen 9 7900X", "RAM: 64 GiB total"], "severity": "ok"},
                        {"title": "GPU visibility", "lines": ["NVIDIA RTX 4080"], "severity": "ok"},
                        {"title": "Storage", "lines": ["/ on /dev/nvme0n1p2", "/data on /dev/sdb1"], "severity": "ok"},
                    ],
                    "raw_available": True,
                    "summary": "Hardware inventory is ready.",
                    "confidence": 1.0,
                    "next_questions": ["How much memory am I using?", "How is my storage?"],
                },
            }

        def resource_handler(context, user_id=None):  # type: ignore[no-untyped-def]
            _ = context
            _ = user_id
            calls["resource"] += 1
            return {
                "status": "ok",
                "text": "CPU load 1m 0.42; memory 41.0% used",
                "payload": {"loads": {"1m": 0.42}, "memory": {"used": 26, "total": 64}},
                "cards_payload": {
                    "cards": [{"title": "Live machine stats", "lines": ["CPU load 1m=0.42", "Memory in use: 41%"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "Live machine stats are ready.",
                    "confidence": 1.0,
                    "next_questions": ["What is using my memory the most?"],
                },
            }

        def storage_handler(context, user_id=None):  # type: ignore[no-untyped-def]
            _ = context
            _ = user_id
            calls["storage"] += 1
            return {
                "status": "ok",
                "text": "Disk status snapshot ready.",
                "payload": {"mounts": [{"mountpoint": "/", "used_pct": 58.0}, {"mountpoint": "/data", "used_pct": 63.0}]},
                "cards_payload": {
                    "cards": [{"title": "Storage", "lines": ["/ is 58% used", "/data is 63% used"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "Disk status snapshot ready.",
                    "confidence": 1.0,
                    "next_questions": ["How is my storage?"],
                },
            }

        orchestrator.skills["hardware_report"].functions["hardware_report"].handler = hardware_handler
        orchestrator.skills["resource_governor"].functions["resource_report"].handler = resource_handler
        orchestrator.skills["storage_governor"].functions["storage_report"].handler = storage_handler
        return calls

    def test_safe_mode_background_controls_are_disabled(self) -> None:
        runtime = self._runtime(llm_automation_enabled=True, model_scout_enabled=True, model_watch_enabled=True)

        self.assertIsNone(runtime._scheduler_thread)
        self.assertFalse(compute_notification_send_policy(runtime)["allow_send_effective"])
        self.assertEqual("safe_mode", compute_notification_send_policy(runtime)["allow_reason"])
        self.assertFalse(compute_self_heal_apply_policy(runtime)["allow_apply_effective"])
        self.assertEqual("safe_mode", compute_self_heal_apply_policy(runtime)["allow_reason"])
        self.assertFalse(compute_autopilot_bootstrap_apply_policy(runtime)["allow_apply_effective"])
        self.assertEqual("safe_mode", compute_autopilot_bootstrap_apply_policy(runtime)["allow_reason"])

        scout_ok, scout_body = runtime.run_model_scout(trigger="scheduler")
        self.assertTrue(scout_ok)
        self.assertTrue(scout_body["skipped"])
        self.assertEqual("safe_mode", scout_body["reason"])
        self.assertFalse(scout_body["notification_emitted"])

        watch_ok, watch_body = runtime.run_model_watch_once(trigger="scheduler")
        self.assertTrue(watch_ok)
        self.assertTrue(watch_body["skipped"])
        self.assertEqual("safe_mode", watch_body["reason"])
        self.assertFalse(watch_body["proposal_notification_emitted"])

    def test_safe_mode_transcript_stays_on_pinned_local_model(self) -> None:
        runtime = self._runtime()
        inference_calls: list[dict[str, object]] = []

        def _fake_route_inference(**kwargs):  # type: ignore[no-untyped-def]
            inference_calls.append(dict(kwargs))
            user_text = str(kwargs.get("user_text") or "").strip().lower()
            if "joke" in user_text:
                text = "Why did the agent stay local? To keep the latency down."
            else:
                text = "Hello."
            return {
                "ok": True,
                "text": text,
                "provider": kwargs.get("provider_override"),
                "model": kwargs.get("model_override"),
                "duration_ms": 1,
                "attempts": [],
            }

        transcript = [
            ("hello", "generic_chat"),
            ("tell me a joke", "generic_chat"),
            ("can you help me", "generic_chat"),
            ("what model are you using?", "model_status"),
            ("is everything working with the agent?", "runtime_status"),
            ("configure ollama", "setup_flow"),
        ]
        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            results = []
            for utterance, expected_route in transcript:
                ok, body = runtime.chat(
                    {
                        "messages": [{"role": "user", "content": utterance}],
                        "source_surface": "api",
                    }
                )
                self.assertTrue(ok, msg=utterance)
                self.assertEqual(expected_route, body["meta"]["route"], msg=utterance)
                self.assertNotEqual("needs_clarification", body.get("error_kind"), msg=utterance)
                self.assertNotIn("Which of these is your goal", body["assistant"]["content"], msg=utterance)
                self.assertNotIn("I couldn't read that from the runtime state.", body["assistant"]["content"], msg=utterance)
                results.append((utterance, body))

        self.assertEqual(3, len(inference_calls))
        for call in inference_calls:
            self.assertEqual("ollama", call["provider_override"])
            self.assertEqual("ollama:qwen3.5:4b", call["model_override"])

        hello_body = results[0][1]
        joke_body = results[1][1]
        model_body = results[3][1]
        runtime_body = results[4][1]
        configure_body = results[5][1]

        self.assertEqual("ollama:qwen3.5:4b", hello_body["meta"]["model"])
        self.assertEqual("ollama:qwen3.5:4b", joke_body["meta"]["model"])
        self.assertIn("ollama:qwen3.5:4b", model_body["assistant"]["content"])
        self.assertNotIn("qwen2.5:7b-instruct", model_body["assistant"]["content"])
        self.assertIn("healthy and ready", runtime_body["assistant"]["content"])
        self.assertIn("qwen3.5:4b", configure_body["assistant"]["content"])
        self.assertEqual("setup_complete", configure_body["setup"]["type"])

    def test_safe_mode_pins_same_chat_target_for_api_and_telegram(self) -> None:
        runtime = self._runtime()
        inference_calls: list[dict[str, object]] = []

        def _fake_route_inference(**kwargs):  # type: ignore[no-untyped-def]
            inference_calls.append(dict(kwargs))
            return {
                "ok": True,
                "text": "reply",
                "provider": kwargs.get("provider_override"),
                "model": kwargs.get("model_override"),
                "duration_ms": 1,
                "attempts": [],
            }

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            for source_surface in ("api", "telegram"):
                ok, body = runtime.chat(
                    {
                        "messages": [{"role": "user", "content": "tell me a joke"}],
                        "source_surface": source_surface,
                    }
                )
                self.assertTrue(ok)
                self.assertEqual("generic_chat", body["meta"]["route"])
                self.assertEqual("ollama:qwen3.5:4b", body["meta"]["model"])

        self.assertEqual(2, len(inference_calls))
        self.assertEqual(
            [call["model_override"] for call in inference_calls],
            ["ollama:qwen3.5:4b", "ollama:qwen3.5:4b"],
        )

    def test_safe_mode_exposes_raw_recovery_only_when_pinned_llm_is_unavailable(self) -> None:
        runtime = self._runtime()
        runtime._health_monitor.state = {
            "providers": {
                "ollama": {"status": "down", "last_checked_at": 123},
            },
            "models": {
                "ollama:qwen3.5:4b": {"provider_id": "ollama", "status": "down", "last_checked_at": 123},
                "ollama:qwen2.5:7b-instruct": {"provider_id": "ollama", "status": "down", "last_checked_at": 123},
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch("agent.orchestrator.route_inference") as route_inference:
            payload = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "source_surface": "api",
                    "user_id": "api:recovery",
                    "thread_id": "api:recovery:thread",
                },
            )

        assistant_text = str((payload.get("assistant") or {}).get("content") or payload.get("message") or "")
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        route_inference.assert_not_called()
        self.assertEqual("generic_chat", meta.get("route"))
        self.assertFalse(bool(meta.get("used_llm", False)))
        self.assertIn("Start Ollama locally", assistant_text)

    def test_safe_mode_http_and_telegram_golden_prompts_are_consistent(self) -> None:
        runtime = self._runtime()
        inference_calls: list[dict[str, object]] = []

        class _FakeDoctorReport:
            trace_id = "doctor-safe-mode"
            summary_status = "OK"
            next_action = "none"
            checks = [
                type("Check", (), {"status": "OK"})(),
                type("Check", (), {"status": "OK"})(),
            ]

        def _fake_route_inference(**kwargs):  # type: ignore[no-untyped-def]
            inference_calls.append(dict(kwargs))
            user_text = str(kwargs.get("user_text") or "").strip().lower()
            if "joke" in user_text:
                text = "Why did the agent stay local? To keep the latency down."
            else:
                text = "Hello."
            return {
                "ok": True,
                "text": text,
                "provider": kwargs.get("provider_override"),
                "model": kwargs.get("model_override"),
                "duration_ms": 1,
                "attempts": [],
            }

        def _fake_observe(self, user_id: str, text: str, nl_decision: dict[str, object]) -> OrchestratorResponse:  # type: ignore[no-untyped-def]
            lowered = text.lower()
            if "storage" in lowered:
                return OrchestratorResponse(
                    "Disk status snapshot ready.\n\n*Storage Report*\n- No storage snapshots found yet.",
                    {"summary": "Disk status snapshot ready."},
                )
            return OrchestratorResponse(
                "CPU load 1m 0.00; memory 0.0% used.\n\n*Resource Report*\n- No resource snapshots found yet.",
                {"summary": "CPU load 1m 0.00; memory 0.0% used."},
            )

        prompts = [
            ("hello", "generic_chat", True, False),
            ("tell me a joke", "generic_chat", True, False),
            ("can you help me", "generic_chat", True, False),
            ("what model are you using?", "model_status", False, False),
            ("is everything working with the agent?", "runtime_status", False, False),
            ("runtime", "runtime_status", False, False),
            ("configure ollama", "setup_flow", False, False),
            ("openrouter health?", "provider_status", False, False),
            ("how much memory am I using?", "operational_status", False, True),
            ("how much RAM am I currently using?", "operational_status", False, True),
            ("how is my storage?", "operational_status", False, True),
        ]

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference), patch(
            "agent.orchestrator.run_doctor_report",
            return_value=_FakeDoctorReport(),
        ), patch.object(
            runtime.orchestrator().__class__,
            "_handle_nl_observe",
            new=_fake_observe,
        ):
            for prompt, expected_route, expect_llm, expect_tools in prompts:
                api_payload = self._invoke_chat_http(
                    runtime,
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "source_surface": "api",
                        "user_id": "api:default",
                        "thread_id": "api:default:thread",
                    },
                )
                telegram_payload = self._invoke_telegram_text(runtime, text=prompt, chat_id="42")

                api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                tg_text = str(telegram_payload.get("text") or "")

                self.assertEqual(expected_route, api_meta.get("route"), msg=prompt)
                self.assertEqual(expected_route, telegram_payload.get("selected_route"), msg=prompt)
                self.assertEqual(expect_llm, bool(api_meta.get("used_llm", False)), msg=prompt)
                self.assertEqual(expect_llm, bool(telegram_payload.get("used_llm", False)), msg=prompt)
                self.assertEqual(expect_tools, bool(api_meta.get("used_tools")), msg=prompt)
                self.assertEqual(expect_tools, bool(telegram_payload.get("used_tools")), msg=prompt)
                self.assertNotEqual("needs_clarification", api_payload.get("error_kind"), msg=prompt)
                self.assertNotIn("Which of these is your goal", api_text, msg=prompt)
                self.assertNotIn("Which of these is your goal", tg_text, msg=prompt)
                self.assertNotIn("Tell me whether you want chat, ask, or model check/switch.", api_text, msg=prompt)
                self.assertNotIn("Tell me whether you want chat, ask, or model check/switch.", tg_text, msg=prompt)
                self.assertNotIn("continuing the current thread", api_text.lower(), msg=prompt)
                self.assertNotIn("continuing the current thread", tg_text.lower(), msg=prompt)
                self.assertNotIn("I couldn't read that from the runtime state.", api_text, msg=prompt)
                self.assertNotIn("I couldn't read that from the runtime state.", tg_text, msg=prompt)
                self.assertNotIn("I am DeepSeek", api_text, msg=prompt)
                self.assertNotIn("I am GPT", api_text, msg=prompt)
                self.assertNotIn("I am DeepSeek", tg_text, msg=prompt)
                self.assertNotIn("I am GPT", tg_text, msg=prompt)
                self.assertNotIn("Chat LLM is unavailable.", api_text, msg=prompt)
                self.assertNotIn("Chat LLM is unavailable.", tg_text, msg=prompt)
                self.assertNotIn("Run: python -m agent setup", api_text, msg=prompt)
                self.assertNotIn("Run: python -m agent setup", tg_text, msg=prompt)

                if expected_route == "generic_chat":
                    self.assertEqual("ollama:qwen3.5:4b", api_meta.get("model"), msg=prompt)
                if expected_route == "model_status":
                    self.assertIn("ollama:qwen3.5:4b", api_text, msg=prompt)
                if expected_route == "runtime_status":
                    self.assertIn("healthy and ready", api_text, msg=prompt)
                if prompt == "configure ollama":
                    self.assertEqual("setup_complete", (api_payload.get("setup") or {}).get("type"), msg=prompt)
                    self.assertIn("qwen3.5:4b", api_text, msg=prompt)

        self.assertEqual(6, len(inference_calls))
        for call in inference_calls:
            self.assertEqual("ollama", call["provider_override"])
            self.assertEqual("ollama:qwen3.5:4b", call["model_override"])

    def test_safe_mode_assistant_guard_sanitizes_identity_and_internal_errors(self) -> None:
        runtime = self._runtime()

        def _fake_route_inference(**kwargs):  # type: ignore[no-untyped-def]
            user_text = str(kwargs.get("user_text") or "").strip().lower()
            if "identity" in user_text:
                return {
                    "ok": True,
                    "text": "I am DeepSeek, and I can help with that.",
                    "provider": kwargs.get("provider_override"),
                    "model": kwargs.get("model_override"),
                    "duration_ms": 1,
                    "attempts": [],
                }
            return {
                "ok": False,
                "text": "Chat LLM is unavailable.\nNext: Run: python -m agent setup",
                "provider": kwargs.get("provider_override"),
                "model": kwargs.get("model_override"),
                "duration_ms": 1,
                "attempts": [],
                "error_kind": "llm_unavailable",
            }

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            identity_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "identity test"}],
                    "source_surface": "api",
                    "user_id": "api:assistant-guard",
                    "thread_id": "api:assistant-guard:thread",
                },
            )
            identity_tg = self._invoke_telegram_text(runtime, text="identity test", chat_id="guard")
            error_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "tell me a joke"}],
                    "source_surface": "api",
                    "user_id": "api:assistant-guard",
                    "thread_id": "api:assistant-guard:thread",
                },
            )
            error_tg = self._invoke_telegram_text(runtime, text="tell me a joke", chat_id="guard")

        identity_api_text = str((identity_api.get("assistant") or {}).get("content") or identity_api.get("message") or "")
        identity_tg_text = str(identity_tg.get("text") or "")
        error_api_text = str((error_api.get("assistant") or {}).get("content") or error_api.get("message") or "")
        error_tg_text = str(error_tg.get("text") or "")

        self.assertIn("Personal Agent", identity_api_text)
        self.assertIn("Personal Agent", identity_tg_text)
        self.assertNotIn("I am DeepSeek", identity_api_text)
        self.assertNotIn("I am DeepSeek", identity_tg_text)
        self.assertIn("Something went wrong while answering that", error_api_text)
        self.assertIn("Something went wrong while answering that", error_tg_text)
        self.assertNotIn("Chat LLM is unavailable.", error_api_text)
        self.assertNotIn("Chat LLM is unavailable.", error_tg_text)
        self.assertNotIn("python -m agent setup", error_api_text)
        self.assertNotIn("python -m agent setup", error_tg_text)

    def test_safe_mode_unmatched_input_stays_grounded_or_bounded_across_api_and_telegram(self) -> None:
        runtime = self._runtime()
        prompts = {
            "system report": "runtime_status",
            "system status": "runtime_status",
            "give me a system report": "runtime_status",
            "agent health report": "runtime_status",
            "help?": "assistant_clarification",
            "fix it": "setup_flow",
            "what is happening": "runtime_status",
            "do the thing": "assistant_clarification",
            "????": "assistant_fallback",
            "asdfasdf": "assistant_clarification",
            "uhh": "assistant_clarification",
        }

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt, expected_route in prompts.items():
                with self.subTest(prompt=prompt):
                    api_payload = self._invoke_chat_http(
                        runtime,
                        {
                            "messages": [{"role": "user", "content": prompt}],
                            "source_surface": "api",
                            "user_id": "api:unmatched",
                            "thread_id": "api:unmatched:thread",
                        },
                    )
                    telegram_payload = self._invoke_telegram_text(runtime, text=prompt, chat_id="84")

                    api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                    api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                    tg_text = str(telegram_payload.get("text") or "")

                    self.assertEqual(expected_route, api_meta.get("route"), msg=prompt)
                    self.assertEqual(expected_route, telegram_payload.get("selected_route"), msg=prompt)
                    self.assertFalse(bool(api_meta.get("used_llm", False)), msg=prompt)
                    self.assertFalse(bool(telegram_payload.get("used_llm", False)), msg=prompt)
                    self.assertNotEqual("generic_chat", api_meta.get("route"), msg=prompt)
                    self.assertNotEqual("generic_chat", telegram_payload.get("selected_route"), msg=prompt)
                    self.assertNotIn("i am deepseek", api_text.lower(), msg=prompt)
                    self.assertNotIn("i am gpt", api_text.lower(), msg=prompt)
                    self.assertNotIn("created by openai", api_text.lower(), msg=prompt)
                    self.assertNotIn("created by anthropic", api_text.lower(), msg=prompt)
                    self.assertNotIn("i am deepseek", tg_text.lower(), msg=prompt)
                    self.assertNotIn("i am gpt", tg_text.lower(), msg=prompt)
                    self.assertNotIn("created by openai", tg_text.lower(), msg=prompt)
                    self.assertNotIn("created by anthropic", tg_text.lower(), msg=prompt)
                    self.assertNotIn("what can i help you with", api_text.lower(), msg=prompt)
                    self.assertNotIn("what can i help you with", tg_text.lower(), msg=prompt)

                    if expected_route == "runtime_status":
                        self.assertIn("healthy and ready", api_text.lower(), msg=prompt)
                    if expected_route == "setup_flow":
                        self.assertIn("setup looks okay right now", api_text.lower(), msg=prompt)
                    if expected_route == "assistant_clarification":
                        self.assertIn("runtime status, model status, setup help, or a direct task", api_text.lower(), msg=prompt)
                    if expected_route == "assistant_fallback":
                        self.assertIn("i’m not sure what you want yet", api_text.lower(), msg=prompt)

    def test_safe_mode_interpretation_followups_use_previous_memory_report_and_stronger_model(self) -> None:
        runtime = self._runtime()
        explanation_calls: list[dict[str, object]] = []

        def _memory_report(self, user_id: str, text: str, nl_decision: dict[str, object]) -> OrchestratorResponse:  # type: ignore[no-untyped-def]
            _ = self
            _ = user_id
            _ = text
            _ = nl_decision
            return OrchestratorResponse(
                "Memory snapshot ready.\n\n*Resource Report*\n- browser: 6.5 GiB RSS\n- postgres: 1.2 GiB RSS",
                {
                    "summary": "Memory use is elevated because browser uses 6.5 GiB RSS and postgres uses 1.2 GiB RSS.",
                    "cards": [
                        {
                            "title": "Top memory processes",
                            "lines": ["browser: 6.5 GiB RSS", "postgres: 1.2 GiB RSS"],
                            "severity": "warn",
                        },
                        {
                            "title": "Memory overview",
                            "lines": ["82% of RAM is currently in use."],
                            "severity": "warn",
                        },
                    ],
                    "next_questions": ["Close heavy browser tabs", "Check memory again later"],
                },
            )

        def _fake_route_inference(**kwargs):  # type: ignore[no-untyped-def]
            explanation_calls.append(dict(kwargs))
            user_text = str(kwargs.get("user_text") or "").strip().lower()
            if "using up my memory the most" in user_text:
                text = "The biggest memory user is the browser at about 6.5 GiB RSS, with postgres well behind it."
            elif "should i worry" in user_text:
                text = "You should pay attention because RAM use is elevated, but it is not automatically an emergency from this snapshot alone."
            else:
                text = "Run free -h and top first. The browser is using 6.5 GiB RSS and postgres is using 1.2 GiB RSS."
            return {
                "ok": True,
                "text": text,
                "provider": kwargs.get("provider_override"),
                "model": kwargs.get("model_override"),
                "duration_ms": 1,
                "attempts": [],
            }

        with patch.object(
            runtime.orchestrator().__class__,
            "_handle_nl_observe",
            new=_memory_report,
        ), patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            initial = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "how much memory am I using?"}],
                    "source_surface": "api",
                    "user_id": "api:interp",
                    "thread_id": "api:interp:thread",
                },
            )
            top = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "so what is using up my memory the most?"}],
                    "source_surface": "api",
                    "user_id": "api:interp",
                    "thread_id": "api:interp:thread",
                },
            )
            explain = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "explain it to me"}],
                    "source_surface": "api",
                    "user_id": "api:interp",
                    "thread_id": "api:interp:thread",
                },
            )
            worry = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "should I worry?"}],
                    "source_surface": "api",
                    "user_id": "api:interp",
                    "thread_id": "api:interp:thread",
                },
            )
            concern = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "is there anything to be concerned about there?"}],
                    "source_surface": "api",
                    "user_id": "api:interp",
                    "thread_id": "api:interp:thread",
                },
            )

        initial_text = str((initial.get("assistant") or {}).get("content") or initial.get("message") or "")
        top_text = str((top.get("assistant") or {}).get("content") or top.get("message") or "")
        explain_text = str((explain.get("assistant") or {}).get("content") or explain.get("message") or "")
        worry_text = str((worry.get("assistant") or {}).get("content") or worry.get("message") or "")
        concern_text = str((concern.get("assistant") or {}).get("content") or concern.get("message") or "")

        self.assertEqual("operational_status", initial["meta"]["route"])
        self.assertIn("browser: 6.5 GiB RSS", initial_text)

        for payload, text in ((top, top_text), (explain, explain_text), (worry, worry_text), (concern, concern_text)):
            self.assertEqual("interpretation_followup", payload["meta"]["route"])
            self.assertTrue(payload["meta"]["used_llm"])
            self.assertNotIn("Memory snapshot ready.", text)
            self.assertNotIn("Chat LLM is unavailable.", text)
            self.assertNotIn("Run free -h", text)
            self.assertNotIn(" top ", f" {text.lower()} ")
            self.assertNotIn(" ps ", f" {text.lower()} ")
            self.assertRegex(text.lower(), r"(likely|usually|means|main driver|rather than)")
            self.assertRegex(text.lower(), r"(mildly high|normal|concerning|not automatically a crisis)")

        self.assertIn("browser", top_text.lower())
        self.assertIn("important part", explain_text.lower())
        self.assertIn("likely your web browser", explain_text.lower())
        self.assertIn("main issue", worry_text.lower())
        self.assertRegex(concern_text.lower(), r"(main issue|important part|pay attention|not automatically a crisis)")

        self.assertEqual(4, len(explanation_calls))
        for call in explanation_calls:
            self.assertEqual("ollama", call["provider_override"])
            self.assertEqual("ollama:qwen2.5:7b-instruct", call["model_override"])
            prompt_messages = call.get("messages") if isinstance(call.get("messages"), list) else []
            prompt_text = "\n".join(str(message.get("content") or "") for message in prompt_messages if isinstance(message, dict))
            self.assertIn("browser uses 6.5 GiB RSS", prompt_text)
            self.assertIn("postgres uses 1.2 GiB RSS", prompt_text)

    def test_safe_mode_interpretation_followup_falls_back_to_grounded_summary_when_explanation_fails(self) -> None:
        runtime = self._runtime()

        def _memory_report(self, user_id: str, text: str, nl_decision: dict[str, object]) -> OrchestratorResponse:  # type: ignore[no-untyped-def]
            _ = self
            _ = user_id
            _ = text
            _ = nl_decision
            return OrchestratorResponse(
                "Memory snapshot ready.\n\n*Resource Report*\n- browser: 6.5 GiB RSS\n- postgres: 1.2 GiB RSS",
                {
                    "summary": "Memory use is elevated because browser uses 6.5 GiB RSS and postgres uses 1.2 GiB RSS.",
                    "cards": [
                        {
                            "title": "Top memory processes",
                            "lines": ["browser: 6.5 GiB RSS", "postgres: 1.2 GiB RSS"],
                            "severity": "warn",
                        }
                    ],
                    "next_questions": ["Close heavy browser tabs"],
                },
            )

        with patch.object(
            runtime.orchestrator().__class__,
            "_handle_nl_observe",
            new=_memory_report,
        ), patch(
            "agent.orchestrator.route_inference",
            return_value={
                "ok": False,
                "text": "Chat LLM is unavailable.\nNext: Run: python -m agent setup",
                "provider": "ollama",
                "model": "ollama:qwen2.5:7b-instruct",
                "duration_ms": 1,
                "attempts": [],
                "error_kind": "llm_unavailable",
            },
        ):
            self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "how much memory am I using?"}],
                    "source_surface": "api",
                    "user_id": "api:interp-fallback",
                    "thread_id": "api:interp-fallback:thread",
                },
            )
            followup = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "is there anything to be concerned about there?"}],
                    "source_surface": "api",
                    "user_id": "api:interp-fallback",
                    "thread_id": "api:interp-fallback:thread",
                },
            )

        followup_text = str((followup.get("assistant") or {}).get("content") or followup.get("message") or "")
        self.assertEqual("interpretation_followup", followup["meta"]["route"])
        self.assertFalse(followup["meta"]["used_llm"])
        self.assertIn("I can still tell you the basics from the data I already gathered", followup_text)
        self.assertIn("browser", followup_text.lower())
        self.assertRegex(followup_text.lower(), r"(likely|usually|means|main driver|rather than)")
        self.assertRegex(followup_text.lower(), r"(mildly high|normal|concerning)")
        self.assertNotIn("Chat LLM is unavailable.", followup_text)
        self.assertNotIn("python -m agent setup", followup_text)
        self.assertNotIn("Run free -h", followup_text)
        self.assertNotIn(" top ", f" {followup_text.lower()} ")

    def test_safe_mode_thread_choice_reply_resumes_pending_request_instead_of_greeting(self) -> None:
        runtime = self._runtime()

        def _api_chat(text: str, *, user_id: str, thread_id: str) -> dict[str, object]:
            return self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": text}],
                    "source_surface": "api",
                    "user_id": user_id,
                    "thread_id": thread_id,
                },
            )

        def _fake_observe(self, user_id: str, text: str, nl_decision: dict[str, object]) -> OrchestratorResponse:  # type: ignore[no-untyped-def]
            lowered = text.lower()
            if "storage" in lowered or "disk" in lowered:
                return OrchestratorResponse(
                    "Disk status snapshot ready.\n\n*Storage Report*\n- No storage snapshots found yet.",
                    {"summary": "Disk status snapshot ready."},
                )
            return OrchestratorResponse(
                "CPU load 1m 0.00; memory 0.0% used.\n\n*Resource Report*\n- No resource snapshots found yet.",
                {"summary": "CPU load 1m 0.00; memory 0.0% used."},
            )

        runtime.set_thread_integrity_prompt(
            source="api",
            user_id="api:thread-choice",
            pending_text="how much memory am i using?",
            payload_template={
                "messages": [{"role": "user", "content": "how much memory am i using?"}],
                "source_surface": "api",
                "user_id": "api:thread-choice",
                "thread_id": "api:thread-choice:thread",
            },
        )
        with patch.object(
            runtime.orchestrator().__class__,
            "_handle_nl_observe",
            new=_fake_observe,
        ):
            same_payload = _api_chat("same thread", user_id="api:thread-choice", thread_id="api:thread-choice:thread")

        same_meta = same_payload.get("meta") if isinstance(same_payload.get("meta"), dict) else {}
        same_text = str((same_payload.get("assistant") or {}).get("content") or same_payload.get("message") or "")
        self.assertEqual("operational_status", same_meta.get("route"))
        self.assertNotIn("continuing the current thread", same_text.lower())
        self.assertNotIn("hello", same_text.lower())
        self.assertIn("memory", same_text.lower())

        runtime.set_thread_integrity_prompt(
            source="api",
            user_id="api:thread-choice",
            pending_text="how much RAM am i currently using?",
            payload_template={
                "messages": [{"role": "user", "content": "how much RAM am i currently using?"}],
                "source_surface": "api",
                "user_id": "api:thread-choice",
                "thread_id": "api:thread-choice:thread",
            },
        )
        with patch.object(
            runtime.orchestrator().__class__,
            "_handle_nl_observe",
            new=_fake_observe,
        ):
            new_payload = _api_chat("new thread", user_id="api:thread-choice", thread_id="api:thread-choice:thread")

        new_meta = new_payload.get("meta") if isinstance(new_payload.get("meta"), dict) else {}
        new_text = str((new_payload.get("assistant") or {}).get("content") or new_payload.get("message") or "")
        self.assertEqual("operational_status", new_meta.get("route"))
        self.assertNotIn("continuing the current thread", new_text.lower())
        self.assertNotIn("hello", new_text.lower())
        self.assertIn("memory", new_text.lower())

    def test_safe_mode_assistant_first_conversation_suppresses_router_chooser_for_normal_chat(self) -> None:
        runtime = self._runtime()
        inference_calls: list[dict[str, object]] = []

        def _api_chat(text: str, *, user_id: str, thread_id: str) -> dict[str, object]:
            return self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": text}],
                    "source_surface": "api",
                    "user_id": user_id,
                    "thread_id": thread_id,
                },
            )

        def _telegram_chat(text: str, *, chat_id: str) -> dict[str, object]:
            return self._invoke_telegram_text(runtime, text=text, chat_id=chat_id)

        def _fake_route_inference(**kwargs):  # type: ignore[no-untyped-def]
            inference_calls.append(dict(kwargs))
            user_text = str(kwargs.get("user_text") or "").strip().lower()
            if "help" in user_text:
                text = "I can chat, answer questions about the runtime, help with setup, and check local system status."
            elif user_text in {"a", "b", "chat", "model switch"}:
                text = f"I’m treating '{user_text}' as a normal assistant message."
            else:
                text = "I can help."
            return {
                "ok": True,
                "text": text,
                "provider": kwargs.get("provider_override"),
                "model": kwargs.get("model_override"),
                "duration_ms": 1,
                "attempts": [],
            }

        prompts = [
            ("can you help me", "api:assistant", "api:assistant:thread", "assistant", "i can chat"),
            ("model switch", "api:assistant", "api:assistant:thread", "assistant", "normal assistant message"),
            ("chat", "api:assistant", "api:assistant:thread", "assistant", "normal assistant message"),
            ("A", "api:assistant", "api:assistant:thread", "assistant", "normal assistant message"),
            ("B", "api:assistant", "api:assistant:thread", "assistant", "normal assistant message"),
        ]

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            for prompt, user_id, thread_id, chat_id, expected_text in prompts:
                api_payload = _api_chat(prompt, user_id=user_id, thread_id=thread_id)
                tg_payload = _telegram_chat(prompt, chat_id=chat_id)
                api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                tg_text = str(tg_payload.get("text") or "")

                self.assertEqual("generic_chat", api_meta.get("route"), msg=prompt)
                self.assertEqual("generic_chat", tg_payload.get("selected_route"), msg=prompt)
                self.assertTrue(bool(api_meta.get("used_llm", False)), msg=prompt)
                self.assertTrue(bool(tg_payload.get("used_llm", False)), msg=prompt)
                self.assertNotEqual("needs_clarification", api_payload.get("error_kind"), msg=prompt)
                self.assertNotIn("chat, ask, or model check/switch", api_text, msg=prompt)
                self.assertNotIn("chat, ask, or model check/switch", tg_text, msg=prompt)
                self.assertNotIn("Choose A, B, or C", api_text, msg=prompt)
                self.assertNotIn("A)", api_text, msg=prompt)
                self.assertNotIn("I couldn't read that from the runtime state.", api_text, msg=prompt)
                self.assertNotIn("I couldn't read that from the runtime state.", tg_text, msg=prompt)
                self.assertIn(expected_text, api_text.lower(), msg=prompt)
                self.assertIn(expected_text, tg_text.lower(), msg=prompt)
                self.assertEqual("ollama:qwen3.5:4b", api_meta.get("model"), msg=prompt)

        self.assertEqual(10, len(inference_calls))
        for call in inference_calls:
            self.assertEqual("ollama", call["provider_override"])
            self.assertEqual("ollama:qwen3.5:4b", call["model_override"])

    def test_safe_mode_capability_answers_are_grounded_in_real_subsystems(self) -> None:
        runtime = self._runtime()
        prompts = [
            "what can you do?",
            "what skills do you have?",
            "what abilities do you have?",
            "what skills/abilities do you have?",
            "what agentic abilities do you have?",
            "what skills do you have access to?",
            "what abilities do you have access to?",
            "what tools do you have?",
            "what can you help me with?",
        ]

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt in prompts:
                api_payload = self._invoke_chat_http(
                    runtime,
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "source_surface": "api",
                        "user_id": "api:capabilities",
                        "thread_id": "api:capabilities:thread",
                    },
                )
                telegram_payload = self._invoke_telegram_text(runtime, text=prompt, chat_id="capabilities")

                api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                tg_text = str(telegram_payload.get("text") or "")

                self.assertEqual("assistant_capabilities", api_meta.get("route"), msg=prompt)
                self.assertEqual("assistant_capabilities", telegram_payload.get("selected_route"), msg=prompt)
                self.assertFalse(bool(api_meta.get("used_llm", False)), msg=prompt)
                self.assertFalse(bool(telegram_payload.get("used_llm", False)), msg=prompt)

                for text in (api_text, tg_text):
                    lowered = text.lower()
                    self.assertIn("system inspection", lowered, msg=prompt)
                    self.assertIn("runtime and model status", lowered, msg=prompt)
                    self.assertIn("provider repair and switching", lowered, msg=prompt)
                    self.assertIn("local memory", lowered, msg=prompt)
                    self.assertIn("scheduler and daily brief", lowered, msg=prompt)
                    self.assertIn("safe mode", lowered, msg=prompt)
                    self.assertNotIn("i can process text", lowered, msg=prompt)
                    self.assertNotIn("i can generate responses", lowered, msg=prompt)
                    self.assertNotIn("available 24/7", lowered, msg=prompt)
                    self.assertNotIn("many kinds of questions", lowered, msg=prompt)
                    self.assertNotIn("i am gpt", lowered, msg=prompt)
                    self.assertNotIn("i am deepseek", lowered, msg=prompt)
                    self.assertNotIn("i'm having trouble", lowered, msg=prompt)
                    self.assertNotIn("something went wrong while answering that", lowered, msg=prompt)

    def test_safe_mode_model_availability_and_memory_questions_are_grounded(self) -> None:
        runtime = self._runtime()
        runtime.orchestrator().db.set_preference("response_style", "concise")
        runtime.orchestrator().db.set_preference("daily_brief_enabled", "off")
        runtime.orchestrator().db.add_open_loop("renew passport", "2026-03-20", priority=2)

        def _memory_report(self, user_id: str, text: str, nl_decision: dict[str, object]) -> OrchestratorResponse:  # type: ignore[no-untyped-def]
            _ = self
            _ = user_id
            _ = text
            _ = nl_decision
            return OrchestratorResponse(
                "Memory snapshot ready.\n\n*Resource Report*\n- browser: 6.5 GiB RSS\n- postgres: 1.2 GiB RSS",
                {
                    "summary": "82% of RAM is currently in use.",
                    "cards": [
                        {
                            "title": "Top memory processes",
                            "lines": ["browser: 6.5 GiB RSS", "postgres: 1.2 GiB RSS"],
                            "severity": "warn",
                        }
                    ],
                },
            )

        prompts = [
            ("are there others available to switch to easily?", "model_status"),
            ("what other models are available?", "model_status"),
            ("what other models are ready to switch to?", "model_status"),
            ("what ollama models do we have downloaded?", "model_status"),
            ("do we have any other local models?", "model_status"),
            ("do we have any other models downloaded?", "model_status"),
            ("what is currently in your memory files?", "agent_memory"),
            ("what do you remember?", "agent_memory"),
            ("show my open loops", "agent_memory"),
            ("how much memory am I using?", "operational_status"),
            ("how much RAM am I using?", "operational_status"),
        ]

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")), patch.object(
            runtime.orchestrator().__class__,
            "_handle_nl_observe",
            new=_memory_report,
        ):
            for prompt, expected_route in prompts:
                api_payload = self._invoke_chat_http(
                    runtime,
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "source_surface": "api",
                        "user_id": "api:grounding",
                        "thread_id": "api:grounding:thread",
                    },
                )
                telegram_payload = self._invoke_telegram_text(runtime, text=prompt, chat_id="grounding")

                api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                tg_text = str(telegram_payload.get("text") or "")

                self.assertEqual(expected_route, api_meta.get("route"), msg=prompt)
                self.assertEqual(expected_route, telegram_payload.get("selected_route"), msg=prompt)
                if expected_route in {"model_status", "agent_memory"}:
                    self.assertFalse(bool(api_meta.get("used_llm", False)), msg=prompt)
                    self.assertFalse(bool(telegram_payload.get("used_llm", False)), msg=prompt)
                if expected_route == "model_status":
                    for text in (api_text, tg_text):
                        lowered = text.lower()
                        self.assertIn("ollama:qwen3.5:4b", lowered, msg=prompt)
                        self.assertIn("ollama:qwen2.5:7b-instruct", lowered, msg=prompt)
                        self.assertNotIn("claude", lowered, msg=prompt)
                        self.assertNotIn("deepseek", lowered, msg=prompt)
                        self.assertNotIn("llava", lowered, msg=prompt)
                        self.assertNotIn("alibaba cloud", lowered, msg=prompt)
                        self.assertNotIn("real-time access", lowered, msg=prompt)
                        self.assertNotIn("hugging face", lowered, msg=prompt)
                        self.assertNotIn("not installed yet", lowered, msg=prompt)
                        if any(token in prompt.lower() for token in ("downloaded", "local", "installed", "ollama")):
                            self.assertIn("local", lowered, msg=prompt)
                if expected_route == "agent_memory":
                    for text in (api_text, tg_text):
                        lowered = text.lower()
                        self.assertNotIn("resource report", lowered, msg=prompt)
                        self.assertNotIn("ram is currently in use", lowered, msg=prompt)
                        if "open loops" in prompt.lower():
                            self.assertIn("renew passport", lowered, msg=prompt)
                            self.assertIn("open loops", lowered, msg=prompt)
                        else:
                            self.assertTrue(
                                "useful memory" in lowered
                                or "do not have much saved about you yet" in lowered
                                or "preferences i know" in lowered,
                                msg=prompt,
                            )
                            self.assertIn("open loops", lowered, msg=prompt)
                if expected_route == "operational_status":
                    for text in (api_text, tg_text):
                        lowered = text.lower()
                        self.assertIn("resource report", lowered, msg=prompt)
                        self.assertNotIn("local saved memory", lowered, msg=prompt)
                        self.assertNotIn("open loops", lowered, msg=prompt)

    def test_safe_mode_model_lifecycle_questions_are_grounded(self) -> None:
        runtime = self._runtime()
        save_model_manager_state(
            model_manager_state_path_for_runtime(runtime),
            {
                "schema_version": 1,
                "targets": {
                    "ollama:llava:7b": {
                        "target_key": "ollama:llava:7b",
                        "target_type": "model",
                        "provider_id": "ollama",
                        "model_id": "ollama:llava:7b",
                        "state": "downloading",
                        "message": "Installing ollama:llava:7b.",
                    },
                    "ollama:deepseek-r1:7b": {
                        "target_key": "ollama:deepseek-r1:7b",
                        "target_type": "model",
                        "provider_id": "ollama",
                        "model_id": "ollama:deepseek-r1:7b",
                        "state": "failed",
                        "message": "Ollama create failed.",
                        "error_kind": "ollama_create_failed",
                    },
                },
            },
        )

        prompts = [
            (
                "is ollama:qwen2.5:7b-instruct installed?",
                ("ollama:qwen2.5:7b-instruct", "installed and ready"),
            ),
            (
                "what models are downloading?",
                ("ollama:llava:7b", "downloading right now"),
            ),
            (
                "did ollama:qwen2.5:7b-instruct install successfully?",
                ("ollama:qwen2.5:7b-instruct", "installed and ready"),
            ),
            (
                "what model installs failed?",
                ("ollama:deepseek-r1:7b", "failed"),
            ),
            (
                "what is the status of ollama:deepseek-r1:7b?",
                ("ollama:deepseek-r1:7b", "reason: ollama create failed"),
            ),
        ]

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt, expected_terms in prompts:
                api_payload = self._invoke_chat_http(
                    runtime,
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "source_surface": "api",
                        "user_id": "api:lifecycle",
                        "thread_id": "api:lifecycle:thread",
                    },
                )
                telegram_payload = self._invoke_telegram_text(runtime, text=prompt, chat_id="lifecycle")

                api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                tg_text = str(telegram_payload.get("text") or "")

                self.assertEqual("model_status", api_meta.get("route"), msg=prompt)
                self.assertEqual("model_status", telegram_payload.get("selected_route"), msg=prompt)
                self.assertFalse(bool(api_meta.get("used_llm", False)), msg=prompt)
                self.assertFalse(bool(telegram_payload.get("used_llm", False)), msg=prompt)
                for text in (api_text.lower(), tg_text.lower()):
                    for expected in expected_terms:
                        self.assertIn(expected, text, msg=prompt)
                    self.assertNotIn("chat llm is unavailable", text, msg=prompt)
                    self.assertNotIn("python -m agent setup", text, msg=prompt)

    def test_safe_mode_model_lifecycle_explains_when_acquisition_is_blocked(self) -> None:
        runtime = self._runtime()
        truth = runtime.runtime_truth_service()
        lifecycle_payload = {
            "counts": {
                "not_installed": 1,
                "queued": 0,
                "downloading": 0,
                "installed": 0,
                "installed_not_ready": 0,
                "ready": 0,
                "failed": 0,
            },
            "models": [
                {
                    "target_key": "ollama:qwen2.5:3b-instruct",
                    "target_type": "model",
                    "provider_id": "ollama",
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "lifecycle_state": "not_installed",
                    "acquirable": True,
                    "acquisition_state": "blocked_by_policy",
                    "acquisition_reason": "approved for local acquisition, but SAFE MODE currently blocks installs",
                    "availability_state": "install_blocked",
                    "availability_reason": "approved for local acquisition, but SAFE MODE currently blocks installs",
                }
            ],
            "queued_targets": [],
            "downloading_targets": [],
            "failed_targets": [],
            "active_operations": [],
        }

        with patch.object(runtime, "runtime_truth_service", return_value=truth), patch.object(
            truth,
            "model_lifecycle_status",
            return_value=lifecycle_payload,
        ), patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            api_payload = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what is the status of ollama:qwen2.5:3b-instruct?"}],
                    "source_surface": "api",
                    "user_id": "api:lifecycle",
                    "thread_id": "api:lifecycle:thread",
                },
            )
            telegram_payload = self._invoke_telegram_text(
                runtime,
                text="what is the status of ollama:qwen2.5:3b-instruct?",
                chat_id="lifecycle",
            )

        api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "").lower()
        tg_text = str(telegram_payload.get("text") or "").lower()
        self.assertIn("does not let me download or install it", api_text)
        self.assertIn("does not let me download or install it", tg_text)

    def test_safe_mode_install_style_prompt_for_missing_target_stays_grounded(self) -> None:
        runtime = self._runtime()
        truth = runtime.runtime_truth_service()
        lifecycle_payload = {
            "counts": {
                "not_installed": 0,
                "queued": 0,
                "downloading": 0,
                "installed": 0,
                "installed_not_ready": 0,
                "ready": 0,
                "failed": 0,
            },
            "models": [],
            "queued_targets": [],
            "downloading_targets": [],
            "failed_targets": [],
            "active_operations": [],
        }

        with patch.object(runtime, "runtime_truth_service", return_value=truth), patch.object(
            truth,
            "model_lifecycle_status",
            return_value=lifecycle_payload,
        ), patch.object(
            truth,
            "model_controller_policy_status",
            return_value={"allow_install_pull": False},
        ), patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            api_payload = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "can you install ollama:qwen2.5-coder:7b?"}],
                    "source_surface": "api",
                    "user_id": "api:install-style",
                    "thread_id": "api:install-style:thread",
                },
            )
            telegram_payload = self._invoke_telegram_text(
                runtime,
                text="can you install ollama:qwen2.5-coder:7b?",
                chat_id="install-style",
            )

        api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
        api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "").lower()
        tg_text = str(telegram_payload.get("text") or "").lower()

        self.assertEqual("model_status", api_meta.get("route"))
        self.assertEqual("model_status", telegram_payload.get("selected_route"))
        self.assertFalse(bool(api_meta.get("used_llm", False)))
        self.assertFalse(bool(telegram_payload.get("used_llm", False)))
        self.assertIn("blocking install/download/import actions", api_text)
        self.assertIn("blocking install/download/import actions", tg_text)
        self.assertNotIn("alibaba cloud", api_text)
        self.assertNotIn("alibaba cloud", tg_text)
        self.assertNotIn("large language model", api_text)
        self.assertNotIn("large language model", tg_text)

    def test_safe_mode_mode_query_stays_grounded(self) -> None:
        runtime = self._runtime()

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            api_payload = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what mode am i in?"}],
                    "source_surface": "api",
                    "user_id": "api:mode",
                    "thread_id": "api:mode:thread",
                },
            )
            telegram_payload = self._invoke_telegram_text(
                runtime,
                text="what mode am i in?",
                chat_id="mode",
            )

        api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
        api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
        tg_text = str(telegram_payload.get("text") or "")

        self.assertEqual("model_policy_status", api_meta.get("route"))
        self.assertEqual("model_policy_status", telegram_payload.get("selected_route"))
        self.assertFalse(bool(api_meta.get("used_llm", False)))
        self.assertFalse(bool(telegram_payload.get("used_llm", False)))
        self.assertIn("Mode: SAFE MODE.", api_text)
        self.assertIn("Mode: SAFE MODE.", tg_text)
        self.assertIn("Status: SAFE MODE is the current baseline.", api_text)
        self.assertIn("Status: SAFE MODE is the current baseline.", tg_text)
        self.assertIn("Blocked: remote switching and install/download/import.", api_text)
        self.assertIn("Blocked: remote switching and install/download/import.", tg_text)
        self.assertIn("Controlled Mode only starts if you turn it on explicitly.", api_text)
        self.assertIn("Controlled Mode only starts if you turn it on explicitly.", tg_text)
        self.assertIn("approval", api_text.lower())
        self.assertIn("approval", tg_text.lower())

    def test_safe_mode_memory_summaries_use_real_saved_context(self) -> None:
        runtime = self._runtime()
        orchestrator = runtime.orchestrator()

        def _seed_memory(user_id: str, thread_id: str) -> None:
            orchestrator.db.set_preference("response_style", "concise")
            orchestrator.db.set_preference("daily_brief_enabled", "off")
            orchestrator.db.add_open_loop("finish the safe mode docs", "2026-03-20", priority=2)
            project_id = orchestrator.db.add_project("personal-agent", "assistant-first stabilization")
            orchestrator.db.add_task(project_id, "Finish personality pass", 30, 5)
            orchestrator._memory_runtime.set_current_topic(user_id, topic="safe mode stabilization")
            orchestrator._memory_runtime.record_user_request(
                user_id,
                "stabilize the assistant-first safe mode baseline",
            )
            orchestrator.db.add_thread_anchor(
                thread_id,
                "Safe mode stabilization",
                json.dumps(["assistant-first baseline", "grounded memory summaries"], ensure_ascii=True),
                "Finish the transcript harness.",
            )

        _seed_memory("api:memory-summary", "api:memory-summary:thread")
        _seed_memory("telegram:memory-summary", "telegram:memory-summary:thread")

        prompts = [
            ("what do you remember about me?", "memory_summary"),
            ("what are we working on?", "working_context"),
            ("what do you know about my system?", "system_context"),
        ]

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt, expected_kind in prompts:
                api_payload = self._invoke_chat_http(
                    runtime,
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "source_surface": "api",
                        "user_id": "api:memory-summary",
                        "thread_id": "api:memory-summary:thread",
                    },
                )
                telegram_payload = self._invoke_telegram_text(runtime, text=prompt, chat_id="memory-summary")

                api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                api_setup = api_payload.get("setup") if isinstance(api_payload.get("setup"), dict) else {}
                api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                tg_text = str(telegram_payload.get("text") or "")

                self.assertEqual("agent_memory", api_meta.get("route"), msg=prompt)
                self.assertEqual("agent_memory", telegram_payload.get("selected_route"), msg=prompt)
                self.assertFalse(bool(api_meta.get("used_llm", False)), msg=prompt)
                self.assertFalse(bool(telegram_payload.get("used_llm", False)), msg=prompt)
                self.assertEqual(expected_kind, api_setup.get("kind"), msg=prompt)

                for text in (api_text, tg_text):
                    lowered = text.lower()
                    self.assertNotIn("database", lowered, msg=prompt)
                    self.assertNotIn("sqlite", lowered, msg=prompt)
                    if expected_kind == "memory_summary":
                        self.assertIn("preferences i know", lowered, msg=prompt)
                        self.assertIn("open loops i am tracking", lowered, msg=prompt)
                        self.assertIn("current working context", lowered, msg=prompt)
                    elif expected_kind == "working_context":
                        self.assertIn("safe mode stabilization", lowered, msg=prompt)
                        self.assertIn("finish the safe mode docs", lowered, msg=prompt)
                        self.assertIn("pick up from that context", lowered, msg=prompt)
                    else:
                        self.assertIn("running locally on", lowered, msg=prompt)
                        self.assertIn("ollama:qwen3.5:4b", lowered, msg=prompt)
                        self.assertIn("ollama", lowered, msg=prompt)

    def test_safe_mode_memory_summary_is_honest_when_memory_is_thin(self) -> None:
        runtime = self._runtime()

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            api_payload = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what do you remember about me?"}],
                    "source_surface": "api",
                    "user_id": "api:memory-thin",
                    "thread_id": "api:memory-thin:thread",
                },
            )

        api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
        api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
        lowered = api_text.lower()

        self.assertEqual("agent_memory", api_meta.get("route"))
        self.assertFalse(bool(api_meta.get("used_llm", False)))
        self.assertIn("do not have much saved about you yet", lowered)
        self.assertNotIn("database", lowered)
        self.assertNotIn("sqlite", lowered)

    def test_safe_mode_time_date_and_model_scout_action_prompts_use_grounded_capabilities(self) -> None:
        runtime = self._runtime(model_scout_enabled=False)
        runtime.add_provider_model(
            "ollama",
            {
                "model": "nanbeige-chat:4b",
                "capabilities": ["chat"],
                "quality_rank": 5,
                "available": True,
                "max_context_tokens": 8192,
            },
        )
        runtime._health_monitor.state["models"]["ollama:nanbeige-chat:4b"] = {
            "provider_id": "ollama",
            "status": "ok",
            "last_checked_at": 123,
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]
        fixed_now = datetime(2026, 3, 17, 14, 5, tzinfo=timezone.utc)

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")), patch.object(
            runtime.orchestrator(),
            "_assistant_local_now",
            return_value=fixed_now,
        ):
            time_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what time is it?"}],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )
            time_tg = self._invoke_telegram_text(runtime, text="what time is it?", chat_id="actions")
            natural_time_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "hey can you check the time for me please"}],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )
            current_time_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what's the time right now"}],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )
            date_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what day is it?"}],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )
            models_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what other models are available?"}],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )
            scout_direct_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "can you run the model scout and see if any of those nanbeige models are actually any good for us?",
                        }
                    ],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )
            scout_direct_tg = self._invoke_telegram_text(
                runtime,
                text="can you run the model scout and see if any of those nanbeige models are actually any good for us?",
                chat_id="actions",
            )
            scout_plain_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "run the model scout"}],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )
            scout_plain_tg = self._invoke_telegram_text(runtime, text="run the model scout", chat_id="actions")
            scout_retry_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "try the model scout again"}],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )
            scout_followup_api = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "check them"}],
                    "source_surface": "api",
                    "user_id": "api:actions",
                    "thread_id": "api:actions:thread",
                },
            )

        time_meta = time_api.get("meta") if isinstance(time_api.get("meta"), dict) else {}
        date_meta = date_api.get("meta") if isinstance(date_api.get("meta"), dict) else {}
        natural_time_meta = natural_time_api.get("meta") if isinstance(natural_time_api.get("meta"), dict) else {}
        current_time_meta = current_time_api.get("meta") if isinstance(current_time_api.get("meta"), dict) else {}
        models_meta = models_api.get("meta") if isinstance(models_api.get("meta"), dict) else {}
        scout_direct_meta = scout_direct_api.get("meta") if isinstance(scout_direct_api.get("meta"), dict) else {}
        scout_plain_meta = scout_plain_api.get("meta") if isinstance(scout_plain_api.get("meta"), dict) else {}
        scout_retry_meta = scout_retry_api.get("meta") if isinstance(scout_retry_api.get("meta"), dict) else {}
        scout_followup_meta = scout_followup_api.get("meta") if isinstance(scout_followup_api.get("meta"), dict) else {}

        time_text = str((time_api.get("assistant") or {}).get("content") or time_api.get("message") or "")
        date_text = str((date_api.get("assistant") or {}).get("content") or date_api.get("message") or "")
        natural_time_text = str((natural_time_api.get("assistant") or {}).get("content") or natural_time_api.get("message") or "")
        current_time_text = str((current_time_api.get("assistant") or {}).get("content") or current_time_api.get("message") or "")
        models_text = str((models_api.get("assistant") or {}).get("content") or models_api.get("message") or "")
        scout_direct_text = str((scout_direct_api.get("assistant") or {}).get("content") or scout_direct_api.get("message") or "")
        scout_plain_text = str((scout_plain_api.get("assistant") or {}).get("content") or scout_plain_api.get("message") or "")
        scout_retry_text = str((scout_retry_api.get("assistant") or {}).get("content") or scout_retry_api.get("message") or "")
        scout_followup_text = str((scout_followup_api.get("assistant") or {}).get("content") or scout_followup_api.get("message") or "")
        scout_direct_tg_text = str(scout_direct_tg.get("text") or "")
        scout_plain_tg_text = str(scout_plain_tg.get("text") or "")

        self.assertEqual("action_tool", time_meta.get("route"))
        self.assertEqual("action_tool", time_tg.get("selected_route"))
        self.assertFalse(bool(time_meta.get("used_llm", False)))
        self.assertFalse(bool(time_tg.get("used_llm", False)))
        self.assertEqual(["local_time"], list(time_meta.get("used_tools") or []))
        self.assertIn("2:05 PM", time_text)
        self.assertIn("UTC", time_text)
        self.assertNotIn("Chat LLM is unavailable.", time_text)

        for meta, text in ((natural_time_meta, natural_time_text), (current_time_meta, current_time_text)):
            self.assertEqual("action_tool", meta.get("route"))
            self.assertFalse(bool(meta.get("used_llm", False)))
            self.assertEqual(["local_time"], list(meta.get("used_tools") or []))
            self.assertIn("2:05 PM", text)
            self.assertIn("UTC", text)
            self.assertNotIn("Chat LLM is unavailable.", text)

        self.assertEqual("action_tool", date_meta.get("route"))
        self.assertFalse(bool(date_meta.get("used_llm", False)))
        self.assertIn("Tuesday, March 17, 2026", date_text)
        self.assertIn("UTC", date_text)

        self.assertEqual("model_status", models_meta.get("route"))
        self.assertIn("ollama:nanbeige-chat:4b", models_text)

        self.assertEqual("action_tool", scout_direct_meta.get("route"))
        self.assertEqual("action_tool", scout_direct_tg.get("selected_route"))
        self.assertFalse(bool(scout_direct_meta.get("used_llm", False)))
        self.assertFalse(bool(scout_direct_tg.get("used_llm", False)))
        self.assertEqual(["model_scout"], list(scout_direct_meta.get("used_tools") or []))
        self.assertIn("nanbeige", scout_direct_text.lower())
        self.assertIn("ollama:qwen3.5:4b", scout_direct_text.lower())
        self.assertIn("nanbeige", scout_direct_tg_text.lower())
        self.assertNotIn("disabled", scout_direct_text.lower())
        self.assertNotIn("disabled", scout_direct_tg_text.lower())
        self.assertNotIn("what are you referring to", scout_direct_text.lower())
        self.assertNotIn("what are you referring to", scout_direct_tg_text.lower())

        for meta, text in ((scout_plain_meta, scout_plain_text), (scout_retry_meta, scout_retry_text)):
            self.assertEqual("action_tool", meta.get("route"))
            self.assertFalse(bool(meta.get("used_llm", False)))
            self.assertEqual(["model_scout"], list(meta.get("used_tools") or []))
            self.assertIn("qwen3.5:4b", text.lower())
            self.assertIn("qwen2.5:7b-instruct", text.lower())
            self.assertIn("test ollama:qwen2.5:7b-instruct without adopting it", text.lower())
            self.assertNotIn("disabled", text.lower())
        self.assertEqual("action_tool", scout_plain_tg.get("selected_route"))
        self.assertFalse(bool(scout_plain_tg.get("used_llm", False)))
        self.assertIn("qwen2.5:7b-instruct", scout_plain_tg_text.lower())
        self.assertIn("test ollama:qwen2.5:7b-instruct without adopting it", scout_plain_tg_text.lower())
        self.assertNotIn("disabled", scout_plain_tg_text.lower())

        self.assertEqual("action_tool", scout_followup_meta.get("route"))
        self.assertFalse(bool(scout_followup_meta.get("used_llm", False)))
        self.assertIn("nanbeige", scout_followup_text.lower())
        self.assertIn("qwen2.5:7b-instruct", scout_followup_text.lower())
        self.assertNotIn("what are you referring to", scout_followup_text.lower())

    def test_safe_mode_model_scout_v2_advises_then_controller_can_test_switch_and_roll_back(self) -> None:
        runtime = self._runtime(model_scout_enabled=False)

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")), patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True, "provider": "ollama", "model_id": "ollama:qwen2.5:7b-instruct"}),
        ), patch.object(
            runtime,
            "rollback_defaults",
            side_effect=AssertionError("assistant switch back must not use operator rollback"),
        ):
            recommendation = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "is there a better model I should use?"}],
                    "source_surface": "api",
                    "user_id": "api:scout-v2",
                    "thread_id": "api:scout-v2:thread",
                },
            )
            test_target = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "test this model without adopting it"}],
                    "source_surface": "api",
                    "user_id": "api:scout-v2",
                    "thread_id": "api:scout-v2:thread",
                },
            )
            switched = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch temporarily"}],
                    "source_surface": "api",
                    "user_id": "api:scout-v2",
                    "thread_id": "api:scout-v2:thread",
                },
            )
            after_switch = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:scout-v2",
                    "thread_id": "api:scout-v2:thread",
                },
            )
            rollback = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch back"}],
                    "source_surface": "api",
                    "user_id": "api:scout-v2",
                    "thread_id": "api:scout-v2:thread",
                },
            )
            rolled_back = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:scout-v2",
                    "thread_id": "api:scout-v2:thread",
                },
            )

        recommendation_meta = recommendation.get("meta") if isinstance(recommendation.get("meta"), dict) else {}
        recommendation_text = str((recommendation.get("assistant") or {}).get("content") or recommendation.get("message") or "")
        test_text = str((test_target.get("assistant") or {}).get("content") or test_target.get("message") or "")
        switched_text = str((switched.get("assistant") or {}).get("content") or switched.get("message") or "")
        after_switch_text = str((after_switch.get("assistant") or {}).get("content") or after_switch.get("message") or "")
        rollback_text = str((rollback.get("assistant") or {}).get("content") or rollback.get("message") or "")
        rolled_back_text = str((rolled_back.get("assistant") or {}).get("content") or rolled_back.get("message") or "")

        self.assertEqual("action_tool", recommendation_meta.get("route"))
        self.assertFalse(bool(recommendation_meta.get("used_llm", False)))
        self.assertEqual(["model_scout"], list(recommendation_meta.get("used_tools") or []))
        self.assertIn("ollama:qwen2.5:7b-instruct", recommendation_text.lower())
        self.assertIn("no change has been made.", recommendation_text.lower())
        self.assertIn(
            "you can test it, switch to it temporarily, or make it the default if you want.",
            recommendation_text.lower(),
        )
        self.assertEqual("ollama:qwen3.5:4b", str(runtime.runtime_truth_service().current_chat_target_status().get("model") or "").strip())
        self.assertIn("without switching", test_text.lower())
        self.assertIn("ollama:qwen2.5:7b-instruct", switched_text.lower())
        self.assertIn("ollama:qwen2.5:7b-instruct", after_switch_text.lower())
        self.assertEqual("Now using ollama:qwen3.5:4b for chat.", rollback_text)
        self.assertIn("ollama:qwen3.5:4b", rolled_back_text.lower())

    def test_safe_mode_cheap_cloud_recommendation_prompts_stay_grounded_and_local_only(self) -> None:
        runtime = self._runtime(model_scout_enabled=False)
        prompts = (
            "what cheap cloud model should I use?",
            "what low-cost cloud model should I use for coding?",
            "what budget remote model should I use?",
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt in prompts:
                with self.subTest(prompt=prompt):
                    api_payload = self._invoke_chat_http(
                        runtime,
                        {
                            "messages": [{"role": "user", "content": prompt}],
                            "source_surface": "api",
                            "user_id": "api:cheap-cloud",
                            "thread_id": "api:cheap-cloud:thread",
                        },
                    )
                    telegram_payload = self._invoke_telegram_text(
                        runtime,
                        text=prompt,
                        chat_id="cheap-cloud",
                    )

                    api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                    api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "").lower()
                    tg_text = str(telegram_payload.get("text") or "").lower()

                    self.assertEqual("action_tool", api_meta.get("route"))
                    self.assertEqual("action_tool", telegram_payload.get("selected_route"))
                    self.assertFalse(bool(api_meta.get("used_llm", False)))
                    self.assertFalse(bool(telegram_payload.get("used_llm", False)))
                    self.assertIn("cheap cloud recommendation:", api_text)
                    self.assertIn("cheap cloud recommendation:", tg_text)
                    self.assertIn("reason: remote recommendations are not usable in this mode.", api_text)
                    self.assertIn("reason: remote recommendations are not usable in this mode.", tg_text)
                    self.assertIn("mode: safe mode.", api_text)
                    self.assertIn("mode: safe mode.", tg_text)
                    self.assertIn("no change has been made.", api_text)
                    self.assertIn("no change has been made.", tg_text)
                    self.assertIn("in safe mode, remote actions are blocked.", api_text)
                    self.assertIn("in safe mode, remote actions are blocked.", tg_text)
                    self.assertNotIn("compared with current:", api_text)
                    self.assertNotIn("compared with current:", tg_text)
                    self.assertIn("local-only right now", api_text)
                    self.assertIn("local-only right now", tg_text)
                    self.assertNotIn("cheap cloud option:", api_text)
                    self.assertNotIn("cheap cloud option:", tg_text)
                    self.assertNotIn("alibaba cloud", api_text)
                    self.assertNotIn("alibaba cloud", tg_text)
                    self.assertNotIn("curl ", api_text)
                    self.assertNotIn("curl ", tg_text)
                    self.assertNotIn("ollama pull", api_text)
                    self.assertNotIn("ollama pull", tg_text)

    def test_safe_mode_premium_recommendation_prompts_stay_grounded_and_local_only(self) -> None:
        runtime = self._runtime(model_scout_enabled=False)
        prompts = (
            "what premium coding model should I use?",
            "what premium model should I use for research?",
        )

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt in prompts:
                with self.subTest(prompt=prompt):
                    api_payload = self._invoke_chat_http(
                        runtime,
                        {
                            "messages": [{"role": "user", "content": prompt}],
                            "source_surface": "api",
                            "user_id": "api:premium-role",
                            "thread_id": "api:premium-role:thread",
                        },
                    )
                    telegram_payload = self._invoke_telegram_text(
                        runtime,
                        text=prompt,
                        chat_id="premium-role",
                    )

                    api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                    api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "").lower()
                    tg_text = str(telegram_payload.get("text") or "").lower()

                    self.assertEqual("action_tool", api_meta.get("route"))
                    self.assertEqual("action_tool", telegram_payload.get("selected_route"))
                    self.assertFalse(bool(api_meta.get("used_llm", False)))
                    self.assertFalse(bool(telegram_payload.get("used_llm", False)))
                    self.assertIn("premium", api_text)
                    self.assertIn("premium", tg_text)
                    self.assertIn("reason: remote recommendations are not usable in this mode.", api_text)
                    self.assertIn("reason: remote recommendations are not usable in this mode.", tg_text)
                    self.assertIn("mode: safe mode.", api_text)
                    self.assertIn("mode: safe mode.", tg_text)
                    self.assertIn("no change has been made.", api_text)
                    self.assertIn("no change has been made.", tg_text)
                    self.assertIn("in safe mode, remote actions are blocked.", api_text)
                    self.assertIn("in safe mode, remote actions are blocked.", tg_text)
                    self.assertNotIn("compared with current:", api_text)
                    self.assertNotIn("compared with current:", tg_text)
                    self.assertIn("local-only right now", api_text)
                    self.assertIn("local-only right now", tg_text)
                    self.assertNotIn("premium coding option:", api_text)
                    self.assertNotIn("premium research option:", api_text)
                    self.assertNotIn("premium coding option:", tg_text)
                    self.assertNotIn("premium research option:", tg_text)
                    self.assertNotIn("trouble reading the current runtime state", api_text)
                    self.assertNotIn("trouble reading the current runtime state", tg_text)
                    self.assertNotIn("alibaba cloud", api_text)
                    self.assertNotIn("alibaba cloud", tg_text)
                    self.assertNotIn("curl ", api_text)
                    self.assertNotIn("curl ", tg_text)

    def test_safe_mode_explicit_controller_phrases_execute_real_controller_paths(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "deepseek-r1:7b",
                "capabilities": ["chat"],
                "quality_rank": 7,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime._health_monitor.state["models"]["ollama:deepseek-r1:7b"] = {
            "provider_id": "ollama",
            "status": "ok",
            "last_checked_at": 123,
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")), patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True, "provider": "ollama", "model_id": "ollama:qwen2.5:7b-instruct", "message": "Hello!"}),
        ):
            test_target = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "test ollama:qwen2.5:7b-instruct without adopting it"}],
                    "source_surface": "api",
                    "user_id": "api:controller-explicit",
                    "thread_id": "api:controller-explicit:thread",
                },
            )
            temporary = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch temporarily to ollama:qwen2.5:7b-instruct"}],
                    "source_surface": "api",
                    "user_id": "api:controller-explicit",
                    "thread_id": "api:controller-explicit:thread",
                },
            )
            default_after_temporary = str(runtime.registry_document.get("defaults", {}).get("chat_model") or "").strip()
            switch_back = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch back"}],
                    "source_surface": "api",
                    "user_id": "api:controller-explicit",
                    "thread_id": "api:controller-explicit:thread",
                },
            )
            after_rollback = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:controller-explicit",
                    "thread_id": "api:controller-explicit:thread",
                },
            )
            temporary_for_default = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch temporarily to ollama:qwen2.5:7b-instruct"}],
                    "source_surface": "api",
                    "user_id": "api:controller-default",
                    "thread_id": "api:controller-default:thread",
                },
            )
            make_default = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "make ollama:deepseek-r1:7b the default"}],
                    "source_surface": "api",
                    "user_id": "api:controller-default",
                    "thread_id": "api:controller-default:thread",
                },
            )
            default_after_make_default = str(runtime.registry_document.get("defaults", {}).get("chat_model") or "").strip()
            active_after_default = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:controller-default",
                    "thread_id": "api:controller-default:thread",
                },
            )
            switch_back_after_default = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch back"}],
                    "source_surface": "api",
                    "user_id": "api:controller-default",
                    "thread_id": "api:controller-default:thread",
                },
            )

        test_meta = test_target.get("meta") if isinstance(test_target.get("meta"), dict) else {}
        test_text = str((test_target.get("assistant") or {}).get("content") or test_target.get("message") or "")
        temporary_meta = temporary.get("meta") if isinstance(temporary.get("meta"), dict) else {}
        temporary_text = str((temporary.get("assistant") or {}).get("content") or temporary.get("message") or "")
        temporary_for_default_meta = temporary_for_default.get("meta") if isinstance(temporary_for_default.get("meta"), dict) else {}
        default_meta = make_default.get("meta") if isinstance(make_default.get("meta"), dict) else {}
        default_text = str((make_default.get("assistant") or {}).get("content") or make_default.get("message") or "")
        active_after_default_text = str((active_after_default.get("assistant") or {}).get("content") or active_after_default.get("message") or "")
        switch_back_after_default_text = str((switch_back_after_default.get("assistant") or {}).get("content") or switch_back_after_default.get("message") or "")
        switch_back_meta = switch_back.get("meta") if isinstance(switch_back.get("meta"), dict) else {}
        switch_back_text = str((switch_back.get("assistant") or {}).get("content") or switch_back.get("message") or "")
        after_rollback_text = str((after_rollback.get("assistant") or {}).get("content") or after_rollback.get("message") or "")

        self.assertEqual("action_tool", test_meta.get("route"))
        self.assertFalse(bool(test_meta.get("used_llm", False)))
        self.assertIn("I tested ollama:qwen2.5:7b-instruct without switching.", test_text)
        self.assertIn("responded successfully", test_text)
        self.assertNotIn("hello!", test_text.lower())

        self.assertEqual("model_status", temporary_meta.get("route"))
        self.assertFalse(bool(temporary_meta.get("used_llm", False)))
        self.assertEqual("Temporarily using ollama:qwen2.5:7b-instruct for chat.", temporary_text)
        self.assertEqual(
            "ollama:qwen2.5:7b-instruct",
            str(runtime.runtime_truth_service().current_chat_target_status().get("model") or "").strip(),
        )
        self.assertEqual("ollama:qwen2.5:7b-instruct", default_after_temporary)

        self.assertEqual("model_status", switch_back_meta.get("route"))
        self.assertFalse(bool(switch_back_meta.get("used_llm", False)))
        self.assertEqual("Now using ollama:qwen3.5:4b for chat.", switch_back_text)
        self.assertIn("ollama:qwen3.5:4b", after_rollback_text.lower())

        self.assertEqual("model_status", temporary_for_default_meta.get("route"))
        self.assertFalse(bool(temporary_for_default_meta.get("used_llm", False)))
        self.assertEqual("model_status", default_meta.get("route"))
        self.assertFalse(bool(default_meta.get("used_llm", False)))
        self.assertIn("ollama:deepseek-r1:7b is now the default chat model.", default_text)
        self.assertIn("Chat is still using ollama:qwen2.5:7b-instruct.", default_text)
        self.assertEqual("ollama:deepseek-r1:7b", default_after_make_default)
        self.assertIn("ollama:qwen2.5:7b-instruct", active_after_default_text.lower())
        self.assertIn("do not have a recent trial model switch to roll back", switch_back_after_default_text.lower())
        self.assertIn("changing the default alone does not create a trial rollback", switch_back_after_default_text.lower())

    def test_safe_mode_chat_switch_blocks_remote_target_at_shared_boundary(self) -> None:
        runtime = self._runtime()
        os.environ["OPENROUTER_API_KEY"] = "sk-test"
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "quality_rank": 10,
                "available": True,
                "max_context_tokens": 131072,
            },
        )
        runtime._health_monitor.state["providers"]["openrouter"] = {
            "status": "ok",
            "last_checked_at": 123,
        }
        runtime._health_monitor.state["models"]["openrouter:openai/gpt-4o-mini"] = {
            "provider_id": "openrouter",
            "status": "ok",
            "last_checked_at": 123,
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            blocked = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch to openrouter:openai/gpt-4o-mini"}],
                    "source_surface": "api",
                    "user_id": "api:safe-mode-remote-switch",
                    "thread_id": "api:safe-mode-remote-switch:thread",
                },
            )

        blocked_meta = blocked.get("meta") if isinstance(blocked.get("meta"), dict) else {}
        blocked_text = str((blocked.get("assistant") or {}).get("content") or blocked.get("message") or "")
        self.assertEqual("model_status", blocked_meta.get("route"))
        self.assertFalse(bool(blocked.get("ok", True)))
        self.assertEqual("safe_mode_remote_switch_blocked", blocked.get("error_kind"))
        self.assertEqual("safe_mode_remote_switch_blocked", blocked_meta.get("error"))
        self.assertIn("Only local chat models can be selected right now.", blocked_text)
        target_status = runtime.safe_mode_target_status()
        self.assertEqual("ollama:qwen3.5:4b", target_status.get("effective_model"))
        self.assertTrue(bool(target_status.get("effective_local")))
        self.assertFalse(bool(target_status.get("explicit_override_active")))

    def test_safe_mode_help_me_get_this_working_without_prior_context_stays_in_grounded_setup_flow(self) -> None:
        runtime = self._runtime()
        truth = runtime.runtime_truth_service()
        inventory_payload = {
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
                    "active": True,
                    "availability_reason": "model health is down",
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                },
            ],
            "usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "other_usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "not_ready_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                }
            ],
        }

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
                "qualification_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                "degraded_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
            },
        ), patch.object(
            truth,
            "provider_status",
            return_value={
                "provider": "ollama",
                "provider_label": "Ollama",
                "known": True,
                "enabled": True,
                "local": True,
                "configured": True,
                "active": True,
                "secret_present": False,
                "health_status": "ok",
                "health_reason": None,
                "model_id": "ollama:qwen3.5:4b",
                "model_ids": ["ollama:qwen3.5:4b", "ollama:qwen2.5:3b-instruct"],
                "current_provider": "ollama",
                "current_model_id": "ollama:qwen3.5:4b",
                "effective_provider": "ollama",
                "effective_model_id": "ollama:qwen3.5:4b",
                "effective_active": True,
                "qualification_reason": None,
                "degraded_reason": None,
            },
        ), patch.object(
            truth,
            "model_inventory_status",
            return_value={
                "active_provider": "ollama",
                "active_model": "ollama:qwen3.5:4b",
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "models": [dict(row) for row in inventory_payload["models"]],
                "local_installed_models": [
                    dict(row)
                    for row in inventory_payload["models"]
                    if bool(row.get("local", False)) and bool(row.get("available", False))
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
                "models": [dict(row) for row in inventory_payload["models"]],
                "ready_now_models": [dict(row) for row in inventory_payload["usable_models"]],
                "other_ready_now_models": [dict(row) for row in inventory_payload["other_usable_models"]],
                "not_ready_models": [dict(row) for row in inventory_payload["not_ready_models"]],
            },
        ), patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True}),
        ):
            response = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "Help me get this working"}],
                    "source_surface": "api",
                    "user_id": "api:direct-help",
                    "thread_id": "api:direct-help:thread",
                },
            )

        response_text = str((response.get("assistant") or {}).get("content") or response.get("message") or "")
        response_meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
        lowered = response_text.lower()

        self.assertEqual("setup_flow", response_meta.get("route"))
        self.assertFalse(bool(response_meta.get("used_llm", False)))
        self.assertIn("setup needs attention right now", lowered)
        self.assertIn("chat is configured for ollama:qwen3.5:4b on ollama", lowered)
        self.assertIn("that model is not healthy right now", lowered)
        self.assertNotIn("docker", lowered)
        self.assertNotIn("env", lowered)
        self.assertNotIn("what can i help you with", lowered)

    def test_safe_mode_model_scout_hf_discovery_is_truthful_about_download_candidates(self) -> None:
        runtime = self._runtime(model_watch_hf_enabled=True)
        fake_hf_scan_body = {
            "ok": True,
            "trigger": "manual",
            "scan": {
                "ok": True,
                "enabled": True,
                "updates": [
                    {
                        "repo_id": "nanbeige/Nanbeige2-16B-Chat-GGUF",
                        "installability": "installable_ollama",
                    }
                ],
                "discovered_count": 1,
            },
            "proposal_created": True,
            "proposal": {
                "repo_id": "nanbeige/Nanbeige2-16B-Chat-GGUF",
                "installability": "installable_ollama",
            },
        }

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")), patch.object(
            runtime,
            "model_watch_hf_status",
            return_value={"ok": True, "enabled": True, "tracked_repos": 1, "discovered_count": 0},
        ), patch.object(
            runtime,
            "model_watch_hf_scan",
            return_value=(True, fake_hf_scan_body),
        ):
            payload = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "can you use the model scout to check for a better model we should download?"}],
                    "source_surface": "api",
                    "user_id": "api:scout-discovery",
                    "thread_id": "api:scout-discovery:thread",
                },
            )

        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        text = str((payload.get("assistant") or {}).get("content") or payload.get("message") or "")
        lowered = text.lower()

        self.assertEqual("action_tool", meta.get("route"))
        self.assertFalse(bool(meta.get("used_llm", False)))
        self.assertEqual(["model_scout"], list(meta.get("used_tools") or []))
        self.assertIn("nanbeige/nanbeige2-16b-chat-gguf", lowered)
        self.assertIn("not installed yet", lowered)
        self.assertNotIn("already installed", lowered)

    def test_safe_mode_direct_model_switch_requests_use_deterministic_switch_path(self) -> None:
        runtime = self._runtime()
        runtime.add_provider_model(
            "ollama",
            {
                "model": "deepseek-r1:7b",
                "capabilities": ["chat"],
                "quality_rank": 7,
                "available": True,
                "max_context_tokens": 32768,
            },
        )
        runtime._health_monitor.state["models"]["ollama:deepseek-r1:7b"] = {
            "provider_id": "ollama",
            "status": "ok",
            "last_checked_at": 123,
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            qwen_switch = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch to qwen2.5:7b-instruct"}],
                    "source_surface": "api",
                    "user_id": "api:switch",
                    "thread_id": "api:switch:thread",
                },
            )
            qwen_status = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:switch",
                    "thread_id": "api:switch:thread",
                },
            )
            deepseek_switch = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "use deepseek-r1:7b"}],
                    "source_surface": "api",
                    "user_id": "api:switch",
                    "thread_id": "api:switch:thread",
                },
            )
            deepseek_status = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:switch",
                    "thread_id": "api:switch:thread",
                },
            )
            switch_back = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "switch back"}],
                    "source_surface": "api",
                    "user_id": "api:switch",
                    "thread_id": "api:switch:thread",
                },
            )
            restored_status = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:switch",
                    "thread_id": "api:switch:thread",
                },
            )

        qwen_meta = qwen_switch.get("meta") if isinstance(qwen_switch.get("meta"), dict) else {}
        qwen_text = str((qwen_switch.get("assistant") or {}).get("content") or qwen_switch.get("message") or "")
        qwen_status_text = str((qwen_status.get("assistant") or {}).get("content") or qwen_status.get("message") or "")
        deepseek_meta = deepseek_switch.get("meta") if isinstance(deepseek_switch.get("meta"), dict) else {}
        deepseek_text = str((deepseek_switch.get("assistant") or {}).get("content") or deepseek_switch.get("message") or "")
        deepseek_status_text = str((deepseek_status.get("assistant") or {}).get("content") or deepseek_status.get("message") or "")
        switch_back_meta = switch_back.get("meta") if isinstance(switch_back.get("meta"), dict) else {}
        switch_back_text = str((switch_back.get("assistant") or {}).get("content") or switch_back.get("message") or "")
        restored_status_text = str((restored_status.get("assistant") or {}).get("content") or restored_status.get("message") or "")

        self.assertEqual("model_status", qwen_meta.get("route"))
        self.assertFalse(bool(qwen_meta.get("used_llm", False)))
        self.assertIn("ollama:qwen2.5:7b-instruct", qwen_text.lower())
        self.assertNotIn("deepseek", qwen_text.lower())
        self.assertIn("ollama:qwen2.5:7b-instruct", qwen_status_text.lower())

        self.assertEqual("model_status", deepseek_meta.get("route"))
        self.assertFalse(bool(deepseek_meta.get("used_llm", False)))
        self.assertIn("ollama:deepseek-r1:7b", deepseek_text.lower())
        self.assertNotIn("what are you referring to", deepseek_text.lower())
        self.assertIn("ollama:deepseek-r1:7b", deepseek_status_text.lower())

        self.assertEqual("model_status", switch_back_meta.get("route"))
        self.assertFalse(bool(switch_back_meta.get("used_llm", False)))
        self.assertIn("ollama:qwen2.5:7b-instruct", switch_back_text.lower())
        self.assertIn("ollama:qwen2.5:7b-instruct", restored_status_text.lower())

    def test_safe_mode_repair_followup_reuses_recent_unhealthy_runtime_context(self) -> None:
        runtime = self._runtime()
        truth = runtime.runtime_truth_service()

        def _provider_status(provider_id: str) -> dict[str, object]:
            provider_key = str(provider_id).strip().lower()
            if provider_key == "ollama":
                return {
                    "provider": "ollama",
                    "provider_label": "Ollama",
                    "known": True,
                    "enabled": True,
                    "local": True,
                    "configured": True,
                    "active": True,
                    "secret_present": False,
                    "health_status": "down",
                    "health_reason": "timeout while reaching Ollama",
                    "model_id": "ollama:qwen3.5:4b",
                    "model_ids": ["ollama:qwen3.5:4b"],
                    "current_provider": "ollama",
                    "current_model_id": "ollama:qwen3.5:4b",
                    "effective_provider": "ollama",
                    "effective_model_id": "ollama:qwen3.5:4b",
                    "effective_active": True,
                    "qualification_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                    "degraded_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                }
            return {
                "provider": provider_key,
                "provider_label": provider_key.title(),
                "known": False,
                "enabled": False,
                "local": False,
                "configured": False,
                "active": False,
                "secret_present": False,
                "health_status": "unknown",
                "health_reason": None,
                "model_id": None,
                "model_ids": [],
                "current_provider": "ollama",
                "current_model_id": "ollama:qwen3.5:4b",
                "effective_provider": "ollama",
                "effective_model_id": "ollama:qwen3.5:4b",
                "effective_active": False,
                "qualification_reason": None,
                "degraded_reason": None,
            }

        with patch.object(
            truth,
            "current_chat_target_status",
            return_value={
                "provider": "ollama",
                "model": "ollama:qwen3.5:4b",
                "ready": False,
                "health_status": "down",
                "provider_health_status": "down",
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
                "qualification_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                "degraded_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
            },
        ), patch.object(
            truth,
            "provider_status",
            side_effect=_provider_status,
        ), patch.object(
            runtime,
            "test_provider",
            return_value=(False, {"ok": False, "error": "timeout"}),
        ):
            first = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:repair",
                    "thread_id": "api:repair:thread",
                },
            )
            second = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "can you repair it?"}],
                    "source_surface": "api",
                    "user_id": "api:repair",
                    "thread_id": "api:repair:thread",
                },
            )

        first_text = str((first.get("assistant") or {}).get("content") or first.get("message") or "")
        second_text = str((second.get("assistant") or {}).get("content") or second.get("message") or "")
        second_meta = second.get("meta") if isinstance(second.get("meta"), dict) else {}

        self.assertIn("not responding right now", first_text.lower())
        self.assertEqual("setup_flow", second_meta.get("route"))
        self.assertNotEqual("needs_clarification", second.get("error_kind"))
        self.assertNotIn("Which of these is your goal", second_text)
        self.assertNotIn("Tell me whether you want chat, ask, or model check/switch.", second_text)
        self.assertNotIn("continuing the current thread", second_text.lower())
        self.assertIn("ollama is currently down", second_text.lower())
        self.assertIn("reconfigure ollama", second_text.lower())

    def test_safe_mode_unhealthy_model_followups_stay_in_grounded_repair_flow(self) -> None:
        runtime = self._runtime()
        truth = runtime.runtime_truth_service()
        inventory_payload = {
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
                    "active": True,
                    "availability_reason": "model health is down",
                },
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                },
            ],
            "usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "other_usable_models": [
                {
                    "model_id": "ollama:qwen2.5:3b-instruct",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": True,
                    "active": False,
                    "availability_reason": "ready",
                }
            ],
            "not_ready_models": [
                {
                    "model_id": "ollama:qwen3.5:4b",
                    "provider_id": "ollama",
                    "local": True,
                    "available": True,
                    "usable_now": False,
                    "active": True,
                    "availability_reason": "model health is down",
                }
            ],
        }

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
                "qualification_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
                "degraded_reason": "Configured default ollama:qwen3.5:4b on ollama is not currently healthy.",
            },
        ), patch.object(
            truth,
            "provider_status",
            return_value={
                "provider": "ollama",
                "provider_label": "Ollama",
                "known": True,
                "enabled": True,
                "local": True,
                "configured": True,
                "active": True,
                "secret_present": False,
                "health_status": "ok",
                "health_reason": None,
                "model_id": "ollama:qwen3.5:4b",
                "model_ids": ["ollama:qwen3.5:4b", "ollama:qwen2.5:3b-instruct"],
                "current_provider": "ollama",
                "current_model_id": "ollama:qwen3.5:4b",
                "effective_provider": "ollama",
                "effective_model_id": "ollama:qwen3.5:4b",
                "effective_active": True,
                "qualification_reason": None,
                "degraded_reason": None,
            },
        ), patch.object(
            truth,
            "model_inventory_status",
            return_value={
                "active_provider": "ollama",
                "active_model": "ollama:qwen3.5:4b",
                "configured_provider": "ollama",
                "configured_model": "ollama:qwen3.5:4b",
                "models": [dict(row) for row in inventory_payload["models"]],
                "local_installed_models": [
                    dict(row)
                    for row in inventory_payload["models"]
                    if bool(row.get("local", False)) and bool(row.get("available", False))
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
                "models": [dict(row) for row in inventory_payload["models"]],
                "ready_now_models": [dict(row) for row in inventory_payload["usable_models"]],
                "other_ready_now_models": [dict(row) for row in inventory_payload["other_usable_models"]],
                "not_ready_models": [dict(row) for row in inventory_payload["not_ready_models"]],
            },
        ), patch.object(
            runtime,
            "test_provider",
            return_value=(True, {"ok": True}),
        ):
            first = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what model are you using?"}],
                    "source_surface": "api",
                    "user_id": "api:repair-flow",
                    "thread_id": "api:repair-flow:thread",
                },
            )
            second = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "Help me get this working"}],
                    "source_surface": "api",
                    "user_id": "api:repair-flow",
                    "thread_id": "api:repair-flow:thread",
                },
            )
            third = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "What needs attention?"}],
                    "source_surface": "api",
                    "user_id": "api:repair-flow",
                    "thread_id": "api:repair-flow:thread",
                },
            )
            fourth = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "Check setup and explain what's wrong"}],
                    "source_surface": "api",
                    "user_id": "api:repair-flow",
                    "thread_id": "api:repair-flow:thread",
                },
            )

        first_text = str((first.get("assistant") or {}).get("content") or first.get("message") or "")
        self.assertIn("not healthy right now", first_text.lower())

        for payload in (second, third, fourth):
            text = str((payload.get("assistant") or {}).get("content") or payload.get("message") or "")
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            lowered = text.lower()
            self.assertEqual("setup_flow", meta.get("route"))
            self.assertNotEqual("needs_clarification", payload.get("error_kind"))
            self.assertIn("ollama is reachable", lowered)
            self.assertIn("current chat model ollama:qwen3.5:4b is not healthy right now", lowered)
            self.assertIn("1) recheck ollama:qwen3.5:4b now", lowered)
            self.assertIn("2) switch to ollama:qwen2.5:3b-instruct", lowered)
            self.assertNotIn("what can i help you with", lowered)
            self.assertNotIn("no chat model available", lowered)
            self.assertNotIn("start ollama locally", lowered)
            self.assertNotIn("install a local chat model", lowered)
            self.assertNotIn("which of these is your goal", lowered)
            self.assertNotIn("tell me whether you want chat, ask, or model check/switch", lowered)
            self.assertNotIn("continuing the current thread", lowered)

    def test_safe_mode_model_scout_is_honest_when_no_models_are_available(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path, model_scout_enabled=False))

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")), patch.object(
            runtime.runtime_truth_service(),
            "model_scout_v2_status",
            return_value={
                "type": "model_scout_v2",
                "active_provider": None,
                "active_model": None,
                "current_candidate": None,
                "recommended_candidate": None,
                "better_candidates": [],
                "candidate_rows": [],
                "not_ready_models": [],
                "selection": {
                    "ordered_candidates": [],
                    "rejected_candidates": [],
                    "switch_recommended": False,
                    "decision_reason": "no_candidate",
                    "decision_detail": "No chat-capable models are currently available.",
                },
                "source": "test-empty",
            },
        ):
            payload = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "can you run the model scout"}],
                    "source_surface": "api",
                    "user_id": "api:scout-disabled",
                    "thread_id": "api:scout-disabled:thread",
                },
            )

        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        text = str((payload.get("assistant") or {}).get("content") or payload.get("message") or "")
        lowered = text.lower()

        self.assertEqual("action_tool", meta.get("route"))
        self.assertFalse(bool(meta.get("used_llm", False)))
        self.assertEqual(["model_scout"], list(meta.get("used_tools") or []))
        self.assertIn("do not currently see any ready chat models", lowered)
        self.assertIn("configure a provider", lowered)
        self.assertNotIn("what are you referring to", lowered)

    def test_safe_mode_machine_stats_and_hardware_prompts_use_grounded_native_skills(self) -> None:
        runtime = self._runtime()
        calls = self._install_machine_skill_fakes(runtime)

        prompts = [
            ("what other pc stats can you find?", ("hardware_report", "resource_report", "storage_report")),
            ("can you tell what CPU and GPU I have?", ("hardware_report",)),
            ("can you see the GPU?", ("hardware_report",)),
            ("can you dig deeper into my system?", ("hardware_report", "resource_report", "storage_report")),
            ("run a system check", ("hardware_report", "resource_report", "storage_report")),
        ]

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt, expected_tools in prompts:
                api_payload = self._invoke_chat_http(
                    runtime,
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "source_surface": "api",
                        "user_id": "api:machine",
                        "thread_id": "api:machine:thread",
                    },
                )
                telegram_payload = self._invoke_telegram_text(runtime, text=prompt, chat_id="machine")

                api_meta = api_payload.get("meta") if isinstance(api_payload.get("meta"), dict) else {}
                api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                tg_text = str(telegram_payload.get("text") or "")

                self.assertEqual("operational_status", api_meta.get("route"), msg=prompt)
                self.assertEqual("operational_status", telegram_payload.get("selected_route"), msg=prompt)
                self.assertFalse(bool(api_meta.get("used_llm", False)), msg=prompt)
                self.assertFalse(bool(telegram_payload.get("used_llm", False)), msg=prompt)
                self.assertEqual(list(expected_tools), list(api_meta.get("used_tools") or []), msg=prompt)

                for text in (api_text, tg_text):
                    lowered = text.lower()
                    self.assertIn("amd ryzen 9 7900x", lowered, msg=prompt)
                    self.assertIn("nvidia rtx 4080", lowered, msg=prompt)
                    self.assertIn("64 gib", lowered, msg=prompt)
                    self.assertNotIn("bot status", lowered, msg=prompt)
                    self.assertNotIn("scheduler", lowered, msg=prompt)
                    self.assertNotIn("database", lowered, msg=prompt)
                    self.assertNotIn("chat llm is unavailable", lowered, msg=prompt)

        self.assertGreaterEqual(calls["hardware"], 3)
        self.assertGreaterEqual(calls["resource"], 1)
        self.assertGreaterEqual(calls["storage"], 1)

    def test_safe_mode_deeper_system_followup_reuses_grounded_inspection_path(self) -> None:
        runtime = self._runtime()
        calls = self._install_machine_skill_fakes(runtime)

        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            first = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "what other pc stats can you find?"}],
                    "source_surface": "api",
                    "user_id": "api:machine-followup",
                    "thread_id": "api:machine-followup:thread",
                },
            )
            second = self._invoke_chat_http(
                runtime,
                {
                    "messages": [{"role": "user", "content": "can you run a check and see if you can learn more?"}],
                    "source_surface": "api",
                    "user_id": "api:machine-followup",
                    "thread_id": "api:machine-followup:thread",
                },
            )

        self.assertEqual("operational_status", first["meta"]["route"])
        self.assertEqual("operational_status", second["meta"]["route"])
        self.assertFalse(bool(second["meta"].get("used_llm", False)))
        self.assertEqual(["hardware_report", "resource_report", "storage_report"], list(second["meta"].get("used_tools") or []))
        second_text = str((second.get("assistant") or {}).get("content") or second.get("message") or "").lower()
        self.assertIn("amd ryzen 9 7900x", second_text)
        self.assertIn("nvidia rtx 4080", second_text)
        self.assertIn("/data", second_text)
        self.assertNotIn("bot status", second_text)
        self.assertGreaterEqual(calls["hardware"], 2)
        self.assertGreaterEqual(calls["resource"], 2)
        self.assertGreaterEqual(calls["storage"], 2)

    def test_safe_mode_day_planning_stays_user_facing(self) -> None:
        runtime = self._runtime()
        orchestrator = runtime.orchestrator()
        orchestrator.db.add_open_loop("renew passport", "2026-03-20", priority=1)
        orchestrator.db.add_task(None, "Finish report draft", 45, 4)
        orchestrator.db.add_task(None, "Reply to priority email", 10, 3)

        prompts = ("can you help me plan my day?", "what should i work on today?")
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")):
            for prompt in prompts:
                api_payload = self._invoke_chat_http(
                    runtime,
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "source_surface": "api",
                        "user_id": "api:plan",
                        "thread_id": "api:plan:thread",
                    },
                )
                telegram_payload = self._invoke_telegram_text(runtime, text=prompt, chat_id="plan")

                api_text = str((api_payload.get("assistant") or {}).get("content") or api_payload.get("message") or "")
                tg_text = str(telegram_payload.get("text") or "")

                self.assertFalse(bool((api_payload.get("meta") or {}).get("used_llm", False)), msg=prompt)
                for text in (api_text, tg_text):
                    lowered = text.lower()
                    self.assertIn("renew passport", lowered, msg=prompt)
                    self.assertIn("finish report draft", lowered, msg=prompt)
                    self.assertNotIn("db tasks table", lowered, msg=prompt)
                    self.assertNotIn("internal task", lowered, msg=prompt)
                    self.assertNotIn("chat llm is unavailable", lowered, msg=prompt)
