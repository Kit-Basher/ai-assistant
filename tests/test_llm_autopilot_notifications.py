from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config


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
        autopilot_notify_enabled=True,
        autopilot_notify_rate_limit_seconds=0,
        autopilot_notify_dedupe_window_seconds=0,
        llm_notifications_allow_send=None,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _state(*, default_model: str | None, health_status: str = "ok") -> dict[str, object]:
    return {
        "defaults": {
            "routing_mode": "auto",
            "default_provider": "ollama",
            "default_model": default_model,
            "allow_remote_fallback": True,
        },
        "providers": {
            "ollama": {
                "enabled": True,
                "available": True,
                "health": {
                    "status": health_status,
                    "cooldown_until": None,
                    "down_since": None,
                    "failure_streak": 0,
                },
            }
        },
        "models": {
            "ollama:qwen2.5:3b-instruct": {
                "enabled": True,
                "available": True,
                "routable": True,
                "health": {
                    "status": health_status,
                    "cooldown_until": None,
                    "down_since": None,
                    "failure_streak": 0,
                },
            }
        },
    }


class TestLLMAutopilotNotifications(unittest.TestCase):
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

    def test_no_changes_produces_no_notification_and_audit_reason(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b-instruct")
        after = _state(default_model="ollama:qwen2.5:3b-instruct")
        with patch("agent.api_server.time.time", return_value=10_000):
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=[],
                trigger="scheduler",
            )
        self.assertEqual("no_changes", result["reason"])
        self.assertEqual([], runtime.llm_notifications(limit=5)["notifications"])
        entries = runtime.get_audit(limit=5)["entries"]
        self.assertEqual("llm.autopilot.notify", entries[0]["action"])
        self.assertEqual("no_changes", entries[0]["reason"])

    def test_defaults_change_notifies_and_dedupe_skips_repeat(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=3600,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b")
        after = _state(default_model="ollama:qwen2.5:3b-instruct")
        with patch.object(runtime, "_send_telegram_message", return_value=None), patch.object(
            runtime, "_resolve_telegram_target", return_value=("token", "chat-1")
        ), patch("agent.api_server.time.time", return_value=20_000):
            first = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
            second = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
        self.assertEqual("sent", first["reason"])
        self.assertEqual("skipped_dedupe", second["reason"])
        rows = runtime.llm_notifications(limit=5)["notifications"]
        self.assertEqual(2, len(rows))
        self.assertIn("Defaults: default_model -> ollama:qwen2.5:3b-instruct", rows[0]["message"])
        entries = runtime.get_audit(limit=5)["entries"]
        self.assertEqual("skipped_dedupe", entries[0]["reason"])

    def test_rate_limit_skips_second_cycle(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                autopilot_notify_rate_limit_seconds=3600,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b")
        after_first = _state(default_model="ollama:qwen2.5:3b-instruct")
        after_second = _state(default_model="ollama:qwen2.5:7b-instruct")
        with patch.object(runtime, "_send_telegram_message", return_value=None), patch.object(
            runtime, "_resolve_telegram_target", return_value=("token", "chat-1")
        ), patch("agent.api_server.time.time", return_value=30_000):
            first = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after_first,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
            second = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=after_first,
                after_state=after_second,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
        self.assertEqual("sent", first["reason"])
        self.assertEqual("skipped_rate_limit", second["reason"])
        entries = runtime.get_audit(limit=5)["entries"]
        self.assertEqual("skipped_rate_limit", entries[0]["reason"])

    def test_quiet_hours_skip_is_recorded(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                autopilot_notify_quiet_start_hour=1,
                autopilot_notify_quiet_end_hour=6,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b")
        after = _state(default_model="ollama:qwen2.5:3b-instruct")
        with patch("agent.api_server.time.time", return_value=1_706_836_800):
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
        self.assertEqual("skipped_quiet_hours", result["reason"])
        entries = runtime.get_audit(limit=5)["entries"]
        self.assertEqual("skipped_quiet_hours", entries[0]["reason"])

    def test_non_loopback_without_send_permission_uses_local_delivery_only(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                llm_notifications_allow_send=None,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("0.0.0.0", 8765)
        before = _state(default_model="ollama:qwen2.5:3b")
        after = _state(default_model="ollama:qwen2.5:3b-instruct")
        with patch("agent.api_server.time.time", return_value=40_000), patch.object(
            runtime, "_resolve_telegram_target", return_value=("token", "chat-1")
        ), patch.object(runtime, "_send_telegram_message", return_value=None) as send_mock:
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
        self.assertEqual("sent", result["reason"])
        self.assertEqual("local", result["delivered_to"])
        send_mock.assert_not_called()
        rows = runtime.llm_notifications(limit=5)["notifications"]
        self.assertEqual(1, len(rows))
        self.assertEqual("local", rows[0]["delivered_to"])
        entries = runtime.get_audit(limit=5)["entries"]
        self.assertEqual("allow", entries[0]["decision"])
        self.assertEqual("sent", entries[0]["reason"])

    def test_loopback_auto_send_allowed_when_telegram_configured(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                llm_notifications_allow_send=None,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b")
        after = _state(default_model="ollama:qwen2.5:3b-instruct")
        with patch.object(runtime, "_send_telegram_message", return_value=None), patch.object(
            runtime, "_resolve_telegram_target", return_value=("token", "chat-1")
        ), patch("agent.api_server.time.time", return_value=50_000):
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
        self.assertEqual("sent", result["reason"])
        self.assertEqual("telegram", result["delivered_to"])
        rows = runtime.llm_notifications(limit=5)["notifications"]
        self.assertEqual(1, len(rows))
        self.assertEqual("telegram", rows[0]["delivered_to"])
        summary = runtime.llm_health_summary()["health"]
        self.assertEqual("sent", summary["notifications"]["last_outcome"])
        self.assertTrue(summary["notifications"]["last_hash"])

    def test_fixit_prompt_delivery_persists_state_and_audits_prompt(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        runtime.add_provider_model(
            "ollama",
            {
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "available": True,
            },
        )
        runtime.add_provider_model(
            "openrouter",
            {
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat"],
                "available": False,
            },
        )
        runtime.set_default_chat_model("ollama:qwen2.5:3b-instruct")
        runtime._health_monitor.state["providers"] = {
            "openrouter": {
                "status": "down",
                "last_error_kind": "provider_unavailable",
                "status_code": 502,
                "last_checked_at": 100,
                "cooldown_until": 0,
                "down_since": 100,
                "failure_streak": 12,
                "next_probe_at": 160,
            },
            "ollama": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
        }
        runtime._health_monitor.state["models"] = {
            "ollama:qwen2.5:3b-instruct": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
            "openrouter:openai/gpt-4o-mini": {
                "status": "down",
                "last_error_kind": "provider_unavailable",
                "status_code": 502,
                "last_checked_at": 100,
                "cooldown_until": 0,
                "down_since": 100,
                "failure_streak": 12,
                "next_probe_at": 160,
            },
        }
        runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]

        with patch.object(runtime, "llm_status", side_effect=AssertionError("should not be called")), patch.object(
            runtime, "_resolve_telegram_target", return_value=("token", "123456789")
        ), patch.object(runtime, "_send_telegram_message", return_value=None):
            result = runtime._deliver_autopilot_notification(  # type: ignore[attr-defined]
                message="OpenRouter is unavailable.",
                allow_remote=True,
            )
        self.assertTrue(result.ok)
        self.assertEqual("telegram", result.delivered_to)

        wizard_state = runtime._llm_fixit_store.load()  # type: ignore[attr-defined]
        self.assertTrue(bool(wizard_state.get("active")))
        self.assertEqual("awaiting_choice", str(wizard_state.get("step") or ""))
        choices = wizard_state.get("choices") if isinstance(wizard_state.get("choices"), list) else []
        self.assertTrue(choices)
        self.assertEqual("local_only", str((choices[0] or {}).get("id") or ""))

        entries = runtime.get_audit(limit=20)["entries"]
        prompt_rows = [row for row in entries if row.get("action") == "telegram.fixit.prompt_shown"]
        self.assertTrue(prompt_rows)
        prompt_row = prompt_rows[0]
        params = (
            prompt_row.get("params_redacted")
            if isinstance(prompt_row.get("params_redacted"), dict)
            else {}
        )
        self.assertEqual("***6789", str(params.get("chat_id_redacted") or ""))
        self.assertEqual("needs_user_choice", str(params.get("status") or ""))
        self.assertEqual("openrouter_down", str(params.get("issue_code") or ""))
        self.assertEqual("awaiting_choice", str(params.get("step") or ""))

    def test_loopback_without_telegram_falls_back_to_local_delivery(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                llm_notifications_allow_send=None,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b")
        after = _state(default_model="ollama:qwen2.5:3b-instruct")
        with patch.object(runtime, "_resolve_telegram_target", return_value=(None, None)), patch(
            "agent.api_server.time.time", return_value=60_000
        ):
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
        self.assertEqual("sent", result["reason"])
        self.assertEqual("local", result["delivered_to"])
        rows = runtime.llm_notifications(limit=5)["notifications"]
        self.assertEqual("local", rows[0]["delivered_to"])

    def test_notification_includes_catalog_and_cleanup_extra_changes_deterministically(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b-instruct")
        after = _state(default_model="ollama:qwen2.5:3b-instruct")
        with patch.object(runtime, "_resolve_telegram_target", return_value=(None, None)), patch(
            "agent.api_server.time.time",
            return_value=70_000,
        ):
            first = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["catalog_refresh"],
                extra_changes=[
                    "Cleanup: model ollama:llama3 marked unavailable (missing_from_catalog)",
                    "Catalog: pricing updated for openrouter:openai/gpt-4o-mini",
                    "Cleanup: model ollama:llama3 marked unavailable (missing_from_catalog)",
                ],
                trigger="scheduler",
            )
            second = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["catalog_refresh"],
                extra_changes=[
                    "Cleanup: model ollama:llama3 marked unavailable (missing_from_catalog)",
                    "Catalog: pricing updated for openrouter:openai/gpt-4o-mini",
                ],
                trigger="scheduler",
            )
        self.assertEqual("sent", first["reason"])
        self.assertEqual("sent", second["reason"])
        rows = runtime.llm_notifications(limit=5)["notifications"]
        self.assertEqual(2, len(rows))
        self.assertEqual(rows[0]["dedupe_hash"], rows[1]["dedupe_hash"])
        message = str(rows[0]["message"])
        self.assertIn("Catalog: pricing updated for openrouter:openai/gpt-4o-mini", message)
        self.assertIn("Cleanup: model ollama:llama3 marked unavailable (missing_from_catalog)", message)
        self.assertLess(
            message.index("Catalog: pricing updated for openrouter:openai/gpt-4o-mini"),
            message.index("Cleanup: model ollama:llama3 marked unavailable (missing_from_catalog)"),
        )

    def test_health_only_noise_without_threshold_crossing_is_suppressed(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b-instruct", health_status="down")
        after = copy.deepcopy(before)
        after["providers"]["ollama"]["health"]["failure_streak"] = 2
        after["providers"]["ollama"]["health"]["cooldown_until"] = 123456
        after["models"]["ollama:qwen2.5:3b-instruct"]["health"]["failure_streak"] = 2
        after["models"]["ollama:qwen2.5:3b-instruct"]["health"]["cooldown_until"] = 123456
        with patch("agent.api_server.time.time", return_value=80_000):
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["health_probe"],
                trigger="scheduler",
            )
        self.assertEqual("no_changes", result["reason"])
        self.assertEqual([], runtime.llm_notifications(limit=5)["notifications"])

    def test_health_threshold_crossing_sends_notification(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        before = _state(default_model="ollama:qwen2.5:3b-instruct", health_status="down")
        before["providers"]["ollama"]["health"]["failure_streak"] = 2
        before["models"]["ollama:qwen2.5:3b-instruct"]["health"]["failure_streak"] = 2
        after = copy.deepcopy(before)
        after["providers"]["ollama"]["health"]["failure_streak"] = 3
        after["models"]["ollama:qwen2.5:3b-instruct"]["health"]["failure_streak"] = 3
        with patch("agent.api_server.time.time", return_value=81_000):
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["health_probe"],
                trigger="scheduler",
            )
        self.assertEqual("sent", result["reason"])
        self.assertEqual(1, len(runtime.llm_notifications(limit=5)["notifications"]))

    def test_telegram_delivery_uses_friendly_summary_not_raw_diff(self) -> None:
        runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                autopilot_notify_rate_limit_seconds=0,
                autopilot_notify_dedupe_window_seconds=0,
            )
        )
        runtime.set_listening("127.0.0.1", 8765)
        runtime._health_monitor.state["providers"] = {
            "ollama": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
            "openrouter": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "last_checked_at": 100,
                "cooldown_until": None,
                "down_since": None,
                "failure_streak": 0,
                "next_probe_at": 160,
            },
        }
        before = _state(default_model="ollama:qwen2.5:3b")
        after = _state(default_model="ollama:qwen2.5:3b-instruct")
        sent_messages: list[str] = []

        def _capture_send(_token: str, _chat_id: str, text: str) -> None:
            sent_messages.append(text)

        healthy_status = {
            "ok": True,
            "default_provider": "ollama",
            "default_model": "ollama:qwen2.5:3b-instruct",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "safe_mode": {"paused": False, "reason": "not_paused"},
            "providers": [
                {"id": "ollama", "enabled": True, "health": {"status": "ok", "last_error_kind": None, "status_code": None}},
                {"id": "openrouter", "enabled": True, "health": {"status": "ok", "last_error_kind": None, "status_code": None}},
            ],
            "models": [
                {
                    "id": "ollama:qwen2.5:3b-instruct",
                    "provider": "ollama",
                    "enabled": True,
                    "available": True,
                    "routable": True,
                    "health": {"status": "ok", "last_error_kind": None, "status_code": None},
                }
            ],
        }
        with patch.object(runtime, "_resolve_telegram_target", return_value=("token", "chat-1")), patch.object(
            runtime, "_send_telegram_message", side_effect=_capture_send
        ), patch.object(runtime, "llm_status", return_value=healthy_status), patch(
            "agent.api_server.time.time", return_value=82_000
        ):
            result = runtime._process_scheduler_notification_cycle(  # type: ignore[attr-defined]
                before_state=before,
                after_state=after,
                reasons=["autoconfig_repair"],
                trigger="scheduler",
            )
        self.assertEqual("sent", result["reason"])
        self.assertEqual(1, len(sent_messages))
        self.assertIn("Default model changed to", sent_messages[0])
        self.assertNotIn("LLM Autopilot updated configuration", sent_messages[0])
        self.assertNotIn("Defaults: default_model ->", sent_messages[0])


if __name__ == "__main__":
    unittest.main()
