from __future__ import annotations

import unittest

from agent.recovery_contract import (
    RECOVERY_API_DOWN,
    RECOVERY_DEGRADED_READ_ONLY,
    RECOVERY_LLM_UNAVAILABLE,
    RECOVERY_LOCK_CONFLICT,
    RECOVERY_TELEGRAM_DOWN,
    RECOVERY_TOKEN_INVALID,
    detect_recovery_mode,
    recovery_next_action,
    recovery_summary,
)


class TestRecoveryContract(unittest.TestCase):
    def test_api_down(self) -> None:
        mode = detect_recovery_mode(api_reachable=False)
        self.assertEqual(RECOVERY_API_DOWN, mode)
        self.assertIn("personal-agent-api.service", recovery_next_action(mode))

    def test_telegram_down(self) -> None:
        ready_payload = {
            "ready": False,
            "telegram": {"enabled": True, "configured": True, "state": "stopped"},
            "runtime_status": {"runtime_mode": "DEGRADED"},
        }
        mode = detect_recovery_mode(ready_payload=ready_payload, api_reachable=True)
        self.assertEqual(RECOVERY_TELEGRAM_DOWN, mode)
        self.assertIn("personal-agent-telegram.service", recovery_next_action(mode))

    def test_token_invalid(self) -> None:
        ready_payload = {"telegram": {"enabled": True, "configured": False, "state": "disabled_missing_token"}}
        mode = detect_recovery_mode(ready_payload=ready_payload)
        self.assertEqual(RECOVERY_TOKEN_INVALID, mode)
        self.assertIn("telegram:bot_token", recovery_next_action(mode))

    def test_telegram_disabled_optional_does_not_trigger_telegram_recovery(self) -> None:
        ready_payload = {
            "ready": True,
            "telegram": {"enabled": False, "configured": False, "state": "disabled_optional"},
            "runtime_status": {"runtime_mode": "READY"},
        }
        llm_status = {
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "active_provider_health": {"status": "ok"},
            "active_model_health": {"status": "ok"},
        }
        mode = detect_recovery_mode(ready_payload=ready_payload, llm_status=llm_status)
        self.assertNotEqual(RECOVERY_TELEGRAM_DOWN, mode)
        self.assertNotEqual(RECOVERY_TOKEN_INVALID, mode)

    def test_lock_conflict(self) -> None:
        mode = detect_recovery_mode(failure_code="lock_conflict")
        self.assertEqual(RECOVERY_LOCK_CONFLICT, mode)
        self.assertIn("duplicate Telegram pollers", recovery_next_action(mode))

    def test_llm_unavailable(self) -> None:
        llm_status = {
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "active_provider_health": {"status": "down"},
            "active_model_health": {"status": "down"},
        }
        mode = detect_recovery_mode(llm_status=llm_status)
        self.assertEqual(RECOVERY_LLM_UNAVAILABLE, mode)
        self.assertIn("python -m agent setup", recovery_next_action(mode))
        self.assertIn("unavailable", recovery_summary(mode).lower())

    def test_degraded_read_only(self) -> None:
        ready_payload = {"runtime_status": {"runtime_mode": "DEGRADED"}}
        llm_status = {
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "active_provider_health": {"status": "ok"},
            "active_model_health": {"status": "ok"},
            "safe_mode": {"safe_mode": True},
        }
        mode = detect_recovery_mode(ready_payload=ready_payload, llm_status=llm_status)
        self.assertEqual(RECOVERY_DEGRADED_READ_ONLY, mode)
        self.assertEqual("Run: python -m agent doctor", recovery_next_action(mode))


if __name__ == "__main__":
    unittest.main()
