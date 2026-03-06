from __future__ import annotations

import unittest

from agent.onboarding_contract import (
    ONBOARDING_DEGRADED,
    ONBOARDING_LLM_MISSING,
    ONBOARDING_NOT_STARTED,
    ONBOARDING_READY,
    ONBOARDING_SERVICES_DOWN,
    ONBOARDING_TOKEN_MISSING,
    detect_onboarding_state,
    onboarding_next_action,
    onboarding_steps,
    onboarding_summary,
)


class TestOnboardingContract(unittest.TestCase):
    def test_detect_not_started(self) -> None:
        self.assertEqual(ONBOARDING_NOT_STARTED, detect_onboarding_state())

    def test_detect_token_missing(self) -> None:
        ready_payload = {
            "ready": False,
            "phase": "ready",
            "telegram": {"configured": False, "state": "disabled_missing_token"},
            "runtime_status": {"runtime_mode": "BOOTSTRAP_REQUIRED", "failure_code": "telegram_token_missing"},
        }
        self.assertEqual(ONBOARDING_TOKEN_MISSING, detect_onboarding_state(ready_payload=ready_payload))
        self.assertEqual(
            "Run: python -m agent.secrets set telegram:bot_token",
            onboarding_next_action(ONBOARDING_TOKEN_MISSING, ready_payload=ready_payload),
        )

    def test_detect_llm_missing(self) -> None:
        status = {
            "default_provider": "ollama",
            "resolved_default_model": None,
            "active_provider_health": {"status": "down"},
            "active_model_health": {"status": "down"},
        }
        self.assertEqual(ONBOARDING_LLM_MISSING, detect_onboarding_state(llm_status=status))
        self.assertIn("No chat model available", onboarding_summary(ONBOARDING_LLM_MISSING))

    def test_detect_services_down(self) -> None:
        ready_payload = {
            "ready": False,
            "phase": "starting",
            "telegram": {"configured": True, "state": "stopped"},
            "runtime_status": {"runtime_mode": "DEGRADED", "failure_code": "startup_check_failed"},
        }
        self.assertEqual(ONBOARDING_SERVICES_DOWN, detect_onboarding_state(ready_payload=ready_payload))
        self.assertEqual(
            "Run: systemctl --user restart personal-agent-telegram.service",
            onboarding_next_action(ONBOARDING_SERVICES_DOWN, ready_payload=ready_payload),
        )

    def test_detect_ready(self) -> None:
        ready_payload = {
            "ready": True,
            "phase": "ready",
            "telegram": {"configured": True, "state": "running"},
            "runtime_status": {"runtime_mode": "READY", "failure_code": None},
        }
        status = {
            "default_provider": "ollama",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "active_provider_health": {"status": "ok"},
            "active_model_health": {"status": "ok"},
        }
        self.assertEqual(
            ONBOARDING_READY,
            detect_onboarding_state(ready_payload=ready_payload, llm_status=status),
        )
        self.assertEqual("No action needed.", onboarding_next_action(ONBOARDING_READY))

    def test_detect_degraded(self) -> None:
        ready_payload = {
            "ready": False,
            "phase": "ready",
            "telegram": {"configured": True, "state": "running"},
            "runtime_status": {"runtime_mode": "DEGRADED", "failure_code": None},
        }
        status = {
            "default_provider": "ollama",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "active_provider_health": {"status": "ok"},
            "active_model_health": {"status": "ok"},
        }
        self.assertEqual(
            ONBOARDING_DEGRADED,
            detect_onboarding_state(ready_payload=ready_payload, llm_status=status),
        )
        self.assertIn("python -m agent doctor", onboarding_next_action(ONBOARDING_DEGRADED))
        self.assertEqual(3, len(onboarding_steps(ONBOARDING_DEGRADED)))


if __name__ == "__main__":
    unittest.main()
