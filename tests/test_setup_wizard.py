from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.doctor import _check_telegram_token
from agent.runtime_contract import normalize_user_facing_status
from agent.setup_wizard import build_setup_result, render_setup_text, run_setup_wizard


class TestSetupWizard(unittest.TestCase):
    def test_build_setup_result_token_missing(self) -> None:
        ready_payload = {
            "ready": False,
            "phase": "ready",
            "telegram": {"enabled": True, "configured": False, "state": "disabled_missing_token"},
            "runtime_status": {"runtime_mode": "BOOTSTRAP_REQUIRED", "failure_code": "telegram_token_missing"},
        }
        result = build_setup_result(
            ready_payload=ready_payload,
            llm_status={},
            api_reachable=True,
            dry_run=True,
            trace_id="setup-test-1",
        )
        self.assertEqual("TOKEN_MISSING", result.onboarding_state)
        self.assertEqual("TOKEN_INVALID", result.recovery_mode)
        self.assertEqual("Run: python -m agent.secrets set telegram:bot_token", result.next_action)
        self.assertTrue(result.dry_run)
        text = render_setup_text(result)
        self.assertIn("1) State: TOKEN_MISSING", text)
        self.assertIn("Dry-run: no changes were applied.", text)

    def test_run_setup_wizard_api_down(self) -> None:
        def _fetch(**_kwargs: object) -> tuple[bool, dict[str, object] | str]:
            return False, "URLError:connection refused"

        result = run_setup_wizard(
            fetch_json=_fetch,
            dry_run=True,
            service_probe_fn=lambda: {
                "checked": True,
                "available": True,
                "active_state": "inactive",
                "confidence": "confirmed",
            },
        )
        self.assertFalse(result.api_reachable)
        self.assertEqual("API_DOWN", result.recovery_mode)
        self.assertEqual("SERVICES_DOWN", result.onboarding_state)
        self.assertEqual("API service is down.", result.why)
        self.assertIn("personal-agent-api.service", result.next_action)

    def test_run_setup_wizard_uses_telegram_runtime_state_when_enabled_but_stopped(self) -> None:
        def _fetch(**kwargs: object) -> tuple[bool, dict[str, object] | str]:
            path = str(kwargs.get("path") or "")
            if path == "/ready":
                return True, {
                    "ready": True,
                    "phase": "ready",
                    "telegram": {"enabled": False, "configured": False, "state": "disabled_optional"},
                    "runtime_status": {"runtime_mode": "READY", "failure_code": None},
                }
            if path == "/llm/status":
                return True, {
                    "default_provider": "ollama",
                    "resolved_default_model": "ollama:qwen2.5:3b-instruct",
                    "active_provider_health": {"status": "ok"},
                    "active_model_health": {"status": "ok"},
                }
            raise AssertionError(f"unexpected path: {path}")

        with patch(
            "agent.setup_wizard.get_telegram_runtime_state",
            return_value={
                "enabled": True,
                "token_configured": True,
                "effective_state": "enabled_stopped",
                "ready_state": "stopped",
            },
        ):
            result = run_setup_wizard(fetch_json=_fetch, dry_run=True)
        self.assertEqual("SERVICES_DOWN", result.onboarding_state)
        self.assertEqual("TELEGRAM_DOWN", result.recovery_mode)
        self.assertIn("personal-agent-telegram.service", result.next_action)

    def test_run_setup_wizard_prefers_embedded_ready_llm_payload(self) -> None:
        calls: list[str] = []

        def _fetch(**kwargs: object) -> tuple[bool, dict[str, object] | str]:
            path = str(kwargs.get("path") or "")
            calls.append(path)
            if path == "/ready":
                return True, {
                    "ready": True,
                    "phase": "ready",
                    "telegram": {"enabled": False, "configured": False, "state": "disabled_optional"},
                    "runtime_status": {"runtime_mode": "READY", "failure_code": None},
                    "llm": {
                        "default_provider": "ollama",
                        "resolved_default_model": "ollama:qwen2.5:3b-instruct",
                        "active_provider_health": {"status": "ok"},
                        "active_model_health": {"status": "ok"},
                    },
                }
            if path == "/llm/status":
                raise AssertionError("run_setup_wizard should use embedded ready.llm before /llm/status")
            raise AssertionError(f"unexpected path: {path}")

        result = run_setup_wizard(fetch_json=_fetch, dry_run=True)

        self.assertTrue(result.api_reachable)
        self.assertEqual(["/ready"], calls)

    def test_no_contradictory_next_action_for_token_missing(self) -> None:
        status = normalize_user_facing_status(
            ready=False,
            bootstrap_required=True,
            failure_code="telegram_token_missing",
            phase="degraded",
            provider=None,
            model=None,
            local_providers={"ollama"},
        )
        setup_result = build_setup_result(
            ready_payload={
                "ready": False,
                "phase": "degraded",
                "telegram": {"enabled": True, "configured": False, "state": "disabled_missing_token"},
                "runtime_status": status,
            },
            llm_status={},
            api_reachable=True,
            trace_id="setup-test-2",
        )
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": ""}, clear=False):
            with patch("agent.doctor.SecretStore.get_secret", return_value=""):
                doctor_check = _check_telegram_token(online=False)
        self.assertEqual("Run: python -m agent.secrets set telegram:bot_token", setup_result.next_action)
        self.assertEqual("Run: python -m agent.secrets set telegram:bot_token", str(status.get("next_action")))
        self.assertEqual(setup_result.next_action, doctor_check.next_action)

    def test_ready_when_telegram_disabled_optional(self) -> None:
        result = build_setup_result(
            ready_payload={
                "ready": True,
                "phase": "ready",
                "telegram": {"enabled": False, "configured": False, "state": "disabled_optional"},
                "runtime_status": {"runtime_mode": "READY", "failure_code": None},
            },
            llm_status={
                "default_provider": "ollama",
                "resolved_default_model": "ollama:qwen2.5:3b-instruct",
                "active_provider_health": {"status": "ok"},
                "active_model_health": {"status": "ok"},
            },
            api_reachable=True,
            dry_run=True,
            trace_id="setup-test-optional",
        )
        self.assertEqual("READY", result.onboarding_state)
        self.assertEqual([], result.suggestions)

    def test_build_setup_result_runtime_truth_beats_failed_probe(self) -> None:
        result = build_setup_result(
            ready_payload={
                "ready": True,
                "phase": "ready",
                "telegram": {"enabled": True, "configured": True, "state": "running"},
                "runtime_status": {"runtime_mode": "READY", "failure_code": None},
            },
            llm_status={
                "default_provider": "ollama",
                "resolved_default_model": "ollama:qwen3.5:4b",
                "active_provider_health": {"status": "ok"},
                "active_model_health": {"status": "ok"},
            },
            api_reachable=False,
            transport_error="URLError:timed out",
            dry_run=True,
            trace_id="setup-test-runtime-truth",
            service_probe_fn=lambda: {
                "checked": True,
                "available": True,
                "active_state": "active",
                "confidence": "confirmed",
            },
        )
        self.assertEqual("READY", result.onboarding_state)
        self.assertEqual("runtime_truth", result.diagnosis_source)
        self.assertEqual("confirmed", result.diagnosis_confidence)
        self.assertNotIn("API service is down", result.why)

    def test_build_setup_result_transport_failure_with_active_service_is_uncertain(self) -> None:
        result = build_setup_result(
            ready_payload={},
            llm_status={},
            api_reachable=False,
            transport_error="URLError:timed out",
            dry_run=True,
            trace_id="setup-test-uncertain",
            service_probe_fn=lambda: {
                "checked": True,
                "available": True,
                "active_state": "active",
                "confidence": "confirmed",
            },
        )
        self.assertEqual("DEGRADED", result.onboarding_state)
        self.assertEqual("UNKNOWN_FAILURE", result.recovery_mode)
        self.assertEqual("Run: python -m agent status", result.next_action)
        self.assertEqual("service_check", result.diagnosis_source)
        self.assertEqual("uncertain", result.diagnosis_confidence)
        self.assertEqual("The API may be restarting or temporarily unavailable.", result.why)


if __name__ == "__main__":
    unittest.main()
