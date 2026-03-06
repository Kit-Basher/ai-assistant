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
            "telegram": {"configured": False, "state": "disabled_missing_token"},
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

        result = run_setup_wizard(fetch_json=_fetch, dry_run=True)
        self.assertFalse(result.api_reachable)
        self.assertEqual("API_DOWN", result.recovery_mode)
        self.assertEqual("SERVICES_DOWN", result.onboarding_state)
        self.assertIn("personal-agent-api.service", result.next_action)

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
                "telegram": {"configured": False, "state": "disabled_missing_token"},
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


if __name__ == "__main__":
    unittest.main()
