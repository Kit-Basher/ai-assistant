from __future__ import annotations

import unittest

from agent.ux.llm_fixit_wizard import evaluate_wizard_decision


def _status_payload(
    *,
    safe_mode_paused: bool = False,
    openai_error_kind: str | None = None,
    openai_status_code: int | None = None,
    openrouter_status: str = "ok",
    openrouter_failure_streak: int = 0,
    any_routable: bool = True,
    allow_remote_fallback: bool = True,
    last_chat_model: str | None = None,
) -> dict[str, object]:
    providers = [
        {
            "id": "ollama",
            "enabled": True,
            "health": {
                "status": "ok",
                "last_error_kind": None,
                "status_code": None,
                "failure_streak": 0,
                "cooldown_until": None,
            },
        },
        {
            "id": "openrouter",
            "enabled": True,
            "health": {
                "status": openrouter_status,
                "last_error_kind": "provider_unavailable" if openrouter_status == "down" else None,
                "status_code": 502 if openrouter_status == "down" else None,
                "failure_streak": openrouter_failure_streak,
                "cooldown_until": None,
            },
        },
        {
            "id": "openai",
            "enabled": True,
            "health": {
                "status": "down" if openai_error_kind else "ok",
                "last_error_kind": openai_error_kind,
                "status_code": openai_status_code,
                "failure_streak": 0,
                "cooldown_until": None,
            },
        },
    ]
    models = [
        {
            "id": "ollama:llama3",
            "provider": "ollama",
            "enabled": True,
            "available": bool(any_routable),
            "routable": bool(any_routable),
            "health": {
                "status": "ok" if any_routable else "down",
                "last_error_kind": None,
                "status_code": None,
                "failure_streak": 0,
                "cooldown_until": None,
            },
        }
    ]
    return {
        "ok": True,
        "default_provider": "ollama",
        "default_model": "ollama:llama3",
        "resolved_default_model": "ollama:llama3",
        "last_chat_model": last_chat_model,
        "allow_remote_fallback": bool(allow_remote_fallback),
        "safe_mode": {
            "paused": bool(safe_mode_paused),
            "reason": "flip_flop_default_model" if safe_mode_paused else "not_paused",
            "next_retry": None,
            "cooldown_until": None,
            "last_transition_at": 1_700_000_000 if safe_mode_paused else None,
        },
        "providers": providers,
        "models": models,
    }


class TestLLMFixitWizardRules(unittest.TestCase):
    def test_openai_unauthorized_prefers_local_only(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(openai_error_kind="http_401", openai_status_code=401),
        )
        self.assertEqual("needs_user_choice", decision.status)
        self.assertEqual("openai_unauthorized", decision.issue_code)
        self.assertEqual(3, len(decision.choices))
        self.assertTrue(decision.choices[0].recommended)
        self.assertEqual("local_only", decision.choices[0].id)
        self.assertEqual("Which option should I take?", decision.question)

    def test_openrouter_down_prefers_local_only(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(openrouter_status="down", openrouter_failure_streak=20),
        )
        self.assertEqual("needs_user_choice", decision.status)
        self.assertEqual("openrouter_down", decision.issue_code)
        self.assertEqual("local_only", decision.choices[0].id)
        self.assertTrue(decision.choices[0].recommended)

    def test_remote_warnings_suppressed_when_remote_fallback_disabled_and_local_is_healthy(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(
                openrouter_status="down",
                openrouter_failure_streak=20,
                allow_remote_fallback=False,
            ),
        )
        self.assertEqual("ok", decision.status)
        self.assertEqual("ok", decision.issue_code)
        self.assertEqual([], decision.choices)

    def test_openrouter_down_without_key_offers_add_key(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(openrouter_status="down", openrouter_failure_streak=20),
            context={"openrouter_secret_present": False, "openrouter_last_test": None},
        )
        self.assertEqual("openrouter_down", decision.issue_code)
        self.assertEqual("OpenRouter needs an API key.", decision.message)
        self.assertEqual("local_only", decision.choices[0].id)
        self.assertEqual("add_openrouter_key", decision.choices[1].id)

    def test_openrouter_down_with_401_offers_update_key(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(openrouter_status="down", openrouter_failure_streak=20),
            context={
                "openrouter_secret_present": True,
                "openrouter_last_test": {
                    "ok": False,
                    "status_code": 401,
                    "error_kind": "auth_error",
                    "human_reason": "Authentication failed for provider.",
                },
            },
        )
        self.assertEqual("openrouter_down", decision.issue_code)
        self.assertEqual("Your OpenRouter key looks invalid (401).", decision.message)
        self.assertEqual("update_openrouter_key", decision.choices[1].id)

    def test_openrouter_down_with_402_offers_switch_provider(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(openrouter_status="down", openrouter_failure_streak=20),
            context={
                "openrouter_secret_present": True,
                "openrouter_last_test": {
                    "ok": False,
                    "status_code": 402,
                    "error_kind": "payment_required",
                    "human_reason": "Provider test hit a credits/limit issue.",
                },
            },
        )
        self.assertEqual("openrouter_down", decision.issue_code)
        self.assertEqual("OpenRouter returned payment required (402).", decision.message)
        self.assertEqual("switch_provider", decision.choices[1].id)

    def test_openrouter_down_network_failure_offers_retry(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(openrouter_status="down", openrouter_failure_streak=20),
            context={
                "openrouter_secret_present": True,
                "openrouter_last_test": {
                    "ok": False,
                    "status_code": 503,
                    "error_kind": "provider_unavailable",
                    "human_reason": "Provider server error.",
                },
            },
        )
        self.assertEqual("openrouter_down", decision.issue_code)
        self.assertEqual("OpenRouter appears down or unreachable.", decision.message)
        self.assertEqual("retry_openrouter_test", decision.choices[1].id)

    def test_safe_mode_paused_prompts_unpause(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(safe_mode_paused=True),
        )
        self.assertEqual("safe_mode_paused", decision.issue_code)
        self.assertEqual("unpause_autopilot", decision.choices[0].id)
        self.assertTrue(decision.choices[0].recommended)
        self.assertIn("paused", decision.message)

    def test_no_routable_model_offers_install_small_and_medium(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(any_routable=False, allow_remote_fallback=False),
        )
        self.assertEqual("no_routable_model", decision.issue_code)
        self.assertEqual("install_local_small", decision.choices[0].id)
        self.assertTrue(decision.choices[0].recommended)
        self.assertEqual("install_local_medium", decision.choices[1].id)
        self.assertEqual("details", decision.choices[2].id)

    def test_no_routable_model_with_last_chat_model_offers_undo(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(
                any_routable=False,
                allow_remote_fallback=False,
                last_chat_model="ollama:qwen2.5:3b-instruct",
            ),
        )
        self.assertEqual("no_routable_model", decision.issue_code)
        self.assertEqual("rollback_chat_model", decision.choices[2].id)

    def test_no_routable_model_without_last_chat_model_hides_undo(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(
                any_routable=False,
                allow_remote_fallback=False,
                last_chat_model=None,
            ),
        )
        choice_ids = [choice.id for choice in decision.choices]
        self.assertNotIn("rollback_chat_model", choice_ids)
        self.assertIn("details", choice_ids)


if __name__ == "__main__":
    unittest.main()
