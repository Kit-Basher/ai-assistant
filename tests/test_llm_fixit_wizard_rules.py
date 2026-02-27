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

    def test_safe_mode_paused_prompts_unpause(self) -> None:
        decision = evaluate_wizard_decision(
            _status_payload(safe_mode_paused=True),
        )
        self.assertEqual("safe_mode_paused", decision.issue_code)
        self.assertEqual("unpause_autopilot", decision.choices[0].id)
        self.assertTrue(decision.choices[0].recommended)
        self.assertIn("paused", decision.message)


if __name__ == "__main__":
    unittest.main()
