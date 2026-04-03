from __future__ import annotations

import tempfile
import unittest

from agent.ux.llm_fixit_wizard import LLMFixitWizardStore, OperatorRecoveryStore
from telegram_adapter.bot import (
    _map_fixit_reply_to_payload,
    maybe_handle_llm_fixit_reply,
    maybe_handle_operator_recovery_reply,
)


class TestTelegramFixitWizardReplies(unittest.TestCase):
    def test_choice_mapping_numeric_index(self) -> None:
        state = {
            "active": True,
            "step": "awaiting_choice",
            "choices": [
                {"id": "local_only", "label": "Use local-only"},
                {"id": "repair_openrouter", "label": "Repair OpenRouter"},
                {"id": "details", "label": "Show details"},
            ],
        }
        payload, hint = _map_fixit_reply_to_payload(state, "2")
        self.assertEqual({"answer": "repair_openrouter"}, payload)
        self.assertIsNone(hint)

    def test_confirmation_mapping_yes(self) -> None:
        state = {
            "active": True,
            "step": "awaiting_confirm",
            "choices": [],
        }
        payload, hint = _map_fixit_reply_to_payload(state, "1")
        self.assertEqual({"confirm": True}, payload)
        self.assertIsNone(hint)

    def test_confirmation_mapping_no(self) -> None:
        state = {
            "active": True,
            "step": "awaiting_confirm",
            "choices": [],
        }
        payload, hint = _map_fixit_reply_to_payload(state, "2")
        self.assertEqual({"confirm": False}, payload)
        self.assertIsNone(hint)

    def test_maybe_handle_invokes_llm_fixit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LLMFixitWizardStore(path=f"{tmpdir}/fixit.json")
            store.save(
                {
                    "active": True,
                    "issue_hash": "abc",
                    "issue_code": "openrouter_down",
                    "step": "awaiting_confirm",
                    "question": "Apply?",
                    "choices": [],
                    "pending_plan": [{"id": "01", "kind": "safe_action", "action": "health.run", "reason": "test"}],
                    "pending_confirm_token": "token",
                    "pending_created_ts": 1,
                    "pending_expires_ts": 9999999999,
                    "pending_issue_code": "openrouter_down",
                    "last_prompt_ts": 1,
                }
            )
            captured: dict[str, object] = {}

            def _llm_fixit(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
                captured.update(payload)
                return True, {"ok": True, "message": "Applied safe LLM fixes.", "error_kind": None}

            message = maybe_handle_llm_fixit_reply(
                llm_fixit_fn=_llm_fixit,
                wizard_store=store,
                audit_log=None,
                chat_id="123456",
                text="yes",
                log_path=None,
            )
            self.assertEqual("Applied safe LLM fixes.", message)
            self.assertEqual(True, captured.get("confirm"))
            self.assertEqual("telegram", captured.get("actor"))

    def test_maybe_handle_operator_recovery_invokes_generic_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OperatorRecoveryStore(path=f"{tmpdir}/fixit.json")
            store.save(
                {
                    "active": True,
                    "issue_hash": "abc",
                    "issue_code": "openrouter_down",
                    "step": "awaiting_confirm",
                    "question": "Apply?",
                    "choices": [],
                    "pending_plan": [{"id": "01", "kind": "safe_action", "action": "health.run", "reason": "test"}],
                    "pending_confirm_token": "token",
                    "pending_created_ts": 1,
                    "pending_expires_ts": 9999999999,
                    "pending_issue_code": "openrouter_down",
                    "last_prompt_ts": 1,
                }
            )
            captured: dict[str, object] = {}

            def _operator_recovery(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
                captured.update(payload)
                return True, {"ok": True, "message": "Applied operator recovery.", "error_kind": None}

            message = maybe_handle_operator_recovery_reply(
                operator_recovery_fn=_operator_recovery,
                recovery_store=store,
                audit_log=None,
                chat_id="123456",
                text="yes",
                log_path=None,
            )
            self.assertEqual("Applied operator recovery.", message)
            self.assertEqual(True, captured.get("confirm"))
            self.assertEqual("telegram", captured.get("actor"))


if __name__ == "__main__":
    unittest.main()
