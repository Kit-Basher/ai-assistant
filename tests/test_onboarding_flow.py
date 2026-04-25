from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from agent.onboarding_flow import (
    classify_onboarding_reply,
    is_onboarding_entry_trigger,
    onboarding_completed_key,
    onboarding_intent_to_capability,
    normalize_onboarding_intent,
    resolve_onboarding_intent,
    should_offer_onboarding,
)
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class _FakeChatLLM:
    def __init__(self, *, enabled: bool, text: str = "LLM reply") -> None:
        self._enabled = bool(enabled)
        self._text = text
        self.chat_calls: list[dict[str, object]] = []

    def enabled(self) -> bool:
        return self._enabled

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return {"ok": True, "text": self._text, "provider": "ollama", "model": "llama3"}


class _FakePackStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def list_external_packs(self) -> list[dict[str, object]]:
        return list(self._rows)


class TestOnboardingFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self, *, llm_enabled: bool = True) -> tuple[Orchestrator, _FakeChatLLM]:
        llm = _FakeChatLLM(enabled=llm_enabled, text="LLM reply")
        orch = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        return orch, llm

    def test_fresh_greeting_offers_onboarding_once(self) -> None:
        orch, llm = self._orchestrator(llm_enabled=True)

        first = orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        self.assertIn("tailor suggestions", first.text.lower())
        self.assertEqual([], llm.chat_calls)

        second = orch.handle_message("yes", "user1")
        self.assertIn("what do you mainly want help with?", second.text.lower())
        self.assertIn("coding, system / pc tasks, creative / writing, general use, or not sure", second.text.lower())
        self.assertEqual([], llm.chat_calls)

    def test_skip_marks_completion_and_resumes_normal_chat(self) -> None:
        orch, llm = self._orchestrator(llm_enabled=True)

        prompt = orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        self.assertIn("tailor suggestions", prompt.text.lower())

        skipped = orch.handle_message("skip", "user1")
        self.assertIn("just ask me anything", skipped.text.lower())
        self.assertEqual("true", str(self.db.get_user_pref(onboarding_completed_key("user1")) or "").strip().lower())

        with patch.object(orch, "_interpret_previous_result_followup", return_value=None), patch.object(
            orch, "_deep_system_followup_response", return_value=None
        ), patch.object(orch, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orch, "_handle_action_tool_intent", return_value=None
        ), patch.object(
            orch, "_grounded_system_fallback_response", return_value=None
        ), patch.object(
            orch, "_safe_mode_containment_response", return_value=None
        ):
            follow_up = orch.handle_message("tell me something useful", "user1")
        self.assertEqual("LLM reply", follow_up.text)
        self.assertEqual(1, len(llm.chat_calls))

    def test_freeform_request_abandons_onboarding_and_resumes_normal_chat(self) -> None:
        orch, llm = self._orchestrator(llm_enabled=True)

        prompt = orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        self.assertIn("tailor suggestions", prompt.text.lower())

        with patch.object(orch, "_interpret_previous_result_followup", return_value=None), patch.object(
            orch, "_deep_system_followup_response", return_value=None
        ), patch.object(orch, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orch, "_handle_action_tool_intent", return_value=None
        ), patch.object(
            orch, "_grounded_system_fallback_response", return_value=None
        ), patch.object(
            orch, "_safe_mode_containment_response", return_value=None
        ):
            response = orch.handle_message("what can you do?", "user1")
        self.assertEqual("LLM reply", response.text)
        self.assertEqual(1, len(llm.chat_calls))
        self.assertEqual("true", str(self.db.get_user_pref(onboarding_completed_key("user1")) or "").strip().lower())

    def test_intent_flow_uses_existing_recommendation_pipeline(self) -> None:
        orch, llm = self._orchestrator(llm_enabled=True)

        orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        orch.handle_message("yes", "user1")

        stub_recommendation = {
            "ok": True,
            "capability_required": "dev_tools",
            "capability_label": "coding tools",
            "status": "missing",
            "installed_pack": None,
            "recommended_pack": {
                "name": "Local Dev Tools",
                "reason": "best_fit_for_machine",
                "installable": True,
                "usable": False,
                "tradeoff_note": "lighter",
                "normalized_state": {"installable": True},
            },
            "alternate_pack": None,
            "comparison_mode": "single_recommendation",
            "fallback": "install_preview",
            "next_step": "If you want, say yes and I'll show the install details.",
            "warnings": [],
            "source_errors": [],
            "queries": [],
        }
        with patch("agent.orchestrator.recommend_onboarding_capability", return_value=stub_recommendation) as mock_recommend:
            response = orch.handle_message("coding", "user1")

        mock_recommend.assert_called_once()
        self.assertEqual("coding", mock_recommend.call_args.args[0])
        self.assertIn("I can add capabilities for coding.", response.text)
        self.assertIn("say yes and I'll show the install details", response.text)
        self.assertEqual("true", str(self.db.get_user_pref(onboarding_completed_key("user1")) or "").strip().lower())
        self.assertEqual([], llm.chat_calls)

    def test_completed_state_prevents_repeat_prompt(self) -> None:
        orch, llm = self._orchestrator(llm_enabled=True)
        self.db.set_user_pref(onboarding_completed_key("user1"), "true")

        response = orch.handle_message("hello", "user1")
        self.assertEqual("LLM reply", response.text)
        self.assertEqual(1, len(llm.chat_calls))
        self.assertNotIn("tailor suggestions", response.text.lower())

    def test_onboarding_detection_helpers_are_deterministic(self) -> None:
        self.assertTrue(is_onboarding_entry_trigger("hello"))
        self.assertEqual("dev_tools", onboarding_intent_to_capability("coding"))
        self.assertEqual("system_tools", onboarding_intent_to_capability("system"))
        self.assertEqual("creative_tools", onboarding_intent_to_capability("creative"))
        self.assertIsNone(onboarding_intent_to_capability("general"))
        self.assertIsNone(onboarding_intent_to_capability("not_sure"))
        self.assertEqual("general", resolve_onboarding_intent("general use"))
        self.assertEqual("not_sure", resolve_onboarding_intent("not sure"))
        self.assertEqual("coding", normalize_onboarding_intent("programming"))
        self.assertEqual("coding", normalize_onboarding_intent("python help"))
        self.assertEqual("system", normalize_onboarding_intent("linux help"))
        self.assertEqual("system", normalize_onboarding_intent("fix my pc"))
        self.assertEqual("creative", normalize_onboarding_intent("brainstorming"))
        self.assertEqual("creative", normalize_onboarding_intent("worldbuilding"))
        self.assertEqual("general", normalize_onboarding_intent("everyday stuff"))
        self.assertEqual("not_sure", normalize_onboarding_intent("don't know"))
        self.assertEqual("coding", normalize_onboarding_intent("mostly coding and scripts"))
        self.assertEqual("system", normalize_onboarding_intent("help with my linux setup"))
        self.assertEqual("creative", normalize_onboarding_intent("writing stories"))
        self.assertEqual("general", normalize_onboarding_intent("coding and writing"))
        self.assertEqual("not_sure", normalize_onboarding_intent("whatever"))
        self.assertEqual({"kind": "yes"}, classify_onboarding_reply("yes", stage="entry"))
        self.assertEqual({"kind": "skip"}, classify_onboarding_reply("skip", stage="entry"))
        self.assertEqual({"kind": "intent", "intent": "coding"}, classify_onboarding_reply("coding", stage="intent"))
        self.assertEqual(
            {"kind": "intent", "intent": "general"},
            classify_onboarding_reply("coding and writing", stage="intent"),
        )

    def test_onboarding_intent_normalization_is_stable_across_repeated_calls(self) -> None:
        samples = [
            "mostly coding and scripts",
            "help with my linux setup",
            "writing stories and brainstorming",
            "everything",
            "not sure yet",
        ]
        for sample in samples:
            first = normalize_onboarding_intent(sample)
            second = normalize_onboarding_intent(sample)
            self.assertEqual(first, second)

    def test_should_offer_onboarding_fails_closed_when_external_packs_exist(self) -> None:
        pack_store = _FakePackStore(
            [
                {
                    "pack_id": "pack.voice.local_fast",
                    "name": "Local Voice",
                    "status": "normalized",
                    "enabled": False,
                }
            ]
        )
        self.assertFalse(should_offer_onboarding(self.db, pack_store, "user1"))
