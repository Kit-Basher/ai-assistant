from __future__ import annotations

import os
import tempfile
import time
import unittest

from tests.chat_eval_harness import all_route_cases, evaluate_route_case, run_conversation_evals
from agent.orchestrator import Orchestrator
from agent.shell_skill import ShellSkill
from agent.setup_chat_flow import classify_runtime_chat_route
from memory.db import MemoryDB
from agent.memory_contract import PENDING_STATUS_ABORTED, PENDING_STATUS_WAITING_FOR_USER
from tests.test_orchestrator import _FakeChatLLM, _FakeRuntimeTruthService, _FrontdoorRuntimeAdapter


class TestAdversarialChatRoutingClassifier(unittest.TestCase):
    def assertRoute(
        self,
        message: str,
        *,
        intent: str,
        route: str | None = None,
        kind: str | None = None,
    ) -> None:
        decision = classify_runtime_chat_route(message)

        self.assertEqual(intent, decision.get("semantic_intent"), message)
        if route is not None:
            self.assertEqual(route, decision.get("route"), message)
        if kind is not None:
            self.assertEqual(kind, decision.get("kind"), message)
        self.assertIn("confidence", decision)
        self.assertIn("evidence", decision)
        self.assertIn("stale_context_cleared", decision)
        self.assertIsInstance(decision.get("semantic"), dict)

    def test_runtime_and_local_status_inputs_route_to_deterministic_status(self) -> None:
        cases = (
            ("status", "runtime_status", "runtime_status"),
            ("runtime check", "runtime_status", "runtime_status"),
            ("give me a runtime check", "runtime_status", "runtime_status"),
            ("are you working?", "runtime_status", "runtime_status"),
            ("are you alive?", "runtime_status", "runtime_status"),
            ("what model are you using?", "model_status", "describe_current_model"),
            ("is search working?", "action_tool", "safe_web_search_status"),
            ("is telegram working?", "runtime_status", "telegram_status"),
            ("check if telegram is set up", "runtime_status", "telegram_status"),
        )
        for message, route, kind in cases:
            with self.subTest(message=message):
                self.assertRoute(message, intent="status_check", route=route, kind=kind)

    def test_public_lookup_inputs_route_to_metadata_search_when_enabled(self) -> None:
        cases = (
            "what is dots.tts",
            "dots tts any good?",
            "pi.dev?",
            "nex agi adaptive thinking?",
            "kwite?",
            "is kwite still around?",
            "what is fallen london?",
            "what is the latest qwen tts?",
        )
        for message in cases:
            with self.subTest(message=message):
                self.assertRoute(message, intent="web_search", route="action_tool", kind="safe_web_search")

    def test_timeless_and_provided_text_inputs_do_not_force_search(self) -> None:
        cases = (
            "what is photosynthesis?",
            "why is the sky blue?",
            "explain recursion simply",
            "what is a hyperlink?",
            "rewrite this: what is dots.tts",
            "summarize this: pi.dev is a website with a short landing page",
            "make this sound nicer: is telegram working?",
        )
        for message in cases:
            with self.subTest(message=message):
                decision = classify_runtime_chat_route(message)
                self.assertNotEqual("safe_web_search", decision.get("kind"), message)
                self.assertEqual("answer_directly", decision.get("semantic_intent"), message)

    def test_explicit_do_not_search_is_honored(self) -> None:
        self.assertRoute(
            "do not search, what is dots.tts?",
            intent="answer_directly",
            route="action_tool",
            kind="safe_web_search_suppressed",
        )

    def test_mutation_inputs_are_not_generic_chat(self) -> None:
        cases = (
            ("install htop", "package_or_system_mutation_preview", "shell_install_package"),
            ("install dots.tts", "package_or_system_mutation_preview", "shell_install_package"),
            ("can you install dots.tts?", "package_or_system_mutation_preview", "shell_install_package"),
        )
        for message, intent, kind in cases:
            with self.subTest(message=message):
                self.assertRoute(message, intent=intent, route="action_tool", kind=kind)

    def test_ambiguous_referents_ask_one_followup(self) -> None:
        cases = (
            "what is it?",
            "that thing",
            "the tts thing",
            "that tts thing people are talking about",
        )
        for message in cases:
            with self.subTest(message=message):
                self.assertRoute(message, intent="ask_clarifying_question", route="assistant_clarification")

    def test_pack_relevance_does_not_hijack_public_lookup_or_status(self) -> None:
        public_lookup = classify_runtime_chat_route("what is dots.tts")
        linux_prompt = classify_runtime_chat_route("my Linux laptop is slow after resume")
        telegram_status = classify_runtime_chat_route("is telegram working?")

        self.assertEqual("safe_web_search", public_lookup.get("kind"))
        self.assertNotEqual("pack_capability_recommendation", public_lookup.get("kind"))
        self.assertNotEqual("pack_capability_recommendation", telegram_status.get("kind"))
        self.assertNotEqual("safe_web_search", linux_prompt.get("kind"))

    def test_corrections_reroute_fresh(self) -> None:
        cases = (
            ("no, I meant search", "status_check", "safe_web_search_status"),
            ("no, check telegram", "status_check", "telegram_status"),
            ("I said runtime check", "status_check", "runtime_status"),
        )
        for message, intent, kind in cases:
            with self.subTest(message=message):
                self.assertRoute(message, intent=intent, kind=kind)


class TestChatEvalHarness(unittest.TestCase):
    def test_intent_invariant_matrix_cases_pass(self) -> None:
        cases = all_route_cases(include_generated=False, include_fixtures=True)
        self.assertGreaterEqual(len(cases), 45)
        for case in cases:
            with self.subTest(case_id=case.case_id, message=case.message):
                result = evaluate_route_case(case)
                self.assertTrue(result.passed, "; ".join(result.failures))

    def test_generated_fuzz_cases_preserve_invariants(self) -> None:
        cases = tuple(case for case in all_route_cases(include_generated=True, include_fixtures=False) if case.generated)
        self.assertGreaterEqual(len(cases), 1000)
        for case in cases:
            with self.subTest(case_id=case.case_id, message=case.message):
                result = evaluate_route_case(case)
                self.assertTrue(result.passed, "; ".join(result.failures))

    def test_multi_turn_conversation_simulator_invariants_pass(self) -> None:
        results = run_conversation_evals()
        self.assertGreaterEqual(len(results), 4)
        for result in results:
            with self.subTest(case_id=result.case.case_id, message=result.case.message):
                self.assertTrue(result.passed, "; ".join(result.failures))


class TestAdversarialChatRoutingOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self) -> Orchestrator:
        runtime_truth = _FakeRuntimeTruthService()
        safe_root = os.path.join(self.tmpdir.name, "workspace")
        os.makedirs(safe_root, exist_ok=True)
        runtime_truth.shell_skill = ShellSkill(
            allowed_roots=[safe_root],
            base_dir=safe_root,
            sensitive_roots=[os.path.join(safe_root, "private")],
        )
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=_FakeChatLLM(enabled=True, text="LLM should not be needed"),
            runtime_truth_service=runtime_truth,
            chat_runtime_adapter=_FrontdoorRuntimeAdapter(),
        )

    def _add_pending_clarification(self, orchestrator: Orchestrator, user_id: str = "user1") -> str:
        thread_id = orchestrator._active_thread_id_for_user(user_id)  # noqa: SLF001
        orchestrator._memory_runtime.add_pending_item(  # noqa: SLF001
            user_id,
            {
                "pending_id": "pending-model-choice",
                "thread_id": thread_id,
                "kind": "clarification",
                "status": PENDING_STATUS_WAITING_FOR_USER,
                "origin_tool": "model_selection",
                "question": "Which model do you mean?",
                "created_at": int(time.time()),
                "expires_at": int(time.time() + 600),
            },
        )
        return thread_id

    def test_fresh_runtime_status_beats_stale_pending_clarification(self) -> None:
        orchestrator = self._orchestrator()
        self._add_pending_clarification(orchestrator)

        response = orchestrator.handle_message("give me a runtime check", "user1")

        self.assertEqual("runtime_status", response.data.get("route"))
        self.assertNotIn("Which model do you mean?", response.text)

    def test_explicit_correction_beats_stale_pending_clarification(self) -> None:
        orchestrator = self._orchestrator()
        self._add_pending_clarification(orchestrator)

        response = orchestrator.handle_message("i said do a runtime check", "user1")

        self.assertEqual("runtime_status", response.data.get("route"))
        self.assertNotIn("Which model do you mean?", response.text)

    def test_new_cancel_and_forget_that_clear_stale_pending_state(self) -> None:
        for message in ("new", "cancel", "forget that"):
            with self.subTest(message=message):
                orchestrator = self._orchestrator()
                thread_id = self._add_pending_clarification(orchestrator)

                response = orchestrator.handle_message(message, "user1")
                pending = orchestrator._memory_runtime.list_pending_items(  # noqa: SLF001
                    "user1",
                    thread_id=thread_id,
                    include_expired=True,
                )

                self.assertEqual("assistant_clarification", response.data.get("route"))
                self.assertTrue(response.data.get("runtime_payload", {}).get("stale_context_cleared"))
                self.assertTrue(all(row.get("status") == PENDING_STATUS_ABORTED for row in pending))

    def test_telegram_status_uses_local_status_not_generic_advice(self) -> None:
        orchestrator = self._orchestrator()

        response = orchestrator.handle_message("is telegram working?", "user1")

        self.assertEqual("runtime_status", response.data.get("route"))
        self.assertEqual(["telegram_status"], response.data.get("used_tools"))
        self.assertNotIn("open Telegram", response.text.lower())

    def test_package_install_preview_does_not_execute(self) -> None:
        orchestrator = self._orchestrator()

        response = orchestrator.handle_message("install htop", "user1")

        self.assertEqual("action_tool", response.data.get("route"))
        self.assertIn("say yes to continue", response.text.lower())
        self.assertIn("no to cancel", response.text.lower())
        self.assertFalse(response.data.get("used_llm"))
