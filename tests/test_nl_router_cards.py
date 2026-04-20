import os
import tempfile
import unittest
from unittest.mock import patch

from agent.cards import render_cards_markdown, validate_cards_payload
from agent.nl_router import classify_free_text, nl_route
from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestNLRouterCards(unittest.TestCase):
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

    def test_intent_classification(self) -> None:
        self.assertEqual(classify_free_text("what changed on my disk?"), "EXPLAIN_PREVIOUS")
        self.assertEqual(classify_free_text("how is cpu and memory"), "OBSERVE_PC")
        self.assertEqual(classify_free_text("what other pc stats can you find?"), "OBSERVE_PC")
        self.assertEqual(classify_free_text("can you tell what CPU and GPU I have?"), "OBSERVE_PC")
        self.assertEqual(classify_free_text("can you run a check and see if you can learn more?"), "OBSERVE_PC")
        self.assertEqual(classify_free_text("can you dig deeper into my system?"), "OBSERVE_PC")
        self.assertEqual(classify_free_text("run a system check"), "OBSERVE_PC")
        self.assertEqual(classify_free_text("my download is going slowly, can you tell why?"), "EXPLAIN_PREVIOUS")
        self.assertEqual(classify_free_text("my pc is slow"), "OBSERVE_PC")
        self.assertEqual(classify_free_text("laggy system"), "OBSERVE_PC")
        self.assertEqual(classify_free_text("remember this for later"), "MEMORY_WRITE_REQUEST")
        self.assertEqual(classify_free_text("plan my day"), "PLAN_DAY")
        self.assertEqual(classify_free_text("can you help me plan my day?"), "PLAN_DAY")
        self.assertEqual(classify_free_text("daily brief status"), "DAILY_BRIEF_STATUS")
        self.assertEqual(classify_free_text("what do you remember about me?"), "MEMORY_INSPECT")
        self.assertEqual(classify_free_text("what are we working on?"), "MEMORY_INSPECT")
        self.assertEqual(classify_free_text("what do you know about my system?"), "MEMORY_INSPECT")
        self.assertEqual(
            classify_free_text("My Wi-Fi drops after suspend. Can you help me figure out why?"),
            "DIAGNOSTICS_CAPTURE_REQUEST",
        )
        self.assertEqual(
            classify_free_text("My Bluetooth headphones disconnect after sleep. Can you help me figure out why?"),
            "DIAGNOSTICS_CAPTURE_BLUETOOTH_AUDIO_REQUEST",
        )
        self.assertEqual(
            classify_free_text("My disk is full and I can't save files. Can you help me figure out why?"),
            "DIAGNOSTICS_CAPTURE_STORAGE_DISK_REQUEST",
        )
        self.assertEqual(
            classify_free_text("My printer is offline and print jobs are stuck. Can you help me figure out why?"),
            "DIAGNOSTICS_CAPTURE_PRINTER_CUPS_REQUEST",
        )
        self.assertEqual(
            classify_free_text("My webcam isn't detected after sleep. Can you help me figure out why?"),
            "DIAGNOSTICS_CAPTURE_GENERIC_DEVICE_FALLBACK_REQUEST",
        )
        self.assertNotEqual(
            classify_free_text("what is eating space on my drive?"),
            "DIAGNOSTICS_CAPTURE_STORAGE_DISK_REQUEST",
        )
        self.assertNotEqual(
            classify_free_text("how do I print a document?"),
            "DIAGNOSTICS_CAPTURE_PRINTER_CUPS_REQUEST",
        )
        self.assertNotEqual(
            classify_free_text("can you help me write a script?"),
            "DIAGNOSTICS_CAPTURE_GENERIC_DEVICE_FALLBACK_REQUEST",
        )
        self.assertEqual(classify_free_text("Write a Bash script that finds the 10 largest files under a directory."), "UNKNOWN")
        self.assertEqual(classify_free_text("how do I pair bluetooth headphones?"), "UNKNOWN")
        self.assertEqual(classify_free_text("hello"), "CHITCHAT")
        self.assertEqual(classify_free_text("lorem ipsum"), "UNKNOWN")

    def test_machine_and_hardware_prompts_select_expected_skills(self) -> None:
        broad = nl_route("what other pc stats can you find?")
        ram_vram = nl_route("what do i have for ram and vram right now?")
        hardware = nl_route("can you tell what CPU and GPU I have?")
        gpu = nl_route("can you see the GPU?")
        deeper = nl_route("can you run a check and see if you can learn more?")
        deeper_system = nl_route("can you dig deeper into my system?")
        system_check = nl_route("run a system check")

        self.assertEqual(
            broad["skills"],
            [
                {"skill": "hardware_report", "function": "hardware_report"},
                {"skill": "resource_governor", "function": "resource_report"},
                {"skill": "storage_governor", "function": "storage_report"},
            ],
        )
        self.assertEqual(ram_vram["skills"][0], {"skill": "hardware_report", "function": "hardware_report"})
        self.assertIn({"skill": "resource_governor", "function": "resource_report"}, ram_vram["skills"])
        self.assertEqual(
            hardware["skills"],
            [
                {"skill": "hardware_report", "function": "hardware_report"},
                {"skill": "resource_governor", "function": "resource_report"},
            ],
        )
        self.assertEqual(gpu["skills"], [{"skill": "hardware_report", "function": "hardware_report"}])
        self.assertEqual(
            deeper["skills"],
            [
                {"skill": "hardware_report", "function": "hardware_report"},
                {"skill": "resource_governor", "function": "resource_report"},
                {"skill": "storage_governor", "function": "storage_report"},
            ],
        )
        self.assertEqual(
            deeper_system["skills"],
            [
                {"skill": "hardware_report", "function": "hardware_report"},
                {"skill": "resource_governor", "function": "resource_report"},
                {"skill": "storage_governor", "function": "storage_report"},
            ],
        )
        self.assertEqual(
            system_check["skills"],
            [
                {"skill": "hardware_report", "function": "hardware_report"},
                {"skill": "resource_governor", "function": "resource_report"},
                {"skill": "storage_governor", "function": "storage_report"},
            ],
        )

    def test_slowdown_questions_use_observe_path_not_generic_chat(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        observe_response = OrchestratorResponse(
            "observed",
            {"skip_friction_formatting": True, "cards_payload": {"cards": [], "raw_available": False, "summary": "observed", "confidence": 1.0, "next_questions": []}},
        )

        with (
            patch.object(orchestrator, "_handle_nl_observe", return_value=observe_response) as observe_call,
            patch.object(orchestrator, "_llm_chat", side_effect=AssertionError("generic chat should not be used")),
        ):
            response = orchestrator.handle_message("my download is going slowly, can you tell why?", "user-1")

        self.assertEqual("observed", response.text)
        self.assertEqual(1, observe_call.call_count)

    def test_card_schema_validation(self) -> None:
        payload = {
            "cards": [
                {"title": "Disk usage", "lines": ["/ is 72% used"], "severity": "ok"},
            ],
            "raw_available": True,
            "summary": "Disk looks stable.",
            "confidence": 0.9,
            "next_questions": ["What changed on /home?"],
        }
        ok, err = validate_cards_payload(payload)
        self.assertTrue(ok)
        self.assertIsNone(err)

        bad_payload = {
            "cards": [{"title": "Disk", "lines": ["x"], "severity": "maybe"}],
            "raw_available": True,
            "summary": "x",
            "confidence": 0.9,
            "next_questions": [],
        }
        ok, err = validate_cards_payload(bad_payload)
        self.assertFalse(ok)
        self.assertIn("severity", err or "")

    def test_e2e_disk_question_selects_skill_and_renders_cards(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        called = {"storage": 0, "resource": 0}

        def storage_handler(ctx, user_id=None):
            called["storage"] += 1
            return {
                "status": "ok",
                "text": "storage text",
                "cards_payload": {
                    "cards": [{"title": "Disk usage", "lines": ["/ is 72% used"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "Disk compared against previous snapshot.",
                    "confidence": 0.9,
                    "next_questions": ["Show top growing paths."],
                },
                "payload": {"mounts": []},
            }

        def resource_handler(ctx, user_id=None):
            called["resource"] += 1
            return {
                "status": "ok",
                "text": "resource text",
                "cards_payload": {
                    "cards": [{"title": "CPU load", "lines": ["1m=0.20"], "severity": "ok"}],
                    "raw_available": True,
                    "summary": "CPU compared against previous snapshot.",
                    "confidence": 0.9,
                    "next_questions": [],
                },
                "payload": {},
            }

        orchestrator.skills["storage_governor"].functions["storage_report"].handler = storage_handler
        orchestrator.skills["resource_governor"].functions["resource_report"].handler = resource_handler

        route = nl_route("what changed on my disk?")
        self.assertEqual(route["intent"], "EXPLAIN_PREVIOUS")
        self.assertEqual(route["skills"], [{"skill": "storage_governor", "function": "storage_report"}])

        response = orchestrator.handle_message("what changed on my disk?", "user-1")
        self.assertEqual(called["storage"], 1)
        self.assertEqual(called["resource"], 0)
        data = response.data or {}
        payload = data.get("runtime_payload") if isinstance(data.get("runtime_payload"), dict) else {}
        ok, err = validate_cards_payload(payload)
        self.assertTrue(ok, msg=err)

        rendered = render_cards_markdown(payload)
        self.assertEqual(response.text, rendered)
        self.assertIn("*Disk usage*", response.text)

    def test_nl_does_not_allow_observe_now(self) -> None:
        route = nl_route("observe now")
        self.assertEqual(route["skills"], [])

    def test_nl_path_audits_reads_but_does_not_insert_snapshots(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        before_audit = len(self.db.audit_log_list_recent("user-1", limit=100))
        before_activity = len(self.db.activity_log_list_recent("nl_read", limit=100))
        before_resource = self.db.get_latest_resource_snapshot()
        before_disk_root = self.db.get_latest_disk_snapshot("/")

        orchestrator.handle_message("what changed on my disk?", "user-1")

        after_audit = len(self.db.audit_log_list_recent("user-1", limit=100))
        after_activity = len(self.db.activity_log_list_recent("nl_read", limit=100))
        after_resource = self.db.get_latest_resource_snapshot()
        after_disk_root = self.db.get_latest_disk_snapshot("/")

        self.assertGreater(after_audit, before_audit)
        self.assertGreater(after_activity, before_activity)
        self.assertEqual(before_resource, after_resource)
        self.assertEqual(before_disk_root, after_disk_root)

    def test_nl_refuses_non_read_only_skill(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        orchestrator.skills["storage_governor"].functions["storage_report"].read_only = False
        response = orchestrator.handle_message("show disk status", "user-1")
        self.assertIn("Read-only guard", response.text)

    def test_memory_writes_only_on_explicit_intent(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        self.assertIsNone(self.db.get_preference("response_style"))
        orchestrator.handle_message("remember that i prefer concise answers", "user-1")
        self.assertEqual("concise", self.db.get_preference("response_style"))
        before = self.db.get_preference("response_style")
        orchestrator.handle_message("remember this maybe later", "user-1")
        self.assertEqual(before, self.db.get_preference("response_style"))

    def test_plan_my_day_routes_to_cards(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        self.db.add_task(None, "Write report", 30, 4)
        response = orchestrator.handle_message("plan my day", "user-1")
        self.assertIn("Today priorities", response.text)

    def test_today_followups_quick_wins_and_top3(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        self.db.add_task(None, "Quick task A", 10, 3)
        self.db.add_task(None, "Long task B", 60, 4)
        self.db.add_task(None, "Quick task C", 15, 2)
        self.db.add_task(None, "Task D", 30, 5)

        quick = orchestrator.handle_message("show quick wins", "user-1")
        self.assertIn("Quick task A (10m)", quick.text)
        self.assertIn("Quick task C (15m)", quick.text)
        self.assertNotIn("Long task B", quick.text)

        top3 = orchestrator.handle_message("show top 3 priorities", "user-1")
        data = top3.data or {}
        cards = data.get("cards", [])
        lines = cards[0].get("lines", []) if cards else []
        self.assertLessEqual(len(lines), 3)

    def test_defaults_max_cards_and_next_questions(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        response = orchestrator.handle_message("what changed on my disk and cpu and memory and network", "user-1")
        data = response.data or {}
        self.assertLessEqual(len(data.get("cards", [])), 4)
        self.assertLessEqual(len(data.get("next_questions", [])), 2)

    def test_preference_setters(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        orchestrator.handle_message("set max cards to 3", "user-1")
        self.assertEqual("3", self.db.get_preference("max_cards"))
        orchestrator.handle_message("turn confidence off", "user-1")
        self.assertEqual("off", self.db.get_preference("show_confidence"))
        orchestrator.handle_message("default compare on", "user-1")
        self.assertEqual("on", self.db.get_preference("default_compare_enabled"))

    def test_show_my_preferences_intent(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        self.db.set_preference("max_cards", "3")
        response = orchestrator.handle_message("show my preferences", "user-1")
        self.assertIn("Your preferences", response.text)
        self.assertIn("max_cards: 3", response.text)

    def test_renderer_output_stable(self) -> None:
        payload = {
            "cards": [{"title": "Disk usage (/)", "lines": ["used 1G"], "severity": "ok"}],
            "raw_available": True,
            "summary": "x",
            "confidence": 0.9,
            "next_questions": ["a", "b", "c"],
            "show_confidence": False,
        }
        text1 = render_cards_markdown(payload)
        text2 = render_cards_markdown(payload)
        self.assertEqual(text1, text2)

    def test_summary_is_answer_first_without_compared_for_non_delta(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )

        def storage_handler(ctx, user_id=None):
            return {
                "status": "ok",
                "cards_payload": {"cards": [{"title": "Disk usage (/)", "lines": ["used 40%"], "severity": "ok"}], "raw_available": True},
                "payload": {"mounts": [{"mountpoint": "/", "used_pct": 40.0, "delta_used": None}], "root_top": {"samples": []}, "home_top": {"samples": []}},
            }

        orchestrator.skills["storage_governor"].functions["storage_report"].handler = storage_handler
        response = orchestrator.handle_message("disk status", "user-1")
        self.assertNotIn("Compared", response.text)
        self.assertIn("Disk / is 40.0% used", response.text)

    def test_followups_state_aware_for_preferences(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        self.db.set_preference("max_cards", "3")
        self.db.set_preference("show_confidence", "off")
        response = orchestrator.handle_message("show my preferences", "user-1")
        data = response.data or {}
        questions = data.get("next_questions", [])
        self.assertIn("Set max cards to 4", questions)
        self.assertIn("Turn confidence on", questions)

    def test_memory_inspection_output_schema(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        self.db.set_preference("max_cards", "4")
        self.db.set_preference("response_style", "concise")
        self.db.add_open_loop("finish the docs", "2026-03-20")
        orchestrator._memory_runtime.set_current_topic("user-1", topic="safe mode stabilization")
        response = orchestrator.handle_message("what do you remember about me?", "user-1")
        data = response.data or {}
        payload = data.get("runtime_payload") if isinstance(data.get("runtime_payload"), dict) else {}
        self.assertEqual("agent_memory", data.get("route"))
        self.assertEqual("memory_summary", payload.get("kind"))
        self.assertIn("useful memory", response.text.lower())
        self.assertIn("preferences i know", response.text.lower())
        self.assertIn("open loops i am tracking", response.text.lower())
        self.assertNotIn("database", response.text.lower())


if __name__ == "__main__":
    unittest.main()
