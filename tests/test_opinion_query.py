import os
import tempfile
import unittest
import os

from agent.intent_router import route_message
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB
from skills.opinion import handler as opinion_handler


class TestOpinionQuery(unittest.TestCase):
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

        self._env_backup = dict(os.environ)

        # seed minimal data
        day1 = "2026-02-01"
        day2 = "2026-02-07"
        for mount, first_used, last_used in [("/", 100, 110), ("/data", 200, 250), ("/data2", 300, 330)]:
            self.db.insert_disk_snapshot(
                taken_at=f"{day1}T09:00:00-06:00",
                snapshot_local_date=day1,
                hostname="host-a",
                mountpoint=mount,
                filesystem="/dev/root",
                total_bytes=1000,
                used_bytes=first_used,
                free_bytes=900,
            )
            self.db.insert_disk_snapshot(
                taken_at=f"{day2}T09:00:00-06:00",
                snapshot_local_date=day2,
                hostname="host-a",
                mountpoint=mount,
                filesystem="/dev/root",
                total_bytes=1000,
                used_bytes=last_used,
                free_bytes=900,
            )

        self.db.insert_resource_snapshot(
            taken_at=f"{day1}T09:00:00-06:00",
            snapshot_local_date=day1,
            hostname="host-a",
            load_1m=1.0,
            load_5m=1.2,
            load_15m=1.4,
            mem_total=800,
            mem_used=100,
            mem_free=700,
            swap_total=0,
            swap_used=0,
        )
        self.db.insert_resource_snapshot(
            taken_at=f"{day2}T09:00:00-06:00",
            snapshot_local_date=day2,
            hostname="host-a",
            load_1m=3.0,
            load_5m=2.2,
            load_15m=2.4,
            mem_total=800,
            mem_used=300,
            mem_free=500,
            swap_total=0,
            swap_used=0,
        )

        self.db.insert_network_snapshot(
            taken_at=f"{day1}T09:00:00-06:00",
            snapshot_local_date=day1,
            hostname="host-a",
            default_iface="eth0",
            default_gateway="10.0.0.1",
        )
        self.db.insert_network_snapshot(
            taken_at=f"{day2}T09:00:00-06:00",
            snapshot_local_date=day2,
            hostname="host-a",
            default_iface="eth0",
            default_gateway="10.0.0.2",
        )

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.db.close()
        self.tmpdir.cleanup()

    def test_opinion_vocab_enforced(self) -> None:
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system",
            timeframe=timeframe,
            trigger="what do you think",
        )
        text = result.get("text", "")
        for forbidden in opinion_handler.FORBIDDEN_WORDS:
            self.assertNotIn(forbidden, text.lower())

    def test_deterministic_output(self) -> None:
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result1 = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my storage lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        result2 = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my storage lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        self.assertEqual(result1.get("text"), result2.get("text"))

    def test_opinion_only_with_trigger(self) -> None:
        decision = route_message(
            "user1",
            "status lately",
            {"db": self.db, "timezone": "America/Regina", "chat_id": "user1"},
        )
        self.assertNotEqual(decision.get("skill"), "opinion")

    def test_advice_rejected(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="America/Regina",
            llm_client=None,
            enable_writes=False,
        )
        response = orchestrator.handle_message("/ask_opinion should i clean my disk", "user1")
        self.assertIn("not advice", response.text)

    def test_numeric_basis_present(self) -> None:
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        text = result.get("text", "")
        lines = [line for line in text.splitlines() if line.startswith("- ") and "(basis:" in line]
        self.assertTrue(lines)
        for line in lines:
            has_digit = any(ch.isdigit() for ch in line)
            self.assertTrue(has_digit)

    def test_presentation_flag_off_unchanged(self) -> None:
        os.environ["ENABLE_LLM_PRESENTATION"] = "0"
        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        result = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        text = result.get("text", "")
        os.environ["ENABLE_LLM_PRESENTATION"] = "1"
        result2 = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        self.assertEqual(text, result2.get("text"))

    def test_presentation_successful_rewrite(self) -> None:
        class FakePresentationClient:
            provider = "fake"

            def __init__(self, text: str) -> None:
                self.text = text

            def rewrite(self, _deterministic: str, _must_keep: list[str]):
                return {"text": self.text, "provider": self.provider}

        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        os.environ["ENABLE_LLM_PRESENTATION"] = "0"
        os.environ["LLM_PROVIDER"] = "none"
        base = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        ).get("text", "")

        rewritten = base.replace(
            "Opinionated Assessment:",
            "Opinionated Assessment:\nPresentation-only note.",
        )

        os.environ["ENABLE_LLM_PRESENTATION"] = "1"
        os.environ["LLM_PROVIDER"] = "ollama"
        result = opinion_handler.ask_opinion(
            {
                "db": self.db,
                "timezone": "America/Regina",
                "llm_presentation_client": FakePresentationClient(rewritten),
            },
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        text = result.get("text", "")
        self.assertNotEqual(text, base)
        self.assertIn("Presentation-only note.", text)

        audit = self.db.audit_log_list_recent("user1", limit=1)[0]
        details = audit.get("details", {})
        self.assertTrue(details.get("llm_presentation_attempted"))
        self.assertTrue(details.get("llm_presentation_used"))
        self.assertTrue(details.get("llm_validation_passed"))

    def test_presentation_forbidden_word_rejected(self) -> None:
        class FakePresentationClient:
            provider = "fake"

            def __init__(self, text: str) -> None:
                self.text = text

            def rewrite(self, _deterministic: str, _must_keep: list[str]):
                return {"text": self.text, "provider": self.provider}

        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        os.environ["ENABLE_LLM_PRESENTATION"] = "0"
        os.environ["LLM_PROVIDER"] = "none"
        base = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        ).get("text", "")

        rewritten = base.replace(
            "Confidence & Limits:",
            "Confidence & Limits:\nNo major concerns noted.",
        )

        os.environ["ENABLE_LLM_PRESENTATION"] = "1"
        os.environ["LLM_PROVIDER"] = "ollama"
        result = opinion_handler.ask_opinion(
            {
                "db": self.db,
                "timezone": "America/Regina",
                "llm_presentation_client": FakePresentationClient(rewritten),
            },
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        self.assertEqual(result.get("text"), base)
        audit = self.db.audit_log_list_recent("user1", limit=1)[0]
        details = audit.get("details", {})
        self.assertTrue(details.get("llm_presentation_attempted"))
        self.assertFalse(details.get("llm_presentation_used"))
        self.assertFalse(details.get("llm_validation_passed"))

    def test_presentation_basis_change_rejected(self) -> None:
        class FakePresentationClient:
            provider = "fake"

            def __init__(self, text: str) -> None:
                self.text = text

            def rewrite(self, _deterministic: str, _must_keep: list[str]):
                return {"text": self.text, "provider": self.provider}

        timeframe = {
            "label": "last 7 days",
            "start_date": "2026-02-01",
            "end_date": "2026-02-07",
            "start_ts": None,
            "end_ts": None,
            "user_id": "user1",
            "clarification_required": False,
        }
        os.environ["ENABLE_LLM_PRESENTATION"] = "0"
        os.environ["LLM_PROVIDER"] = "none"
        base = opinion_handler.ask_opinion(
            {"db": self.db, "timezone": "America/Regina"},
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        ).get("text", "")

        rewritten = base.replace("basis:", "basis: CHANGED")

        os.environ["ENABLE_LLM_PRESENTATION"] = "1"
        os.environ["LLM_PROVIDER"] = "ollama"
        result = opinion_handler.ask_opinion(
            {
                "db": self.db,
                "timezone": "America/Regina",
                "llm_presentation_client": FakePresentationClient(rewritten),
            },
            question="what do you think about my system lately",
            timeframe=timeframe,
            trigger="what do you think",
        )
        self.assertEqual(result.get("text"), base)
        audit = self.db.audit_log_list_recent("user1", limit=1)[0]
        details = audit.get("details", {})
        self.assertTrue(details.get("llm_presentation_attempted"))
        self.assertFalse(details.get("llm_presentation_used"))
        self.assertFalse(details.get("llm_validation_passed"))


if __name__ == "__main__":
    unittest.main()
