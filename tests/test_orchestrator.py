import os
import tempfile
import unittest

from agent.knowledge_cache import facts_hash
from agent.orchestrator import Orchestrator, OrchestratorResponse
from memory.db import MemoryDB


class TestOrchestrator(unittest.TestCase):
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

    def test_handle_message_no_longer_raises(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        response = orchestrator.handle_message("hello there", "user1")
        self.assertIsInstance(response, OrchestratorResponse)

    def test_knowledge_query_cache_and_cta(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        response = orchestrator.handle_message("what changed this week", "user1")
        self.assertIn("Want my opinion", response.text)
        entry = orchestrator._knowledge_cache.get_recent("user1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.facts_hash, facts_hash(entry.facts))

    def test_opinion_followup_uses_cached_facts(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        orchestrator.handle_message("what changed this week", "user1")
        entry = orchestrator._knowledge_cache.get_recent("user1")
        response = orchestrator.handle_message("opinion", "user1")
        self.assertIn("source", response.data.get("data", {}))
        self.assertEqual(response.data["data"]["facts_hash"], entry.facts_hash)


if __name__ == "__main__":
    unittest.main()
