import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class TestNarrationToggle(unittest.TestCase):
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

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
            enable_writes=False,
        )

    def test_narration_disabled_raw_report(self) -> None:
        os.environ["ENABLE_NARRATION"] = "0"
        orchestrator = self._orchestrator()

        storage = orchestrator.handle_message("/storage_report", "user1")
        resource = orchestrator.handle_message("/resource_report", "user1")

        self.assertIn("No storage snapshots found yet", storage.text)
        self.assertIn("No resource snapshots found yet", resource.text)
        self.assertFalse(storage.text.startswith("Narration"))
        self.assertFalse(resource.text.startswith("Narration"))

    def test_narration_enabled_llm_unavailable_raw_report(self) -> None:
        os.environ["ENABLE_NARRATION"] = "1"
        os.environ["OLLAMA_MODEL"] = ""
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENAI_MODEL"] = ""

        orchestrator = self._orchestrator()

        storage = orchestrator.handle_message("/storage_report", "user1")
        resource = orchestrator.handle_message("/resource_report", "user1")

        self.assertIn("No storage snapshots found yet", storage.text)
        self.assertIn("No resource snapshots found yet", resource.text)
        self.assertFalse(storage.text.startswith("Narration"))
        self.assertFalse(resource.text.startswith("Narration"))


if __name__ == "__main__":
    unittest.main()
