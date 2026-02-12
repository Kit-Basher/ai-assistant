import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class TestOpenLoops(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)
        self.orch = Orchestrator(
            db=self.db,
            skills_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_open_loop_add_list_done(self) -> None:
        self.orch.handle_message("remember that i need to file taxes by 2026-03-01", "user-1")
        rows = self.db.list_open_loops("open")
        self.assertEqual(1, len(rows))
        self.assertEqual("file taxes", rows[0]["title"])
        self.assertEqual("2026-03-01", rows[0]["due_date"])

        response = self.orch.handle_message("/open_loops", "user-1")
        self.assertIn("Open loops", response.text)
        self.assertIn("file taxes", response.text)

        self.orch.handle_message("mark file taxes done", "user-1")
        done_rows = self.db.list_open_loops("done")
        self.assertEqual(1, len(done_rows))

    def test_open_loop_priority_and_views(self) -> None:
        self.orch.handle_message("remember that ! renew passport by 2026-04-15", "user-1")
        self.orch.handle_message("remember that i need to clean inbox by 2026-06-01", "user-1")
        open_rows = self.db.list_open_loops("open", order="due")
        self.assertEqual(1, open_rows[0]["priority"])
        self.assertEqual("renew passport", open_rows[0]["title"])

        self.orch.handle_message("mark clean inbox done", "user-1")
        due_view = self.orch.handle_message("/open_loops due", "user-1")
        self.assertIn("renew passport", due_view.text)
        self.assertNotIn("clean inbox", due_view.text)

        all_view = self.orch.handle_message("/open_loops all", "user-1")
        self.assertIn("renew passport", all_view.text)
        self.assertIn("clean inbox", all_view.text)

    def test_open_loops_important_order(self) -> None:
        self.orch.handle_message("remember that i need to medium by 2026-05-01", "user-1")
        self.orch.handle_message("remember that ! urgent by 2026-06-01", "user-1")
        view = self.orch.handle_message("/open_loops important", "user-1")
        urgent_idx = view.text.find("urgent")
        medium_idx = view.text.find("medium")
        self.assertTrue(urgent_idx != -1 and medium_idx != -1 and urgent_idx < medium_idx)


if __name__ == "__main__":
    unittest.main()
