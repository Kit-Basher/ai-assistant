import os
import tempfile
import unittest

from memory.db import MemoryDB


class TestDBInit(unittest.TestCase):
    def test_init_and_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = MemoryDB(db_path)
            schema_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
            )
            db.init_schema(schema_path)

            note_id = db.add_note("hello", None, None)
            self.assertIsInstance(note_id, int)

            projects = db.list_projects()
            self.assertEqual(projects, [])
            self.assertEqual(MemoryDB.SCHEMA_VERSION, db.get_schema_version())

            # idempotent init
            db.init_schema(schema_path)
            self.assertEqual(MemoryDB.SCHEMA_VERSION, db.get_schema_version())

            task_id = db.add_task(None, "Write report", 30, 4)
            task = db.get_task(task_id)
            self.assertIsNotNone(task)
            self.assertEqual("todo", task["status"])
            self.assertEqual("Write report", task["title"])

            self.assertTrue(db.mark_task_done(task_id))
            done_task = db.get_task(task_id)
            self.assertIsNotNone(done_task)
            self.assertEqual("done", done_task["status"])

            self.assertFalse(db.mark_task_done(999999))
            self.assertIsNone(db.get_task(999999))

            db.close()


if __name__ == "__main__":
    unittest.main()
