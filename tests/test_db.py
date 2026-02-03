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

            db.close()


if __name__ == "__main__":
    unittest.main()
