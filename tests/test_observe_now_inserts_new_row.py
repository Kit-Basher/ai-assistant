import os
import tempfile
import unittest

from memory.db import MemoryDB
from skills.observe_now.handler import observe_now


class TestObserveNowInsertsSystemFacts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
        self.db.init_schema(schema_path)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_observe_now_inserts_new_row_each_call(self) -> None:
        user_id = "user1"
        ctx = {"db": self.db, "timezone": "UTC", "user_id": user_id}

        observe_now(ctx, user_id=user_id)
        observe_now(ctx, user_id=user_id)

        rows = self.db.list_system_facts_snapshots(user_id, limit=2)
        self.assertEqual(2, len(rows))
        self.assertNotEqual(rows[0]["id"], rows[1]["id"])


if __name__ == "__main__":
    unittest.main()

