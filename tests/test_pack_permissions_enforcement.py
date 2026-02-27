from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


def _write_skill(
    *,
    skills_root: str,
    skill_dir_name: str,
    skill_name: str,
    function_name: str,
    pack_id: str,
    trust: str,
    ifaces: list[str],
) -> None:
    skill_dir = os.path.join(skills_root, skill_dir_name)
    os.makedirs(skill_dir, exist_ok=True)
    with open(os.path.join(skill_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "name": skill_name,
                "description": "pack test skill",
                "version": "0.1.0",
                "permissions": [],
                "functions": [
                    {
                        "name": function_name,
                        "args_schema": {"type": "object", "properties": {}},
                    }
                ],
            },
            handle,
            ensure_ascii=True,
        )
    with open(os.path.join(skill_dir, "handler.py"), "w", encoding="utf-8") as handle:
        handle.write(
            "def %s(ctx):\n"
            "    return {'text': 'ran'}\n" % function_name
        )
    with open(os.path.join(skill_dir, "pack.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "pack_id": pack_id,
                "version": "0.1.0",
                "title": "Pack Test",
                "description": "permissions",
                "entrypoints": [f"skills.{skill_dir_name}:handler"],
                "trust": trust,
                "permissions": {"ifaces": ifaces},
            },
            handle,
            ensure_ascii=True,
        )


class TestPackPermissionsEnforcement(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.db = MemoryDB(self.db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.skills_root = os.path.join(self.tmpdir.name, "skills")
        os.makedirs(self.skills_root, exist_ok=True)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_unapproved_pack_cannot_call_iface(self) -> None:
        _write_skill(
            skills_root=self.skills_root,
            skill_dir_name="danger_skill",
            skill_name="danger_skill",
            function_name="dangerous_call",
            pack_id="danger_pack",
            trust="untrusted",
            ifaces=[],
        )
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_root,
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )
        response = orchestrator._call_skill(
            "user1",
            "danger_skill",
            "dangerous_call",
            {},
            [],
        )
        self.assertIn("not allowed to call dangerous_call", response.text)
        self.assertIn("Approve pack danger_pack for dangerous_call?", response.text)


if __name__ == "__main__":
    unittest.main()
