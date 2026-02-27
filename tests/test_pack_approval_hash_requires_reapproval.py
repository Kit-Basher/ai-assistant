from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.orchestrator import Orchestrator
from agent.packs.manifest import load_manifest
from memory.db import MemoryDB


def _write_skill_with_pack(
    *,
    skills_root: str,
    ifaces: list[str],
) -> str:
    skill_dir = os.path.join(skills_root, "switch_skill")
    os.makedirs(skill_dir, exist_ok=True)
    with open(os.path.join(skill_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "name": "switch_skill",
                "description": "switch test skill",
                "version": "0.1.0",
                "permissions": [],
                "functions": [
                    {
                        "name": "switch_model",
                        "args_schema": {"type": "object", "properties": {}},
                    }
                ],
            },
            handle,
            ensure_ascii=True,
        )
    with open(os.path.join(skill_dir, "handler.py"), "w", encoding="utf-8") as handle:
        handle.write("def switch_model(ctx):\n    return {'text': 'switched'}\n")
    manifest_path = os.path.join(skill_dir, "pack.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "pack_id": "switch_pack",
                "version": "0.1.0",
                "title": "Switch Pack",
                "description": "switch permissions",
                "entrypoints": ["skills.switch_skill:handler"],
                "trust": "trusted",
                "permissions": {"ifaces": ifaces},
            },
            handle,
            ensure_ascii=True,
        )
    return manifest_path


class TestPackApprovalHashRequiresReapproval(unittest.TestCase):
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

    def test_permissions_hash_change_requires_reapproval(self) -> None:
        manifest_path = _write_skill_with_pack(skills_root=self.skills_root, ifaces=["switch_model"])
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_root,
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )

        manifest = load_manifest(manifest_path)
        installed = orchestrator._pack_store.install_pack(manifest, manifest_path=manifest_path, enable=True)
        orchestrator._pack_store.set_approval_hash("switch_pack", str(installed.get("permissions_hash") or ""))

        allowed = orchestrator._call_skill("user1", "switch_skill", "switch_model", {}, [])
        self.assertEqual("switched", allowed.text)

        _write_skill_with_pack(skills_root=self.skills_root, ifaces=["switch_model", "other_iface"])
        orchestrator_reloaded = Orchestrator(
            db=self.db,
            skills_path=self.skills_root,
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )
        denied = orchestrator_reloaded._call_skill("user1", "switch_skill", "switch_model", {}, [])
        self.assertIn("Approve pack switch_pack for switch_model?", denied.text)
        self.assertIn("not allowed to call switch_model", denied.text)


if __name__ == "__main__":
    unittest.main()
