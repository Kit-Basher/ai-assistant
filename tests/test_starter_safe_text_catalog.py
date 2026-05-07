from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agent.packs.registry_discovery import PackRegistryDiscoveryService
from agent.packs.store import PackStore


class TestStarterSafeTextCatalog(unittest.TestCase):
    def test_catalog_source_is_documented_allowlisted_and_text_only(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        storage_root = repo_root / "memory" / "external_packs"
        catalog_path = storage_root / "starter_catalog" / "catalog.json"
        sources_path = storage_root / "registry_sources.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackStore(str(Path(tmpdir) / "packs.db"))
            service = PackRegistryDiscoveryService(
                pack_store=store,
                storage_root=str(storage_root),
                sources_path=str(sources_path),
            )

            sources = service.list_sources()
            starter = next((row for row in sources if row.get("id") == "starter-safe-text"), None)
            self.assertIsNotNone(starter)
            assert starter is not None
            self.assertTrue(starter.get("enabled"))
            self.assertTrue(starter.get("allowlisted"))
            self.assertTrue(starter.get("allowed_by_policy"))
            self.assertEqual("local_catalog", starter.get("kind"))
            self.assertIn("portable text-only", str(starter.get("notes") or "").lower())

            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            packs = payload.get("packs")
            self.assertIsInstance(packs, list)
            assert isinstance(packs, list)
            self.assertEqual(8, len(packs))
            for row in packs:
                self.assertEqual("portable_text_skill", row.get("artifact_type_hint"))
                self.assertTrue(row.get("has_skill_md"))
                pack_path = repo_root / str(row.get("source_url") or "")
                self.assertTrue((pack_path / "SKILL.md").exists())
                metadata = json.loads((pack_path / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual("none", metadata.get("execution"))
                self.assertEqual([], metadata.get("dependencies"))
                self.assertFalse(metadata.get("contains_executable_code"))
                self.assertFalse(metadata.get("contains_mcp_server"))
                self.assertFalse(metadata.get("contains_plugin_handler"))

            listing_payload = service.list_packs("starter-safe-text")
            self.assertEqual(8, listing_payload["count"])
            self.assertTrue(all(row["installable_by_current_policy"] for row in listing_payload["packs"]))


if __name__ == "__main__":
    unittest.main()
