from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.packs.external_ingestion import ExternalPackIngestor
from agent.packs.store import PackStore


class TestPackStoreExternalPackRedaction(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.storage_root = self.root / "external_packs"
        self.store = PackStore(str(self.root / "packs.db"))
        self.ingestor = ExternalPackIngestor(str(self.storage_root))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_removed_pack_tombstone_does_not_store_full_skill_text(self) -> None:
        source = self.root / "source"
        source.mkdir()
        hostile_marker = "HOSTILE_MARKER_DO_NOT_STORE"
        (source / "SKILL.md").write_text(
            f"# Removed Pack\n\nUse this guidance. {hostile_marker}\n",
            encoding="utf-8",
        )
        result, review = self.ingestor.ingest_from_path(str(source))
        row = self.store.record_external_pack(
            canonical_pack=result.pack.to_dict(),
            classification=result.classification,
            status=result.status,
            risk_report=result.risk_report.to_dict(),
            review_envelope=review.to_dict(),
            quarantine_path=result.quarantine_path,
            normalized_path=result.normalized_path,
        )
        pack_id = str(row["pack_id"])

        removed = self.store.remove_external_pack(pack_id, removed_by="test", reason="unit_test")

        assert removed is not None
        tombstone = self.store.get_external_pack_removal(pack_id)
        assert tombstone is not None
        tombstone_text = json.dumps(tombstone, ensure_ascii=True, sort_keys=True)
        self.assertNotIn(hostile_marker, tombstone_text)
        self.assertIsNone(tombstone.get("skill_text"))
        review_envelope = tombstone.get("review_envelope") if isinstance(tombstone.get("review_envelope"), dict) else {}
        audit = review_envelope.get("removed_skill_text") if isinstance(review_envelope.get("removed_skill_text"), dict) else {}
        self.assertTrue(audit.get("sha256"))
        self.assertFalse(bool(audit.get("stored", True)))


if __name__ == "__main__":
    unittest.main()
