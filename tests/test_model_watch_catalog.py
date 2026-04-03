from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.model_watch_catalog import (
    CATALOG_SCHEMA_VERSION,
    build_feature_index,
    build_openrouter_snapshot,
    load_latest_snapshot,
    write_snapshot_atomic,
)


class TestModelWatchCatalog(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.snapshot_path = Path(self.tmpdir.name) / "catalog_snapshot.json"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_snapshot_write_is_atomic_and_loadable(self) -> None:
        snapshot = build_openrouter_snapshot(
            raw_payload={
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "context_length": 128000,
                        "pricing": {"prompt": "0.00000015", "completion": "0.00000060"},
                        "supported_parameters": ["tools"],
                    },
                    {"id": "meta-llama/llama-3.1-8b-instruct"},
                ]
            },
            fetched_at=1700000000,
        )

        with patch("agent.model_watch_catalog.os.replace", wraps=os.replace) as replace_mock:
            saved = write_snapshot_atomic(self.snapshot_path, snapshot)
        self.assertTrue(self.snapshot_path.is_file())
        self.assertGreaterEqual(replace_mock.call_count, 1)
        self.assertEqual(CATALOG_SCHEMA_VERSION, int(saved["schema_version"]))
        self.assertEqual(2, int(saved["model_count"]))

        loaded, error = load_latest_snapshot(self.snapshot_path)
        self.assertIsNone(error)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(CATALOG_SCHEMA_VERSION, int(loaded["schema_version"]))
        self.assertEqual(2, int(loaded["model_count"]))
        ids = [str(row.get("id") or "") for row in loaded.get("models") or []]
        self.assertEqual(sorted(ids), ids)

    def test_feature_index_marks_missing_fields(self) -> None:
        snapshot = build_openrouter_snapshot(
            raw_payload={
                "data": [
                    {
                        "id": "acme/no-meta-model",
                    }
                ]
            },
            fetched_at=1700000001,
        )
        saved = write_snapshot_atomic(self.snapshot_path, snapshot)
        index = build_feature_index(saved)
        row = index["openrouter:acme/no-meta-model"]
        self.assertIn("missing:context_length", row["missing_features"])
        self.assertIn("missing:pricing", row["missing_features"])
        self.assertIn("missing:params_b", row["missing_features"])


if __name__ == "__main__":
    unittest.main()
