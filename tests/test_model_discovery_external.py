from __future__ import annotations

import tempfile
import unittest

from agent.llm.model_discovery_external import (
    external_model_discovery_rows_from_openrouter_snapshot,
    load_external_model_discovery_rows,
)


class TestModelDiscoveryExternal(unittest.TestCase):
    def test_openrouter_snapshot_rows_normalize_to_external_discovery_shape(self) -> None:
        snapshot = {
            "provider": "openrouter",
            "source": "openrouter_models",
            "models": [
                {
                    "id": "openrouter:vendor/cheap-text",
                    "provider_id": "openrouter",
                    "model": "vendor/cheap-text",
                    "context_length": 262144,
                    "modalities": ["text"],
                    "supports_tools": True,
                    "pricing": {
                        "prompt_per_million": 0.2,
                        "completion_per_million": 0.4,
                    },
                },
                {
                    "id": "openrouter:vendor/unknown",
                    "provider_id": "openrouter",
                    "model": "vendor/unknown",
                    "modalities": [],
                    "supports_tools": None,
                    "pricing": {},
                },
            ],
        }

        rows = external_model_discovery_rows_from_openrouter_snapshot(snapshot)
        by_model = {row["model_id"]: row for row in rows}

        cheap = by_model["openrouter:vendor/cheap-text"]
        self.assertEqual("openrouter", cheap["provider_id"])
        self.assertEqual("vendor/cheap-text", cheap["model_name"])
        self.assertEqual(["chat", "tools"], cheap["capabilities"])
        self.assertEqual([], cheap["task_types"])
        self.assertEqual(["text"], cheap["modalities"])
        self.assertEqual(262144, cheap["context_window"])
        self.assertEqual(0.2, cheap["price_in"])
        self.assertEqual(0.4, cheap["price_out"])
        self.assertEqual("external_openrouter_snapshot", cheap["source"])
        self.assertTrue(bool(cheap["non_canonical"]))
        self.assertEqual("not_adopted", cheap["canonical_status"])

        unknown = by_model["openrouter:vendor/unknown"]
        self.assertEqual([], unknown["capabilities"])
        self.assertEqual([], unknown["modalities"])
        self.assertIsNone(unknown["context_window"])
        self.assertIsNone(unknown["price_in"])
        self.assertIsNone(unknown["price_out"])

    def test_load_external_rows_uses_snapshot_loader_and_reports_source_status(self) -> None:
        snapshot = {
            "provider": "openrouter",
            "source": "openrouter_models",
            "models": [
                {
                    "id": "openrouter:vendor/cheap-text",
                    "provider_id": "openrouter",
                    "model": "vendor/cheap-text",
                    "context_length": 131072,
                    "modalities": ["text"],
                    "supports_tools": False,
                    "pricing": {
                        "prompt_per_million": 0.1,
                        "completion_per_million": 0.2,
                    },
                }
            ],
        }

        def _loader(_path: object) -> tuple[dict[str, object] | None, str | None]:
            return snapshot, None

        result = load_external_model_discovery_rows(
            provider_id="openrouter",
            openrouter_snapshot_path=f"{tempfile.gettempdir()}/model_watch_snapshot.json",
            snapshot_loader=_loader,
        )
        self.assertEqual(1, len(result["models"]))
        source_row = result["sources"][0]
        self.assertTrue(bool(source_row["ok"]))
        self.assertEqual("external_openrouter_snapshot", source_row["source"])
        self.assertEqual(1, source_row["model_count"])


if __name__ == "__main__":
    unittest.main()
