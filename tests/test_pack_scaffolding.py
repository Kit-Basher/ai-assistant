from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agent.packs.external_ingestion import ExternalPackIngestor
from agent.packs.scaffolding import build_scaffold_preview, create_generated_scaffold_source, render_scaffold_preview


class TestPackScaffolding(unittest.TestCase):
    def test_youtube_history_scaffold_preview_is_preview_only(self) -> None:
        preview = build_scaffold_preview(
            "youtube_history_search",
            user_goal="Find a watched YouTube video about neurons differentiating during animal infancy.",
        )

        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual("skill_scaffold_preview", preview.get("type"))
        self.assertEqual("youtube_history_search", preview.get("capability"))
        self.assertEqual("YouTube History Search", preview.get("title"))
        self.assertFalse(preview.get("creates_files"))
        self.assertFalse(preview.get("executes_code"))
        manifest = preview.get("proposed_manifest")
        self.assertIsInstance(manifest, dict)
        assert isinstance(manifest, dict)
        self.assertEqual([], manifest.get("permissions_granted"))
        self.assertIn("google_takeout_import", manifest.get("capabilities"))
        adapters = manifest.get("managed_adapters")
        self.assertIsInstance(adapters, list)
        assert isinstance(adapters, list)
        self.assertEqual("local_file_import", adapters[0].get("kind"))
        self.assertEqual([".json", ".html"], adapters[0].get("allowed_extensions"))
        self.assertEqual("user_selected_file_only", adapters[0].get("path_policy"))
        self.assertFalse(adapters[0].get("network_allowed"))
        blocked = " ".join(str(item) for item in preview.get("blocked_actions", []))
        self.assertIn("No OAuth", blocked)
        self.assertIn("No browser history scraping", blocked)
        self.assertIn("No transcript fetching", blocked)
        self.assertIn("No video or audio downloads", blocked)
        rendered = render_scaffold_preview(preview)
        self.assertIn("No files were created", rendered)
        self.assertIn("no code was executed", rendered)

    def test_create_generated_scaffold_source_writes_text_only_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview = build_scaffold_preview(
                "youtube_history_search",
                user_goal="Find a watched YouTube video about neurons differentiating during animal infancy.",
            )
            assert preview is not None

            result = create_generated_scaffold_source(preview, storage_root=tmpdir)

            source_path = Path(str(result["source_path"]))
            self.assertTrue(str(source_path).startswith(str(Path(tmpdir).resolve() / "quarantine" / "generated-")))
            self.assertEqual(["SKILL.md", "manifest.json", "metadata.json"], result.get("files_created"))
            self.assertFalse(result.get("executes_code"))
            self.assertFalse(result.get("approved"))
            self.assertFalse(result.get("enabled"))
            self.assertEqual([], result.get("permissions_granted"))
            self.assertTrue((source_path / "SKILL.md").is_file())
            self.assertTrue((source_path / "manifest.json").is_file())
            self.assertTrue((source_path / "metadata.json").is_file())
            self.assertFalse((source_path / "handler.py").exists())
            self.assertFalse((source_path / "requirements.txt").exists())

            combined = "\n".join(path.read_text(encoding="utf-8") for path in source_path.iterdir() if path.is_file())
            self.assertNotIn("neurons differentiating", combined)
            metadata = json.loads((source_path / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("exclude_raw_history_rows_full_urls_account_identifiers_and_private_search_terms", metadata["support_context_policy"])
            self.assertEqual("local_file_import", metadata["managed_adapters"][0]["kind"])

            ingestor = ExternalPackIngestor(tmpdir)
            normalized, _review = ingestor.ingest_from_path(str(source_path), source_origin="generated_scaffold")
            self.assertIn(normalized.status, {"normalized", "partial_safe_import"})
            normalized_root = Path(str(normalized.normalized_path))
            manifest = json.loads((normalized_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([], manifest["permissions_granted"])
            self.assertEqual("local_file_import", manifest["managed_adapters"][0]["kind"])

    def test_generated_scaffold_with_executable_file_is_stripped_by_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview = build_scaffold_preview("youtube_history_search")
            assert preview is not None
            result = create_generated_scaffold_source(preview, storage_root=tmpdir)
            source_path = Path(str(result["source_path"]))
            (source_path / "handler.py").write_text("print('should not execute')\n", encoding="utf-8")

            ingestor = ExternalPackIngestor(tmpdir)
            normalized, _review = ingestor.ingest_from_path(str(source_path), source_origin="generated_scaffold")

            self.assertIn("handler.py", normalized.stripped_components)
            self.assertFalse((Path(str(normalized.normalized_path)) / "handler.py").exists())
            manifest = json.loads((Path(str(normalized.normalized_path)) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([], manifest["permissions_granted"])


if __name__ == "__main__":
    unittest.main()
