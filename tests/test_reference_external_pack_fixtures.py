from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from agent.packs.external_ingestion import ExternalPackIngestor


FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "reference_packs"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_paths(name: str) -> tuple[Path, dict, dict]:
    root = FIXTURES_ROOT / name
    return root / "source", _read_json(root / "source_manifest.json"), _read_json(root / "expected.json")


def _copy_fixture(source: Path, destination_root: Path, destination_name: str) -> Path:
    destination = destination_root / destination_name
    shutil.copytree(source, destination)
    return destination


class TestReferenceExternalPackFixtures(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.tmpdir.name) / "storage"
        self.ingestor = ExternalPackIngestor(str(self.storage_root))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _ingest_fixture(self, fixture_name: str):
        source, source_manifest, expected = _fixture_paths(fixture_name)
        copied_source = _copy_fixture(source, Path(self.tmpdir.name) / "sources", fixture_name)
        result, review = self.ingestor.ingest_from_path(str(copied_source))
        normalized_root = Path(result.normalized_path or "")
        manifest = _read_json(normalized_root / "manifest.json")
        source_meta = _read_json(normalized_root / "metadata" / "source.json")
        normalization_meta = _read_json(normalized_root / "metadata" / "normalization.json")
        review_meta = _read_json(normalized_root / "metadata" / "review.json")
        return {
            "source_manifest": source_manifest,
            "expected": expected,
            "result": result,
            "review": review,
            "normalized_root": normalized_root,
            "manifest": manifest,
            "source_meta": source_meta,
            "normalization_meta": normalization_meta,
            "review_meta": review_meta,
        }

    def test_reference_fixtures_import_into_expected_outcomes(self) -> None:
        cases = {
            "anthropic_clean_skill",
            "vercel_structured_skill",
            "fragmented_prompt_repo",
            "openclaw_mixed_skill",
            "vercel_blocked_native_skill",
        }
        for case in sorted(cases):
            with self.subTest(case=case):
                payload = self._ingest_fixture(case)
                expected = payload["expected"]
                result = payload["result"]
                review = payload["review"]
                normalized_root = payload["normalized_root"]
                manifest = payload["manifest"]
                source_meta = payload["source_meta"]
                normalization_meta = payload["normalization_meta"]
                review_meta = payload["review_meta"]

                self.assertEqual(expected["classification"], result.classification)
                self.assertEqual(expected["status"], result.status)
                self.assertEqual(expected["outcome_category"], review_meta["outcome_category"])
                self.assertEqual(result.pack.id, manifest["pack_id"])
                self.assertEqual(result.pack.version, manifest["version"])
                self.assertEqual(result.pack.id, source_meta["pack_id"])
                self.assertEqual(result.pack.version, source_meta["version"])
                self.assertEqual(source_meta["source"]["source_key"], manifest["source"]["source_key"])
                self.assertTrue(review.review_required)
                self.assertTrue(review.summary)
                self.assertEqual([], manifest["permissions_granted"])

                for rel_path in expected["must_have_paths"]:
                    self.assertTrue((normalized_root / rel_path).exists(), rel_path)

                for rel_path in expected.get("must_not_have_paths", []):
                    self.assertFalse((normalized_root / rel_path).exists(), rel_path)

                self.assertEqual(list(manifest["capabilities"]["declared"]), list(result.pack.capabilities["declared"]))
                self.assertIn("files_seen", normalization_meta)
                self.assertIn("files_kept", normalization_meta)
                self.assertIn("files_dropped", normalization_meta)
                self.assertEqual(review_meta["review_required"], True)
                for fragment in expected.get("summary_contains", []):
                    self.assertIn(fragment, review.summary)

                if case == "fragmented_prompt_repo":
                    skill_text = (normalized_root / "SKILL.md").read_text(encoding="utf-8")
                    self.assertIn("## Purpose", skill_text)
                    self.assertIn("## When to Use", skill_text)
                    self.assertIn("## Example Prompts", skill_text)

                if case == "anthropic_clean_skill":
                    skill_text = (normalized_root / "SKILL.md").read_text(encoding="utf-8")
                    self.assertIn("## Purpose", skill_text)
                    self.assertIn("## Response Style", skill_text)
                    self.assertIn("Import instructions and reference material only.", review.safe_options[0])

                if case == "vercel_structured_skill":
                    self.assertTrue((normalized_root / "assets" / "source" / "references" / "implementation.md").exists())
                    self.assertTrue((normalized_root / "assets" / "source" / "references" / "patterns.md").exists())
                    self.assertTrue((normalized_root / "assets" / "source" / "references" / "nextjs.md").exists())
                    self.assertTrue((normalized_root / "assets" / "source" / "references" / "css-recipes.md").exists())

                if case == "openclaw_mixed_skill":
                    self.assertGreater(len(result.stripped_components), 0)
                    self.assertIn("Use the imported text and assets only.", review.safe_options[0])

                if case == "vercel_blocked_native_skill":
                    self.assertIn("Block and keep only the audit snapshot.", review.safe_options[0])
                    self.assertIn("blocked it after review", review.summary.lower())

    def test_gold_standard_guides_fragmented_skill_synthesis(self) -> None:
        gold_standard = (Path(__file__).resolve().parents[1] / "agent" / "packs" / "GOLD_STANDARD.md").read_text(encoding="utf-8")
        section_titles = []
        capture = False
        for line in gold_standard.splitlines():
            lowered = line.strip().lower()
            if lowered == "## skill.md required sections":
                capture = True
                continue
            if capture:
                if line.startswith("## "):
                    break
                if line.strip().startswith("- "):
                    section_titles.append(line.strip()[2:].strip())
        self.assertGreaterEqual(len(section_titles), 7)
        payload = self._ingest_fixture("fragmented_prompt_repo")
        skill_text = (payload["normalized_root"] / "SKILL.md").read_text(encoding="utf-8")
        for title in section_titles:
            self.assertIn(f"## {title}", skill_text)

    def test_identical_safe_content_from_different_source_metadata_keeps_canonical_identity(self) -> None:
        source, source_manifest, _expected = _fixture_paths("anthropic_clean_skill")
        source_a = _copy_fixture(source, Path(self.tmpdir.name) / "identity_a", "clean_a")
        source_b = _copy_fixture(source, Path(self.tmpdir.name) / "identity_b", "clean_b")
        source_manifest_b = source_b.parent / "source_manifest.json"
        source_manifest_b.parent.mkdir(parents=True, exist_ok=True)
        source_manifest_b.write_text(
            json.dumps(
                {
                    **source_manifest,
                    "fixture_id": "anthropic_clean_skill_mirror",
                    "source_url": "https://mirror.example.invalid/skills/repo-helper",
                    "why_this_exists": "Mirror copy with the same safe content but different provenance metadata.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result_a, _review_a = self.ingestor.ingest_from_path(str(source_a))
        result_b, _review_b = self.ingestor.ingest_from_path(str(source_b))
        self.assertEqual(result_a.pack.id, result_b.pack.id)
        self.assertEqual(result_a.pack.version, result_b.pack.version)


if __name__ == "__main__":
    unittest.main()
