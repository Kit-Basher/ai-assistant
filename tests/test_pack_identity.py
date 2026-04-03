from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from agent.packs.external_ingestion import ExternalPackIngestor
from agent.packs.remote_fetch import RemotePackFetcher, RemotePackSource
from agent.packs.store import PackStore


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, data in files.items():
            handle.writestr(name, data)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body: bytes, *, url: str) -> None:
        self._body = io.BytesIO(body)
        self._url = url
        self.headers = {"Content-Length": str(len(body))}

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeOpener:
    def __init__(self, mapping: dict[str, _FakeResponse]) -> None:
        self.mapping = mapping

    def open(self, request, timeout: int = 15):  # noqa: ANN001
        url = getattr(request, "full_url", str(request))
        response = self.mapping.get(url)
        if response is None:
            raise RuntimeError(f"unexpected url: {url}")
        return response


class TestPackIdentityAndTrustAnchoring(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.storage_root = str(self.root / "storage")
        self.store = PackStore(str(self.root / "packs.db"))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _record_remote_pack(
        self,
        *,
        source_url: str,
        archive_files: dict[str, bytes],
        source_kind: str = "github_archive",
        ref: str | None = None,
        download_url: str | None = None,
    ) -> dict[str, object]:
        archive = _zip_bytes(archive_files)
        fetch_url = str(download_url or source_url)
        fetcher = RemotePackFetcher(
            self.storage_root,
            opener=_FakeOpener({fetch_url: _FakeResponse(archive, url=fetch_url)}),
        )
        ingestor = ExternalPackIngestor(self.storage_root, remote_fetcher=fetcher)
        result, review = ingestor.ingest_from_remote_source(
            RemotePackSource(kind=source_kind, url=source_url, ref=ref)
        )
        row = self.store.record_external_pack(
            canonical_pack=result.pack.to_dict(),
            classification=result.classification,
            status=result.status,
            risk_report=result.risk_report.to_dict(),
            review_envelope=review.to_dict(),
            quarantine_path=result.quarantine_path,
            normalized_path=result.normalized_path,
        )
        return {
            "result": result,
            "review": review,
            "row": row,
        }

    def _record_local_pack(self, relative_dir: str, skill_text: str) -> dict[str, object]:
        source = self.root / relative_dir
        source.mkdir(parents=True, exist_ok=True)
        (source / "SKILL.md").write_text(skill_text, encoding="utf-8")
        ingestor = ExternalPackIngestor(self.storage_root)
        result, review = ingestor.ingest_from_path(str(source))
        row = self.store.record_external_pack(
            canonical_pack=result.pack.to_dict(),
            classification=result.classification,
            status=result.status,
            risk_report=result.risk_report.to_dict(),
            review_envelope=review.to_dict(),
            quarantine_path=result.quarantine_path,
            normalized_path=result.normalized_path,
        )
        return {
            "result": result,
            "review": review,
            "row": row,
        }

    def test_same_content_different_url_has_same_canonical_id(self) -> None:
        files = {
            "repo-main/SKILL.md": b"# Remote Skill\n\nUse the same normalized content.\n",
            "repo-main/references/guide.md": b"# Guide\n\nShared text.\n",
        }
        first = self._record_remote_pack(
            source_url="https://github.com/example/skill-one/archive/main.zip",
            archive_files=files,
            ref="main",
        )
        second = self._record_remote_pack(
            source_url="https://github.com/another-owner/skill-two/archive/main.zip",
            archive_files=files,
            ref="main",
        )

        first_row = first["row"]
        second_row = second["row"]
        assert isinstance(first_row, dict)
        assert isinstance(second_row, dict)
        self.assertEqual(first_row["canonical_id"], second_row["canonical_id"])
        self.assertNotEqual(first_row["source_fingerprint"], second_row["source_fingerprint"])
        self.assertEqual(1, len(self.store.list_external_packs()))
        merged = self.store.get_external_pack(str(first_row["canonical_id"]))
        assert merged is not None
        self.assertEqual(2, len(merged["source_history"]))

    def test_same_repo_different_commit_has_different_canonical_id(self) -> None:
        first = self._record_remote_pack(
            source_url="https://github.com/example/repo",
            archive_files={"repo-1111111/SKILL.md": b"# Repo Skill\n\nCommit one.\n"},
            source_kind="github_repo",
            ref="1111111",
            download_url="https://github.com/example/repo/archive/1111111.zip",
        )
        second = self._record_remote_pack(
            source_url="https://github.com/example/repo",
            archive_files={"repo-2222222/SKILL.md": b"# Repo Skill\n\nCommit two.\n"},
            source_kind="github_repo",
            ref="2222222",
            download_url="https://github.com/example/repo/archive/2222222.zip",
        )

        first_result = first["result"]
        second_result = second["result"]
        self.assertNotEqual(first_result.pack.id, second_result.pack.id)

    def test_upstream_mutation_increases_risk_and_records_diff(self) -> None:
        source_url = "https://github.com/example/repo/archive/main.zip"
        first = self._record_remote_pack(
            source_url=source_url,
            archive_files={
                "repo-main/SKILL.md": b"# Remote Skill\n\nUse the first instructions.\n",
                "repo-main/references/original.md": b"# Original\n\nKeep this file.\n",
            },
            ref="main",
        )
        second = self._record_remote_pack(
            source_url=source_url,
            archive_files={
                "repo-main/SKILL.md": b"# Remote Skill\n\nUse the changed instructions.\n",
                "repo-main/references/updated.md": b"# Updated\n\nNew reference file.\n",
            },
            ref="main",
        )

        first_row = first["row"]
        second_row = second["row"]
        assert isinstance(first_row, dict)
        assert isinstance(second_row, dict)
        self.assertGreater(float(second_row["risk_score"]), float(first_row["risk_score"]))
        self.assertIn("upstream_content_changed", second_row["risk_flags"])
        self.assertIn("changed since the last time it was seen", second_row["review_envelope"]["summary"])
        self.assertEqual(first_row["canonical_id"], second_row["previous_version"]["canonical_id"])
        change_summary = second_row["change_summary"]
        self.assertIn("references/updated.md", change_summary["new_files"])
        self.assertIn("references/original.md", change_summary["removed_files"])
        self.assertIn("SKILL.md", change_summary["changed_instructions"])

    def test_user_approval_is_tied_to_content_hash_not_name(self) -> None:
        first = self._record_local_pack(
            "local-skill-v1",
            "---\n"
            "id: same-name\n"
            "name: Same Name\n"
            "version: 1.0.0\n"
            "---\n"
            "# Same Name\n\n"
            "Version one.\n",
        )
        first_row = first["row"]
        assert isinstance(first_row, dict)
        approved = self.store.set_external_pack_review_status(
            str(first_row["canonical_id"]),
            local_review_status="approved",
            approve_current_hash=True,
        )
        assert approved is not None
        approved_anchor = approved["trust_anchor"]
        self.assertEqual("approved", approved_anchor["local_review_status"])
        self.assertIn(first_row["content_hash"], approved_anchor["user_approved_hashes"])

        second = self._record_local_pack(
            "local-skill-v2",
            "---\n"
            "id: same-name\n"
            "name: Same Name\n"
            "version: 9.9.9\n"
            "---\n"
            "# Same Name\n\n"
            "Version two with changed content.\n",
        )
        second_row = second["row"]
        assert isinstance(second_row, dict)
        self.assertEqual(first_row["name"], second_row["name"])
        self.assertNotEqual(first_row["canonical_id"], second_row["canonical_id"])
        self.assertNotIn(second_row["content_hash"], second_row["trust_anchor"]["user_approved_hashes"])
        self.assertEqual([], second_row["trust_anchor"]["user_approved_hashes"])

    def test_refetch_identical_archive_does_not_create_duplicate_pack(self) -> None:
        source_url = "https://github.com/example/repo/archive/main.zip"
        archive_files = {"repo-main/SKILL.md": b"# Remote Skill\n\nIdentical content.\n"}
        first = self._record_remote_pack(
            source_url=source_url,
            archive_files=archive_files,
            ref="main",
        )
        second = self._record_remote_pack(
            source_url=source_url,
            archive_files=archive_files,
            ref="main",
        )

        first_row = first["row"]
        second_row = second["row"]
        assert isinstance(first_row, dict)
        assert isinstance(second_row, dict)
        self.assertEqual(first_row["canonical_id"], second_row["canonical_id"])
        self.assertEqual(1, len(self.store.list_external_packs()))


if __name__ == "__main__":
    unittest.main()
