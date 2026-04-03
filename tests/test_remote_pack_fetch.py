from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from agent.packs.external_ingestion import CLASS_NATIVE_CODE_PACK, CLASS_PORTABLE_TEXT_SKILL, STATUS_BLOCKED, STATUS_NORMALIZED, STATUS_PARTIAL_SAFE_IMPORT, ExternalPackIngestor
from agent.packs.remote_fetch import (
    MAX_ARCHIVE_MEMBERS,
    MAX_DOWNLOAD_BYTES,
    RemoteFetchError,
    RemotePackFetcher,
    RemotePackSource,
)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, data in files.items():
            handle.writestr(name, data)
    return buf.getvalue()


def _tar_bytes(entries: list[tuple[str, bytes | None, str]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as handle:
        for name, data, entry_type in entries:
            info = tarfile.TarInfo(name)
            if entry_type == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "target.txt"
                handle.addfile(info)
                continue
            payload = data or b""
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, body: bytes, *, url: str, content_length: int | None = None) -> None:
        self._body = io.BytesIO(body)
        self._url = url
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

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
        self.seen_urls: list[str] = []

    def open(self, request, timeout: int = 15):  # noqa: ANN001
        url = getattr(request, "full_url", str(request))
        self.seen_urls.append(url)
        response = self.mapping.get(url)
        if response is None:
            raise RuntimeError(f"unexpected url: {url}")
        return response


class TestRemotePackFetch(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_github_archive_fetch_extracts_into_quarantine(self) -> None:
        archive = _zip_bytes(
            {
                "repo-main/SKILL.md": b"# Remote Skill\n\nUse only the safe notes.\n",
                "repo-main/references/guide.md": b"# Guide\n\nUseful notes.\n",
            }
        )
        url = "https://github.com/example/repo/archive/main.zip"
        opener = _FakeOpener({url: _FakeResponse(archive, url=url, content_length=len(archive))})
        fetcher = RemotePackFetcher(str(self.root / "storage"), opener=opener)

        result = fetcher.fetch(RemotePackSource(kind="github_archive", url=url, ref="main"))

        self.assertTrue(result.quarantine_path)
        self.assertTrue(Path(result.raw_archive_path).exists())
        self.assertTrue(Path(result.snapshot_path, "SKILL.md").exists())
        self.assertEqual("repo-main", result.source.top_level_dir_name)
        self.assertEqual("main", result.source.ref)
        self.assertIn("The requested ref is not a pinned commit hash.", result.source.provenance_notes)

    def test_zip_slip_rejected(self) -> None:
        archive = _zip_bytes({"../escape.txt": b"nope"})
        url = "https://example.com/escape.zip"
        opener = _FakeOpener({url: _FakeResponse(archive, url=url)})
        fetcher = RemotePackFetcher(str(self.root / "storage"), opener=opener)

        with self.assertRaises(RemoteFetchError) as raised:
            fetcher.fetch(RemotePackSource(kind="generic_archive_url", url=url))

        self.assertEqual("traversal_entries_rejected", raised.exception.error_kind)

    def test_symlink_rejected(self) -> None:
        archive = _tar_bytes([("repo-main/link", None, "symlink")])
        url = "https://example.com/archive.tgz"
        opener = _FakeOpener({url: _FakeResponse(archive, url=url)})
        fetcher = RemotePackFetcher(str(self.root / "storage"), opener=opener)

        with self.assertRaises(RemoteFetchError) as raised:
            fetcher.fetch(RemotePackSource(kind="generic_archive_url", url=url))

        self.assertEqual("symlink_entries_rejected", raised.exception.error_kind)

    def test_oversized_archive_rejected(self) -> None:
        url = "https://example.com/big.zip"
        opener = _FakeOpener({url: _FakeResponse(b"", url=url, content_length=MAX_DOWNLOAD_BYTES + 1)})
        fetcher = RemotePackFetcher(str(self.root / "storage"), opener=opener)

        with self.assertRaises(RemoteFetchError) as raised:
            fetcher.fetch(RemotePackSource(kind="generic_archive_url", url=url))

        self.assertEqual("oversized_archive_rejected", raised.exception.error_kind)

    def test_too_many_files_rejected(self) -> None:
        files = {f"repo-main/file-{idx}.txt": b"x" for idx in range(MAX_ARCHIVE_MEMBERS + 1)}
        archive = _zip_bytes(files)
        url = "https://example.com/many.zip"
        opener = _FakeOpener({url: _FakeResponse(archive, url=url)})
        fetcher = RemotePackFetcher(str(self.root / "storage"), opener=opener)

        with self.assertRaises(RemoteFetchError) as raised:
            fetcher.fetch(RemotePackSource(kind="generic_archive_url", url=url))

        self.assertEqual("too_many_archive_members", raised.exception.error_kind)

    def test_remote_skill_pack_normalizes(self) -> None:
        archive = _zip_bytes({"repo-main/SKILL.md": b"# Remote Skill\n\nAnswer from the docs.\n"})
        url = "https://github.com/example/repo/archive/main.zip"
        opener = _FakeOpener({url: _FakeResponse(archive, url=url)})
        ingestor = ExternalPackIngestor(str(self.root / "storage"), remote_fetcher=RemotePackFetcher(str(self.root / "storage"), opener=opener))

        result, review = ingestor.ingest_from_remote_source(
            RemotePackSource(kind="github_archive", url=url, ref="main")
        )

        self.assertEqual(CLASS_PORTABLE_TEXT_SKILL, result.classification)
        self.assertEqual(STATUS_NORMALIZED, result.status)
        self.assertIn("archive_fetch", result.risk_report.flags)
        self.assertIn("remote_unpinned_ref", result.risk_report.flags)
        self.assertIn("I fetched a snapshot", review.summary)

    def test_remote_skill_pack_with_script_is_partial_safe_import(self) -> None:
        archive = _zip_bytes(
            {
                "repo-main/SKILL.md": b"# Remote Skill\n\nUse the docs only.\n",
                "repo-main/install.sh": b"#!/bin/sh\necho nope\n",
            }
        )
        url = "https://example.com/skill.zip"
        opener = _FakeOpener({url: _FakeResponse(archive, url=url)})
        ingestor = ExternalPackIngestor(str(self.root / "storage"), remote_fetcher=RemotePackFetcher(str(self.root / "storage"), opener=opener))

        result, _review = ingestor.ingest_from_remote_source(
            RemotePackSource(kind="generic_archive_url", url=url)
        )

        self.assertEqual(STATUS_PARTIAL_SAFE_IMPORT, result.status)
        self.assertIn("install.sh", result.stripped_components)

    def test_remote_plugin_pack_is_blocked(self) -> None:
        archive = _zip_bytes(
            {
                "repo-main/package.json": b'{"name":"plugin-pack"}',
                "repo-main/handler.js": b"export const run = () => true;\n",
            }
        )
        url = "https://example.com/plugin.zip"
        opener = _FakeOpener({url: _FakeResponse(archive, url=url)})
        ingestor = ExternalPackIngestor(str(self.root / "storage"), remote_fetcher=RemotePackFetcher(str(self.root / "storage"), opener=opener))

        result, _review = ingestor.ingest_from_remote_source(
            RemotePackSource(kind="generic_archive_url", url=url)
        )

        self.assertEqual(CLASS_NATIVE_CODE_PACK, result.classification)
        self.assertEqual(STATUS_BLOCKED, result.status)

    def test_non_https_remote_source_is_blocked(self) -> None:
        ingestor = ExternalPackIngestor(str(self.root / "storage"))

        result, review = ingestor.ingest_from_remote_source(
            RemotePackSource(kind="generic_archive_url", url="http://example.com/pack.zip")
        )

        self.assertEqual(STATUS_BLOCKED, result.status)
        self.assertIn("non_https_source", result.risk_report.flags)
        self.assertIn("could not safely fetch", review.summary.lower())

    def test_redirect_to_non_https_is_blocked(self) -> None:
        archive = _zip_bytes({"repo-main/SKILL.md": b"# Remote Skill\n"})
        url = "https://example.com/redirect.zip"
        opener = _FakeOpener({url: _FakeResponse(archive, url="http://example.com/redirect.zip")})
        ingestor = ExternalPackIngestor(str(self.root / "storage"), remote_fetcher=RemotePackFetcher(str(self.root / "storage"), opener=opener))

        result, _review = ingestor.ingest_from_remote_source(
            RemotePackSource(kind="generic_archive_url", url=url)
        )

        self.assertEqual(STATUS_BLOCKED, result.status)
        self.assertIn("redirect_to_non_https_blocked", result.risk_report.flags)


if __name__ == "__main__":
    unittest.main()
