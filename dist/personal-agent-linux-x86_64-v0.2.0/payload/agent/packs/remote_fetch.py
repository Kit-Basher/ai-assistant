from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 500
MAX_UNPACKED_BYTES = 50 * 1024 * 1024
CHUNK_SIZE = 64 * 1024

REMOTE_KIND_GITHUB_REPO = "github_repo"
REMOTE_KIND_GITHUB_ARCHIVE = "github_archive"
REMOTE_KIND_GENERIC_ARCHIVE_URL = "generic_archive_url"
ALLOWED_REMOTE_KINDS = {
    REMOTE_KIND_GITHUB_REPO,
    REMOTE_KIND_GITHUB_ARCHIVE,
    REMOTE_KIND_GENERIC_ARCHIVE_URL,
}
ALLOWED_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


def _now_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slugify(value: str) -> str:
    lowered = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "remote-pack"


def _is_commit_like(value: str | None) -> bool:
    ref = str(value or "").strip()
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", ref))


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RemotePackSource:
    kind: str
    url: str
    ref: str | None = None
    commit_hash_resolved: str | None = None
    fetched_at: str | None = None
    transport: str = "https_archive"
    archive_sha256: str | None = None
    top_level_dir_name: str | None = None
    provenance_notes: tuple[str, ...] = ()
    resolved_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemoteFetchResult:
    source: RemotePackSource
    quarantine_path: str
    snapshot_path: str
    raw_archive_path: str
    file_count: int
    total_unpacked_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RemoteFetchError(RuntimeError):
    def __init__(
        self,
        *,
        source: RemotePackSource,
        error_kind: str,
        message: str,
        flags: tuple[str, ...] = (),
        hard_block_reasons: tuple[str, ...] = (),
        archive_sha256: str | None = None,
        quarantine_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.error_kind = error_kind
        self.message = message
        self.flags = tuple(sorted(set(flags)))
        self.hard_block_reasons = tuple(sorted(set(hard_block_reasons)))
        self.archive_sha256 = archive_sha256
        self.quarantine_path = quarantine_path


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        parsed = urllib.parse.urlparse(newurl)
        if str(parsed.scheme or "").lower() != "https":
            raise urllib.error.HTTPError(newurl, code, "redirect_to_non_https_blocked", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class RemotePackFetcher:
    def __init__(self, storage_root: str, *, opener: Any | None = None) -> None:
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.fetch_root = self.storage_root / "fetch"
        self.quarantine_root = self.storage_root / "quarantine"
        self.fetch_root.mkdir(parents=True, exist_ok=True)
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        self._opener = opener or urllib.request.build_opener(_HttpsOnlyRedirectHandler())

    @staticmethod
    def build_source(
        *,
        kind: str,
        url: str,
        ref: str | None = None,
    ) -> RemotePackSource:
        normalized_kind = str(kind or "").strip().lower()
        normalized_url = str(url or "").strip()
        normalized_ref = str(ref or "").strip() or None
        return RemotePackSource(
            kind=normalized_kind,
            url=normalized_url,
            ref=normalized_ref,
        )

    def fetch(self, source: RemotePackSource) -> RemoteFetchResult:
        prepared = self._prepare_source(source)
        download_url = str(prepared.resolved_url or prepared.url).strip()
        parsed = urllib.parse.urlparse(download_url)
        if str(parsed.scheme or "").lower() != "https":
            raise RemoteFetchError(
                source=prepared,
                error_kind="non_https_source",
                message="Only https remote sources are supported.",
                flags=("non_https_source",),
                hard_block_reasons=("non_https_source",),
            )
        if parsed.username or parsed.password:
            raise RemoteFetchError(
                source=prepared,
                error_kind="authenticated_fetch_not_supported",
                message="Authenticated remote fetch is not supported in this pass.",
                flags=("authenticated_fetch_not_supported",),
                hard_block_reasons=("authenticated_fetch_not_supported",),
            )
        downloaded_path, archive_sha256, final_url = self._download_archive(prepared, download_url)
        prepared = RemotePackSource(
            kind=prepared.kind,
            url=prepared.url,
            ref=prepared.ref,
            commit_hash_resolved=prepared.commit_hash_resolved,
            fetched_at=_now_iso(),
            transport=prepared.transport,
            archive_sha256=archive_sha256,
            top_level_dir_name=prepared.top_level_dir_name,
            provenance_notes=prepared.provenance_notes,
            resolved_url=final_url,
        )
        quarantine_dir = self.quarantine_root / f"remote-{archive_sha256[:16]}"
        if quarantine_dir.exists():
            shutil.rmtree(quarantine_dir)
        raw_dir = quarantine_dir / "raw"
        extracted_dir = quarantine_dir / "snapshot"
        raw_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        raw_archive_name = self._raw_archive_name(download_url, final_url)
        raw_archive_path = raw_dir / raw_archive_name
        shutil.move(str(downloaded_path), raw_archive_path)
        file_count, total_unpacked_bytes, top_level_dir_name = self._extract_archive(
            raw_archive_path,
            extracted_dir,
            prepared,
        )
        provenance = RemotePackSource(
            kind=prepared.kind,
            url=prepared.url,
            ref=prepared.ref,
            commit_hash_resolved=prepared.commit_hash_resolved,
            fetched_at=prepared.fetched_at,
            transport=prepared.transport,
            archive_sha256=archive_sha256,
            top_level_dir_name=top_level_dir_name,
            provenance_notes=prepared.provenance_notes,
            resolved_url=final_url,
        )
        (quarantine_dir / "provenance.json").write_text(_safe_json(provenance.to_dict()) + "\n", encoding="utf-8")
        snapshot_path = extracted_dir / top_level_dir_name if top_level_dir_name and (extracted_dir / top_level_dir_name).is_dir() else extracted_dir
        return RemoteFetchResult(
            source=provenance,
            quarantine_path=str(quarantine_dir),
            snapshot_path=str(snapshot_path),
            raw_archive_path=str(raw_archive_path),
            file_count=file_count,
            total_unpacked_bytes=total_unpacked_bytes,
        )

    def _prepare_source(self, source: RemotePackSource) -> RemotePackSource:
        if source.kind not in ALLOWED_REMOTE_KINDS:
            raise RemoteFetchError(
                source=source,
                error_kind="unsupported_remote_source_kind",
                message="That remote source kind is not supported.",
                flags=("unsupported_remote_source_kind",),
                hard_block_reasons=("unsupported_remote_source_kind",),
            )
        parsed = urllib.parse.urlparse(source.url)
        scheme = str(parsed.scheme or "").lower()
        if scheme != "https":
            raise RemoteFetchError(
                source=source,
                error_kind="non_https_source",
                message="Only https remote sources are supported.",
                flags=("non_https_source",),
                hard_block_reasons=("non_https_source",),
            )
        provenance_notes: list[str] = []
        resolved_url = source.url
        commit_hash_resolved = source.commit_hash_resolved
        ref = source.ref
        if source.kind == REMOTE_KIND_GITHUB_REPO:
            owner_repo = [part for part in parsed.path.split("/") if part]
            if len(owner_repo) < 2:
                raise RemoteFetchError(
                    source=source,
                    error_kind="invalid_github_repo_url",
                    message="That GitHub repository URL is not valid.",
                    flags=("invalid_github_repo_url",),
                    hard_block_reasons=("invalid_github_repo_url",),
                )
            owner, repo = owner_repo[0], owner_repo[1]
            effective_ref = ref or "HEAD"
            if ref is None:
                provenance_notes.append("No ref was provided; fetched the current GitHub HEAD snapshot.")
            resolved_url = f"https://github.com/{owner}/{repo}/archive/{effective_ref}.zip"
            if _is_commit_like(effective_ref):
                commit_hash_resolved = effective_ref
            ref = effective_ref
        elif source.kind == REMOTE_KIND_GITHUB_ARCHIVE:
            if parsed.netloc.lower() not in {"github.com", "codeload.github.com"}:
                provenance_notes.append("The URL is not a standard GitHub archive host.")
            inferred_ref = ref or self._infer_ref_from_github_archive(source.url)
            ref = inferred_ref
            if _is_commit_like(inferred_ref):
                commit_hash_resolved = inferred_ref
        else:
            lowered_path = parsed.path.lower()
            if not lowered_path.endswith(ALLOWED_ARCHIVE_SUFFIXES):
                raise RemoteFetchError(
                    source=source,
                    error_kind="unsupported_archive_type",
                    message="Only zip or tar archive URLs are supported.",
                    flags=("unsupported_archive_type",),
                    hard_block_reasons=("unsupported_archive_type",),
                )
        if ref and not _is_commit_like(ref):
            provenance_notes.append("The requested ref is not a pinned commit hash.")
        return RemotePackSource(
            kind=source.kind,
            url=source.url,
            ref=ref,
            commit_hash_resolved=commit_hash_resolved,
            fetched_at=source.fetched_at,
            transport="https_archive",
            archive_sha256=source.archive_sha256,
            top_level_dir_name=source.top_level_dir_name,
            provenance_notes=tuple(sorted(dict.fromkeys(provenance_notes))),
            resolved_url=resolved_url,
        )

    @staticmethod
    def _infer_ref_from_github_archive(url: str) -> str | None:
        path = urllib.parse.urlparse(url).path
        match = re.search(r"/archive/(?:refs/(?:heads|tags)/)?([^/]+)\.(zip|tar\.gz|tgz|tar)$", path)
        if match:
            return match.group(1)
        return None

    def _download_archive(self, source: RemotePackSource, download_url: str) -> tuple[Path, str, str]:
        temp_dir = Path(tempfile.mkdtemp(dir=str(self.fetch_root)))
        archive_path = temp_dir / "downloaded.archive"
        sha256 = hashlib.sha256()
        total_bytes = 0
        request = urllib.request.Request(
            download_url,
            headers={
                "User-Agent": "Personal-Agent/pack-fetch",
                "Accept": "application/zip,application/x-tar,application/gzip,application/octet-stream",
            },
        )
        try:
            with self._opener.open(request, timeout=15) as response, open(archive_path, "wb") as handle:
                final_url = str(response.geturl() or download_url)
                final_parsed = urllib.parse.urlparse(final_url)
                if str(final_parsed.scheme or "").lower() != "https":
                    raise RemoteFetchError(
                        source=source,
                        error_kind="redirect_to_non_https_blocked",
                        message="Remote fetch redirected to a non-https URL and was blocked.",
                        flags=("redirect_to_non_https_blocked",),
                        hard_block_reasons=("redirect_to_non_https_blocked",),
                    )
                content_length = str(getattr(response, "headers", {}).get("Content-Length") or "").strip()
                if content_length:
                    try:
                        if int(content_length) > MAX_DOWNLOAD_BYTES:
                            raise RemoteFetchError(
                                source=source,
                                error_kind="oversized_archive_rejected",
                                message="The remote archive exceeds the download size limit.",
                                flags=("oversized_archive_rejected",),
                                hard_block_reasons=("oversized_archive_rejected",),
                            )
                    except ValueError:
                        pass
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_DOWNLOAD_BYTES:
                        raise RemoteFetchError(
                            source=source,
                            error_kind="oversized_archive_rejected",
                            message="The remote archive exceeds the download size limit.",
                            flags=("oversized_archive_rejected",),
                            hard_block_reasons=("oversized_archive_rejected",),
                        )
                    sha256.update(chunk)
                    handle.write(chunk)
        except RemoteFetchError:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RemoteFetchError(
                source=source,
                error_kind="remote_fetch_failed",
                message=f"Could not safely fetch the remote archive: {exc.__class__.__name__}.",
                flags=("remote_fetch_failed",),
                hard_block_reasons=("remote_fetch_failed",),
            ) from exc
        return archive_path, sha256.hexdigest(), final_url

    @staticmethod
    def _raw_archive_name(original_url: str, final_url: str) -> str:
        path = urllib.parse.urlparse(final_url or original_url).path
        name = Path(path).name
        return name or "archive.zip"

    def _extract_archive(
        self,
        archive_path: Path,
        destination: Path,
        source: RemotePackSource,
    ) -> tuple[int, int, str | None]:
        if zipfile.is_zipfile(archive_path):
            return self._extract_zip(archive_path, destination, source)
        if tarfile.is_tarfile(archive_path):
            return self._extract_tar(archive_path, destination, source)
        raise RemoteFetchError(
            source=source,
            error_kind="unsupported_archive_type",
            message="The fetched file is not a supported archive type.",
            flags=("unsupported_archive_type",),
            hard_block_reasons=("unsupported_archive_type",),
            archive_sha256=source.archive_sha256,
        )

    def _extract_zip(
        self,
        archive_path: Path,
        destination: Path,
        source: RemotePackSource,
    ) -> tuple[int, int, str | None]:
        file_count = 0
        total_bytes = 0
        top_level_parts: set[str] = set()
        seen_paths: set[str] = set()
        with zipfile.ZipFile(archive_path) as handle:
            members = handle.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise RemoteFetchError(
                    source=source,
                    error_kind="too_many_archive_members",
                    message="The remote archive contains too many files.",
                    flags=("too_many_archive_members",),
                    hard_block_reasons=("too_many_archive_members",),
                    archive_sha256=source.archive_sha256,
                )
            for member in members:
                rel_path = self._validated_member_path(member.filename, source=source)
                if rel_path is None:
                    continue
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise RemoteFetchError(
                        source=source,
                        error_kind="symlink_entries_rejected",
                        message="The remote archive contains symlink entries and was blocked.",
                        flags=("symlink_entries_rejected",),
                        hard_block_reasons=("symlink_entries_rejected",),
                        archive_sha256=source.archive_sha256,
                    )
                if rel_path in seen_paths:
                    raise RemoteFetchError(
                        source=source,
                        error_kind="duplicate_archive_member",
                        message="The remote archive contains duplicate file paths.",
                        flags=("unknown_archive_layout",),
                        hard_block_reasons=("unknown_archive_layout",),
                        archive_sha256=source.archive_sha256,
                    )
                seen_paths.add(rel_path)
                if member.is_dir():
                    continue
                file_count += 1
                total_bytes += int(member.file_size or 0)
                if file_count > MAX_ARCHIVE_MEMBERS:
                    raise RemoteFetchError(
                        source=source,
                        error_kind="too_many_archive_members",
                        message="The remote archive contains too many files.",
                        flags=("too_many_archive_members",),
                        hard_block_reasons=("too_many_archive_members",),
                        archive_sha256=source.archive_sha256,
                    )
                if total_bytes > MAX_UNPACKED_BYTES:
                    raise RemoteFetchError(
                        source=source,
                        error_kind="oversized_archive_rejected",
                        message="The remote archive expands beyond the unpacked size limit.",
                        flags=("oversized_archive_rejected",),
                        hard_block_reasons=("oversized_archive_rejected",),
                        archive_sha256=source.archive_sha256,
                    )
                target = destination / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                first_part = PurePosixPath(rel_path).parts[0]
                top_level_parts.add(first_part)
        top_level_dir_name = next(iter(top_level_parts)) if len(top_level_parts) == 1 else None
        return file_count, total_bytes, top_level_dir_name

    def _extract_tar(
        self,
        archive_path: Path,
        destination: Path,
        source: RemotePackSource,
    ) -> tuple[int, int, str | None]:
        file_count = 0
        total_bytes = 0
        top_level_parts: set[str] = set()
        seen_paths: set[str] = set()
        with tarfile.open(archive_path, mode="r:*") as handle:
            members = handle.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise RemoteFetchError(
                    source=source,
                    error_kind="too_many_archive_members",
                    message="The remote archive contains too many files.",
                    flags=("too_many_archive_members",),
                    hard_block_reasons=("too_many_archive_members",),
                    archive_sha256=source.archive_sha256,
                )
            for member in members:
                rel_path = self._validated_member_path(member.name, source=source)
                if rel_path is None:
                    continue
                if member.issym() or member.islnk():
                    raise RemoteFetchError(
                        source=source,
                        error_kind="symlink_entries_rejected",
                        message="The remote archive contains symlink entries and was blocked.",
                        flags=("symlink_entries_rejected",),
                        hard_block_reasons=("symlink_entries_rejected",),
                        archive_sha256=source.archive_sha256,
                    )
                if member.isdev():
                    raise RemoteFetchError(
                        source=source,
                        error_kind="special_archive_entry_rejected",
                        message="The remote archive contains special device entries and was blocked.",
                        flags=("special_archive_entry_rejected",),
                        hard_block_reasons=("special_archive_entry_rejected",),
                        archive_sha256=source.archive_sha256,
                    )
                if rel_path in seen_paths:
                    raise RemoteFetchError(
                        source=source,
                        error_kind="duplicate_archive_member",
                        message="The remote archive contains duplicate file paths.",
                        flags=("unknown_archive_layout",),
                        hard_block_reasons=("unknown_archive_layout",),
                        archive_sha256=source.archive_sha256,
                    )
                seen_paths.add(rel_path)
                if member.isdir():
                    continue
                file_count += 1
                total_bytes += int(member.size or 0)
                if file_count > MAX_ARCHIVE_MEMBERS:
                    raise RemoteFetchError(
                        source=source,
                        error_kind="too_many_archive_members",
                        message="The remote archive contains too many files.",
                        flags=("too_many_archive_members",),
                        hard_block_reasons=("too_many_archive_members",),
                        archive_sha256=source.archive_sha256,
                    )
                if total_bytes > MAX_UNPACKED_BYTES:
                    raise RemoteFetchError(
                        source=source,
                        error_kind="oversized_archive_rejected",
                        message="The remote archive expands beyond the unpacked size limit.",
                        flags=("oversized_archive_rejected",),
                        hard_block_reasons=("oversized_archive_rejected",),
                        archive_sha256=source.archive_sha256,
                    )
                src = handle.extractfile(member)
                if src is None:
                    continue
                target = destination / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                first_part = PurePosixPath(rel_path).parts[0]
                top_level_parts.add(first_part)
        top_level_dir_name = next(iter(top_level_parts)) if len(top_level_parts) == 1 else None
        return file_count, total_bytes, top_level_dir_name

    @staticmethod
    def _validated_member_path(name: str, *, source: RemotePackSource) -> str | None:
        normalized = str(name or "").replace("\\", "/").strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized:
            return None
        pure = PurePosixPath(normalized)
        parts = pure.parts
        if any(part in {"..", ""} for part in parts):
            raise RemoteFetchError(
                source=source,
                error_kind="traversal_entries_rejected",
                message="The remote archive contains unsafe traversal paths and was blocked.",
                flags=("traversal_entries_rejected",),
                hard_block_reasons=("traversal_entries_rejected",),
                archive_sha256=source.archive_sha256,
            )
        if pure.is_absolute() or normalized.startswith("/"):
            raise RemoteFetchError(
                source=source,
                error_kind="traversal_entries_rejected",
                message="The remote archive contains absolute paths and was blocked.",
                flags=("traversal_entries_rejected",),
                hard_block_reasons=("traversal_entries_rejected",),
                archive_sha256=source.archive_sha256,
            )
        return str(pure)
