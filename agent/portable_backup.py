"""Portable, allowlisted Personal Agent backup and isolated-restore contract."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import tarfile
import tempfile
import time
from typing import Any

from agent.version import read_version


MANIFEST_NAME = "personal-agent-backup-manifest.json"
BACKUP_SCHEMA_VERSION = 2
BACKUP_CONTRACT = "personal-agent.portable-backup.v2"
APP_VERSION = read_version()[0]
BACKUP_EXACT_FILES = (
    ".local/share/personal-agent/agent.db",
    ".local/share/personal-agent/confirmation_transactions.sqlite3",
    ".local/share/personal-agent/managed_actions.db",
    ".local/share/personal-agent/llm_registry.json",
    ".local/share/personal-agent/search_runtime_config.json",
    ".local/share/personal-agent/autopilot_state.json",
    ".local/share/personal-agent/model_watch_state.json",
    ".local/share/personal-agent/provider_catalog_state.json",
    ".local/share/personal-agent/mutation_plans_v1.json",
    ".config/systemd/user/personal-agent-api.service",
)
BACKUP_DIRECTORY_ROOTS = (
    ".config/personal-agent",
    ".local/share/personal-agent/external_packs/normalized",
    ".local/share/personal-agent/external_packs/capability-runtime-v1",
    ".local/share/personal-agent/external_packs/visualizers-v1",
    ".local/share/personal-agent/memory",
    ".config/systemd/user/personal-agent-api.service.d",
    ".config/systemd/user/personal-agent-telegram.service.d",
)
EXCLUDED_NAME_FRAGMENTS = ("secret", "token", "password", "cookie", "credential", "private_key")
MAX_FILES = 10_000
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class BackupValidation:
    ok: bool
    error: str | None
    manifest: dict[str, Any]
    files: tuple[str, ...]
    sensitive_files: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class BackupRestoreError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sensitive(relative_path: str) -> bool:
    lowered = relative_path.lower()
    return any(fragment in lowered for fragment in EXCLUDED_NAME_FRAGMENTS)


def _safe_relative(path: str) -> str:
    normalized = Path(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise BackupRestoreError("unsafe_path", "Backup contains an unsafe path.")
    clean = normalized.as_posix()
    if not clean or clean == "." or len(clean) > 512:
        raise BackupRestoreError("unsafe_path", "Backup contains an invalid path.")
    return clean


def _eligible_file(source_home: Path, path: Path) -> tuple[bool, str | None]:
    try:
        metadata = path.lstat()
    except OSError:
        return False, "unreadable"
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        return False, "unsupported_type"
    relative = path.relative_to(source_home).as_posix()
    if _is_sensitive(relative):
        return False, "secret_reentry_required"
    if metadata.st_size > MAX_FILE_BYTES:
        return False, "file_too_large"
    return True, None


def _collect_backup_files(source_home: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    candidates: set[Path] = set()
    for relative in BACKUP_EXACT_FILES:
        candidate = source_home / relative
        if candidate.exists() or candidate.is_symlink():
            candidates.add(candidate)
    for relative in BACKUP_DIRECTORY_ROOTS:
        root = source_home / relative
        if root.is_dir() and not root.is_symlink():
            candidates.update(root.rglob("*"))
    files: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    total = 0
    for path in sorted(candidates):
        relative = path.relative_to(source_home).as_posix()
        eligible, reason = _eligible_file(source_home, path)
        if not eligible:
            if reason and (path.is_file() or path.is_symlink()):
                excluded.append({"path": relative, "reason": reason})
            continue
        size = path.stat().st_size
        total += size
        if len(files) >= MAX_FILES:
            raise BackupRestoreError("file_count_limit", "Portable backup file-count limit exceeded.")
        if total > MAX_TOTAL_BYTES:
            raise BackupRestoreError("total_size_limit", "Portable backup size limit exceeded.")
        files.append({"path": relative, "size": size})
    return files, excluded


def _copy_portable_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as handle:
            header = handle.read(16)
    except OSError as exc:
        raise BackupRestoreError("source_unreadable", "An allowlisted backup source could not be read.") from exc
    if header == b"SQLite format 3\x00":
        try:
            with sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=30.0) as source_db:
                with sqlite3.connect(str(destination), timeout=30.0) as target_db:
                    source_db.backup(target_db)
                    integrity = target_db.execute("PRAGMA integrity_check").fetchone()
                    if integrity is None or str(integrity[0]).lower() != "ok":
                        raise BackupRestoreError("sqlite_snapshot_invalid", "A SQLite backup snapshot failed integrity validation.")
        except sqlite3.Error as exc:
            raise BackupRestoreError("sqlite_snapshot_failed", "A SQLite backup snapshot could not be created.") from exc
    else:
        shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)


def create_backup(source_home: Path, archive_path: Path, *, app_version: str = APP_VERSION) -> Path:
    source_home = source_home.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f".{archive_path.name}.partial")
    temporary.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="personal-agent-portable-snapshot-", dir=str(archive_path.parent)) as raw_stage:
        stage = Path(raw_stage)
        source_rows, excluded = _collect_backup_files(source_home)
        files: list[dict[str, Any]] = []
        total = 0
        for source_row in source_rows:
            relative = _safe_relative(str(source_row["path"]))
            destination = stage / relative
            _copy_portable_file(source_home / relative, destination)
            size = destination.stat().st_size
            total += size
            if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise BackupRestoreError("size_limit", "Portable backup size limit exceeded after consistent snapshotting.")
            files.append({"path": relative, "size": size, "sha256": _sha256(destination), "mode": 0o600})
        manifest = {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "contract": BACKUP_CONTRACT,
            "app": "personal-agent",
            "app_version": app_version,
            "created_at_epoch": int(time.time()),
            "included_classes": ["state_database", "configuration", "model_registry", "pack_state", "memory_state", "service_configuration"],
            "excluded": excluded,
            "excluded_classes": ["runtime_releases", "caches", "logs", "quarantine", "model_artifacts", "arbitrary_home_data", "plaintext_or_machine_bound_secrets"],
            "secret_restoration": "reenter secrets in Setup after restore; machine-bound encrypted secret files are not portable",
            "files": files,
            "sqlite_capture": "online_backup_with_integrity_check",
            "service_actions": "none",
            "restore_policy": "validated_staging_to_empty_or_identical_explicit_target_only",
            "limits": {"files": MAX_FILES, "file_bytes": MAX_FILE_BYTES, "total_bytes": MAX_TOTAL_BYTES},
        }
        manifest_bytes = json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8")
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                info = tarfile.TarInfo(MANIFEST_NAME)
                info.size = len(manifest_bytes)
                info.mode = 0o600
                info.mtime = int(time.time())
                archive.addfile(info, io.BytesIO(manifest_bytes))
                for row in files:
                    relative = _safe_relative(str(row["path"]))
                    info = archive.gettarinfo(str(stage / relative), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o600
                    with (stage / relative).open("rb") as source:
                        archive.addfile(info, source)
            os.replace(temporary, archive_path)
            os.chmod(archive_path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)
    return archive_path


def _read_manifest(archive: tarfile.TarFile) -> dict[str, Any]:
    try:
        member = archive.getmember(MANIFEST_NAME)
    except KeyError as exc:
        raise BackupRestoreError("manifest_missing", "Backup manifest is missing.") from exc
    if not member.isfile() or member.size > 1024 * 1024:
        raise BackupRestoreError("manifest_invalid", "Backup manifest is invalid.")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise BackupRestoreError("manifest_unreadable", "Backup manifest cannot be read.")
    try:
        parsed = json.loads(extracted.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupRestoreError("manifest_invalid", "Backup manifest is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise BackupRestoreError("manifest_invalid", "Backup manifest must be an object.")
    return parsed


def validate_backup(archive_path: Path, *, expected_app_version: str | None = APP_VERSION, strict_version: bool = True) -> BackupValidation:
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            manifest = _read_manifest(archive)
            if int(manifest.get("schema_version") or 0) != BACKUP_SCHEMA_VERSION or manifest.get("contract") != BACKUP_CONTRACT:
                raise BackupRestoreError("schema_version_unsupported", "Backup schema is not supported.")
            if manifest.get("app") != "personal-agent":
                raise BackupRestoreError("wrong_app", "Backup belongs to another application.")
            warnings: list[str] = []
            version = str(manifest.get("app_version") or "").strip()
            if expected_app_version and version != expected_app_version:
                if strict_version:
                    raise BackupRestoreError("version_mismatch", "Backup application version does not match.")
                warnings.append("version_mismatch")
            rows = [row for row in (manifest.get("files") or []) if isinstance(row, dict)]
            if len(rows) > MAX_FILES:
                raise BackupRestoreError("file_count_limit", "Backup file-count limit exceeded.")
            expected = {_safe_relative(str(row.get("path") or "")) for row in rows}
            archive_paths: set[str] = set()
            total_size = 0
            for member in archive.getmembers():
                name = _safe_relative(member.name)
                if not member.isfile():
                    raise BackupRestoreError("unsupported_member_type", "Backup contains a link or special member.")
                if name in archive_paths:
                    raise BackupRestoreError("duplicate_member", "Backup contains duplicate members.")
                archive_paths.add(name)
                total_size += int(member.size)
                if member.size > MAX_FILE_BYTES or total_size > MAX_TOTAL_BYTES:
                    raise BackupRestoreError("size_limit", "Backup size limit exceeded.")
            unexpected = archive_paths - expected - {MANIFEST_NAME}
            if unexpected:
                raise BackupRestoreError("unexpected_member", "Backup contains an unlisted member.")
            if expected - archive_paths:
                raise BackupRestoreError("archive_file_missing", "Backup is missing a listed member.")
            for row in rows:
                relative = _safe_relative(str(row.get("path") or ""))
                extracted = archive.extractfile(relative)
                if extracted is None or hashlib.sha256(extracted.read()).hexdigest() != str(row.get("sha256") or ""):
                    raise BackupRestoreError("archive_hash_mismatch", "Backup member integrity failed.")
            files = tuple(sorted(expected))
            return BackupValidation(True, None, manifest, files, tuple(path for path in files if _is_sensitive(path)), tuple(warnings))
    except (tarfile.TarError, OSError, EOFError):
        return BackupValidation(False, "corrupt_backup", {}, (), ())
    except BackupRestoreError as exc:
        return BackupValidation(False, exc.code, {}, (), ())


def dry_run_restore(archive_path: Path, *, expected_app_version: str | None = APP_VERSION) -> dict[str, Any]:
    validation = validate_backup(archive_path, expected_app_version=expected_app_version)
    if not validation.ok:
        return {"ok": False, "error": validation.error, "mutated": False}
    return {"ok": True, "mutated": False, "file_count": len(validation.files), "sensitive_file_count": 0, "sensitive_files": [], "included_classes": validation.manifest.get("included_classes", []), "secret_restoration": validation.manifest.get("secret_restoration"), "service_actions": "none"}


def restore_to_temp_state(archive_path: Path, target_home: Path, *, expected_app_version: str | None = APP_VERSION) -> dict[str, Any]:
    validation = validate_backup(archive_path, expected_app_version=expected_app_version)
    if not validation.ok:
        return {"ok": False, "error": validation.error, "mutated_live_state": False}
    target_home.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="personal-agent-restore-stage-", dir=str(target_home.parent)))
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            for relative in validation.files:
                destination = stage / _safe_relative(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(relative)
                if extracted is None:
                    return {"ok": False, "error": "archive_file_unreadable", "mutated_live_state": False}
                data = extracted.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    return {"ok": False, "error": "size_limit", "mutated_live_state": False}
                destination.write_bytes(data)
        conflicts = [relative for relative in validation.files if (target_home / relative).exists() and (not (target_home / relative).is_file() or _sha256(target_home / relative) != _sha256(stage / relative))]
        if conflicts:
            return {"ok": False, "error": "restore_target_conflict", "mutated_live_state": False, "conflict_count": len(conflicts)}
        for relative in validation.files:
            destination = target_home / relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / relative, destination)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    required = [".config/personal-agent/config.json", ".local/share/personal-agent/agent.db", ".local/share/personal-agent/search_runtime_config.json", ".config/systemd/user/personal-agent-api.service"]
    missing = [path for path in required if not (target_home / path).is_file()]
    return {"ok": not missing, "error": "restored_file_missing" if missing else None, "missing": missing, "restored_root": str(target_home), "restored_file_count": len(validation.files), "mutated_live_state": False, "service_actions": "none", "secrets_require_reentry": True}
