#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sqlite3
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_NAME = "personal-agent-backup-manifest.json"
BACKUP_SCHEMA_VERSION = 1
APP_VERSION = (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip()
BACKUP_ROOTS = (
    ".config/personal-agent",
    ".local/share/personal-agent",
    ".config/systemd/user",
)
SENSITIVE_NAME_FRAGMENTS = (
    "secret",
    "token",
    "key",
    "password",
    "secrets.enc.json",
)


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
    return any(fragment in lowered for fragment in SENSITIVE_NAME_FRAGMENTS)


def _safe_relative(path: str) -> str:
    normalized = Path(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise BackupRestoreError("unsafe_path", f"Backup path is not safe: {path}")
    clean = normalized.as_posix()
    if not clean or clean == ".":
        raise BackupRestoreError("unsafe_path", f"Backup path is not safe: {path}")
    return clean


def create_backup(source_home: Path, archive_path: Path, *, app_version: str = APP_VERSION) -> Path:
    source_home = source_home.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for root_label in BACKUP_ROOTS:
        root = source_home / root_label
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(source_home).as_posix()
            files.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                    "sensitive": _is_sensitive(relative),
                }
            )
    manifest = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "app": "personal-agent",
        "app_version": app_version,
        "created_at_epoch": int(time.time()),
        "backup_roots": list(BACKUP_ROOTS),
        "files": files,
        "service_actions": "none",
        "restore_policy": "temp_or_explicit_target_only",
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        manifest_info = tarfile.TarInfo(MANIFEST_NAME)
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = int(time.time())
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for row in files:
            rel = _safe_relative(str(row["path"]))
            archive.add(source_home / rel, arcname=rel, recursive=False)
    return archive_path


def _read_manifest(archive: tarfile.TarFile) -> dict[str, Any]:
    try:
        member = archive.getmember(MANIFEST_NAME)
    except KeyError as exc:
        raise BackupRestoreError("manifest_missing", "Backup manifest is missing.") from exc
    extracted = archive.extractfile(member)
    if extracted is None:
        raise BackupRestoreError("manifest_unreadable", "Backup manifest cannot be read.")
    try:
        parsed = json.loads(extracted.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupRestoreError("manifest_invalid", "Backup manifest is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise BackupRestoreError("manifest_invalid", "Backup manifest must be a JSON object.")
    return parsed


def validate_backup(
    archive_path: Path,
    *,
    expected_app_version: str | None = APP_VERSION,
    strict_version: bool = True,
) -> BackupValidation:
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            manifest = _read_manifest(archive)
            if int(manifest.get("schema_version") or 0) != BACKUP_SCHEMA_VERSION:
                raise BackupRestoreError("schema_version_unsupported", "Backup schema version is not supported.")
            if str(manifest.get("app") or "") != "personal-agent":
                raise BackupRestoreError("wrong_app", "Backup is not a Personal Agent backup.")
            warnings: list[str] = []
            version = str(manifest.get("app_version") or "").strip()
            if expected_app_version and version != expected_app_version:
                message = f"Backup version {version or 'unknown'} does not match expected {expected_app_version}."
                if strict_version:
                    raise BackupRestoreError("version_mismatch", message)
                warnings.append("version_mismatch")
            manifest_files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
            file_rows = [row for row in manifest_files if isinstance(row, dict)]
            expected_paths = {_safe_relative(str(row.get("path") or "")) for row in file_rows}
            archive_paths: set[str] = set()
            for member in archive.getmembers():
                name = _safe_relative(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise BackupRestoreError("unsupported_member_type", f"Backup contains unsupported member: {name}")
                archive_paths.add(name)
            missing = expected_paths - archive_paths
            if missing:
                raise BackupRestoreError("archive_file_missing", f"Backup is missing listed file: {sorted(missing)[0]}")
            for row in file_rows:
                rel = _safe_relative(str(row.get("path") or ""))
                extracted = archive.extractfile(rel)
                if extracted is None:
                    raise BackupRestoreError("archive_file_unreadable", f"Backup file cannot be read: {rel}")
                digest = hashlib.sha256(extracted.read()).hexdigest()
                if digest != str(row.get("sha256") or ""):
                    raise BackupRestoreError("archive_hash_mismatch", f"Backup file hash mismatch: {rel}")
            files = tuple(sorted(expected_paths))
            sensitive = tuple(sorted(path for path in files if _is_sensitive(path)))
            return BackupValidation(True, None, manifest, files, sensitive, tuple(warnings))
    except (tarfile.TarError, OSError, EOFError) as exc:
        return BackupValidation(False, "corrupt_backup", {}, (), (), ())
    except BackupRestoreError as exc:
        return BackupValidation(False, exc.code, {}, (), (), ())


def dry_run_restore(archive_path: Path, *, expected_app_version: str | None = APP_VERSION) -> dict[str, Any]:
    validation = validate_backup(archive_path, expected_app_version=expected_app_version, strict_version=True)
    if not validation.ok:
        return {"ok": False, "error": validation.error, "mutated": False}
    return {
        "ok": True,
        "mutated": False,
        "file_count": len(validation.files),
        "sensitive_file_count": len(validation.sensitive_files),
        "sensitive_files": ["<redacted>" for _ in validation.sensitive_files],
        "backup_roots": validation.manifest.get("backup_roots", []),
        "service_actions": "none",
    }


def restore_to_temp_state(
    archive_path: Path,
    target_home: Path,
    *,
    expected_app_version: str | None = APP_VERSION,
) -> dict[str, Any]:
    validation = validate_backup(archive_path, expected_app_version=expected_app_version, strict_version=True)
    if not validation.ok:
        return {"ok": False, "error": validation.error, "mutated_live_state": False}
    target_home.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        for relative in validation.files:
            _safe_relative(relative)
            destination = target_home / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(relative)
            if extracted is None:
                return {"ok": False, "error": "archive_file_unreadable", "mutated_live_state": False}
            destination.write_bytes(extracted.read())
    expected = [
        ".config/personal-agent/config.json",
        ".local/share/personal-agent/secrets.enc.json",
        ".local/share/personal-agent/agent.db",
        ".local/share/personal-agent/runtime_state/search_runtime_config.json",
        ".config/systemd/user/personal-agent-api.service",
    ]
    missing = [path for path in expected if not (target_home / path).is_file()]
    return {
        "ok": not missing,
        "error": "restored_file_missing" if missing else None,
        "missing": missing,
        "restored_root": str(target_home),
        "restored_file_count": len(validation.files),
        "mutated_live_state": False,
        "service_actions": "none",
    }


def _write_fixture_state(home: Path) -> None:
    config_dir = home / ".config" / "personal-agent"
    state_dir = home / ".local" / "share" / "personal-agent"
    systemd_dir = home / ".config" / "systemd" / "user"
    (state_dir / "runtime_state").mkdir(parents=True, exist_ok=True)
    (state_dir / "packs").mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    systemd_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text('{"profile":"proof"}\n', encoding="utf-8")
    (state_dir / "secrets.enc.json").write_text('{"telegram":"SUPER_SECRET_TOKEN_VALUE"}\n', encoding="utf-8")
    (state_dir / "runtime_state" / "search_runtime_config.json").write_text(
        '{"enabled":false,"provider":"searxng"}\n',
        encoding="utf-8",
    )
    (state_dir / "packs" / "registry_sources.json").write_text('{"sources":[]}\n', encoding="utf-8")
    (systemd_dir / "personal-agent-api.service").write_text("[Service]\nExecStart=/bin/false\n", encoding="utf-8")
    db_path = state_dir / "agent.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO memories (id, body) VALUES (?, ?)", ("proof", "redacted proof memory"))
        conn.commit()


def run_proof() -> tuple[bool, list[tuple[str, bool, str]]]:
    rows: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="personal-agent-backup-proof-") as tmp:
        root = Path(tmp)
        source_home = root / "source-home"
        restore_home = root / "restore-home"
        _write_fixture_state(source_home)
        archive_path = root / "personal-agent-backup.tar.gz"
        create_backup(source_home, archive_path)

        validation = validate_backup(archive_path)
        rows.append(("valid backup validates", validation.ok, validation.error or f"files={len(validation.files)}"))

        dry_run = dry_run_restore(archive_path)
        dry_text = json.dumps(dry_run, sort_keys=True)
        rows.append(("dry-run restore succeeds without mutation", bool(dry_run.get("ok")) and not bool(dry_run.get("mutated")), dry_text))
        rows.append(("dry-run redacts secret content", "SUPER_SECRET_TOKEN_VALUE" not in dry_text, "secret content redacted"))

        restore = restore_to_temp_state(archive_path, restore_home)
        rows.append(("restore into temp state succeeds", bool(restore.get("ok")) and not bool(restore.get("mutated_live_state")), json.dumps(restore, sort_keys=True)))
        restored_secret = (restore_home / ".local" / "share" / "personal-agent" / "secrets.enc.json").read_text(encoding="utf-8")
        rows.append(("secret file restored locally", "SUPER_SECRET_TOKEN_VALUE" in restored_secret, "secret preserved in temp restore target"))

        corrupt = root / "corrupt.tar.gz"
        corrupt.write_bytes(b"not a valid tar archive")
        corrupt_validation = validate_backup(corrupt)
        rows.append(("corrupt backup fails safely", not corrupt_validation.ok and corrupt_validation.error == "corrupt_backup", corrupt_validation.error or "unexpected ok"))

        mismatch = root / "version-mismatch.tar.gz"
        create_backup(source_home, mismatch, app_version="0.0.0-mismatch")
        mismatch_validation = validate_backup(mismatch, expected_app_version=APP_VERSION, strict_version=True)
        rows.append(("version mismatch refused", not mismatch_validation.ok and mismatch_validation.error == "version_mismatch", mismatch_validation.error or "unexpected ok"))

    return all(ok for _, ok, _ in rows), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded Personal Agent backup/restore proof.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    args = parser.parse_args()
    ok, rows = run_proof()
    if args.json:
        print(json.dumps({"ok": ok, "checks": [{"name": name, "ok": check_ok, "detail": detail} for name, check_ok, detail in rows]}, indent=2, sort_keys=True))
    else:
        print("# Personal Agent Backup/Restore Proof")
        for name, check_ok, detail in rows:
            print(f"{'PASS' if check_ok else 'FAIL'} {name}: {detail}")
        print(f"\n{'PASS' if ok else 'FAIL'} backup_restore_proof")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
