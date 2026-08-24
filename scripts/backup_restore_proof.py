#!/usr/bin/env python3
"""Portable backup/restore proof using the shipped runtime contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import tempfile

from agent.portable_backup import (
    APP_VERSION,
    BACKUP_CONTRACT,
    BACKUP_DIRECTORY_ROOTS,
    BACKUP_EXACT_FILES,
    BACKUP_SCHEMA_VERSION,
    BackupRestoreError,
    BackupValidation,
    create_backup,
    dry_run_restore,
    restore_to_temp_state,
    validate_backup,
)


def _write_fixture_state(home: Path) -> None:
    config = home / ".config/personal-agent"
    state_root = home / ".local/share/personal-agent"
    systemd = home / ".config/systemd/user"
    (state_root / "external_packs/normalized").mkdir(parents=True, exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)
    systemd.mkdir(parents=True, exist_ok=True)
    (config / "config.json").write_text('{"profile":"proof"}\n', encoding="utf-8")
    (state_root / "secrets.enc.json").write_text('{"ciphertext":"MACHINE_BOUND_SECRET_MATERIAL"}\n', encoding="utf-8")
    (state_root / "search_runtime_config.json").write_text('{"enabled":false,"provider":"searxng"}\n', encoding="utf-8")
    (state_root / "external_packs/normalized/registry_sources.json").write_text('{"sources":[]}\n', encoding="utf-8")
    (systemd / "personal-agent-api.service").write_text("[Service]\nExecStart=/bin/false\n", encoding="utf-8")
    with sqlite3.connect(state_root / "agent.db") as connection:
        connection.execute("CREATE TABLE proof_memory (id TEXT PRIMARY KEY, body TEXT)")
        connection.execute("INSERT INTO proof_memory VALUES (?, ?)", ("proof", "restored proof memory"))


def run_proof() -> tuple[bool, list[tuple[str, bool, str]]]:
    rows: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="personal-agent-backup-proof-") as tmp:
        root = Path(tmp)
        source, target = root / "source-home", root / "restore-home"
        _write_fixture_state(source)
        archive = create_backup(source, root / "personal-agent-backup.tar.gz")
        validation = validate_backup(archive)
        rows.append(("allowlisted backup validates", validation.ok, validation.error or f"files={len(validation.files)}"))
        rows.append(("SQLite online snapshot recorded", validation.manifest.get("sqlite_capture") == "online_backup_with_integrity_check", str(validation.manifest.get("sqlite_capture"))))
        rows.append(("machine-bound secrets excluded", not any("secrets.enc" in path for path in validation.files), "secrets require UI re-entry"))
        dry = dry_run_restore(archive)
        rows.append(("dry run is non-mutating", bool(dry.get("ok")) and not dry.get("mutated"), json.dumps(dry, sort_keys=True)))
        restored = restore_to_temp_state(archive, target)
        rows.append(("isolated restore succeeds", bool(restored.get("ok")), json.dumps(restored, sort_keys=True)))
        with sqlite3.connect(target / ".local/share/personal-agent/agent.db") as connection:
            restored_memory = connection.execute("SELECT body FROM proof_memory WHERE id='proof'").fetchone()
            integrity = str((connection.execute("PRAGMA integrity_check").fetchone() or ["unknown"])[0])
        rows.append(("restored SQLite state and integrity", restored_memory == ("restored proof memory",) and integrity == "ok", f"integrity={integrity}"))
        repeated = restore_to_temp_state(archive, target)
        rows.append(("identical restore is idempotent", bool(repeated.get("ok")), json.dumps(repeated, sort_keys=True)))
        corrupt = root / "corrupt.tar.gz"
        corrupt.write_bytes(b"not a tar")
        rows.append(("truncated backup fails safely", validate_backup(corrupt).error == "corrupt_backup", "no restore attempted"))
        mismatch = create_backup(source, root / "future.tar.gz", app_version="99.0.0")
        rows.append(("incompatible version fails safely", validate_backup(mismatch).error == "version_mismatch", "no restore attempted"))
    return all(ok for _, ok, _ in rows), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Personal Agent portable backup/restore proof.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ok, rows = run_proof()
    report = {"schema_version": "personal-agent.backup-restore-proof.v2", "ok": ok, "checks": [{"name": name, "ok": passed, "detail": detail} for name, passed, detail in rows]}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for row in report["checks"]:
            print(f"{'PASS' if row['ok'] else 'FAIL'} {row['name']}: {row['detail']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
