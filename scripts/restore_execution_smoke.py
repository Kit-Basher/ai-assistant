#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.orchestrator import Orchestrator
from agent.executor_registry import restore_backup_v1
from memory.db import MemoryDB


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str


def _pass(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=True, evidence=evidence.strip()[:1200], command=command)


def _fail(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=False, evidence=evidence.strip()[:1600], command=command)


def _write_json(path: Path, payload: dict[str, Any]) -> int:
    text = json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _create_backup(root: Path, *, value: str, name: str = "personal-agent-backup-restore-fixture") -> Path:
    backup = root / ".local/share/personal-agent/backups" / name
    backup.mkdir(parents=True)
    files = {
        "backup_summary.json": {"backup_schema_version": "backup.v1"},
        "diagnostics_summary.json": {"ok": True},
        "executor_registry_journal_summary.json": {"entries": []},
        "memory_anchors_summary.json": {"mode": "summary_only_raw_memory_text_excluded"},
        "pack_metadata_summary.json": {"raw_pack_text": "excluded"},
        "preferences_summary.json": {
            "mode": "allowlisted_restore_export",
            "preferences": [
                {"key": "system_resource_baseline_v1", "value": value, "restore_supported": True},
                {"key": "unsupported.secretish", "value": "should-not-restore", "restore_supported": False},
            ],
        },
        "runtime_config_summary.json": {"version": {"git_commit": "fixture"}},
        "state_database_summary.json": {"mode": "summary_only_raw_database_excluded"},
        "support_bundle_style_summary.json": {"redaction": "same redaction helper as Support Bundle v2"},
    }
    file_sizes = {name: _write_json(backup / name, payload) for name, payload in files.items()}
    manifest = {
        "backup_schema_version": "backup.v1",
        "created_at": "2026-07-02T00:00:00+00:00",
        "runtime_commit": "fixture",
        "runtime_instance": "isolated",
        "included_files": sorted([*files.keys(), "manifest.json"]),
        "excluded_files": ["raw secret-store files", "raw logs and full support bundles", "arbitrary home directory files"],
        "file_sizes": file_sizes,
        "total_size_bytes": sum(file_sizes.values()),
        "restore_status": "dry_run_only",
        "live_restore": "restore_not_enabled",
    }
    _write_json(backup / "manifest.json", manifest)
    return backup


def _read_pref(db_path: Path, key: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row is not None else None
    finally:
        conn.close()


def run() -> list[Check]:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="personal-agent-restore-smoke-") as raw:
        tmp = Path(raw)
        fake_home = tmp / "home"
        fake_home.mkdir()
        db = MemoryDB(str(fake_home / ".local/share/personal-agent/agent.db"))
        db.init_schema(str(ROOT / "memory/schema.sql"))
        try:
            db.set_preference("system_resource_baseline_v1", "old-baseline")
            backup = _create_backup(fake_home, value="restored-baseline")
            orchestrator = Orchestrator(
                db=db,
                skills_path=str(ROOT / "skills"),
                log_path=str(tmp / "events.log"),
                timezone="UTC",
                llm_client=None,
            )
            with patch("pathlib.Path.home", return_value=fake_home):
                preview = orchestrator.handle_message(f"restore from backup: {backup}", "restore-smoke", chat_context={"thread_id": "restore"})
                preview_payload = preview.data.get("runtime_payload", {})
                raw_plan = preview_payload.get("canonical_plan") if isinstance(preview_payload, dict) else {}
                plan = raw_plan if isinstance(raw_plan, dict) else {}
                checks.append(
                    _pass("restore preview is enabled and validated", preview.text, "Orchestrator.handle_message restore preview")
                    if plan.get("action_type") == "operator.restore"
                    and plan.get("executor_status") == "enabled"
                    and "safety snapshot" in preview.text.lower()
                    else _fail("restore preview is enabled and validated", json.dumps(preview.data, sort_keys=True)[:1600], "Orchestrator.handle_message restore preview")
                )
                confirmed = orchestrator.handle_message("yes", "restore-smoke", chat_context={"thread_id": "restore"})
                result = confirmed.data.get("runtime_payload", {}).get("executor_result", {})
                checks.append(
                    _pass("restore confirmation mutates isolated state", confirmed.text, 'POST-like chat "yes"')
                    if result.get("ok") and result.get("mutated") and result.get("executor_id") == "operator.restore.v1"
                    else _fail("restore confirmation mutates isolated state", json.dumps(confirmed.data, sort_keys=True)[:1600], 'POST-like chat "yes"')
                )
                checks.append(
                    _pass("allowlisted preference restored", db.get_preference("system_resource_baseline_v1") or "", "MemoryDB.get_preference")
                    if db.get_preference("system_resource_baseline_v1") == "restored-baseline"
                    else _fail("allowlisted preference restored", str(db.get_preference("system_resource_baseline_v1")), "MemoryDB.get_preference")
                )
                checks.append(
                    _pass("unsupported preference not restored", "unsupported preference absent", "MemoryDB.get_preference")
                    if db.get_preference("unsupported.secretish") is None
                    else _fail("unsupported preference not restored", str(db.get_preference("unsupported.secretish")), "MemoryDB.get_preference")
                )
                details = result.get("details") if isinstance(result, dict) and isinstance(result.get("details"), dict) else {}
                snapshot = Path(str(details.get("snapshot_path") or ""))
                checks.append(
                    _pass("safety snapshot created", str(snapshot), "inspect restore snapshot")
                    if snapshot.is_dir() and (snapshot / "manifest.json").is_file() and (snapshot / "preferences_snapshot.json").is_file()
                    else _fail("safety snapshot created", str(details), "inspect restore snapshot")
                )
                duplicate = orchestrator.handle_message("yes", "restore-smoke", chat_context={"thread_id": "restore"})
                checks.append(
                    _pass("duplicate confirmation does not reapply", duplicate.text, 'POST-like chat duplicate "yes"')
                    if "current action" in duplicate.text.lower()
                    else _fail("duplicate confirmation does not reapply", duplicate.text, 'POST-like chat duplicate "yes"')
                )

                rollback_db = MemoryDB(str(fake_home / ".local/share/personal-agent/rollback-agent.db"))
                rollback_db.init_schema(str(ROOT / "memory/schema.sql"))
                rollback_db.set_preference("system_resource_baseline_v1", "rollback-old")
                rollback_backup = _create_backup(fake_home, value="rollback-new", name="personal-agent-backup-restore-rollback-fixture")
                import agent.executor_registry as registry_module

                real_current = registry_module._current_preferences
                calls = 0

                def _current_with_bad_verification(path: Path, keys: list[str]) -> dict[str, str | None]:
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        return {"system_resource_baseline_v1": "wrong-after-apply"}
                    return real_current(path, keys)

                try:
                    with patch("agent.executor_registry._current_preferences", side_effect=_current_with_bad_verification):
                        rollback_result = restore_backup_v1(
                            {"plan_id": "restore-smoke-rollback", "action_type": "operator.restore"},
                            {
                                "pending_id": "restore-smoke-rollback",
                                "state_root": str(Path(rollback_db.db_path).parent),
                                "db_path": str(Path(rollback_db.db_path)),
                                "backup_root": str(fake_home / ".local/share/personal-agent/backups"),
                                "restore_backup_path": str(rollback_backup),
                            },
                        )
                    checks.append(
                        _pass("post-apply verification failure rolls back", json.dumps(rollback_result, sort_keys=True)[:1200], "restore_backup_v1 forced verification failure")
                        if rollback_result.get("error_code") == "restore_failed_rolled_back"
                        and _read_pref(Path(rollback_db.db_path), "system_resource_baseline_v1") == "rollback-old"
                        else _fail("post-apply verification failure rolls back", json.dumps(rollback_result, sort_keys=True)[:1600], "restore_backup_v1 forced verification failure")
                    )
                finally:
                    rollback_db.close()
        finally:
            db.close()
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore v1 isolated execution smoke.")
    parser.parse_args()
    checks = run()
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed
    print("# Personal Agent Restore Execution Smoke")
    for check in checks:
        print()
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
    print()
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"RESTORE_EXECUTION_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
