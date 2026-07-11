#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.capability_policy import capability_for_action_type  # noqa: E402
from agent.executor_registry import (  # noqa: E402
    BACKUP_SCHEMA_VERSION,
    ExecutorRegistry,
    ExecutorSpec,
    RESTORE_V1_CAPABILITY,
    create_additive_backup,
    create_redacted_support_bundle,
    restore_backup_v1,
)
from agent.mutation_plan import MUTATION_PLAN_SCHEMA_VERSION, build_mutation_plan, validate_mutation_plan  # noqa: E402


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


def _pass(name: str, detail: str = "") -> Check:
    return Check("PASS", name, detail)


def _warn(name: str, detail: str = "") -> Check:
    return Check("WARN", name, detail)


def _fail(name: str, detail: str = "") -> Check:
    return Check("FAIL", name, detail)


def _plan(action_type: str, *, plan_id: str, target: str) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "action_type": action_type,
        "target": target,
        "risk_level": "medium",
        "executor_status": "enabled",
        "high_risk_confirmed": True,
    }


def _registry(tmp: Path) -> ExecutorRegistry:
    registry = ExecutorRegistry(tmp / "executor-journal.jsonl")
    registry.register(
        ExecutorSpec(
            executor_id="operator.backup.v1",
            action_type="operator.backup",
            status="enabled",
            run=create_additive_backup,
            rollback_available=True,
            rollback_hint="Remove only the created backup artifact.",
            capability_id="backup.create",
        )
    )
    registry.register(
        ExecutorSpec(
            executor_id="operator.restore.v1",
            action_type="operator.restore",
            status="enabled",
            run=restore_backup_v1,
            rollback_available=True,
            rollback_hint="Use the pre-restore safety snapshot.",
            capability_id="restore.execute",
        )
    )
    registry.register(
        ExecutorSpec(
            executor_id="operator.support_bundle.v1",
            action_type="operator.support_bundle",
            status="enabled",
            run=create_redacted_support_bundle,
            rollback_available=True,
            rollback_hint="Remove only the created support bundle artifact.",
            capability_id="support_bundle.create",
        )
    )
    return registry


def _init_pref_db(db_path: Path, value: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            conn.execute(
                "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
                ("system_resource_baseline_v1", value, "2026-07-10T00:00:00+00:00"),
            )
    finally:
        conn.close()


def _read_pref(db_path: Path) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT value FROM preferences WHERE key = ?", ("system_resource_baseline_v1",)).fetchone()
        return str(row[0]) if row else None
    finally:
        conn.close()


def _build_backup(registry: ExecutorRegistry, tmp: Path) -> tuple[Check, Path | None, dict[str, Any]]:
    backup_root = tmp / "backups"
    plan = _plan("operator.backup", plan_id="migration-backup", target="safety backup")
    action = {
        "pending_id": "migration-backup",
        "backup_root": str(backup_root),
        "diagnostics": {"version": {"git_commit": "fixture", "runtime_instance": "isolated"}},
        "backup_sources": {
            "preferences": {
                "preferences": [{"key": "system_resource_baseline_v1", "value": "restored-value"}],
                "restore_supported_keys": ["system_resource_baseline_v1", "system_resource_baseline_context_v1"],
            },
            "memory": {"summary_count": 0},
            "state_database": {"path": str(tmp / "state" / "memory.db"), "size_bytes": 0},
        },
        "executor_journal_recent": [],
    }
    result = registry.execute_confirmed_plan(plan=plan, action=action).to_dict()
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    artifact = Path(str(details.get("artifact_path") or "")) if details.get("artifact_path") else None
    if result.get("ok") and result.get("mutated") and result.get("capability_id") == "backup.create" and result.get("mutation_plan_schema_version") == MUTATION_PLAN_SCHEMA_VERSION and artifact and artifact.is_dir():
        return _pass("backup create uses Universal Plan", json.dumps(result, sort_keys=True)[:1000]), artifact, result
    return _fail("backup create uses Universal Plan", json.dumps(result, sort_keys=True)[:1400]), artifact, result


def _restore_backup(registry: ExecutorRegistry, tmp: Path, backup_path: Path) -> Check:
    state_root = tmp / "restore-state"
    db_path = state_root / "memory.db"
    _init_pref_db(db_path, "old-value")
    plan = _plan("operator.restore", plan_id="migration-restore", target=str(backup_path))
    plan["risk_level"] = "high"
    action = {
        "pending_id": "migration-restore",
        "state_root": str(state_root),
        "db_path": str(db_path),
        "backup_root": str(backup_path.parent),
        "restore_backup_path": str(backup_path),
    }
    result = registry.execute_confirmed_plan(plan=plan, action=action).to_dict()
    if result.get("ok") and result.get("mutated") and result.get("capability_id") == "restore.execute" and _read_pref(db_path) == "restored-value":
        return _pass("restore fixture executes and verifies", json.dumps(result, sort_keys=True)[:1000])
    return _fail("restore fixture executes and verifies", json.dumps(result, sort_keys=True)[:1400])


def _support_bundle(registry: ExecutorRegistry) -> Check:
    plan = _plan("operator.support_bundle", plan_id="migration-support", target="support bundle")
    action = {
        "pending_id": "migration-support",
        "diagnostics": {
            "version": {"git_commit": "fixture", "runtime_instance": "isolated"},
            "telegram_status": {"token": "123456:telegram-secret-token"},
            "ready": {"message": "Bearer abc.def.ghi"},
        },
        "executor_journal_recent": [{"api_key": "sk-secret"}],
    }
    result = registry.execute_confirmed_plan(plan=plan, action=action).to_dict()
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    artifact = Path(str(details.get("artifact_path") or "")) if details.get("artifact_path") else None
    combined = ""
    if artifact and artifact.is_dir():
        combined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in artifact.glob("*.json"))
        shutil.rmtree(artifact, ignore_errors=True)
    redacted = all(secret not in combined for secret in ("123456:telegram-secret-token", "abc.def.ghi", "sk-secret"))
    if result.get("ok") and result.get("mutated") and result.get("capability_id") == "support_bundle.create" and redacted:
        return _pass("support-bundle creation is authorized and redacted", json.dumps(result, sort_keys=True)[:1000])
    return _fail("support-bundle creation is authorized and redacted", json.dumps(result, sort_keys=True)[:1400])


def _memory_plan() -> Check:
    plan = build_mutation_plan(
        plan_id="migration-memory-forget",
        capability_id="memory.forget",
        executor_id="operator.memory.forget.v1",
        expires_at_epoch=4_102_444_800,
        thread_id="migration-thread",
        session_id="migration-session",
        target_snapshot={"scope": "fixture", "record_count": 1, "fingerprint": "memory-fixture"},
        mutation_inventory=[{"kind": "memory_record", "count": 1, "operation": "forget"}],
        preserved_resources=[{"kind": "memory_category", "name": "preferences"}],
        recovery={"rollback_supported": False, "truth": "forget is irreversible without an export checkpoint"},
    )
    try:
        validate_mutation_plan(plan)
    except Exception as exc:  # noqa: BLE001
        return _fail("memory forget requires Universal Plan", f"{exc.__class__.__name__}: {exc}")
    if capability_for_action_type("memory.delete_all") == "memory.forget" and plan.get("schema_version") == MUTATION_PLAN_SCHEMA_VERSION:
        return _pass("memory forget requires Universal Plan", f"fingerprint={plan.get('plan_fingerprint')}")
    return _fail("memory forget requires Universal Plan", json.dumps(plan, sort_keys=True)[:1200])


def run() -> list[Check]:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="personal-agent-executor-auth-") as raw:
        tmp = Path(raw)
        registry = _registry(tmp)
        backup_check, backup_path, backup_result = _build_backup(registry, tmp)
        checks.append(backup_check)
        if backup_path is not None:
            manifest = json.loads((backup_path / "manifest.json").read_text(encoding="utf-8"))
            checks.append(
                _pass("backup receipt validates", f"schema={manifest.get('backup_schema_version')} restore={manifest.get('restore_status')}")
                if manifest.get("backup_schema_version") == BACKUP_SCHEMA_VERSION and manifest.get("restore_status") == RESTORE_V1_CAPABILITY
                else _fail("backup receipt validates", json.dumps(manifest, sort_keys=True)[:1200])
            )
            checks.append(_restore_backup(registry, tmp, backup_path))
        else:
            checks.append(_fail("backup receipt validates", json.dumps(backup_result, sort_keys=True)[:1200]))
            checks.append(_fail("restore fixture executes and verifies", "backup artifact missing"))
        checks.append(_support_bundle(registry))
        checks.append(_memory_plan())

        direct_backup = create_additive_backup(_plan("operator.backup", plan_id="direct-backup", target="backup"), {"pending_id": "direct-backup", "backup_root": str(tmp / "direct-backups")})
        direct_support = create_redacted_support_bundle(_plan("operator.support_bundle", plan_id="direct-support", target="support"), {"pending_id": "direct-support"})
        checks.append(
            _pass("direct lower-level bypass blocked", f"backup={direct_backup.get('error_code')} support={direct_support.get('error_code')}")
            if direct_backup.get("error_code") == "generic_bypass_blocked" and direct_support.get("error_code") == "generic_bypass_blocked"
            else _fail("direct lower-level bypass blocked", json.dumps({"backup": direct_backup, "support": direct_support}, sort_keys=True)[:1400])
        )

    for action_type, capability_id in (
        ("operator.backup", "backup.create"),
        ("operator.restore", "restore.execute"),
        ("operator.support_bundle", "support_bundle.create"),
        ("memory.delete_all", "memory.forget"),
    ):
        actual = capability_for_action_type(action_type)
        checks.append(_pass(f"{action_type} capability binding", actual) if actual == capability_id else _fail(f"{action_type} capability binding", f"actual={actual!r} expected={capability_id!r}"))
    for legacy in ("communications", "broader skill-pack mutations"):
        checks.append(_warn(f"remaining legacy warning: {legacy}", "future migration batch"))
    return checks


def main() -> int:
    checks = run()
    for check in checks:
        print(f"{check.status}: {check.name}" + (f": {check.detail}" if check.detail else ""))
    passed = sum(1 for check in checks if check.status == "PASS")
    warned = sum(1 for check in checks if check.status == "WARN")
    failed = sum(1 for check in checks if check.status == "FAIL")
    print(f"PASS={passed} WARN={warned} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
