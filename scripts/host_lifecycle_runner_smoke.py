#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.host_lifecycle import (  # noqa: E402
    HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
    HOST_LIFECYCLE_RUNNER_VERSION,
    attach_approved_hash,
    write_json_atomic,
)


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str


def _pass(name: str, evidence: str, command: str) -> Check:
    return Check(name, True, evidence.strip()[:1200], command)


def _fail(name: str, evidence: str, command: str) -> Check:
    return Check(name, False, evidence.strip()[:1600], command)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_release(path: Path, commit: str) -> None:
    _write(path / "agent/BUILD_INFO.json", json.dumps({"git_commit": commit, "version": "fixture"}, sort_keys=True) + "\n")
    _write(path / "README.txt", f"fixture release {commit}\n")


def _current_commit(current: Path) -> str:
    payload = json.loads((current.resolve() / "agent/BUILD_INFO.json").read_text(encoding="utf-8"))
    return str(payload.get("git_commit") or "")


def _git_status_short() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=False)
    return result.stdout.strip()


def _run_runner(operation: str, record: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "scripts/host_lifecycle_runner.py", operation, "--operation-record", str(record)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"ok": False, "raw": proc.stdout[:1000]}
    payload["returncode"] = proc.returncode
    return payload


def _update_fixture(root: Path, *, operation_id: str, force_failure: bool = False) -> tuple[Path, Path]:
    runtime = root / "runtime"
    releases = runtime / "releases"
    releases.mkdir(parents=True)
    _write_release(releases / "release-a", "commit-a")
    _write_release(root / "source-release-b", "commit-b")
    current = runtime / "current"
    current.symlink_to(releases / "release-a")
    state = root / "state"
    record = attach_approved_hash(
        {
            "schema_version": HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
            "runner_version": HOST_LIFECYCLE_RUNNER_VERSION,
            "operation_id": operation_id,
            "operation_type": "update",
            "plan_id": operation_id,
            "created_at": "2026-07-08T00:00:00+00:00",
            "fixture_mode": "strict",
            "state_root": str(state),
            "runtime_root": str(runtime),
            "releases_root": str(releases),
            "current_link": str(current),
            "staged_source_path": str(root / "source-release-b"),
            "target_release_id": f"release-b-{operation_id}",
            "current_runtime_commit": "commit-a",
            "target_commit": "commit-b",
            "operation_state_path": str(state / "host_lifecycle" / "operations" / operation_id / "state.json"),
            "receipt_path": str(state / "host_lifecycle" / "operations" / operation_id / "receipt.json"),
            "force_post_promotion_failure": force_failure,
        }
    )
    record_path = state / "host_lifecycle" / "operations" / operation_id / "operation.json"
    write_json_atomic(record_path, record)
    return record_path, current


def _uninstall_fixture(root: Path, *, operation_id: str, force_failure: bool = False) -> tuple[Path, Path]:
    fixture = root / "fixture-install"
    install = fixture / "personal-agent"
    runtime = install / "runtime"
    release = runtime / "releases/0.2.0"
    service_root = fixture / "config/systemd/user"
    launcher_root = fixture / "share/applications"
    icon_root = fixture / "share/icons"
    for path in (release, service_root, launcher_root, icon_root, install / "backups", fixture / "repo"):
        path.mkdir(parents=True, exist_ok=True)
    (runtime / "current").symlink_to(release)
    _write(release / "agent/BUILD_INFO.json", json.dumps({"git_commit": "fixture-a"}) + "\n")
    _write(runtime / "install-manifest.json", "{}\n")
    _write(service_root / "personal-agent-api.service", "[Service]\n")
    _write(service_root / "unrelated.service", "[Service]\n")
    _write(launcher_root / "personal-agent.desktop", "[Desktop]\n")
    _write(icon_root / "personal-agent.svg", "<svg />\n")
    _write(install / "agent.db", "state\n")
    _write(install / "secrets.enc.json", '{"token":"secret"}\n')
    (install / "models").mkdir()
    (install / "external-packs").mkdir()
    backup = install / "backups" / f"personal-agent-uninstall-backup-{operation_id}"
    _write(backup / "manifest.json", json.dumps({"backup_schema_version": "backup.v1"}) + "\n")
    removable = [
        {"id": "release", "class": "runtime release", "path": str(release), "owned": True, "expected_type": "directory"},
        {"id": "current", "class": "runtime symlink", "path": str(runtime / "current"), "owned": True, "expected_type": "symlink"},
        {"id": "manifest", "class": "install metadata", "path": str(runtime / "install-manifest.json"), "owned": True, "expected_type": "file"},
        {"id": "api-service", "class": "api service unit", "path": str(service_root / "personal-agent-api.service"), "owned": True, "expected_type": "file"},
        {"id": "desktop", "class": "desktop entry", "path": str(launcher_root / "personal-agent.desktop"), "owned": True, "expected_type": "file"},
        {"id": "icon", "class": "desktop icon", "path": str(icon_root / "personal-agent.svg"), "owned": True, "expected_type": "file"},
    ]
    preserved = [
        {"id": "state", "path": str(install / "agent.db")},
        {"id": "secrets", "path": str(install / "secrets.enc.json")},
        {"id": "backups", "path": str(install / "backups")},
        {"id": "repo", "path": str(fixture / "repo")},
        {"id": "models", "path": str(install / "models")},
        {"id": "packs", "path": str(install / "external-packs")},
    ]
    snapshot = {
        "fixture_marker": True,
        "fixture_root": str(fixture),
        "mode": "preserve_data",
        "removable_roots": [str(runtime), str(service_root), str(launcher_root), str(icon_root)],
        "removable_resources": removable,
        "preserved_resources": preserved,
    }
    record = attach_approved_hash(
        {
            "schema_version": HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
            "runner_version": HOST_LIFECYCLE_RUNNER_VERSION,
            "operation_id": operation_id,
            "operation_type": "uninstall",
            "plan_id": operation_id,
            "created_at": "2026-07-08T00:00:00+00:00",
            "fixture_mode": "strict",
            "state_root": str(install),
            "operation_state_path": str(install / "host_lifecycle" / "operations" / operation_id / "state.json"),
            "receipt_path": str(install / "uninstall_receipts" / f"personal-agent-uninstall-{operation_id}.json"),
            "final_backup_path": str(backup),
            "target_snapshot": snapshot,
            "force_failure_after_resource_id": "api-service" if force_failure else None,
        }
    )
    record_path = install / "host_lifecycle" / "operations" / operation_id / "operation.json"
    write_json_atomic(record_path, record)
    return record_path, fixture


def main() -> int:
    checks: list[Check] = []
    before = _git_status_short()
    with tempfile.TemporaryDirectory(prefix="pa-host-lifecycle-") as raw:
        root = Path(raw)
        update_record, current = _update_fixture(root / "update-ok", operation_id="update-ok")
        update_result = _run_runner("update", update_record)
        checks.append(_pass("runner update promotes target", json.dumps(update_result, sort_keys=True), "host_lifecycle_runner.py update") if update_result.get("ok") and _current_commit(current) == "commit-b" else _fail("runner update promotes target", json.dumps(update_result, sort_keys=True), "host_lifecycle_runner.py update"))
        status_result = _run_runner("status", update_record)
        checks.append(_pass("runner status is read-only", json.dumps(status_result, sort_keys=True), "host_lifecycle_runner.py status") if status_result.get("ok") and status_result.get("mutated") is False and status_result.get("has_receipt") else _fail("runner status is read-only", json.dumps(status_result, sort_keys=True), "host_lifecycle_runner.py status"))

        rollback_record, rollback_current = _update_fixture(root / "update-rollback", operation_id="update-rollback", force_failure=True)
        rollback_result = _run_runner("update", rollback_record)
        checks.append(_pass("runner update rolls back on verification failure", json.dumps(rollback_result, sort_keys=True), "host_lifecycle_runner.py update forced failure") if not rollback_result.get("ok") and rollback_result.get("rollback_verified") and _current_commit(rollback_current) == "commit-a" else _fail("runner update rolls back on verification failure", json.dumps(rollback_result, sort_keys=True), "host_lifecycle_runner.py update forced failure"))

        uninstall_record, fixture = _uninstall_fixture(root / "uninstall-ok", operation_id="uninstall-ok")
        uninstall_result = _run_runner("uninstall", uninstall_record)
        removed = not (fixture / "personal-agent/runtime/releases/0.2.0").exists() and not (fixture / "config/systemd/user/personal-agent-api.service").exists()
        preserved = (fixture / "personal-agent/agent.db").exists() and (fixture / "personal-agent/secrets.enc.json").exists() and (fixture / "repo").exists()
        checks.append(_pass("runner uninstall removes fixture and preserves data", json.dumps(uninstall_result, sort_keys=True), "host_lifecycle_runner.py uninstall") if uninstall_result.get("ok") and removed and preserved else _fail("runner uninstall removes fixture and preserves data", json.dumps(uninstall_result, sort_keys=True), "host_lifecycle_runner.py uninstall"))
        repeat_result = _run_runner("uninstall", uninstall_record)
        checks.append(_pass("runner duplicate uninstall is idempotent", json.dumps(repeat_result, sort_keys=True), "host_lifecycle_runner.py uninstall repeat") if repeat_result.get("ok") else _fail("runner duplicate uninstall is idempotent", json.dumps(repeat_result, sort_keys=True), "host_lifecycle_runner.py uninstall repeat"))

        partial_record, _ = _uninstall_fixture(root / "uninstall-partial", operation_id="uninstall-partial", force_failure=True)
        partial_result = _run_runner("uninstall", partial_record)
        checks.append(_pass("runner uninstall partial failure is truthful", json.dumps(partial_result, sort_keys=True), "host_lifecycle_runner.py uninstall forced partial") if not partial_result.get("ok") and partial_result.get("status") == "partial_uninstall" else _fail("runner uninstall partial failure is truthful", json.dumps(partial_result, sort_keys=True), "host_lifecycle_runner.py uninstall forced partial"))

        tampered_record, _ = _update_fixture(root / "tampered", operation_id="tampered")
        payload = json.loads(tampered_record.read_text(encoding="utf-8"))
        payload["target_commit"] = "evil"
        tampered_record.write_text(json.dumps(payload), encoding="utf-8")
        tampered = _run_runner("update", tampered_record)
        checks.append(_pass("tampered operation rejected", json.dumps(tampered, sort_keys=True), "host_lifecycle_runner.py update tampered") if not tampered.get("ok") and "tampered" in str(tampered.get("error_summary", "")).lower() else _fail("tampered operation rejected", json.dumps(tampered, sort_keys=True), "host_lifecycle_runner.py update tampered"))

        command_record, _ = _update_fixture(root / "command-field", operation_id="command-field")
        payload = json.loads(command_record.read_text(encoding="utf-8"))
        payload["command"] = "rm -rf /"
        payload = attach_approved_hash(payload)
        command_record.write_text(json.dumps(payload), encoding="utf-8")
        command_field = _run_runner("update", command_record)
        checks.append(_pass("arbitrary command field rejected", json.dumps(command_field, sort_keys=True), "host_lifecycle_runner.py update command field") if not command_field.get("ok") and "command" in str(command_field.get("error_summary", "")).lower() else _fail("arbitrary command field rejected", json.dumps(command_field, sort_keys=True), "host_lifecycle_runner.py update command field"))

    after = _git_status_short()
    checks.append(_pass("git status unchanged", after or "(clean)", "git status --short") if after == before else _fail("git status unchanged", f"before={before!r}\nafter={after!r}", "git status --short"))
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed
    print("# Personal Agent Host Lifecycle Runner Smoke")
    for check in checks:
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"command: {check.command}")
        print(f"evidence: {check.evidence}\n")
    print(f"SUMMARY PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
