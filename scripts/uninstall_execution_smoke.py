#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.executor_registry import ExecutorRegistry, ExecutorSpec, execute_uninstall_v1


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


def _git_status_short() -> str:
    result = subprocess.run(["git", "status", "--short"], check=False, text=True, capture_output=True, timeout=10)
    return result.stdout.strip()


def _snapshot_hash(snapshot: dict) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(root: Path, *, operation_id: str = "uninstall-smoke") -> tuple[dict, dict]:
    fixture = root / "fixture-install"
    install_root = fixture / "personal-agent"
    runtime = install_root / "runtime"
    releases = runtime / "releases"
    release = releases / "0.2.0"
    service_root = fixture / "config/systemd/user"
    launcher_root = fixture / "share/applications"
    icon_root = fixture / "share/icons"
    repo = fixture / "repo"
    backups = install_root / "backups"
    for path in (release, service_root, launcher_root, icon_root, repo, backups):
        path.mkdir(parents=True, exist_ok=True)
    (runtime / "current").symlink_to(release)
    _write(release / "agent/BUILD_INFO.json", json.dumps({"git_commit": "fixture-a"}) + "\n")
    _write(runtime / "install-manifest.json", "{}\n")
    _write(service_root / "personal-agent-api.service", "[Service]\n")
    _write(service_root / "personal-agent-telegram.service", "[Service]\n")
    _write(launcher_root / "personal-agent.desktop", "[Desktop Entry]\n")
    _write(icon_root / "personal-agent.svg", "<svg />\n")
    _write(install_root / "agent.db", "memory/preferences\n")
    _write(install_root / "secrets.enc.json", '{"telegram_token":"secret"}\n')
    _write(repo / "README.md", "preserved repo\n")
    (install_root / "models").mkdir()
    (install_root / "external-packs").mkdir()
    (backups / "existing-backup").mkdir()
    _write(fixture / "config/systemd/user/unrelated.service", "[Service]\n")
    _write(fixture / "containers/unrelated-container.txt", "not personal agent\n")

    removable = [
        {"id": "release", "class": "runtime release", "path": str(release), "owned": True, "expected_type": "directory"},
        {"id": "current", "class": "runtime symlink", "path": str(runtime / "current"), "owned": True, "expected_type": "symlink"},
        {"id": "manifest", "class": "install metadata", "path": str(runtime / "install-manifest.json"), "owned": True, "expected_type": "file"},
        {"id": "api-service", "class": "api service unit", "path": str(service_root / "personal-agent-api.service"), "owned": True, "expected_type": "file"},
        {"id": "telegram-service", "class": "telegram service unit", "path": str(service_root / "personal-agent-telegram.service"), "owned": True, "expected_type": "file"},
        {"id": "desktop-entry", "class": "desktop entry", "path": str(launcher_root / "personal-agent.desktop"), "owned": True, "expected_type": "file"},
        {"id": "icon", "class": "desktop icon", "path": str(icon_root / "personal-agent.svg"), "owned": True, "expected_type": "file"},
    ]
    preserved = [
        {"id": "state-root", "class": "state root", "path": str(install_root)},
        {"id": "memory-db", "class": "memory/preferences", "path": str(install_root / "agent.db")},
        {"id": "secret-store", "class": "secret store", "path": str(install_root / "secrets.enc.json")},
        {"id": "backup-root", "class": "backup root", "path": str(backups)},
        {"id": "repo", "class": "repository", "path": str(repo)},
        {"id": "models", "class": "model caches", "path": str(install_root / "models")},
        {"id": "external-packs", "class": "external packs", "path": str(install_root / "external-packs")},
    ]
    snapshot = {
        "fixture_marker": True,
        "fixture_root": str(fixture),
        "mode": "preserve_data",
        "state_root": str(install_root),
        "backup_root": str(backups),
        "receipt_root": str(install_root / "uninstall_receipts"),
        "runtime_commit": "fixture-a",
        "removable_roots": [str(runtime), str(service_root), str(launcher_root), str(icon_root)],
        "removable_resources": removable,
        "preserved_resources": preserved,
        "install_metadata": {"version": "0.2.0", "service_prefix": "personal-agent"},
    }
    action = {
        "pending_id": "uninstall-smoke",
        "operation_id": operation_id,
        "uninstall_mode": "preserve_data",
        "uninstall_execution_mode": "fixture_preserve_data",
        "state_root": str(install_root),
        "backup_root": str(backups),
        "receipt_root": str(install_root / "uninstall_receipts"),
        "target_snapshot": snapshot,
    }
    return snapshot, action


def _plan() -> dict:
    return {
        "plan_id": "uninstall-smoke",
        "action_type": "operator.uninstall",
        "target": "Personal Agent fixture uninstall",
        "risk_level": "high",
        "executor_status": "enabled",
    }


def _run_success(root: Path) -> list[Check]:
    checks: list[Check] = []
    snapshot, action = _fixture(root, operation_id="uninstall-success")
    registry = ExecutorRegistry(root / "journal.jsonl")
    registry.register(
        ExecutorSpec(
            executor_id="operator.uninstall.v1",
            action_type="operator.uninstall",
            status="enabled",
            run=execute_uninstall_v1,
            rollback_available=False,
            rollback_hint="Reinstall then restore from the final backup.",
        )
    )
    result = registry.execute_confirmed_plan(plan=_plan(), action=action)
    fixture = Path(snapshot["fixture_root"])
    checks.append(
        _pass("fixture uninstall executes", result.user_message, "ExecutorRegistry.execute_confirmed_plan")
        if result.ok and result.mutated
        else _fail("fixture uninstall executes", json.dumps(result.to_dict(), sort_keys=True), "ExecutorRegistry.execute_confirmed_plan")
    )
    checks.append(
        _pass("runtime and service resources removed", "runtime/service/launcher absent", "inspect fixture")
        if not (fixture / "personal-agent/runtime/releases/0.2.0").exists()
        and not (fixture / "config/systemd/user/personal-agent-api.service").exists()
        and not (fixture / "share/applications/personal-agent.desktop").exists()
        else _fail("runtime and service resources removed", "one or more removable resources remained", "inspect fixture")
    )
    checks.append(
        _pass("user data preserved", "state, secrets, backups, repo, models, packs remain", "inspect fixture")
        if (fixture / "personal-agent/agent.db").exists()
        and (fixture / "personal-agent/secrets.enc.json").exists()
        and (fixture / "personal-agent/backups/existing-backup").exists()
        and (fixture / "repo/README.md").exists()
        and (fixture / "personal-agent/models").exists()
        and (fixture / "personal-agent/external-packs").exists()
        else _fail("user data preserved", "a preserved fixture resource was removed", "inspect fixture")
    )
    checks.append(
        _pass("unrelated resources untouched", "unrelated service/container fixtures remain", "inspect fixture")
        if (fixture / "config/systemd/user/unrelated.service").exists()
        and (fixture / "containers/unrelated-container.txt").exists()
        else _fail("unrelated resources untouched", "unrelated fixture resource changed", "inspect fixture")
    )
    details = result.to_dict().get("details", {})
    receipt = Path(str(details.get("receipt_path") or ""))
    backup = Path(str(details.get("final_backup_path") or ""))
    checks.append(_pass("uninstall receipt exists", str(receipt), "inspect receipt") if receipt.is_file() else _fail("uninstall receipt exists", str(receipt), "inspect receipt"))
    checks.append(_pass("final backup exists", str(backup), "inspect final backup") if (backup / "manifest.json").is_file() else _fail("final backup exists", str(backup), "inspect final backup"))
    second = execute_uninstall_v1(_plan(), action)
    checks.append(
        _pass("repeat helper is idempotent", second["user_message"], "execute_uninstall_v1 repeat")
        if second["details"]["status"] == "completed_verified"
        else _fail("repeat helper is idempotent", json.dumps(second, sort_keys=True), "execute_uninstall_v1 repeat")
    )
    return checks


def _run_blockers(root: Path) -> list[Check]:
    checks: list[Check] = []
    live = execute_uninstall_v1(_plan(), {"pending_id": "uninstall-smoke", "uninstall_execution_mode": "live_guarded"})
    checks.append(
        _pass("live uninstall blocked", live["user_message"], "execute_uninstall_v1 live_guarded")
        if not live["ok"] and not live["mutated"] and live["error_code"] == "uninstall_live_execution_not_enabled"
        else _fail("live uninstall blocked", json.dumps(live, sort_keys=True), "execute_uninstall_v1 live_guarded")
    )
    snapshot, action = _fixture(root / "drift", operation_id="drift")
    action["target_snapshot_hash"] = "bad"
    drift = execute_uninstall_v1(_plan(), action)
    checks.append(
        _pass("snapshot drift blocked", drift["user_message"], "execute_uninstall_v1 snapshot drift")
        if not drift["ok"] and not drift["mutated"] and drift["error_code"] == "uninstall_target_changed_since_preview"
        else _fail("snapshot drift blocked", json.dumps(drift, sort_keys=True), "execute_uninstall_v1 snapshot drift")
    )
    snapshot, action = _fixture(root / "escape", operation_id="escape")
    escape = Path(snapshot["fixture_root"]) / "personal-agent/runtime/escape"
    escape.symlink_to(Path("/tmp"))
    snapshot["removable_resources"].append({"id": "escape", "class": "runtime symlink", "path": str(escape), "owned": True, "expected_type": "symlink"})
    escaped = execute_uninstall_v1(_plan(), action)
    checks.append(
        _pass("symlink escape blocked", escaped["user_message"], "execute_uninstall_v1 symlink escape")
        if not escaped["ok"] and not escaped["mutated"] and escaped["error_code"] == "resource_symlink_escape"
        else _fail("symlink escape blocked", json.dumps(escaped, sort_keys=True), "execute_uninstall_v1 symlink escape")
    )
    snapshot, action = _fixture(root / "partial", operation_id="partial")
    action["force_failure_after_resource_id"] = "api-service"
    partial = execute_uninstall_v1(_plan(), action)
    receipt = partial["details"].get("receipt", {}) if isinstance(partial.get("details"), dict) else {}
    checks.append(
        _pass("partial failure is truthful", partial["user_message"], "execute_uninstall_v1 forced partial")
        if not partial["ok"] and partial["mutated"] and partial["error_code"] == "uninstall_partial" and receipt.get("status") == "partial_uninstall"
        else _fail("partial failure is truthful", json.dumps(partial, sort_keys=True), "execute_uninstall_v1 forced partial")
    )
    return checks


def main() -> int:
    checks: list[Check] = []
    before = _git_status_short()
    with tempfile.TemporaryDirectory(prefix="pa-uninstall-smoke-") as raw:
        root = Path(raw)
        checks.extend(_run_success(root / "success"))
        checks.extend(_run_blockers(root / "blockers"))
    after = _git_status_short()
    checks.append(_pass("git status unchanged", after or "(clean)", "git status --short") if after == before else _fail("git status unchanged", f"before={before!r}\nafter={after!r}", "git status --short"))
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed
    print("# Personal Agent Uninstall Execution Smoke")
    for check in checks:
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"command: {check.command}")
        print(f"evidence: {check.evidence}\n")
    print(f"SUMMARY PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
