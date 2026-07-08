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

from agent.executor_registry import ExecutorRegistry, ExecutorSpec, execute_update_v1


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


def _write_release(path: Path, commit: str) -> None:
    (path / "agent").mkdir(parents=True)
    (path / "agent" / "BUILD_INFO.json").write_text(
        json.dumps({"git_commit": commit, "version": "fixture"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "README.txt").write_text(f"fixture release {commit}\n", encoding="utf-8")


def _current_commit(current_link: Path) -> str:
    payload = json.loads((current_link.resolve() / "agent" / "BUILD_INFO.json").read_text(encoding="utf-8"))
    return str(payload.get("git_commit") or "")


def _git_status_short() -> str:
    result = subprocess.run(["git", "status", "--short"], check=False, text=True, capture_output=True, timeout=10)
    return result.stdout.strip()


def _base_plan() -> dict[str, object]:
    return {
        "plan_id": "update-smoke",
        "action_type": "operator.update",
        "target": "Personal Agent fixture update",
        "risk_level": "medium",
        "executor_status": "enabled",
    }


def _base_action(root: Path, *, operation_id: str = "update-smoke", force_failure: bool = False) -> dict[str, object]:
    runtime = root / "runtime"
    releases = runtime / "releases"
    return {
        "pending_id": "update-smoke",
        "operation_id": operation_id,
        "update_mode": "fixture_staged_release",
        "state_root": str(root / "state"),
        "runtime_root": str(runtime),
        "releases_root": str(releases),
        "current_link": str(runtime / "current"),
        "staged_source_path": str(root / "source-release-b"),
        "target_release_id": f"release-b-{operation_id}",
        "expected_current_commit": "commit-a",
        "target_commit": "commit-b",
        "preview_target_commit": "commit-b",
        "working_tree_clean": True,
        "force_post_promotion_failure": force_failure,
    }


def _prepare_fixture(root: Path) -> None:
    runtime = root / "runtime"
    releases = runtime / "releases"
    releases.mkdir(parents=True)
    _write_release(releases / "release-a", "commit-a")
    _write_release(root / "source-release-b", "commit-b")
    current = runtime / "current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(releases / "release-a")


def _run_fixture_registry_success(root: Path) -> list[Check]:
    checks: list[Check] = []
    _prepare_fixture(root)
    registry = ExecutorRegistry(root / "journal.jsonl")
    registry.register(
        ExecutorSpec(
            executor_id="operator.update.v1",
            action_type="operator.update",
            status="enabled",
            run=execute_update_v1,
            rollback_available=True,
            rollback_hint="Switch runtime/current back to the checkpoint release.",
        )
    )
    result = registry.execute_confirmed_plan(plan=_base_plan(), action=_base_action(root))
    current = root / "runtime" / "current"
    checks.append(
        _pass("fixture update promotes target", result.user_message, "ExecutorRegistry.execute_confirmed_plan")
        if result.ok and result.mutated and _current_commit(current) == "commit-b"
        else _fail("fixture update promotes target", json.dumps(result.to_dict(), sort_keys=True), "ExecutorRegistry.execute_confirmed_plan")
    )
    checks.append(
        _pass("update journal recorded", f"journal_id={result.journal_id}", "journal.recent")
        if result.journal_id and registry.journal.recent(limit=3)
        else _fail("update journal recorded", json.dumps(result.to_dict(), sort_keys=True), "journal.recent")
    )
    checkpoint = root / "state" / "update_checkpoints" / "update-smoke" / "manifest.json"
    checks.append(
        _pass("rollback checkpoint exists", str(checkpoint), "checkpoint manifest")
        if checkpoint.is_file()
        else _fail("rollback checkpoint exists", str(checkpoint), "checkpoint manifest")
    )
    return checks


def _run_fixture_rollback(root: Path) -> list[Check]:
    checks: list[Check] = []
    _prepare_fixture(root)
    result = execute_update_v1(_base_plan(), _base_action(root, operation_id="update-rollback", force_failure=True))
    current = root / "runtime" / "current"
    checks.append(
        _pass("forced verification failure rolls back", result["user_message"], "execute_update_v1(force_post_promotion_failure)")
        if not result["ok"] and result["details"]["status"] == "update_failed_rolled_back" and _current_commit(current) == "commit-a"
        else _fail("forced verification failure rolls back", json.dumps(result, sort_keys=True), "execute_update_v1(force_post_promotion_failure)")
    )
    return checks


def _run_blockers(root: Path) -> list[Check]:
    checks: list[Check] = []
    dirty = execute_update_v1(
        _base_plan(),
        {
            **_base_action(root, operation_id="dirty"),
            "working_tree_clean": False,
            "dirty_files": ["M agent/orchestrator.py"],
        },
    )
    checks.append(
        _pass("dirty tree refusal", dirty["user_message"], "execute_update_v1(working_tree_clean=false)")
        if not dirty["ok"] and not dirty["mutated"] and dirty["error_code"] == "update_dirty_working_tree"
        else _fail("dirty tree refusal", json.dumps(dirty, sort_keys=True), "execute_update_v1(working_tree_clean=false)")
    )
    drift = execute_update_v1(
        _base_plan(),
        {
            **_base_action(root, operation_id="drift"),
            "target_commit": "commit-b",
            "preview_target_commit": "commit-old",
        },
    )
    checks.append(
        _pass("target drift refusal", drift["user_message"], "execute_update_v1(target drift)")
        if not drift["ok"] and not drift["mutated"] and drift["error_code"] == "update_target_changed_since_preview"
        else _fail("target drift refusal", json.dumps(drift, sort_keys=True), "execute_update_v1(target drift)")
    )
    live = execute_update_v1(
        _base_plan(),
        {
            "pending_id": "update-smoke",
            "update_mode": "live_noop",
            "current_runtime_commit": "commit-a",
            "target_commit": "commit-a",
            "preview_target_commit": "commit-a",
            "working_tree_clean": True,
        },
    )
    checks.append(
        _pass("live no-op allowed", live["user_message"], "execute_update_v1(live_noop)")
        if live["ok"] and not live["mutated"] and live["details"]["status"] == "already_current"
        else _fail("live no-op allowed", json.dumps(live, sort_keys=True), "execute_update_v1(live_noop)")
    )
    return checks


def main() -> int:
    checks: list[Check] = []
    before_status = _git_status_short()
    with tempfile.TemporaryDirectory(prefix="pa-update-smoke-") as raw:
        root = Path(raw)
        checks.extend(_run_fixture_registry_success(root / "success"))
        checks.extend(_run_fixture_rollback(root / "rollback"))
        checks.extend(_run_blockers(root / "blockers"))
    after_status = _git_status_short()
    checks.append(
        _pass("git status unchanged", after_status or "(clean)", "git status --short")
        if after_status == before_status
        else _fail("git status unchanged", f"before={before_status!r}\nafter={after_status!r}", "git status --short")
    )
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed
    print("# Personal Agent Update Execution Smoke")
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"## {check.name}: {status}")
        print(f"command: {check.command}")
        print(f"evidence: {check.evidence}\n")
    print(f"SUMMARY PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
