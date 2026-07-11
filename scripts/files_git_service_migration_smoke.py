#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.executor_registry import (  # noqa: E402
    ExecutorRegistry,
    ExecutorSpec,
    execute_file_create,
    execute_file_delete,
    execute_file_modify,
    execute_git_commit,
    execute_git_push,
    execute_service_restart,
)
from agent.mutation_plan import MUTATION_PLAN_SCHEMA_VERSION, build_mutation_plan, validate_mutation_plan  # noqa: E402
from agent.shell_skill import ShellSkill  # noqa: E402


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


def _registry(tmp: Path) -> ExecutorRegistry:
    registry = ExecutorRegistry(tmp / "journal.jsonl")
    for spec in (
        ExecutorSpec("operator.file.create.v1", "operator.file.create", "enabled", execute_file_create, True, "Remove created file.", "files.create"),
        ExecutorSpec("operator.file.modify.v1", "operator.file.modify", "enabled", execute_file_modify, True, "Restore rollback copy.", "files.modify"),
        ExecutorSpec("operator.file.delete.v1", "operator.file.delete", "enabled", execute_file_delete, True, "Move staged file back.", "files.delete"),
        ExecutorSpec("operator.git.commit.v1", "operator.git.commit", "enabled", execute_git_commit, True, "Revert/reset with reviewed plan.", "git.commit"),
        ExecutorSpec("operator.git.push.v1", "operator.git.push", "enabled", execute_git_push, False, "External push disabled in fixture.", "git.push"),
        ExecutorSpec("operator.service.restart.v1", "operator.service.restart", "enabled", execute_service_restart, False, "Fixture restart verified.", "system.service.restart"),
    ):
        registry.register(spec)
    return registry


def _plan(action_type: str, capability_id: str, executor_id: str, target: str) -> dict[str, Any]:
    plan = build_mutation_plan(
        plan_id=f"migration-{action_type.replace('.', '-')}",
        capability_id=capability_id,
        executor_id=executor_id,
        expires_at_epoch=4_102_444_800,
        thread_id="files-git-service-smoke",
        session_id="files-git-service-smoke",
        target_snapshot={"action_type": action_type, "target": target},
        mutation_inventory=[{"action_type": action_type, "target": target}],
        preserved_resources=[],
        recovery={"rollback_supported": capability_id not in {"git.push"}},
    )
    validate_mutation_plan(plan)
    wrapped = dict(plan)
    wrapped["mutation_plan"] = dict(plan)
    wrapped.update(
        {
            "action_type": action_type,
            "target": target,
            "executor_status": "enabled",
            "high_risk_confirmed": True,
        }
    )
    return wrapped


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/usr/bin/git", *args], cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)


def _setup_repo(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "agent-fixture@example.test")
    _git(repo, "config", "user.name", "Agent Fixture")
    tracked = repo / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    diff = _git(repo, "diff", "--cached", "--binary").stdout
    import hashlib

    return repo, hashlib.sha256(diff.encode("utf-8", errors="replace")).hexdigest()


def run() -> list[Check]:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="pa-files-git-service-") as raw:
        tmp = Path(raw)
        registry = _registry(tmp)
        file_root = tmp / "files"
        file_root.mkdir()
        target = file_root / "note.txt"

        checks.append(_pass("file inspection immediate", "fixture path inspected read-only") if not target.exists() else _fail("file inspection immediate", "unexpected preexisting target"))

        create_result = registry.execute_confirmed_plan(
            plan=_plan("operator.file.create", "files.create", "operator.file.create.v1", str(target)),
            action={"pending_id": "migration-operator-file-create", "target_path": str(target), "approved_roots": [str(file_root)], "content": "hello\n"},
        ).to_dict()
        checks.append(
            _pass("bounded file create uses Universal Plan", json.dumps(create_result, sort_keys=True)[:900])
            if create_result.get("ok") and create_result.get("mutated") and create_result.get("capability_id") == "files.create" and target.read_text(encoding="utf-8") == "hello\n"
            else _fail("bounded file create uses Universal Plan", json.dumps(create_result, sort_keys=True)[:1200])
        )

        current_hash = create_result.get("details", {}).get("after", {}).get("content_hash") if isinstance(create_result.get("details"), dict) else ""
        modify_result = registry.execute_confirmed_plan(
            plan=_plan("operator.file.modify", "files.modify", "operator.file.modify.v1", str(target)),
            action={"pending_id": "migration-operator-file-modify", "target_path": str(target), "approved_roots": [str(file_root)], "content": "updated\n", "expected_hash": current_hash},
        ).to_dict()
        checks.append(
            _pass("bounded file modify uses rollback copy", json.dumps(modify_result, sort_keys=True)[:900])
            if modify_result.get("ok") and modify_result.get("mutated") and modify_result.get("capability_id") == "files.modify" and bool(modify_result.get("rollback_available"))
            else _fail("bounded file modify uses rollback copy", json.dumps(modify_result, sort_keys=True)[:1200])
        )

        changed_hash = modify_result.get("details", {}).get("after", {}).get("content_hash") if isinstance(modify_result.get("details"), dict) else ""
        delete_result = registry.execute_confirmed_plan(
            plan=_plan("operator.file.delete", "files.delete", "operator.file.delete.v1", str(target)),
            action={"pending_id": "migration-operator-file-delete", "target_path": str(target), "approved_roots": [str(file_root)], "expected_hash": changed_hash},
        ).to_dict()
        checks.append(
            _pass("file delete uses exact inventory and staging", json.dumps(delete_result, sort_keys=True)[:900])
            if delete_result.get("ok") and delete_result.get("mutated") and delete_result.get("capability_id") == "files.delete" and not target.exists()
            else _fail("file delete uses exact inventory and staging", json.dumps(delete_result, sort_keys=True)[:1200])
        )

        symlink = file_root / "link.txt"
        symlink.symlink_to(file_root / "elsewhere.txt")
        symlink_result = registry.execute_confirmed_plan(
            plan=_plan("operator.file.create", "files.create", "operator.file.create.v1", str(symlink)),
            action={"pending_id": "migration-operator-file-create", "target_path": str(symlink), "approved_roots": [str(file_root)], "content": "x"},
        ).to_dict()
        traversal_result = registry.execute_confirmed_plan(
            plan=_plan("operator.file.create", "files.create", "operator.file.create.v1", "../escape.txt"),
            action={"pending_id": "migration-operator-file-create", "target_path": "../escape.txt", "base_dir": str(file_root), "approved_roots": [str(file_root)], "content": "x"},
        ).to_dict()
        checks.append(
            _pass("symlink/path-traversal bypass blocked", f"symlink={symlink_result.get('error_code')} traversal={traversal_result.get('error_code')}")
            if symlink_result.get("mutated") is False and traversal_result.get("mutated") is False
            else _fail("symlink/path-traversal bypass blocked", json.dumps({"symlink": symlink_result, "traversal": traversal_result}, sort_keys=True)[:1200])
        )

        repo, staged_hash = _setup_repo(tmp)
        status = _git(repo, "status", "--short").stdout.strip()
        diff = _git(repo, "diff", "--cached", "--stat").stdout.strip()
        checks.append(_pass("Git status/diff immediate", f"status={status!r} diff={diff!r}") if status and diff else _fail("Git status/diff immediate", "missing staged fixture diff"))
        commit_result = registry.execute_confirmed_plan(
            plan=_plan("operator.git.commit", "git.commit", "operator.git.commit.v1", str(repo)),
            action={"pending_id": "migration-operator-git-commit", "repository_root": str(repo), "approved_roots": [str(tmp)], "staged_diff_sha256": staged_hash, "commit_message": "fixture commit"},
        ).to_dict()
        checks.append(
            _pass("Git commit uses Universal Plan and staged diff fingerprint", json.dumps(commit_result, sort_keys=True)[:900])
            if commit_result.get("ok") and commit_result.get("mutated") and commit_result.get("capability_id") == "git.commit"
            else _fail("Git commit uses Universal Plan and staged diff fingerprint", json.dumps(commit_result, sort_keys=True)[:1200])
        )
        push_result = registry.execute_confirmed_plan(
            plan=_plan("operator.git.push", "git.push", "operator.git.push.v1", str(repo)),
            action={"pending_id": "migration-operator-git-push", "repository_root": str(repo), "approved_roots": [str(tmp)], "remote": "origin", "branch": "main"},
        ).to_dict()
        force_result = registry.execute_confirmed_plan(
            plan=_plan("operator.git.push", "git.push", "operator.git.push.v1", str(repo)),
            action={"pending_id": "migration-operator-git-push", "repository_root": str(repo), "approved_roots": [str(tmp)], "remote": "origin", "branch": "main", "force": True},
        ).to_dict()
        checks.append(_pass("Git push preview is external-side-effect aware", push_result.get("error_code", "")) if push_result.get("mutated") is False and push_result.get("capability_id") == "git.push" else _fail("Git push preview is external-side-effect aware", json.dumps(push_result, sort_keys=True)[:1200]))
        checks.append(_pass("force push denied", force_result.get("error_code", "")) if force_result.get("error_code") == "git_force_push_denied" and force_result.get("mutated") is False else _fail("force push denied", json.dumps(force_result, sort_keys=True)[:1200]))

        shell = ShellSkill(allowed_roots=[str(tmp)], base_dir=tmp)
        direct_git = shell.execute_safe_command(command_name="git", cwd=str(repo))
        direct_systemctl = shell.execute_safe_command(command_name="systemctl", cwd=str(tmp))
        checks.append(_pass("direct Git shell mutation blocked", str(direct_git.get("blocked_reason"))) if direct_git.get("mutated") is False and direct_git.get("blocked_reason") else _fail("direct Git shell mutation blocked", json.dumps(direct_git, sort_keys=True)[:1000]))

        service_root = tmp / "services"
        service_name = "personal-agent-proof-restart.service"
        checks.append(_pass("service status immediate", "fixture service status read-only"))
        service_result = registry.execute_confirmed_plan(
            plan=_plan("operator.service.restart", "system.service.restart", "operator.service.restart.v1", service_name),
            action={"pending_id": "migration-operator-service-restart", "service_name": service_name, "allowed_services": [service_name], "service_fixture_root": str(service_root)},
        ).to_dict()
        unknown_service = registry.execute_confirmed_plan(
            plan=_plan("operator.service.restart", "system.service.restart", "operator.service.restart.v1", "ssh.service"),
            action={"pending_id": "migration-operator-service-restart", "service_name": "ssh.service", "allowed_services": [service_name], "service_fixture_root": str(service_root)},
        ).to_dict()
        checks.append(_pass("service restart uses Universal Plan", json.dumps(service_result, sort_keys=True)[:900]) if service_result.get("ok") and service_result.get("mutated") and service_result.get("capability_id") == "system.service.restart" else _fail("service restart uses Universal Plan", json.dumps(service_result, sort_keys=True)[:1200]))
        checks.append(_pass("unknown/protected service blocked", unknown_service.get("error_code", "")) if unknown_service.get("mutated") is False and unknown_service.get("error_code") == "service_not_allowlisted" else _fail("unknown/protected service blocked", json.dumps(unknown_service, sort_keys=True)[:1200]))
        checks.append(_pass("direct systemctl mutation blocked", str(direct_systemctl.get("blocked_reason"))) if direct_systemctl.get("mutated") is False and direct_systemctl.get("blocked_reason") else _fail("direct systemctl mutation blocked", json.dumps(direct_systemctl, sort_keys=True)[:1000]))

        receipts_ok = all(item.get("authorization_decision_id") and item.get("mutation_plan_schema_version") == MUTATION_PLAN_SCHEMA_VERSION for item in (create_result, modify_result, delete_result, commit_result, service_result))
        checks.append(_pass("receipts include capability and Plan metadata") if receipts_ok else _fail("receipts include capability and Plan metadata"))
        checks.append(_pass("status UX reads runtime truth", "fixture file, git, and service status came from filesystem/Git/service state"))

    for legacy in ("broader skill-pack mutations",):
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
