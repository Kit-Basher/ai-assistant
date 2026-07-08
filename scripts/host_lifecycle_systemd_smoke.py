#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.host_lifecycle import HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION, HOST_LIFECYCLE_RUNNER_VERSION, attach_approved_hash, write_json_atomic


@dataclass
class Check:
    name: str
    status: str
    evidence: str
    command: str


def _check(name: str, status: str, evidence: str, command: str) -> Check:
    return Check(name, status, evidence.strip()[:1200], command)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_release(path: Path, commit: str) -> None:
    _write(path / "agent/BUILD_INFO.json", json.dumps({"git_commit": commit}) + "\n")


def _current_commit(current: Path) -> str:
    return str(json.loads((current.resolve() / "agent/BUILD_INFO.json").read_text(encoding="utf-8")).get("git_commit") or "")


def _git_status_short() -> str:
    return subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, timeout=10, check=False).stdout.strip()


def _operation(root: Path) -> tuple[Path, Path]:
    runtime = root / "runtime"
    releases = runtime / "releases"
    releases.mkdir(parents=True)
    _write_release(releases / "release-a", "commit-a")
    _write_release(root / "source-release-b", "commit-b")
    current = runtime / "current"
    current.symlink_to(releases / "release-a")
    state = root / "state"
    operation_id = "systemd-update-fixture"
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
            "target_release_id": "release-b-systemd",
            "current_runtime_commit": "commit-a",
            "target_commit": "commit-b",
            "operation_state_path": str(state / "host_lifecycle" / "operations" / operation_id / "state.json"),
            "receipt_path": str(state / "host_lifecycle" / "operations" / operation_id / "receipt.json"),
        }
    )
    record_path = state / "host_lifecycle" / "operations" / operation_id / "operation.json"
    write_json_atomic(record_path, record)
    return record_path, current


def main() -> int:
    checks: list[Check] = []
    before = _git_status_short()
    systemd_run = shutil.which("systemd-run")
    if not systemd_run:
        print("# Personal Agent Host Lifecycle Systemd Smoke")
        print("## systemd-run available: SKIP")
        print("- evidence: systemd-run was not found on PATH.")
        print("- next action: run this installed-host proof on a Debian user-systemd host.")
        print("\nSUMMARY PASS=0 WARN=0 FAIL=0 SKIP=1")
        return 0
    probe = subprocess.run(["systemctl", "--user", "show-environment"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10, check=False)
    if probe.returncode != 0:
        print("# Personal Agent Host Lifecycle Systemd Smoke")
        print("## user systemd available: SKIP")
        print(f"- evidence: {probe.stdout.strip()[:500]}")
        print("- next action: run this installed-host proof inside an active user systemd session.")
        print("\nSUMMARY PASS=0 WARN=0 FAIL=0 SKIP=1")
        return 0
    with tempfile.TemporaryDirectory(prefix="pa-host-systemd-") as raw:
        root = Path(raw)
        record, current = _operation(root)
        unit = "personal-agent-host-lifecycle-fixture-update"
        command = [
            systemd_run,
            "--user",
            "--wait",
            "--collect",
            f"--unit={unit}",
            sys.executable,
            str(ROOT / "scripts/host_lifecycle_runner.py"),
            "update",
            "--operation-record",
            str(record),
        ]
        proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, check=False)
        checks.append(_check("systemd launches host runner", "PASS" if proc.returncode == 0 else "FAIL", proc.stdout, " ".join(command)))
        checks.append(_check("fixture update completed under systemd", "PASS" if _current_commit(current) == "commit-b" else "FAIL", f"current_commit={_current_commit(current)}", "inspect fixture current symlink"))
        checks.append(_check("operation status survives runner exit", "PASS" if (root / "state/host_lifecycle/operations/systemd-update-fixture/state.json").is_file() else "FAIL", str(root / "state/host_lifecycle/operations/systemd-update-fixture/state.json"), "inspect operation state"))
        checks.append(_check("receipt survives runner exit", "PASS" if (root / "state/host_lifecycle/operations/systemd-update-fixture/receipt.json").is_file() else "FAIL", str(root / "state/host_lifecycle/operations/systemd-update-fixture/receipt.json"), "inspect receipt"))
    after = _git_status_short()
    checks.append(_check("git status unchanged", "PASS" if after == before else "FAIL", after or "(clean)", "git status --short"))
    passed = sum(1 for check in checks if check.status == "PASS")
    failed = sum(1 for check in checks if check.status == "FAIL")
    print("# Personal Agent Host Lifecycle Systemd Smoke")
    for check in checks:
        print(f"## {check.name}: {check.status}")
        print(f"command: {check.command}")
        print(f"evidence: {check.evidence}\n")
    print(f"SUMMARY PASS={passed} WARN=0 FAIL={failed} SKIP=0")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
