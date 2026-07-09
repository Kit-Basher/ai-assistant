#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.executor_registry import ExecutorRegistry, ExecutorSpec, execute_uninstall_v1, _snapshot_hash


PRIMARY_URL = "http://127.0.0.1:8765"
PRIMARY_STATE = Path.home() / ".local/share/personal-agent"
PRIMARY_RUNTIME = PRIMARY_STATE / "runtime"
PRIMARY_SERVICE = "personal-agent-api.service"


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str


def _check(name: str, ok: bool, evidence: str, command: str) -> Check:
    return Check(name=name, ok=ok, evidence=evidence.strip()[:1600], command=command)


def _json_get(url: str, *, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read(256 * 1024).decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise RuntimeError("non_object_json")
    return payload


def _git_status_short() -> str:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    return proc.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _assert_isolated(root: Path) -> None:
    root = root.resolve()
    forbidden = [PRIMARY_STATE.resolve(), PRIMARY_RUNTIME.resolve(), ROOT.resolve()]
    if root == Path.home().resolve() or any(root == item or str(root).startswith(str(item) + "/") for item in forbidden):
        raise RuntimeError(f"proof_root_overlaps_primary:{root}")


def _build_production_shaped_fixture(root: Path, operation_id: str) -> tuple[dict, dict]:
    _assert_isolated(root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".primary-uninstall-shaped-proof"
    marker.write_text("primary uninstall shaped proof\n", encoding="utf-8")
    install_root = root / "personal-agent"
    runtime = install_root / "runtime"
    releases = runtime / "releases"
    release = releases / "proof-a"
    current = runtime / "current"
    state = install_root
    service_dir = root / "config/systemd/user"
    launcher_dir = root / "share/applications"
    icon_dir = root / "share/icons/hicolor/scalable/apps"
    repo = root / "repo"
    backups = state / "backups"
    for path in (release, service_dir, launcher_dir, icon_dir, repo, backups, state / "models", state / "external-packs", state / "support-bundles"):
        path.mkdir(parents=True, exist_ok=True)
    current.symlink_to(release)
    _write(release / "agent/BUILD_INFO.json", json.dumps({"git_commit": "proof-a"}) + "\n")
    _write(runtime / "install-manifest.json", json.dumps({"generated_by": "personal-agent-proof"}) + "\n")
    _write(service_dir / "personal-agent-proof-api.service", "[Service]\nExecStart=/bin/false # Personal Agent proof\n")
    _write(service_dir / "personal-agent-proof-telegram.service", "[Service]\nExecStart=/bin/false # Personal Agent proof\n")
    _write(launcher_dir / "personal-agent.desktop", "[Desktop Entry]\nName=Personal Agent Proof\n")
    _write(icon_dir / "personal-agent.svg", "<svg />\n")
    _write(state / "agent.db", "memory baseline context\n")
    _write(state / "preferences.json", '{"theme":"system"}\n')
    _write(state / "secrets.enc.json", '{"telegram_token":"dummy-not-production"}\n')
    _write(backups / "existing-backup/manifest.json", '{"backup_schema_version":"backup.v1"}\n')
    _write(repo / "README.md", "preserved repository\n")
    _write(root / "unrelated-keep.txt", "must remain\n")

    removable = [
        {"id": "runtime-current", "class": "runtime symlink", "path": str(current), "owned": True, "expected_type": "symlink", "ownership_basis": "generated runtime/current symlink"},
        {"id": "runtime-releases", "class": "runtime releases", "path": str(releases), "owned": True, "expected_type": "directory", "ownership_basis": "generated release root"},
        {"id": "install-manifest", "class": "install metadata", "path": str(runtime / "install-manifest.json"), "owned": True, "expected_type": "file", "ownership_basis": "generated install manifest"},
        {"id": "api-service", "class": "api service unit", "path": str(service_dir / "personal-agent-proof-api.service"), "owned": True, "expected_type": "file", "ownership_basis": "generated proof service"},
        {"id": "telegram-service", "class": "telegram service unit", "path": str(service_dir / "personal-agent-proof-telegram.service"), "owned": True, "expected_type": "file", "ownership_basis": "generated proof service"},
        {"id": "desktop-entry", "class": "desktop entry", "path": str(launcher_dir / "personal-agent.desktop"), "owned": True, "expected_type": "file", "ownership_basis": "generated launcher"},
        {"id": "desktop-icon", "class": "desktop icon", "path": str(icon_dir / "personal-agent.svg"), "owned": True, "expected_type": "file", "ownership_basis": "generated icon"},
    ]
    preserved = [
        {"id": "state-root", "class": "state/config root", "path": str(state), "reason": "preserve user state"},
        {"id": "memory-db", "class": "memory/preferences", "path": str(state / "agent.db"), "reason": "preserve memory"},
        {"id": "preferences", "class": "preferences", "path": str(state / "preferences.json"), "reason": "preserve preferences"},
        {"id": "secret-store", "class": "secret store", "path": str(state / "secrets.enc.json"), "reason": "preserve secrets"},
        {"id": "backups", "class": "backup root", "path": str(backups), "reason": "preserve backups"},
        {"id": "repo", "class": "repository checkout", "path": str(repo), "reason": "preserve repository"},
        {"id": "models", "class": "model caches", "path": str(state / "models"), "reason": "preserve models"},
        {"id": "external-packs", "class": "external skill packs", "path": str(state / "external-packs"), "reason": "preserve packs"},
        {"id": "unrelated", "class": "unrelated proof file", "path": str(root / "unrelated-keep.txt"), "reason": "not owned"},
    ]
    snapshot = {
        "fixture_marker": True,
        "fixture_root": str(root),
        "proof_marker_path": str(marker),
        "mode": "preserve_data",
        "runtime_root": str(runtime),
        "current_root": str(current),
        "state_root": str(state),
        "backup_root": str(backups),
        "receipt_root": str(state / "uninstall_receipts"),
        "runtime_commit": "proof-a",
        "removable_roots": [str(runtime), str(service_dir), str(launcher_dir), str(icon_dir)],
        "removable_resources": removable,
        "preserved_resources": preserved,
        "service_names": [],
        "install_metadata": {"profile": "primary-uninstall-shaped-proof", "service_prefix": "personal-agent-proof"},
    }
    action = {
        "pending_id": operation_id,
        "operation_id": operation_id,
        "uninstall_mode": "preserve_data",
        "uninstall_execution_mode": "production_shaped_preserve_data",
        "state_root": str(state),
        "backup_root": str(backups),
        "receipt_root": str(state / "uninstall_receipts"),
        "proof_marker_path": str(marker),
        "target_snapshot": snapshot,
        "target_snapshot_hash": _snapshot_hash(snapshot),
        "service_names": [],
    }
    return snapshot, action


def _plan(operation_id: str) -> dict:
    return {
        "plan_id": operation_id,
        "action_type": "operator.uninstall",
        "target": "Personal Agent production-shaped uninstall proof",
        "risk_level": "high",
        "executor_status": "enabled",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-primary-uninstall-shaped-proof", action="store_true")
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    checks: list[Check] = []
    if not args.allow_primary_uninstall_shaped_proof:
        print("SKIP: pass --allow-primary-uninstall-shaped-proof to run the isolated production-shaped proof")
        return 2

    before_git = _git_status_short()
    try:
        version = _json_get(f"{PRIMARY_URL}/version")
        ready = _json_get(f"{PRIMARY_URL}/ready")
        checks.append(_check("primary snapshot healthy", bool(ready.get("ready")) and version.get("git_commit") == args.expected_commit, json.dumps(version, sort_keys=True), "GET /ready /version"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("primary snapshot healthy", False, f"{exc.__class__.__name__}: {exc}", "GET /ready /version"))

    retained_root = ""
    with tempfile.TemporaryDirectory(prefix="pa-primary-uninstall-shaped-") as raw:
        root = Path(raw).resolve()
        retained_root = str(root)
        operation_id = "primary-uninstall-shaped-proof"
        snapshot, action = _build_production_shaped_fixture(root, operation_id)
        checks.append(_check("proof root isolated from primary", not str(root).startswith(str(PRIMARY_STATE.resolve())), str(root), "path containment preflight"))

        registry = ExecutorRegistry(root / "executor_journal.jsonl")
        registry.register(
            ExecutorSpec(
                executor_id="operator.uninstall.v1",
                action_type="operator.uninstall",
                status="enabled",
                run=execute_uninstall_v1,
                rollback_available=False,
                rollback_hint="Reinstall then restore from final backup.",
            )
        )
        result = registry.execute_confirmed_plan(plan=_plan(operation_id), action=action)
        result_dict = result.to_dict()
        details = result_dict.get("details", {})
        checks.append(_check("production-shaped uninstall executes", result.ok and result.mutated, json.dumps(result_dict, sort_keys=True)[:1200], "ExecutorRegistry.execute_confirmed_plan"))
        checks.append(_check("runtime and generated files removed", not (root / "personal-agent/runtime/current").exists() and not (root / "share/applications/personal-agent.desktop").exists(), "runtime/current and launcher absent", "inspect proof root"))
        checks.append(_check("preserved data remains", (root / "personal-agent/agent.db").exists() and (root / "personal-agent/secrets.enc.json").exists() and (root / "repo/README.md").exists() and (root / "unrelated-keep.txt").exists(), "state, secrets, repo, unrelated file remain", "inspect proof root"))
        receipt = Path(str(details.get("receipt_path") or ""))
        backup = Path(str(details.get("final_backup_path") or ""))
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
        checks.append(_check("receipt survives and is redacted", receipt.is_file() and "dummy-not-production" not in receipt.read_text(encoding="utf-8", errors="replace"), str(receipt), "inspect receipt"))
        checks.append(_check("final backup validates", (backup / "manifest.json").is_file(), str(backup), "inspect backup"))
        checks.append(_check("post-uninstall status without API", receipt_payload.get("status") == "completed_verified", json.dumps({"status": receipt_payload.get("status"), "removed": len(receipt_payload.get("removed_resources", []))}, sort_keys=True), "receipt status"))

        duplicate = execute_uninstall_v1(_plan(operation_id), action)
        checks.append(_check("duplicate execution is idempotent", duplicate.get("details", {}).get("status") == "completed_verified", json.dumps(duplicate, sort_keys=True)[:1200], "execute_uninstall_v1 duplicate"))

        snapshot2, action2 = _build_production_shaped_fixture(root / "partial", "primary-uninstall-shaped-partial")
        action2["force_failure_after_resource_id"] = "api-service"
        partial = execute_uninstall_v1(_plan("primary-uninstall-shaped-partial"), action2)
        checks.append(_check("partial failure is truthful", partial.get("error_code") == "uninstall_partial" and partial.get("mutated") is True, json.dumps(partial, sort_keys=True)[:1200], "execute_uninstall_v1 forced partial"))

        # Reinstall/restore sanity boundary: preserved repo and final backup remain available.
        reinstall_root = root / "reinstall"
        shutil.copytree(root / "repo", reinstall_root / "repo")
        checks.append(_check("reinstall/restore sanity boundary", (reinstall_root / "repo/README.md").is_file() and (backup / "manifest.json").is_file(), "preserved repo copied; final backup manifest available", "fixture reinstall sanity"))

    after_git = _git_status_short()
    try:
        final_version = _json_get(f"{PRIMARY_URL}/version")
        checks.append(_check("primary remained unchanged", after_git == before_git and final_version.get("git_commit") == args.expected_commit, json.dumps(final_version, sort_keys=True), "git status; GET /version"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("primary remained unchanged", False, f"{exc.__class__.__name__}: {exc}; retained_root={retained_root}", "GET /version"))

    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed
    print("# Personal Agent Primary Preserve-Data Uninstall Enablement Smoke")
    for check in checks:
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}\n")
    print(f"SUMMARY PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
