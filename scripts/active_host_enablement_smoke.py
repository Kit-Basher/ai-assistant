#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
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


PRIMARY_URL = "http://127.0.0.1:8765"
ALT_URL = "http://127.0.0.1:18765"
PROFILE = "active-host-proof"
SERVICE = "personal-agent-active-host-proof-api.service"


@dataclass
class Check:
    name: str
    status: str
    evidence: str
    command: str


def _check(name: str, ok: bool, evidence: str, command: str) -> Check:
    return Check(name, "PASS" if ok else "FAIL", evidence.strip()[:1800], command)


def _warn(name: str, evidence: str, command: str) -> Check:
    return Check(name, "WARN", evidence.strip()[:1800], command)


def _run(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)


def _http_json(base: str, path: str, *, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as response:
        payload = json.loads(response.read(512 * 1024).decode("utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def _post_chat(base: str, message: str, *, thread_id: str = "active-host-proof") -> dict:
    body = json.dumps({"message": message, "thread_id": thread_id}).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.loads(response.read(1024 * 1024).decode("utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def _wait_ready(base: str, commit: str, *, timeout: float = 45.0) -> dict:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            ready = _http_json(base, "/ready", timeout=2.0)
            version = _http_json(base, "/version", timeout=2.0)
            if ready.get("ready") is True and str(version.get("git_commit") or "") == commit:
                return {"ready": ready, "version": version}
            last = f"ready={ready.get('ready')} commit={version.get('git_commit')}"
        except Exception as exc:  # noqa: BLE001
            last = f"{exc.__class__.__name__}: {exc}"
        time.sleep(0.5)
    raise TimeoutError(f"ready timeout for {base} commit={commit}: {last}")


def _git_status_short() -> str:
    return _run(["git", "status", "--short"], timeout=10).stdout.strip()


def _primary_snapshot() -> dict:
    service = _run(["systemctl", "--user", "is-active", "personal-agent-api.service"], timeout=10).stdout.strip()
    ready = _http_json(PRIMARY_URL, "/ready", timeout=5)
    version = _http_json(PRIMARY_URL, "/version", timeout=5)
    return {
        "service": service,
        "ready": ready.get("ready"),
        "git_commit": version.get("git_commit"),
        "runtime_instance": version.get("runtime_instance"),
        "git_status": _git_status_short(),
    }


def _copy_release(dest: Path, commit: str) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        blocked = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        return {name for name in names if name in blocked}

    shutil.copytree(ROOT, dest, ignore=ignore)
    build_info = {
        "git_commit": commit,
        "checkout_git_commit": commit,
        "version": "active-host-proof",
        "proof_profile": PROFILE,
    }
    (dest / "agent" / "BUILD_INFO.json").write_text(json.dumps(build_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    if port == 8765:
        return _free_loopback_port()
    return port


def _write_service(root: Path, current_link: Path, state_root: Path, log_root: Path, commit: str, port: int) -> Path:
    service_dir = Path.home() / ".config/systemd/user"
    service_path = service_dir / SERVICE
    service_dir.mkdir(parents=True, exist_ok=True)
    wrapper = root / "run-active-host-proof-api.sh"
    env = {
        "PERSONAL_AGENT_INSTANCE": "dev",
        "PERSONAL_AGENT_RUNTIME_ROOT": str(current_link),
        "AGENT_API_PORT": str(port),
        "AGENT_DB_PATH": str(state_root / "agent.db"),
        "AGENT_LOG_PATH": str(log_root / "agent.jsonl"),
        "LLM_PROVIDER": "none",
        "TELEGRAM_ENABLED": "0",
        "TELEGRAM_REQUIRED": "0",
        "AGENT_WEBUI_DIST_PATH": str(current_link / "agent/webui/dist"),
        "PYTHONPATH": str(current_link),
        "PERSONAL_AGENT_ACTIVE_HOST_PROOF_ROOT": str(root),
    }
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        + "".join(f"export {key}={json.dumps(value)}\n" for key, value in env.items())
        + f"cd {json.dumps(str(current_link))}\n"
        + "export PERSONAL_AGENT_GIT_COMMIT_OVERRIDE=\"$("
        + f"{json.dumps(sys.executable)} -c 'import json; print(json.load(open(\"agent/BUILD_INFO.json\")).get(\"git_commit\", \"unknown\"))'"
        + ")\"\n"
        + f"exec {json.dumps(sys.executable)} -m agent.api_server --host 127.0.0.1 --port {int(port)}\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    service_path.write_text(
        "\n".join(
            [
                "[Unit]",
                "Description=Personal Agent active-host proof API",
                "",
                "[Service]",
                "Type=simple",
                f"WorkingDirectory={current_link}",
                f"ExecStart={wrapper}",
                "Restart=no",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return service_path


def _operation_common(state_root: Path, operation_id: str, operation_type: str) -> tuple[Path, Path, Path]:
    op_dir = state_root / "host_lifecycle" / "operations" / operation_id
    return op_dir / "operation.json", op_dir / "state.json", op_dir / "receipt.json"


def _write_update_operation(
    *,
    root: Path,
    runtime_root: Path,
    state_root: Path,
    source_release: Path,
    operation_id: str,
    current_commit: str,
    target_commit: str,
    target_release_id: str,
    alt_url: str,
    force_failure: bool = False,
    interrupt_once: bool = False,
) -> Path:
    record_path, state_path, receipt_path = _operation_common(state_root, operation_id, "update")
    record = attach_approved_hash(
        {
            "schema_version": HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
            "runner_version": HOST_LIFECYCLE_RUNNER_VERSION,
            "operation_id": operation_id,
            "operation_type": "update",
            "plan_id": operation_id,
            "created_at": "2026-07-08T00:00:00+00:00",
            "fixture_mode": "active_host_proof",
            "proof_marker_path": str(root / ".active-host-proof"),
            "state_root": str(state_root),
            "runtime_root": str(runtime_root),
            "releases_root": str(runtime_root / "releases"),
            "current_link": str(runtime_root / "current"),
            "staged_source_path": str(source_release),
            "target_release_id": target_release_id,
            "current_runtime_commit": current_commit,
            "target_commit": target_commit,
            "api_service_name": SERVICE,
            "verify_base_url": alt_url,
            "operation_state_path": str(state_path),
            "receipt_path": str(receipt_path),
            "force_post_promotion_failure": force_failure,
            "interrupt_once_after_promotion": interrupt_once,
            "interrupt_marker_path": str(state_root / "host_lifecycle" / "operations" / operation_id / "interrupt-once.marker"),
        }
    )
    write_json_atomic(record_path, record)
    return record_path


def _write_uninstall_operation(root: Path, runtime_root: Path, state_root: Path, service_path: Path, operation_id: str) -> Path:
    backup_root = state_root / "backups"
    final_backup = backup_root / f"personal-agent-uninstall-backup-{operation_id}"
    final_backup.mkdir(parents=True, exist_ok=True)
    write_json_atomic(final_backup / "manifest.json", {"backup_schema_version": "backup.v1", "operation_id": operation_id})
    receipt_root = state_root / "uninstall_receipts"
    launcher = root / "launchers" / "personal-agent-active-host-proof.desktop"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("[Desktop Entry]\nName=Proof\n", encoding="utf-8")
    preserved = [
        {"id": "state-db", "path": str(state_root / "agent.db")},
        {"id": "secrets", "path": str(state_root / "secrets.enc.json")},
        {"id": "backups", "path": str(backup_root)},
        {"id": "repo", "path": str(root / "repo")},
        {"id": "models", "path": str(state_root / "models")},
        {"id": "packs", "path": str(state_root / "external-packs")},
        {"id": "unrelated", "path": str(root / "unrelated-keep.txt")},
    ]
    removable_roots = [str(runtime_root), str(service_path.parent), str(launcher.parent)]
    removable = [
        {"id": "current", "class": "runtime symlink", "path": str(runtime_root / "current")},
        {"id": "releases", "class": "runtime releases", "path": str(runtime_root / "releases")},
        {"id": "api-service", "class": "api service", "path": str(service_path)},
        {"id": "launcher", "class": "launcher", "path": str(launcher)},
    ]
    snapshot = {
        "fixture_marker": True,
        "fixture_root": str(root),
        "mode": "preserve_data",
        "state_root": str(state_root),
        "backup_root": str(backup_root),
        "receipt_root": str(receipt_root),
        "removable_roots": removable_roots,
        "removable_resources": removable,
        "preserved_resources": preserved,
    }
    record_path, state_path, receipt_path = _operation_common(state_root, operation_id, "uninstall")
    record = attach_approved_hash(
        {
            "schema_version": HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
            "runner_version": HOST_LIFECYCLE_RUNNER_VERSION,
            "operation_id": operation_id,
            "operation_type": "uninstall",
            "plan_id": operation_id,
            "created_at": "2026-07-08T00:00:00+00:00",
            "fixture_mode": "active_host_proof",
            "proof_marker_path": str(root / ".active-host-proof"),
            "target_snapshot": snapshot,
            "final_backup_path": str(final_backup),
            "service_names": [SERVICE],
            "operation_state_path": str(state_path),
            "receipt_path": str(receipt_path),
        }
    )
    write_json_atomic(record_path, record)
    return record_path


def _run_runner_via_systemd(operation: str, record: Path, unit_suffix: str, *, wait: bool = True, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    command = [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit=personal-agent-active-host-proof-runner-{unit_suffix}",
    ]
    if wait:
        command.append("--wait")
    command.extend([sys.executable, str(ROOT / "scripts/host_lifecycle_runner.py"), operation, "--operation-record", str(record)])
    return _run(command, timeout=timeout)


def _cleanup(root: Path, service_path: Path | None) -> str:
    lines: list[str] = []
    for action in ("stop", "disable", "reset-failed"):
        proc = _run(["systemctl", "--user", action, SERVICE], timeout=20)
        lines.append(f"{action}:{proc.returncode}")
    if service_path and service_path.exists():
        try:
            service_path.unlink()
            lines.append("service_file_removed")
        except OSError as exc:
            lines.append(f"service_file_remove_failed:{exc.__class__.__name__}")
    _run(["systemctl", "--user", "daemon-reload"], timeout=20)
    try:
        if root.exists():
            shutil.rmtree(root)
            lines.append("proof_root_removed")
    except OSError as exc:
        lines.append(f"proof_root_remove_failed:{exc.__class__.__name__}:{root}")
    return "; ".join(lines)


def main() -> int:
    checks: list[Check] = []
    before_status = _git_status_short()
    service_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="pa-active-host-proof-") as raw:
        root = Path(raw)
        retained_root = root
        try:
            alt_port = _free_loopback_port()
            alt_url = f"http://127.0.0.1:{alt_port}"
            (root / ".active-host-proof").write_text(PROFILE + "\n", encoding="utf-8")
            primary_before = _primary_snapshot()
            checks.append(_check("primary snapshot healthy before proof", primary_before.get("ready") is True and primary_before.get("git_commit"), json.dumps(primary_before, sort_keys=True), "GET primary /ready /version"))
            if primary_before.get("git_status") != before_status:
                checks.append(_check("primary git baseline stable", False, json.dumps(primary_before), "git status --short"))

            runtime = root / "runtime"
            releases = runtime / "releases"
            state = root / "state"
            logs = root / "logs"
            for path in (releases, state, logs, root / "repo", state / "models", state / "external-packs", state / "backups"):
                path.mkdir(parents=True, exist_ok=True)
            (state / "secrets.enc.json").write_text('{"dummy":"not-a-real-token"}\n', encoding="utf-8")
            (root / "repo" / "README.md").write_text("preserved alternate repo\n", encoding="utf-8")
            (root / "unrelated-keep.txt").write_text("keep\n", encoding="utf-8")
            release_a = releases / "release-a"
            release_b_src = root / "staged-release-b"
            _copy_release(release_a, "active-host-A")
            _copy_release(release_b_src, "active-host-B")
            current = runtime / "current"
            current.symlink_to(release_a)
            service_path = _write_service(root, current, state, logs, "active-host-A", alt_port)
            checks.append(_check("alternate profile isolated from primary", str(root).startswith("/tmp/") and SERVICE != "personal-agent-api.service" and alt_port != 8765, f"root={root} service={SERVICE} url={alt_url}", "isolation preflight"))

            _run(["systemctl", "--user", "daemon-reload"], timeout=20)
            start = _run(["systemctl", "--user", "start", SERVICE], timeout=30)
            checks.append(_check("alternate Release A service starts", start.returncode == 0, start.stdout, f"systemctl --user start {SERVICE}"))
            ready_a = _wait_ready(alt_url, "active-host-A", timeout=60)
            checks.append(_check("alternate Release A serves real API", ready_a["version"].get("git_commit") == "active-host-A", json.dumps(ready_a["version"], sort_keys=True), "GET alternate /version"))
            chat = _post_chat(alt_url, "hello")
            text = str(chat.get("text") or chat.get("message") or chat)
            checks.append(_check("alternate deterministic chat works", "ready to help" in text.lower() or "hi" in text.lower(), text, "POST alternate /chat"))
            preview = _post_chat(alt_url, "update the assistant", thread_id="active-host-update-preview")
            checks.append(_check("alternate Plan Mode update preview renders", "Plan Mode v2" in str(preview), str(preview), "POST alternate /chat update"))

            update_record = _write_update_operation(
                root=root,
                runtime_root=runtime,
                state_root=state,
                source_release=release_b_src,
                operation_id="active-host-update-a-to-b",
                current_commit="active-host-A",
                target_commit="active-host-B",
                target_release_id="release-b",
                alt_url=alt_url,
            )
            update_proc = _run_runner_via_systemd("update", update_record, "update", timeout=120)
            update_receipt_path = state / "host_lifecycle/operations/active-host-update-a-to-b/receipt.json"
            update_receipt_payload = update_receipt_path.read_text(encoding="utf-8") if update_receipt_path.is_file() else ""
            checks.append(_check("real alternate update runner exits cleanly", update_proc.returncode == 0, (update_proc.stdout + "\n" + update_receipt_payload), "systemd-run host_lifecycle_runner update"))
            ready_b = _wait_ready(alt_url, "active-host-B", timeout=60)
            checks.append(_check("real alternate update A to B verified by HTTP", ready_b["version"].get("git_commit") == "active-host-B" and current.resolve().name == "release-b", json.dumps(ready_b["version"], sort_keys=True), "GET alternate /ready /version"))
            update_receipt = json.loads(update_receipt_path.read_text(encoding="utf-8"))
            checks.append(_check("update receipt completed", update_receipt.get("status") == "completed_verified", json.dumps(update_receipt, sort_keys=True), "read update receipt"))

            # Browser/API reconnect is represented here by the same live HTTP client observing service restart and recovery.
            post_update_chat = _post_chat(alt_url, "is the assistant healthy?", thread_id="active-host-after-update")
            checks.append(_check("post-update reconnect chat works", bool(post_update_chat), str(post_update_chat), "POST alternate /chat after update"))

            # Rollback proof: reset to A, promote B with forced verification failure.
            _run(["systemctl", "--user", "stop", SERVICE], timeout=30)
            if current.exists() or current.is_symlink():
                current.unlink()
            current.symlink_to(release_a)
            _run(["systemctl", "--user", "start", SERVICE], timeout=30)
            _wait_ready(alt_url, "active-host-A", timeout=60)
            rollback_record = _write_update_operation(
                root=root,
                runtime_root=runtime,
                state_root=state,
                source_release=release_b_src,
                operation_id="active-host-update-rollback",
                current_commit="active-host-A",
                target_commit="active-host-B",
                target_release_id="release-b-rollback",
                force_failure=True,
                alt_url=alt_url,
            )
            rollback_proc = _run_runner_via_systemd("update", rollback_record, "rollback", timeout=120)
            rollback_receipt = json.loads((state / "host_lifecycle/operations/active-host-update-rollback/receipt.json").read_text(encoding="utf-8"))
            ready_after_rollback = _wait_ready(alt_url, "active-host-A", timeout=60)
            checks.append(_check("forced update failure rolls back serving runtime", rollback_proc.returncode == 2 and rollback_receipt.get("rollback_verified") is True and ready_after_rollback["version"].get("git_commit") == "active-host-A", json.dumps(rollback_receipt, sort_keys=True), "systemd-run host_lifecycle_runner update forced failure"))

            resume_record = _write_update_operation(
                root=root,
                runtime_root=runtime,
                state_root=state,
                source_release=release_b_src,
                operation_id="active-host-update-resume",
                current_commit="active-host-A",
                target_commit="active-host-B",
                target_release_id="release-b-resume",
                interrupt_once=True,
                alt_url=alt_url,
            )
            interrupt_proc = _run_runner_via_systemd("update", resume_record, "resume-first", timeout=120)
            resume_proc = _run_runner_via_systemd("update", resume_record, "resume-second", timeout=120)
            resume_receipt = json.loads((state / "host_lifecycle/operations/active-host-update-resume/receipt.json").read_text(encoding="utf-8"))
            ready_after_resume = _wait_ready(alt_url, "active-host-B", timeout=60)
            checks.append(_check("interrupted runner resumes without duplicate promotion", interrupt_proc.returncode == 2 and resume_proc.returncode == 0 and resume_receipt.get("resume") == "target_already_promoted" and ready_after_resume["version"].get("git_commit") == "active-host-B", json.dumps(resume_receipt, sort_keys=True), "systemd-run host_lifecycle_runner update resume"))

            uninstall_record = _write_uninstall_operation(root, runtime, state, service_path, "active-host-uninstall")
            uninstall_preview = _post_chat(alt_url, "uninstall the assistant", thread_id="active-host-uninstall-preview")
            checks.append(_check("alternate uninstall preview renders", "preserve" in str(uninstall_preview).lower() and "disconnect" in str(uninstall_preview).lower(), str(uninstall_preview), "POST alternate /chat uninstall"))
            uninstall_proc = _run_runner_via_systemd("uninstall", uninstall_record, "uninstall", timeout=120)
            uninstall_receipt = json.loads((state / "host_lifecycle/operations/active-host-uninstall/receipt.json").read_text(encoding="utf-8"))
            api_gone = False
            try:
                _http_json(alt_url, "/ready", timeout=2)
            except Exception:
                api_gone = True
            preserved_ok = all(
                path.exists()
                for path in [
                    state / "agent.db",
                    state / "secrets.enc.json",
                    state / "backups",
                    state / "models",
                    state / "external-packs",
                    root / "repo" / "README.md",
                    root / "unrelated-keep.txt",
                ]
            )
            removed_ok = not (runtime / "current").exists() and not (runtime / "releases").exists() and not service_path.exists()
            checks.append(_check("real alternate uninstall removes runtime after API shutdown", uninstall_proc.returncode == 0 and api_gone and removed_ok, json.dumps(uninstall_receipt, sort_keys=True), "systemd-run host_lifecycle_runner uninstall"))
            checks.append(_check("alternate uninstall preserves data and receipt", preserved_ok and uninstall_receipt.get("status") == "completed_verified", json.dumps({"receipt": str(state / "host_lifecycle/operations/active-host-uninstall/receipt.json"), "preserved_ok": preserved_ok}, sort_keys=True), "inspect preserved resources"))
            status_proc = _run([sys.executable, str(ROOT / "scripts/host_lifecycle_runner.py"), "status", "--operation-record", str(uninstall_record)], timeout=30)
            checks.append(_check("post-uninstall status works without API", status_proc.returncode == 0 and "completed_verified" in status_proc.stdout, status_proc.stdout, "host_lifecycle_runner.py status"))

            # Reinstall sanity: restore A service and verify preserved state/backup.
            releases.mkdir(parents=True, exist_ok=True)
            _copy_release(releases / "release-a-reinstall", "active-host-A")
            current.symlink_to(releases / "release-a-reinstall")
            service_path = _write_service(root, current, state, logs, "active-host-A", alt_port)
            _run(["systemctl", "--user", "daemon-reload"], timeout=20)
            _run(["systemctl", "--user", "start", SERVICE], timeout=30)
            reinstall_ready = _wait_ready(alt_url, "active-host-A", timeout=60)
            final_backup_manifest = state / "backups" / "personal-agent-uninstall-backup-active-host-uninstall" / "manifest.json"
            checks.append(_check("alternate reinstall sanity works", reinstall_ready["version"].get("git_commit") == "active-host-A" and final_backup_manifest.is_file() and (state / "secrets.enc.json").is_file(), json.dumps(reinstall_ready["version"], sort_keys=True), "restart alternate Release A after uninstall"))

            primary_after = _primary_snapshot()
            checks.append(_check("primary remains unchanged after active-host proof", primary_after.get("ready") is True and primary_after.get("git_commit") == primary_before.get("git_commit") and primary_after.get("service") == primary_before.get("service"), json.dumps({"before": primary_before, "after": primary_after}, sort_keys=True), "primary final snapshot"))
            after_status = _git_status_short()
            checks.append(_check("git status unchanged except intended source changes", after_status == before_status, f"before={before_status!r}\nafter={after_status!r}", "git status --short"))
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("active-host proof exception", False, f"{exc.__class__.__name__}: {exc}; retained_root={retained_root}", "active_host_enablement_smoke"))
        finally:
            cleanup = _cleanup(root, service_path)
            checks.append(_check("alternate cleanup attempted", "proof_root_removed" in cleanup, cleanup, "cleanup alternate profile"))

    passed = sum(1 for check in checks if check.status == "PASS")
    warned = sum(1 for check in checks if check.status == "WARN")
    failed = sum(1 for check in checks if check.status == "FAIL")
    skipped = sum(1 for check in checks if check.status == "SKIP")
    print("# Personal Agent Active-Host Enablement Smoke")
    for check in checks:
        print(f"## {check.name}: {check.status}")
        print(f"command: {check.command}")
        print(f"evidence: {check.evidence}\n")
    print(f"SUMMARY PASS={passed} WARN={warned} FAIL={failed} SKIP={skipped}")
    print("ACTIVE_HOST_ENABLEMENT_SMOKE: " + ("pass" if failed == 0 else "fail"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
