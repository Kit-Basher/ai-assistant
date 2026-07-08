from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request
import uuid


HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION = "host_lifecycle_operation.v1"
HOST_LIFECYCLE_RUNNER_VERSION = "host_lifecycle_runner.v1"
HOST_LIFECYCLE_ALLOWED_STAGES = {
    "created",
    "validated",
    "checkpoint_ready",
    "staging",
    "ready_for_handoff",
    "stopping_services",
    "promoting",
    "removing_runtime",
    "starting_services",
    "verifying",
    "rolling_back",
    "finalizing",
    "completed",
    "completed_with_warnings",
    "failed_before_mutation",
    "failed_after_mutation",
    "rollback_completed",
    "rollback_failed",
    "partial",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def approved_payload_hash(record: dict[str, Any]) -> str:
    approved = dict(record)
    approved.pop("approved_hash", None)
    approved.pop("current_stage", None)
    approved.pop("updated_at", None)
    approved.pop("last_error", None)
    return canonical_hash(approved)


def attach_approved_hash(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload["approved_hash"] = approved_payload_hash(payload)
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    os.chmod(temp, mode)
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("operation_record_not_object")
    return parsed


def safe_path_label(path: Any) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    home = str(Path.home())
    if raw == home:
        return "~"
    if raw.startswith(home + "/"):
        return "~/" + raw[len(home) + 1 :]
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if len(parts) <= 5:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{'/'.join(parts[:2])}/.../{parts[-1]}#sha256:{digest}"


def _is_contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _release_commit(path: Path) -> str:
    build = path / "agent" / "BUILD_INFO.json"
    try:
        payload = json.loads(build.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("git_commit") or "")


def _replace_symlink(link: Path, target: Path) -> None:
    temp = link.with_name(f"{link.name}.tmp-{uuid.uuid4().hex[:8]}")
    if temp.exists() or temp.is_symlink():
        temp.unlink()
    temp.symlink_to(target)
    os.replace(temp, link)


def _remove_target(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _bounded_run(args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "args": [str(item) for item in args[:6]],
        "returncode": proc.returncode,
        "output": (proc.stdout or "")[:2000],
    }


def _validate_proof_service_name(name: str) -> str:
    service = str(name or "").strip()
    if not service.startswith("personal-agent-active-host-proof-") or not service.endswith(".service"):
        raise ValueError("host_lifecycle_service_not_allowlisted")
    if any(ch in service for ch in {"/", "\\", "\x00", " ", "\t", "\n", ";", "&", "|"}):
        raise ValueError("host_lifecycle_service_invalid")
    return service


def _systemctl_user(action: str, service: str, *, timeout: int = 30) -> dict[str, Any]:
    if action not in {"start", "stop", "restart", "disable", "reset-failed", "daemon-reload"}:
        raise ValueError("host_lifecycle_systemctl_action_rejected")
    if action == "daemon-reload":
        return _bounded_run(["systemctl", "--user", "daemon-reload"], timeout=timeout)
    service = _validate_proof_service_name(service)
    return _bounded_run(["systemctl", "--user", action, service], timeout=timeout)


def _http_json(url: str, *, timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read(256 * 1024).decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise ValueError("host_lifecycle_http_non_object")
    return payload


def _wait_http_ready(base_url: str, *, expected_commit: str | None = None, timeout: float = 30.0) -> dict[str, Any]:
    base = str(base_url or "").rstrip("/")
    if not base.startswith("http://127.0.0.1:"):
        raise ValueError("host_lifecycle_verify_url_not_loopback")
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            ready = _http_json(f"{base}/ready", timeout=2.0)
            version = _http_json(f"{base}/version", timeout=2.0)
            commit = str(version.get("git_commit") or "")
            if ready.get("ready") is True and (not expected_commit or commit == expected_commit):
                return {"ready": ready, "version": version}
            last_error = f"ready={ready.get('ready')} commit={commit}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
        time.sleep(0.5)
    raise TimeoutError(f"host_lifecycle_http_verify_timeout:{last_error[:500]}")


def validate_operation_record_path(path: Path) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ValueError("operation_record_symlink_rejected")
    resolved = raw.resolve()
    allowed_roots = [
        Path("/tmp").resolve(),
        (Path.home() / ".local/share/personal-agent/host_lifecycle").resolve(),
        (Path.home() / ".local/share/personal-agent").resolve(),
    ]
    if not any(_is_contained(resolved, root) for root in allowed_roots):
        raise ValueError("operation_record_outside_approved_root")
    return resolved


def load_and_validate_operation(path: str | Path, *, expected_type: str | None = None) -> tuple[Path, dict[str, Any]]:
    record_path = validate_operation_record_path(Path(path))
    record = read_json(record_path)
    if record.get("schema_version") != HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION:
        raise ValueError("unsupported_host_lifecycle_schema")
    if record.get("runner_version") != HOST_LIFECYCLE_RUNNER_VERSION:
        raise ValueError("unsupported_host_lifecycle_runner_version")
    operation_type = str(record.get("operation_type") or "").strip().lower()
    if operation_type not in {"update", "uninstall"}:
        raise ValueError("unknown_host_lifecycle_operation")
    if expected_type and operation_type != expected_type:
        raise ValueError("operation_type_mismatch")
    if "command" in record or "shell" in record or "argv" in record:
        raise ValueError("arbitrary_command_field_rejected")
    approved_hash = str(record.get("approved_hash") or "")
    if not approved_hash or approved_hash != approved_payload_hash(record):
        raise ValueError("operation_record_tampered")
    expires_at = str(record.get("expires_at") or "")
    if expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expiry:
                raise ValueError("operation_record_expired")
        except ValueError as exc:
            if str(exc) == "operation_record_expired":
                raise
            raise ValueError("operation_expiry_invalid") from exc
    return record_path, record


def _stage(record: dict[str, Any], stage: str, extra: dict[str, Any] | None = None) -> None:
    if stage not in HOST_LIFECYCLE_ALLOWED_STAGES:
        raise ValueError(f"unknown_stage:{stage}")
    state_path = Path(str(record.get("operation_state_path") or "")).expanduser().resolve()
    payload = {
        "schema_version": HOST_LIFECYCLE_OPERATION_SCHEMA_VERSION,
        "runner_version": HOST_LIFECYCLE_RUNNER_VERSION,
        "operation_id": record.get("operation_id"),
        "operation_type": record.get("operation_type"),
        "current_stage": stage,
        "updated_at": utc_now_iso(),
        **(extra or {}),
    }
    write_json_atomic(state_path, payload)


def _receipt(record: dict[str, Any], payload: dict[str, Any]) -> None:
    receipt_path = Path(str(record.get("receipt_path") or "")).expanduser().resolve()
    write_json_atomic(receipt_path, payload)


def _run_update(record: dict[str, Any]) -> dict[str, Any]:
    fixture_mode = str(record.get("fixture_mode") or "")
    if fixture_mode not in {"strict", "active_host_proof"}:
        raise ValueError("update_live_host_handoff_not_enabled")
    if fixture_mode == "active_host_proof":
        marker = Path(str(record.get("proof_marker_path") or "")).expanduser().resolve()
        runtime_root_for_marker = Path(str(record.get("runtime_root") or "")).expanduser().resolve()
        if not marker.is_file() or not _is_contained(marker, runtime_root_for_marker.parent):
            raise ValueError("active_host_proof_marker_missing")
    runtime_root = Path(str(record["runtime_root"])).expanduser().resolve()
    releases_root = Path(str(record["releases_root"])).expanduser().resolve()
    current_link = Path(str(record["current_link"])).expanduser()
    current_link = current_link.parent.resolve() / current_link.name
    source_release = Path(str(record["staged_source_path"])).expanduser().resolve()
    state_root = Path(str(record["state_root"])).expanduser().resolve()
    target_release_id = str(record["target_release_id"])
    target_commit = str(record["target_commit"])
    expected_current_commit = str(record["current_runtime_commit"])
    operation_id = str(record["operation_id"])
    service_name = str(record.get("api_service_name") or "").strip()
    verify_base_url = str(record.get("verify_base_url") or "").strip()
    if service_name:
        _validate_proof_service_name(service_name)
    if not _is_contained(releases_root, runtime_root) or not _is_contained(current_link.parent, runtime_root):
        raise ValueError("update_path_escape")
    if not source_release.is_dir() or source_release.is_symlink():
        raise ValueError("update_staged_source_missing")
    if not current_link.is_symlink():
        raise ValueError("update_current_not_symlink")

    _stage(record, "validated")
    previous_target = current_link.resolve()
    previous_commit = _release_commit(previous_target)
    if expected_current_commit and previous_commit != expected_current_commit and previous_commit == target_commit:
        checkpoint_dir = state_root / "update_checkpoints" / operation_id
        checkpoint = read_json(checkpoint_dir / "manifest.json") if (checkpoint_dir / "manifest.json").is_file() else {}
        _stage(record, "verifying", {"resume": "target_already_promoted", "checkpoint": checkpoint})
        service_restart: dict[str, Any] | None = None
        if service_name:
            _stage(record, "starting_services", {"resume": "target_already_promoted", "service": service_name})
            service_restart = _systemctl_user("restart", service_name, timeout=45)
            restart_code = service_restart.get("returncode")
            if int(restart_code if restart_code is not None else 1) != 0:
                raise RuntimeError("update_resume_service_restart_failed")
        http_verification = None
        if verify_base_url:
            http_verification = _wait_http_ready(verify_base_url, expected_commit=target_commit, timeout=45.0)
        result = {
            "ok": True,
            "mutated": False,
            "status": "completed_verified",
            "operation_id": operation_id,
            "operation_type": "update",
            "previous_runtime_commit": expected_current_commit,
            "target_commit": target_commit,
            "promoted_release": str(previous_target),
            "verified_commit": target_commit,
            "service_restart": service_restart,
            "http_verification": http_verification,
            "checkpoint": checkpoint,
            "resume": "target_already_promoted",
            "finished_at": utc_now_iso(),
        }
        _stage(record, "completed", result)
        _receipt(record, result)
        return result
    if expected_current_commit and previous_commit != expected_current_commit:
        raise ValueError("update_current_changed_since_preview")
    if _release_commit(source_release) != target_commit:
        raise ValueError("update_target_metadata_mismatch")
    checkpoint_dir = state_root / "update_checkpoints" / operation_id
    checkpoint = {
        "operation_id": operation_id,
        "created_at": utc_now_iso(),
        "previous_release_path": str(previous_target),
        "previous_runtime_commit": previous_commit,
        "current_link": str(current_link),
        "target_commit": target_commit,
    }
    write_json_atomic(checkpoint_dir / "manifest.json", checkpoint)
    _stage(record, "checkpoint_ready", {"checkpoint": checkpoint})

    final_release = releases_root / target_release_id
    stage_release = releases_root / f"{target_release_id}.staging-{uuid.uuid4().hex[:8]}"
    if not final_release.exists():
        _stage(record, "staging", {"staged_release": str(final_release)})
        shutil.copytree(source_release, stage_release, symlinks=False)
        if _release_commit(stage_release) != target_commit:
            shutil.rmtree(stage_release, ignore_errors=True)
            raise ValueError("update_pre_promotion_check_failed")
        os.replace(stage_release, final_release)
    elif _release_commit(final_release) != target_commit:
        raise ValueError("update_existing_release_mismatch")

    _stage(record, "promoting", {"promoted_release": str(final_release), "checkpoint": checkpoint})
    _replace_symlink(current_link, final_release)
    if bool(record.get("interrupt_once_after_promotion")):
        marker_path = Path(str(record.get("interrupt_marker_path") or "")).expanduser().resolve()
        if not marker_path.is_file():
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(utc_now_iso() + "\n", encoding="utf-8")
            interrupted = {
                "ok": False,
                "mutated": True,
                "status": "interrupted_after_promotion",
                "operation_id": operation_id,
                "operation_type": "update",
                "previous_runtime_commit": previous_commit,
                "target_commit": target_commit,
                "promoted_release": str(final_release),
                "checkpoint": checkpoint,
                "finished_at": utc_now_iso(),
            }
            _stage(record, "failed_after_mutation", interrupted)
            _receipt(record, interrupted)
            return interrupted
    service_restart: dict[str, Any] | None = None
    if service_name:
        _stage(record, "starting_services", {"service": service_name, "promoted_release": str(final_release), "checkpoint": checkpoint})
        service_restart = _systemctl_user("restart", service_name, timeout=45)
        restart_code = service_restart.get("returncode")
        if int(restart_code if restart_code is not None else 1) != 0:
            _stage(record, "rolling_back", {"service_restart": service_restart, "checkpoint": checkpoint})
            _replace_symlink(current_link, previous_target)
            _systemctl_user("restart", service_name, timeout=45)
            rollback_commit = _release_commit(current_link.resolve())
            rollback_verified = rollback_commit == previous_commit
            if verify_base_url:
                try:
                    _wait_http_ready(verify_base_url, expected_commit=previous_commit, timeout=30.0)
                except Exception:
                    rollback_verified = False
            result = {
                "ok": False,
                "mutated": True,
                "status": "update_failed_rolled_back" if rollback_verified else "update_failed_rollback_failed",
                "operation_id": operation_id,
                "operation_type": "update",
                "previous_runtime_commit": previous_commit,
                "target_commit": target_commit,
                "rollback_commit": rollback_commit,
                "rollback_verified": rollback_verified,
                "service_restart": service_restart,
                "checkpoint": checkpoint,
                "finished_at": utc_now_iso(),
            }
            _stage(record, "rollback_completed" if rollback_verified else "rollback_failed", result)
            _receipt(record, result)
            return result
    _stage(record, "verifying", {"promoted_release": str(final_release), "checkpoint": checkpoint})
    verified_commit = _release_commit(current_link.resolve())
    http_verification: dict[str, Any] | None = None
    http_failed = False
    if verify_base_url and verified_commit == target_commit and not bool(record.get("force_post_promotion_failure")):
        try:
            http_verification = _wait_http_ready(verify_base_url, expected_commit=target_commit, timeout=45.0)
        except Exception as exc:  # noqa: BLE001
            http_failed = True
            http_verification = {"error": exc.__class__.__name__, "summary": str(exc)[:500]}
    if bool(record.get("force_post_promotion_failure")) or verified_commit != target_commit or http_failed:
        _stage(record, "rolling_back", {"verified_commit": verified_commit, "checkpoint": checkpoint})
        _replace_symlink(current_link, previous_target)
        if service_name:
            _systemctl_user("restart", service_name, timeout=45)
        rollback_commit = _release_commit(current_link.resolve())
        rollback_verified = rollback_commit == previous_commit
        if verify_base_url:
            try:
                _wait_http_ready(verify_base_url, expected_commit=previous_commit, timeout=45.0)
            except Exception:
                rollback_verified = False
        status = "rollback_completed" if rollback_verified else "rollback_failed"
        result = {
            "ok": False,
            "mutated": True,
            "status": "update_failed_rolled_back" if rollback_verified else "update_failed_rollback_failed",
            "operation_id": operation_id,
            "operation_type": "update",
            "previous_runtime_commit": previous_commit,
            "target_commit": target_commit,
            "verified_commit": verified_commit,
            "rollback_commit": rollback_commit,
            "rollback_verified": rollback_verified,
            "http_verification": http_verification,
            "checkpoint": checkpoint,
            "finished_at": utc_now_iso(),
        }
        _stage(record, status, result)
        _receipt(record, result)
        return result

    result = {
        "ok": True,
        "mutated": True,
        "status": "completed_verified",
        "operation_id": operation_id,
        "operation_type": "update",
        "previous_runtime_commit": previous_commit,
        "target_commit": target_commit,
        "promoted_release": str(final_release),
        "verified_commit": verified_commit,
        "service_restart": service_restart,
        "http_verification": http_verification,
        "checkpoint": checkpoint,
        "finished_at": utc_now_iso(),
    }
    _stage(record, "completed", result)
    _receipt(record, result)
    return result


def _run_uninstall(record: dict[str, Any]) -> dict[str, Any]:
    fixture_mode = str(record.get("fixture_mode") or "")
    if fixture_mode not in {"strict", "active_host_proof"}:
        raise ValueError("uninstall_live_host_handoff_not_enabled")
    snapshot = record.get("target_snapshot") if isinstance(record.get("target_snapshot"), dict) else {}
    if not snapshot.get("fixture_marker") or snapshot.get("mode") != "preserve_data":
        raise ValueError("uninstall_fixture_marker_missing")
    fixture_root = Path(str(snapshot.get("fixture_root") or "")).expanduser().resolve()
    if fixture_mode == "active_host_proof":
        marker = Path(str(record.get("proof_marker_path") or "")).expanduser().resolve()
        if not marker.is_file() or not _is_contained(marker, fixture_root):
            raise ValueError("active_host_proof_marker_missing")
    final_backup_path = Path(str(record.get("final_backup_path") or "")).expanduser().resolve()
    if not _is_contained(final_backup_path, fixture_root) or not (final_backup_path / "manifest.json").is_file():
        raise ValueError("uninstall_final_backup_invalid")
    resources = snapshot.get("removable_resources") if isinstance(snapshot.get("removable_resources"), list) else []
    preserved = snapshot.get("preserved_resources") if isinstance(snapshot.get("preserved_resources"), list) else []
    removable_roots = [Path(str(item)).expanduser().resolve() for item in snapshot.get("removable_roots", []) if str(item)]
    if not removable_roots:
        raise ValueError("uninstall_no_removable_roots")

    _stage(record, "validated")
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    service_results: list[dict[str, Any]] = []
    services = [str(item).strip() for item in record.get("service_names", []) if str(item).strip()] if isinstance(record.get("service_names"), list) else []
    if services:
        _stage(record, "stopping_services", {"services": services})
        for service in services:
            _validate_proof_service_name(service)
            stop_result = _systemctl_user("stop", service, timeout=45)
            disable_result = _systemctl_user("disable", service, timeout=45)
            service_results.append({"service": service, "stop": stop_result, "disable": disable_result})
        service_results.append({"daemon_reload_before_removal": _systemctl_user("daemon-reload", "", timeout=30)})
    _stage(record, "removing_runtime")
    force_failure_after = str(record.get("force_failure_after_resource_id") or "")
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        resource_id = str(resource.get("id") or resource.get("path") or "unknown")
        path = Path(str(resource.get("path") or "")).expanduser()
        if not path.is_absolute():
            path = path.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        root_ok = any(_is_contained(resolved_path, root) or path == root for root in removable_roots)
        service_unit_ok = (
            fixture_mode == "active_host_proof"
            and resolved_path.parent == (Path.home() / ".config/systemd/user").resolve()
            and resolved_path.name.startswith("personal-agent-active-host-proof-")
            and resolved_path.name.endswith(".service")
        )
        if not root_ok or (not _is_contained(resolved_path, fixture_root) and not service_unit_ok):
            failures.append({"id": resource_id, "path": str(path), "error": "uninstall_path_escape"})
            break
        if not path.exists() and not path.is_symlink():
            skipped.append({"id": resource_id, "path": str(path), "reason": "already_absent"})
            continue
        try:
            _remove_target(path)
            removed.append({"id": resource_id, "path": str(path), "class": resource.get("class")})
            if force_failure_after and resource_id == force_failure_after:
                raise RuntimeError("forced_uninstall_partial_failure")
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": resource_id, "path": str(path), "error": exc.__class__.__name__})
            break

    _stage(record, "verifying")
    preserved_checked: list[dict[str, Any]] = []
    for resource in preserved:
        if not isinstance(resource, dict):
            continue
        raw = str(resource.get("path") or "")
        path = Path(raw).expanduser()
        preserved_checked.append({"id": str(resource.get("id") or raw), "path": str(path), "exists": path.exists() or path.is_symlink()})
    status = "partial" if failures else "completed"
    if services:
        service_results.append({"daemon_reload_after_removal": _systemctl_user("daemon-reload", "", timeout=30)})
    result = {
        "ok": not failures,
        "mutated": bool(removed),
        "status": "partial_uninstall" if failures else "completed_verified",
        "operation_id": record.get("operation_id"),
        "operation_type": "uninstall",
        "mode": "preserve_data",
        "removed_resources": removed,
        "skipped_resources": skipped,
        "failed_resources": failures,
        "service_results": service_results,
        "preserved_resources": preserved,
        "preserved_checked": preserved_checked,
        "final_backup_path": str(final_backup_path),
        "finished_at": utc_now_iso(),
        "reinstall_guidance": "Reinstall from the preserved Personal Agent repository, then restore supported state from the final uninstall backup.",
    }
    _stage(record, status, result)
    _receipt(record, result)
    return result


def run_operation_file(path: str | Path, *, expected_type: str | None = None) -> dict[str, Any]:
    _, record = load_and_validate_operation(path, expected_type=expected_type)
    operation_type = str(record.get("operation_type"))
    if operation_type == "update":
        return _run_update(record)
    if operation_type == "uninstall":
        return _run_uninstall(record)
    raise ValueError("unknown_host_lifecycle_operation")


def status_operation_file(path: str | Path, *, expected_type: str | None = None) -> dict[str, Any]:
    _, record = load_and_validate_operation(path, expected_type=expected_type)
    state_path = Path(str(record.get("operation_state_path") or "")).expanduser()
    receipt_path = Path(str(record.get("receipt_path") or "")).expanduser()
    state = read_json(state_path) if state_path.is_file() else {}
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    return {
        "ok": True,
        "mutated": False,
        "status": str(state.get("current_stage") or receipt.get("status") or record.get("current_stage") or "created"),
        "operation_id": record.get("operation_id"),
        "operation_type": record.get("operation_type"),
        "state_path": str(state_path),
        "receipt_path": str(receipt_path),
        "has_state": bool(state),
        "has_receipt": bool(receipt),
        "receipt_status": receipt.get("status"),
    }
