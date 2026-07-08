from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
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
    if str(record.get("fixture_mode") or "") != "strict":
        raise ValueError("update_live_host_handoff_not_enabled")
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
    if not _is_contained(releases_root, runtime_root) or not _is_contained(current_link.parent, runtime_root):
        raise ValueError("update_path_escape")
    if not source_release.is_dir() or source_release.is_symlink():
        raise ValueError("update_staged_source_missing")
    if not current_link.is_symlink():
        raise ValueError("update_current_not_symlink")

    _stage(record, "validated")
    previous_target = current_link.resolve()
    previous_commit = _release_commit(previous_target)
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
    _stage(record, "verifying", {"promoted_release": str(final_release), "checkpoint": checkpoint})
    verified_commit = _release_commit(current_link.resolve())
    if bool(record.get("force_post_promotion_failure")) or verified_commit != target_commit:
        _stage(record, "rolling_back", {"verified_commit": verified_commit, "checkpoint": checkpoint})
        _replace_symlink(current_link, previous_target)
        rollback_commit = _release_commit(current_link.resolve())
        rollback_verified = rollback_commit == previous_commit
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
        "checkpoint": checkpoint,
        "finished_at": utc_now_iso(),
    }
    _stage(record, "completed", result)
    _receipt(record, result)
    return result


def _run_uninstall(record: dict[str, Any]) -> dict[str, Any]:
    if str(record.get("fixture_mode") or "") != "strict":
        raise ValueError("uninstall_live_host_handoff_not_enabled")
    snapshot = record.get("target_snapshot") if isinstance(record.get("target_snapshot"), dict) else {}
    if not snapshot.get("fixture_marker") or snapshot.get("mode") != "preserve_data":
        raise ValueError("uninstall_fixture_marker_missing")
    fixture_root = Path(str(snapshot.get("fixture_root") or "")).expanduser().resolve()
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
    _stage(record, "removing_runtime")
    force_failure_after = str(record.get("force_failure_after_resource_id") or "")
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        resource_id = str(resource.get("id") or resource.get("path") or "unknown")
        path = Path(str(resource.get("path") or "")).expanduser()
        if not path.is_absolute():
            path = path.resolve(strict=False)
        root_ok = any(_is_contained(path.resolve(strict=False), root) or path == root for root in removable_roots)
        if not root_ok or not _is_contained(path.resolve(strict=False), fixture_root):
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
