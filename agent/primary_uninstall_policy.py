from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
from typing import Any
import uuid


PRIMARY_UNINSTALL_SCHEMA_VERSION = 1
PRIMARY_UNINSTALL_CAPABILITY = "primary_preserve_data_uninstall"
PRIMARY_UNINSTALL_MARKER_NAME = "primary_uninstall_enabled.json"
PRIMARY_UNINSTALL_IDENTITY_NAME = "installation_identity.json"
PRIMARY_UNINSTALL_DEFAULT_DAYS = 30
PRIMARY_UNINSTALL_MAX_DAYS = 90
PRIMARY_UNINSTALL_MAX_BYTES = 16 * 1024
PRIMARY_UNINSTALL_SERVICE = "personal-agent-api.service"
PRIMARY_UNINSTALL_CREATED_BY = "local_operator_cli"
PRIMARY_UNINSTALL_ALLOWED_KEYS = {
    "schema_version",
    "capability",
    "enabled",
    "installation_id",
    "repository_path",
    "primary_service",
    "created_at",
    "expires_at",
    "created_by",
    "policy",
    "nonce",
    "uid",
    "host_id",
    "integrity",
}
PRIMARY_UNINSTALL_ALLOWED_POLICY_KEYS = {"mode", "purge_allowed"}
PRIMARY_UNINSTALL_ALLOWED_INTEGRITY_KEYS = {"algorithm", "payload_sha256"}


@dataclass(frozen=True)
class PrimaryUninstallPolicyContext:
    repository_path: Path
    state_root: Path
    host_lifecycle_root: Path
    marker_path: Path
    identity_path: Path
    installation_id: str
    uid: int
    host_id: str
    primary_service: str = PRIMARY_UNINSTALL_SERVICE


@dataclass(frozen=True)
class PrimaryUninstallPolicyStatus:
    enabled: bool
    reason: str
    marker_path: str
    schema_version: int | None = None
    expires_at: str | None = None
    installation_id_matches: bool = False
    repository_path_matches: bool = False
    service_matches: bool = False
    permissions_ok: bool = False
    policy_mode: str | None = None
    purge_allowed: bool = False
    fingerprint: str | None = None
    details: dict[str, Any] | None = None

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "marker_path": self.marker_path,
            "schema_version": self.schema_version,
            "expires_at": self.expires_at,
            "installation_id_matches": self.installation_id_matches,
            "repository_path_matches": self.repository_path_matches,
            "service_matches": self.service_matches,
            "permissions_ok": self.permissions_ok,
            "policy_mode": self.policy_mode,
            "purge_allowed": self.purge_allowed,
            "fingerprint": self.fingerprint,
            "details": self.details or {},
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(raw: Any) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("timestamp_missing")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp_timezone_missing")
    return parsed.astimezone(timezone.utc)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1].resolve()


def default_state_root() -> Path:
    return (Path.home() / ".local/share/personal-agent").resolve()


def _read_machine_id() -> str:
    for raw in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            text = Path(raw).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashlib.sha256(socket.gethostname().encode("utf-8", errors="replace")).hexdigest()


def _reject_duplicate_json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate_json_key:{key}")
        out[key] = value
    return out


def _json_loads_strict(text: str) -> dict[str, Any]:
    parsed = json.loads(text, object_pairs_hook=_reject_duplicate_json_object_pairs)
    if not isinstance(parsed, dict):
        raise ValueError("marker_not_object")
    return parsed


def canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    clean.pop("integrity", None)
    return clean


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(canonical_payload(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def marker_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    temp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temp, flags, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temp.exists() or temp.is_symlink():
            try:
                temp.unlink()
            except OSError:
                pass


def _read_identity(path: Path) -> str | None:
    try:
        st = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return None
    try:
        payload = _json_loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    installation_id = str(payload.get("installation_id") or "").strip()
    if installation_id:
        return installation_id
    return None


def ensure_installation_identity(context: PrimaryUninstallPolicyContext | None = None) -> str:
    ctx = context or build_policy_context(create_identity=False)
    existing = _read_identity(ctx.identity_path)
    if existing:
        return existing
    payload = {
        "schema_version": 1,
        "installation_id": str(uuid.uuid4()),
        "created_at": iso_utc(utc_now()),
        "repository_path": str(ctx.repository_path),
        "uid": ctx.uid,
        "host_id": ctx.host_id,
    }
    _write_json_atomic(ctx.identity_path, payload)
    return str(payload["installation_id"])


def build_policy_context(
    *,
    state_root: Path | None = None,
    repository_path: Path | None = None,
    create_identity: bool = False,
    installation_id: str | None = None,
) -> PrimaryUninstallPolicyContext:
    repo = (repository_path or repository_root()).expanduser().resolve()
    root = (state_root or default_state_root()).expanduser().resolve()
    host_lifecycle = (root / "host_lifecycle").resolve()
    marker = host_lifecycle / PRIMARY_UNINSTALL_MARKER_NAME
    identity = host_lifecycle / PRIMARY_UNINSTALL_IDENTITY_NAME
    host_id = _read_machine_id()
    uid = os.getuid()
    resolved_installation_id = str(installation_id or "").strip()
    if not resolved_installation_id:
        resolved_installation_id = _read_identity(identity) or ""
    ctx = PrimaryUninstallPolicyContext(
        repository_path=repo,
        state_root=root,
        host_lifecycle_root=host_lifecycle,
        marker_path=marker,
        identity_path=identity,
        installation_id=resolved_installation_id,
        uid=uid,
        host_id=host_id,
    )
    if create_identity and not ctx.installation_id:
        resolved_installation_id = ensure_installation_identity(ctx)
        ctx = PrimaryUninstallPolicyContext(
            repository_path=repo,
            state_root=root,
            host_lifecycle_root=host_lifecycle,
            marker_path=marker,
            identity_path=identity,
            installation_id=resolved_installation_id,
            uid=uid,
            host_id=host_id,
        )
    return ctx


def _fail(ctx: PrimaryUninstallPolicyContext, reason: str, **details: Any) -> PrimaryUninstallPolicyStatus:
    return PrimaryUninstallPolicyStatus(
        enabled=False,
        reason=reason,
        marker_path=str(ctx.marker_path),
        details={k: v for k, v in details.items() if v is not None},
    )


def _validate_filesystem(ctx: PrimaryUninstallPolicyContext) -> tuple[bool, str, dict[str, Any]]:
    marker = ctx.marker_path
    approved_root = ctx.host_lifecycle_root
    try:
        parent_resolved = marker.parent.resolve(strict=False)
        marker_resolved_parent = marker.resolve(strict=False).parent
    except OSError:
        return False, "marker_path_unresolvable", {}
    if parent_resolved != approved_root or marker_resolved_parent != approved_root:
        return False, "marker_outside_approved_host_lifecycle_root", {"approved_root": str(approved_root)}
    try:
        root_st = os.lstat(approved_root)
    except OSError:
        return False, "host_lifecycle_root_missing", {}
    if stat.S_ISLNK(root_st.st_mode) or not stat.S_ISDIR(root_st.st_mode):
        return False, "host_lifecycle_root_not_directory", {}
    if root_st.st_uid != ctx.uid:
        return False, "host_lifecycle_root_wrong_owner", {"owner_uid": root_st.st_uid, "expected_uid": ctx.uid}
    if root_st.st_mode & 0o077:
        return False, "host_lifecycle_root_permissions_too_broad", {"mode": oct(stat.S_IMODE(root_st.st_mode))}
    try:
        st = os.lstat(marker)
    except OSError:
        return False, "marker_missing", {}
    if stat.S_ISLNK(st.st_mode):
        return False, "marker_symlink_rejected", {}
    if not stat.S_ISREG(st.st_mode):
        return False, "marker_not_regular_file", {}
    if st.st_uid != ctx.uid:
        return False, "marker_wrong_owner", {"owner_uid": st.st_uid, "expected_uid": ctx.uid}
    if st.st_mode & 0o077:
        return False, "marker_permissions_too_broad", {"mode": oct(stat.S_IMODE(st.st_mode))}
    if getattr(st, "st_nlink", 1) != 1:
        return False, "marker_hardlink_rejected", {"nlink": getattr(st, "st_nlink", 1)}
    if st.st_size > PRIMARY_UNINSTALL_MAX_BYTES:
        return False, "marker_oversized", {"size": st.st_size, "max_size": PRIMARY_UNINSTALL_MAX_BYTES}
    return True, "ok", {"mode": oct(stat.S_IMODE(st.st_mode)), "size": st.st_size}


def validate_primary_uninstall_marker(
    context: PrimaryUninstallPolicyContext | None = None,
    *,
    expected_fingerprint: str | None = None,
    now: datetime | None = None,
) -> PrimaryUninstallPolicyStatus:
    ctx = context or build_policy_context()
    fs_ok, fs_reason, fs_details = _validate_filesystem(ctx)
    if not fs_ok:
        return _fail(ctx, fs_reason, filesystem=fs_details)
    try:
        raw = ctx.marker_path.read_bytes()
    except OSError as exc:
        return _fail(ctx, "marker_unreadable", exception=exc.__class__.__name__)
    try:
        text = raw.decode("utf-8")
        payload = _json_loads_strict(text)
    except UnicodeDecodeError:
        return _fail(ctx, "marker_not_utf8")
    except json.JSONDecodeError:
        return _fail(ctx, "marker_malformed_json")
    except ValueError as exc:
        return _fail(ctx, str(exc))

    unknown = sorted(set(payload) - PRIMARY_UNINSTALL_ALLOWED_KEYS)
    if unknown:
        return _fail(ctx, "marker_unknown_fields", fields=unknown)
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        return _fail(ctx, "marker_policy_missing")
    unknown_policy = sorted(set(policy) - PRIMARY_UNINSTALL_ALLOWED_POLICY_KEYS)
    if unknown_policy:
        return _fail(ctx, "marker_policy_unknown_fields", fields=unknown_policy)
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict):
        return _fail(ctx, "marker_integrity_missing")
    unknown_integrity = sorted(set(integrity) - PRIMARY_UNINSTALL_ALLOWED_INTEGRITY_KEYS)
    if unknown_integrity:
        return _fail(ctx, "marker_integrity_unknown_fields", fields=unknown_integrity)

    fingerprint = marker_fingerprint(payload)
    if expected_fingerprint and expected_fingerprint != fingerprint:
        return _fail(ctx, "marker_changed_since_preview", expected_fingerprint=expected_fingerprint, actual_fingerprint=fingerprint)
    schema_version = payload.get("schema_version")
    if schema_version != PRIMARY_UNINSTALL_SCHEMA_VERSION:
        return _fail(ctx, "marker_schema_version_unsupported", schema_version=schema_version)
    if payload.get("capability") != PRIMARY_UNINSTALL_CAPABILITY:
        return _fail(ctx, "marker_capability_mismatch", schema_version=schema_version)
    if payload.get("enabled") is not True:
        return _fail(ctx, "marker_not_enabled", schema_version=schema_version)
    if payload.get("created_by") != PRIMARY_UNINSTALL_CREATED_BY:
        return _fail(ctx, "marker_created_by_mismatch", schema_version=schema_version)
    if not str(payload.get("nonce") or "").strip():
        return _fail(ctx, "marker_nonce_missing", schema_version=schema_version)
    if integrity.get("algorithm") != "sha256":
        return _fail(ctx, "marker_integrity_algorithm_unsupported", schema_version=schema_version)
    actual_payload_hash = payload_sha256(payload)
    if integrity.get("payload_sha256") != actual_payload_hash:
        return _fail(ctx, "marker_integrity_mismatch", schema_version=schema_version)
    if payload.get("installation_id") != ctx.installation_id or not ctx.installation_id:
        return _fail(ctx, "marker_installation_id_mismatch", schema_version=schema_version)
    if payload.get("repository_path") != str(ctx.repository_path):
        return _fail(ctx, "marker_repository_path_mismatch", schema_version=schema_version)
    if payload.get("primary_service") != ctx.primary_service:
        return _fail(ctx, "marker_primary_service_mismatch", schema_version=schema_version)
    if payload.get("uid") != ctx.uid:
        return _fail(ctx, "marker_uid_mismatch", schema_version=schema_version)
    if payload.get("host_id") != ctx.host_id:
        return _fail(ctx, "marker_host_id_mismatch", schema_version=schema_version)
    if policy.get("mode") != "preserve_data":
        return _fail(ctx, "marker_policy_mode_not_preserve_data", schema_version=schema_version)
    if policy.get("purge_allowed") is not False:
        return _fail(ctx, "marker_policy_purge_allowed", schema_version=schema_version)
    try:
        created = parse_utc_timestamp(payload.get("created_at"))
        expires = parse_utc_timestamp(payload.get("expires_at"))
    except (TypeError, ValueError) as exc:
        return _fail(ctx, f"marker_timestamp_invalid:{exc}", schema_version=schema_version)
    current = (now or utc_now()).astimezone(timezone.utc)
    if created > current + timedelta(minutes=5):
        return _fail(ctx, "marker_created_at_in_future", schema_version=schema_version, created_at=iso_utc(created))
    if expires <= created:
        return _fail(ctx, "marker_expiry_not_after_created", schema_version=schema_version)
    if expires - created > timedelta(days=PRIMARY_UNINSTALL_MAX_DAYS, minutes=1):
        return _fail(ctx, "marker_expiry_exceeds_maximum", schema_version=schema_version)
    if current >= expires:
        return _fail(ctx, "marker_expired", schema_version=schema_version, expires_at=iso_utc(expires))

    return PrimaryUninstallPolicyStatus(
        enabled=True,
        reason="enabled",
        marker_path=str(ctx.marker_path),
        schema_version=PRIMARY_UNINSTALL_SCHEMA_VERSION,
        expires_at=iso_utc(expires),
        installation_id_matches=True,
        repository_path_matches=True,
        service_matches=True,
        permissions_ok=True,
        policy_mode="preserve_data",
        purge_allowed=False,
        fingerprint=fingerprint,
        details={"filesystem": fs_details},
    )


def build_primary_uninstall_marker_payload(ctx: PrimaryUninstallPolicyContext, *, expires_in_days: int) -> dict[str, Any]:
    if expires_in_days < 1:
        raise ValueError("expiry_must_be_at_least_one_day")
    if expires_in_days > PRIMARY_UNINSTALL_MAX_DAYS:
        raise ValueError("expiry_exceeds_maximum")
    created = utc_now()
    payload: dict[str, Any] = {
        "schema_version": PRIMARY_UNINSTALL_SCHEMA_VERSION,
        "capability": PRIMARY_UNINSTALL_CAPABILITY,
        "enabled": True,
        "installation_id": ctx.installation_id,
        "repository_path": str(ctx.repository_path),
        "primary_service": ctx.primary_service,
        "created_at": iso_utc(created),
        "expires_at": iso_utc(created + timedelta(days=expires_in_days)),
        "created_by": PRIMARY_UNINSTALL_CREATED_BY,
        "policy": {"mode": "preserve_data", "purge_allowed": False},
        "nonce": uuid.uuid4().hex,
        "uid": ctx.uid,
        "host_id": ctx.host_id,
    }
    payload["integrity"] = {"algorithm": "sha256", "payload_sha256": payload_sha256(payload)}
    return payload


def enable_primary_uninstall_marker(
    *,
    expires_in_days: int = PRIMARY_UNINSTALL_DEFAULT_DAYS,
    context: PrimaryUninstallPolicyContext | None = None,
) -> PrimaryUninstallPolicyStatus:
    ctx = context or build_policy_context(create_identity=True)
    payload = build_primary_uninstall_marker_payload(ctx, expires_in_days=expires_in_days)
    _write_json_atomic(ctx.marker_path, payload)
    return validate_primary_uninstall_marker(ctx)


def disable_primary_uninstall_marker(context: PrimaryUninstallPolicyContext | None = None) -> PrimaryUninstallPolicyStatus:
    ctx = context or build_policy_context()
    try:
        st = os.lstat(ctx.marker_path)
    except OSError:
        return _fail(ctx, "marker_missing")
    if stat.S_ISLNK(st.st_mode):
        return _fail(ctx, "marker_symlink_rejected")
    archive = ctx.marker_path.with_name(f"{ctx.marker_path.name}.disabled-{uuid.uuid4().hex[:8]}")
    try:
        os.replace(ctx.marker_path, archive)
        os.chmod(archive, 0o600)
    except OSError as exc:
        return _fail(ctx, "marker_disable_failed", exception=exc.__class__.__name__)
    return _fail(ctx, "marker_missing")


def consume_primary_uninstall_marker(context: PrimaryUninstallPolicyContext | None = None) -> dict[str, Any]:
    status = validate_primary_uninstall_marker(context)
    if not status.enabled:
        return {"consumed": False, "reason": status.reason}
    disabled = disable_primary_uninstall_marker(context)
    return {"consumed": disabled.reason == "marker_missing", "reason": disabled.reason, "fingerprint": status.fingerprint}
