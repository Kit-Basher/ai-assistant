from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from agent.capability_policy import POLICY_SCHEMA_VERSION, stable_fingerprint
from agent.mutation_plan import (
    MUTATION_PLAN_STATUS_PENDING,
    MutationPlanStore,
    build_mutation_confirmation,
    build_mutation_plan,
    validate_mutation_plan,
)

if TYPE_CHECKING:
    from agent.executor_registry import ExecutorRegistry


SKILL_PERMISSION_SCHEMA_VERSION = 1
SUPPORTED_MANIFEST_SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*$")
_PERMISSION_RE = re.compile(r"^(read|invoke)\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]+)+$")


class SkillPermissionError(ValueError):
    pass


@dataclass(frozen=True)
class SkillPackIdentity:
    skill_pack_id: str
    publisher_id: str
    package_name: str
    version: str
    manifest_version: int
    install_source: str
    install_path: str
    content_fingerprint: str
    signature_status: str
    bundled_or_external: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_pack_id": self.skill_pack_id,
            "publisher_id": self.publisher_id,
            "package_name": self.package_name,
            "version": self.version,
            "manifest_version": int(self.manifest_version),
            "install_source": self.install_source,
            "install_path": _safe_path_label(self.install_path),
            "content_fingerprint": self.content_fingerprint,
            "signature_status": self.signature_status,
            "bundled_or_external": self.bundled_or_external,
            "enabled": bool(self.enabled),
        }


@dataclass(frozen=True)
class PermissionDefinition:
    permission_id: str
    capability_id: str | None
    action_type: str | None
    executor_id: str | None
    effect: str
    risk_level: str
    allowed_target_classes: tuple[str, ...] = ()
    universal_plan_required: bool = False
    confirmation_required: bool = False
    background_allowed: bool = False
    bundled_allowed: bool = True
    external_allowed: bool = True
    version_bound: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "permission_id": self.permission_id,
            "capability_id": self.capability_id,
            "action_type": self.action_type,
            "executor_id": self.executor_id,
            "effect": self.effect,
            "risk_level": self.risk_level,
            "allowed_target_classes": list(self.allowed_target_classes),
            "universal_plan_required": bool(self.universal_plan_required),
            "confirmation_required": bool(self.confirmation_required),
            "background_allowed": bool(self.background_allowed),
            "bundled_allowed": bool(self.bundled_allowed),
            "external_allowed": bool(self.external_allowed),
            "version_bound": bool(self.version_bound),
        }


@dataclass(frozen=True)
class SkillPermissionGrant:
    grant_id: str
    skill_pack_id: str
    publisher_id: str
    version: str
    content_fingerprint: str
    permission_id: str
    target_scope: dict[str, Any]
    granted_at: str
    granted_by: str
    expires_at: str | None = None
    revoked_at: str | None = None
    grant_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "skill_pack_id": self.skill_pack_id,
            "publisher_id": self.publisher_id,
            "version": self.version,
            "content_fingerprint": self.content_fingerprint,
            "permission_id": self.permission_id,
            "target_scope": dict(self.target_scope),
            "granted_at": self.granted_at,
            "granted_by": self.granted_by,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
            "grant_reason": self.grant_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillPermissionGrant":
        return cls(
            grant_id=str(payload.get("grant_id") or ""),
            skill_pack_id=str(payload.get("skill_pack_id") or ""),
            publisher_id=str(payload.get("publisher_id") or ""),
            version=str(payload.get("version") or ""),
            content_fingerprint=str(payload.get("content_fingerprint") or ""),
            permission_id=str(payload.get("permission_id") or ""),
            target_scope=dict(payload.get("target_scope") if isinstance(payload.get("target_scope"), dict) else {}),
            granted_at=str(payload.get("granted_at") or ""),
            granted_by=str(payload.get("granted_by") or ""),
            expires_at=str(payload.get("expires_at") or "").strip() or None,
            revoked_at=str(payload.get("revoked_at") or "").strip() or None,
            grant_reason=str(payload.get("grant_reason") or ""),
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path_label(path: Any) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    home = str(Path.home())
    if raw == home:
        return "~"
    if raw.startswith(home + "/"):
        return "~/" + raw[len(home) + 1 :]
    return raw


def _validate_id(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if _ID_RE.fullmatch(normalized) is None:
        raise SkillPermissionError(f"invalid_{field_name}")
    return normalized


def build_permission_registry() -> dict[str, PermissionDefinition]:
    entries = [
        PermissionDefinition("read.notifications.inspect", None, None, None, "read_only", "low", ("notification_history",)),
        PermissionDefinition("read.files.inspect", None, None, None, "read_only", "low", ("file", "directory")),
        PermissionDefinition("read.git.inspect", None, None, None, "read_only", "low", ("git_repository",)),
        PermissionDefinition("invoke.notification.local.send", "notification.local.send", "operator.notification.local.send", "operator.notification.local.send.v1", "mutating", "medium", ("local_notification",), True, True),
        PermissionDefinition("invoke.notification.external.send", "notification.external.send", "operator.notification.telegram.send", "operator.notification.telegram.send.v1", "mutating", "high", ("telegram_notification",), True, True, background_allowed=False),
        PermissionDefinition("invoke.files.create", "files.create", "operator.file.create", "operator.file.create.v1", "mutating", "medium", ("file",), True, True),
        PermissionDefinition("invoke.backup.create", "backup.create", "operator.backup", "operator.backup.v1", "mutating", "medium", ("backup",), True, True),
        PermissionDefinition("invoke.git.commit", "git.commit", "operator.git.commit", "operator.git.commit.v1", "mutating", "medium", ("git_repository",), True, True),
    ]
    return {entry.permission_id: entry for entry in entries}


def validate_skill_manifest(payload: dict[str, Any], *, install_path: str = "", expected_skill_pack_id: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SkillPermissionError("manifest_root_must_be_object")
    schema_version = int(payload.get("schema_version") or 0)
    if schema_version != SUPPORTED_MANIFEST_SCHEMA_VERSION:
        raise SkillPermissionError("unsupported_manifest_schema_version")
    skill_pack_id = _validate_id(str(payload.get("skill_pack_id") or ""), "skill_pack_id")
    if expected_skill_pack_id and skill_pack_id != expected_skill_pack_id:
        raise SkillPermissionError("skill_pack_id_mismatch")
    publisher_id = _validate_id(str(payload.get("publisher_id") or ""), "publisher_id")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise SkillPermissionError("name_required")
    version = str(payload.get("version") or "").strip()
    if not version:
        raise SkillPermissionError("version_required")
    permissions = payload.get("declared_permissions")
    if not isinstance(permissions, list):
        raise SkillPermissionError("declared_permissions_required")
    normalized_permissions: list[str] = []
    registry = build_permission_registry()
    for raw in permissions:
        permission_id = str(raw or "").strip()
        if "*" in permission_id:
            raise SkillPermissionError("wildcard_permission_denied")
        if _PERMISSION_RE.fullmatch(permission_id) is None:
            raise SkillPermissionError("invalid_permission_id")
        if permission_id not in registry:
            raise SkillPermissionError("unknown_permission")
        normalized_permissions.append(permission_id)
    if len(normalized_permissions) != len(set(normalized_permissions)):
        raise SkillPermissionError("duplicate_permission")
    network_domains = _normalize_str_list(payload.get("network_domains"), "network_domains")
    for domain in network_domains:
        if "*" in domain:
            raise SkillPermissionError("wildcard_domain_denied")
    filesystem_roots = _normalize_str_list(payload.get("filesystem_roots"), "filesystem_roots")
    canonical_roots: list[str] = []
    home = Path.home().resolve()
    for root in filesystem_roots:
        candidate = Path(root).expanduser().resolve(strict=False)
        if candidate == Path("/") or candidate == home:
            raise SkillPermissionError("broad_filesystem_root_denied")
        canonical_roots.append(str(candidate))
    background_tasks = payload.get("background_tasks")
    if background_tasks is None:
        background_tasks = []
    if not isinstance(background_tasks, list):
        raise SkillPermissionError("background_tasks_must_be_list")
    for row in background_tasks:
        if not isinstance(row, dict) or not row.get("task_id"):
            raise SkillPermissionError("background_task_declaration_invalid")
        if row.get("permission_id") not in normalized_permissions:
            raise SkillPermissionError("background_task_permission_not_declared")
    install_resolved = str(Path(install_path).expanduser().resolve(strict=False)) if install_path else ""
    if install_path and Path(install_path).is_symlink():
        raise SkillPermissionError("symlink_install_root_denied")
    normalized = {
        "schema_version": schema_version,
        "skill_pack_id": skill_pack_id,
        "publisher_id": publisher_id,
        "name": name,
        "version": version,
        "entrypoints": _normalize_str_list(payload.get("entrypoints"), "entrypoints"),
        "declared_permissions": sorted(normalized_permissions),
        "read_only_surfaces": _normalize_str_list(payload.get("read_only_surfaces"), "read_only_surfaces"),
        "network_domains": network_domains,
        "filesystem_roots": sorted(set(canonical_roots)),
        "provider_accounts": _normalize_str_list(payload.get("provider_accounts"), "provider_accounts"),
        "background_tasks": background_tasks,
        "configuration_schema": payload.get("configuration_schema") if isinstance(payload.get("configuration_schema"), dict) else {},
        "install_path": install_resolved,
    }
    normalized["content_fingerprint"] = stable_fingerprint(normalized)
    return normalized


def build_skill_identity(manifest: dict[str, Any], *, install_source: str, install_path: str, bundled_or_external: str = "external", signature_status: str = "unsigned", enabled: bool = True) -> SkillPackIdentity:
    normalized = validate_skill_manifest(manifest, install_path=install_path)
    return SkillPackIdentity(
        skill_pack_id=str(normalized["skill_pack_id"]),
        publisher_id=str(normalized["publisher_id"]),
        package_name=str(normalized["name"]),
        version=str(normalized["version"]),
        manifest_version=int(normalized["schema_version"]),
        install_source=str(install_source or "unknown"),
        install_path=str(normalized.get("install_path") or install_path),
        content_fingerprint=str(normalized["content_fingerprint"]),
        signature_status=str(signature_status or "unsigned"),
        bundled_or_external=str(bundled_or_external or "external"),
        enabled=bool(enabled),
    )


def _normalize_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SkillPermissionError(f"{field_name}_must_be_list")
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return sorted(set(out))


class SkillGrantStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SKILL_PERMISSION_SCHEMA_VERSION, "grants": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {"schema_version": SKILL_PERMISSION_SCHEMA_VERSION, "grants": []}
        if not isinstance(payload, dict) or int(payload.get("schema_version") or 0) != SKILL_PERMISSION_SCHEMA_VERSION:
            return {"schema_version": SKILL_PERMISSION_SCHEMA_VERSION, "grants": []}
        grants = payload.get("grants") if isinstance(payload.get("grants"), list) else []
        return {"schema_version": SKILL_PERMISSION_SCHEMA_VERSION, "grants": [row for row in grants if isinstance(row, dict)]}

    def _save(self, payload: dict[str, Any]) -> None:
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def list_grants(self) -> list[dict[str, Any]]:
        return [SkillPermissionGrant.from_dict(row).to_dict() for row in self._load().get("grants", [])]

    def create_grant(
        self,
        *,
        identity: SkillPackIdentity,
        permission_id: str,
        target_scope: dict[str, Any],
        granted_by: str = "local_operator",
        expires_at: str | None = None,
        grant_reason: str = "",
    ) -> dict[str, Any]:
        if permission_id not in build_permission_registry():
            raise SkillPermissionError("unknown_permission")
        if not str(granted_by or "").strip() or str(granted_by or "").strip().lower() in {"user", "assistant", "skill"}:
            raise SkillPermissionError("invalid_granted_by")
        grant = SkillPermissionGrant(
            grant_id=f"grant-{uuid.uuid4().hex[:12]}",
            skill_pack_id=identity.skill_pack_id,
            publisher_id=identity.publisher_id,
            version=identity.version,
            content_fingerprint=identity.content_fingerprint,
            permission_id=permission_id,
            target_scope=dict(target_scope),
            granted_at=utc_now_iso(),
            granted_by=str(granted_by).strip(),
            expires_at=expires_at,
            grant_reason=str(grant_reason or ""),
        )
        payload = self._load()
        grants = [row for row in payload.get("grants", []) if isinstance(row, dict)]
        grants.append(grant.to_dict())
        payload["grants"] = grants
        self._save(payload)
        return grant.to_dict()

    def revoke_grant(self, grant_id: str, *, revoked_by: str = "local_operator") -> dict[str, Any] | None:
        payload = self._load()
        changed: dict[str, Any] | None = None
        grants: list[dict[str, Any]] = []
        for row in payload.get("grants", []):
            if not isinstance(row, dict):
                continue
            next_row = dict(row)
            if str(next_row.get("grant_id") or "") == str(grant_id or "").strip() and not str(next_row.get("revoked_at") or "").strip():
                next_row["revoked_at"] = utc_now_iso()
                next_row["revoked_by"] = str(revoked_by or "local_operator")
                changed = next_row
            grants.append(next_row)
        payload["grants"] = grants
        self._save(payload)
        return SkillPermissionGrant.from_dict(changed).to_dict() if changed else None

    def effective_grant(self, *, identity: SkillPackIdentity, permission_id: str, target: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        now = datetime.now(timezone.utc)
        for row in self.list_grants():
            grant = SkillPermissionGrant.from_dict(row)
            if grant.skill_pack_id != identity.skill_pack_id or grant.publisher_id != identity.publisher_id:
                continue
            if grant.version != identity.version or grant.content_fingerprint != identity.content_fingerprint:
                continue
            if grant.permission_id != permission_id:
                continue
            if grant.revoked_at:
                continue
            if grant.expires_at:
                try:
                    if datetime.fromisoformat(grant.expires_at) <= now:
                        continue
                except ValueError:
                    continue
            if not _target_scope_allows(grant.target_scope, target):
                return None, "target_scope_denied"
            return grant.to_dict(), "allowed"
        return None, "grant_missing"


def _target_scope_allows(scope: dict[str, Any], target: dict[str, Any]) -> bool:
    if not isinstance(scope, dict):
        return False
    root = str(scope.get("root") or "").strip()
    target_path = str(target.get("target_path") or target.get("path") or "").strip()
    if root and target_path:
        root_path = Path(root).expanduser().resolve(strict=False)
        target_resolved = Path(target_path).expanduser().resolve(strict=False)
        try:
            target_resolved.relative_to(root_path)
        except ValueError:
            return False
    exact_chat_hash = str(scope.get("chat_id_sha256") or "").strip()
    target_chat_hash = str(target.get("chat_id_sha256") or "").strip()
    if exact_chat_hash and exact_chat_hash != target_chat_hash:
        return False
    max_bytes = int(scope.get("max_bytes") or 0)
    size = int(target.get("size_bytes") or 0)
    if max_bytes > 0 and size > max_bytes:
        return False
    return True


def diff_skill_permissions(old_manifest: dict[str, Any], new_manifest: dict[str, Any]) -> dict[str, list[str]]:
    old = set(validate_skill_manifest(old_manifest).get("declared_permissions", []))
    new = set(validate_skill_manifest(new_manifest).get("declared_permissions", []))
    return {
        "unchanged": sorted(old & new),
        "newly_requested": sorted(new - old),
        "removed": sorted(old - new),
        "expanded_scope": [],
        "reduced_scope": [],
        "grant_expired": [],
        "grant_invalidated": [],
    }


class SkillPackInvocationBroker:
    def __init__(
        self,
        *,
        grant_store: SkillGrantStore,
        executor_registry: ExecutorRegistry | None = None,
        plan_store: MutationPlanStore | None = None,
        plan_ttl_seconds: int = 600,
    ) -> None:
        self.grant_store = grant_store
        self.executor_registry = executor_registry
        self.plan_store = plan_store or MutationPlanStore(
            grant_store.path.with_name("skill_pack_invocation_plans_v1.json")
        )
        self.plan_ttl_seconds = max(30, min(int(plan_ttl_seconds), 3600))
        self._confirmation_lock = threading.RLock()

    def inspect(self, *, identity: SkillPackIdentity, manifest: dict[str, Any], permission_id: str, target: dict[str, Any]) -> dict[str, Any]:
        allowed, reason, definition, _grant = self._authorize(identity=identity, manifest=manifest, permission_id=permission_id, target=target)
        if not allowed:
            return {"ok": False, "mutated": False, "reason_code": reason, "skill_pack": identity.to_dict(), "permission_id": permission_id}
        if definition.effect != "read_only":
            return {"ok": False, "mutated": False, "reason_code": "permission_not_read_only", "skill_pack": identity.to_dict(), "permission_id": permission_id}
        return {"ok": True, "mutated": False, "status": "inspection", "skill_pack": identity.to_dict(), "permission_id": permission_id, "target": dict(target)}

    def request_action(
        self,
        *,
        identity: SkillPackIdentity,
        manifest: dict[str, Any],
        permission_id: str,
        target: dict[str, Any],
        action_payload: dict[str, Any],
        actor_id: str = "local_user",
        thread_id: str = "skill-pack",
        session_id: str = "skill-pack",
    ) -> dict[str, Any]:
        """Preview a mutation and persist its exact invocation scope.

        This method deliberately never executes.  ``confirm_action`` is the
        only broker path that can dispatch a managed skill-pack mutation.
        """
        allowed, reason, definition, grant = self._authorize(identity=identity, manifest=manifest, permission_id=permission_id, target=target)
        if not allowed:
            return {"ok": False, "mutated": False, "reason_code": reason, "skill_pack": identity.to_dict(), "permission_id": permission_id}
        if definition.effect != "mutating":
            return {"ok": False, "mutated": False, "reason_code": "permission_not_mutating", "skill_pack": identity.to_dict(), "permission_id": permission_id}
        assert definition.capability_id and definition.executor_id and definition.action_type
        plan = build_mutation_plan(
            plan_id=f"skill-{uuid.uuid4().hex[:12]}",
            capability_id=definition.capability_id,
            executor_id=definition.executor_id,
            expires_at_epoch=int(time.time()) + self.plan_ttl_seconds,
            thread_id=thread_id,
            session_id=session_id,
            actor_id=actor_id,
            target_snapshot={
                "skill_pack": identity.to_dict(),
                "permission_id": permission_id,
                "grant_id": str((grant or {}).get("grant_id") or ""),
                "target": dict(target),
                "arguments_fingerprint": stable_fingerprint(action_payload),
            },
            mutation_inventory=[{
                "permission_id": permission_id,
                "target": dict(target),
                "arguments_fingerprint": stable_fingerprint(action_payload),
                "requested_by_skill_pack": identity.skill_pack_id,
            }],
            preserved_resources=[],
            recovery={"rollback_supported": definition.permission_id in {"invoke.files.create", "invoke.backup.create"}},
        )
        validate_mutation_plan(plan)
        self.plan_store.save(plan)
        return {
            "ok": True,
            "mutated": False,
            "status": "confirmation_required",
            "plan": plan,
            "plan_id": plan["plan_id"],
            "expires_at": plan["expires_at"],
            "skill_pack": identity.to_dict(),
            "permission_id": permission_id,
            "grant_id": str((grant or {}).get("grant_id") or ""),
        }

    def confirm_action(
        self,
        *,
        plan_id: str,
        confirmation_id: str,
        identity: SkillPackIdentity,
        manifest: dict[str, Any],
        permission_id: str,
        target: dict[str, Any],
        action_payload: dict[str, Any],
        actor_id: str = "local_user",
        thread_id: str = "skill-pack",
        session_id: str = "skill-pack",
    ) -> dict[str, Any]:
        with self._confirmation_lock:
            plan = self.plan_store.load(plan_id)
            if plan is None:
                return self._blocked(identity, permission_id, "skill_invocation_plan_missing")
            if str(plan.get("status") or "") != MUTATION_PLAN_STATUS_PENDING:
                return self._blocked(identity, permission_id, "skill_invocation_plan_not_pending")
        allowed, reason, definition, grant = self._authorize(
            identity=identity,
            manifest=manifest,
            permission_id=permission_id,
            target=target,
        )
        if not allowed:
            return self._blocked(identity, permission_id, reason)
        if definition.effect != "mutating":
            return self._blocked(identity, permission_id, "permission_not_mutating")
        snapshot = plan.get("target_snapshot") if isinstance(plan.get("target_snapshot"), dict) else {}
        expected_snapshot = {
            "skill_pack": identity.to_dict(),
            "permission_id": permission_id,
            "grant_id": str((grant or {}).get("grant_id") or ""),
            "target": dict(target),
            "arguments_fingerprint": stable_fingerprint(action_payload),
        }
        if stable_fingerprint(snapshot) != stable_fingerprint(expected_snapshot):
            return self._blocked(identity, permission_id, "skill_invocation_scope_changed")
        for field, actual in (("actor_id", actor_id), ("thread_id", thread_id), ("session_id", session_id)):
            if str(plan.get(field) or "") != str(actual or ""):
                return self._blocked(identity, permission_id, f"skill_invocation_{field}_mismatch")
        if self.executor_registry is None:
            return self._blocked(identity, permission_id, "executor_registry_unavailable")
        assert definition.capability_id and definition.executor_id and definition.action_type
        wrapped_plan = {
            **plan,
            "mutation_plan": dict(plan),
            "action_type": definition.action_type,
            "target": str(target.get("target") or target.get("target_path") or permission_id),
            "executor_status": "enabled",
            "high_risk_confirmed": True,
        }
        action = {
            **dict(action_payload),
            "pending_id": wrapped_plan["plan_id"],
            "skill_pack_context": {
                "caller_type": "skill_pack",
                "skill_pack_id": identity.skill_pack_id,
                "skill_pack_version": identity.version,
                "skill_pack_fingerprint": identity.content_fingerprint,
                "permission_id": permission_id,
                "grant_id": str((grant or {}).get("grant_id") or ""),
            },
        }
        confirmation = build_mutation_confirmation(
            plan,
            confirmation_id=confirmation_id,
            actor_id=actor_id,
            thread_id=thread_id,
            session_id=session_id,
        )
        with self._confirmation_lock:
            current = self.plan_store.load(plan_id)
            if current is None or str(current.get("status") or "") != MUTATION_PLAN_STATUS_PENDING:
                return self._blocked(identity, permission_id, "skill_invocation_plan_not_pending")
            still_allowed, current_reason, _current_definition, current_grant = self._authorize(
                identity=identity,
                manifest=manifest,
                permission_id=permission_id,
                target=target,
            )
            if not still_allowed:
                return self._blocked(identity, permission_id, current_reason)
            if str((current_grant or {}).get("grant_id") or "") != str((grant or {}).get("grant_id") or ""):
                return self._blocked(identity, permission_id, "skill_invocation_grant_changed")
            self.plan_store.transition(plan_id, "executing")
            result = self.executor_registry.execute_confirmed_plan(
                plan=wrapped_plan,
                action=action,
                confirmation=confirmation,
            ).to_dict()
            self.plan_store.transition(plan_id, "completed" if result.get("ok") else "failed")
        details = dict(result.get("details") if isinstance(result.get("details"), dict) else {})
        details["skill_pack"] = identity.to_dict()
        details["permission_id"] = permission_id
        details["grant_id"] = str((grant or {}).get("grant_id") or "")
        result["details"] = details
        result["skill_pack_id"] = identity.skill_pack_id
        result["permission_id"] = permission_id
        result["grant_id"] = str((grant or {}).get("grant_id") or "")
        return result

    def cancel_action(
        self,
        *,
        plan_id: str,
        actor_id: str,
        thread_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        with self._confirmation_lock:
            plan = self.plan_store.load(plan_id)
            if plan is None:
                return {"ok": False, "mutated": False, "reason_code": "skill_invocation_plan_missing"}
            for field, actual in (("actor_id", actor_id), ("thread_id", thread_id), ("session_id", session_id)):
                if str(plan.get(field) or "") != str(actual or ""):
                    return {"ok": False, "mutated": False, "reason_code": f"skill_invocation_{field}_mismatch"}
            if str(plan.get("status") or "") != MUTATION_PLAN_STATUS_PENDING:
                return {"ok": False, "mutated": False, "reason_code": "skill_invocation_plan_not_pending"}
            self.plan_store.cancel(plan_id)
        return {"ok": True, "mutated": False, "status": "cancelled", "plan_id": plan_id}

    @staticmethod
    def _blocked(identity: SkillPackIdentity, permission_id: str, reason: str) -> dict[str, Any]:
        return {
            "ok": False,
            "mutated": False,
            "reason_code": reason,
            "skill_pack": identity.to_dict(),
            "permission_id": permission_id,
        }

    def _authorize(
        self,
        *,
        identity: SkillPackIdentity,
        manifest: dict[str, Any],
        permission_id: str,
        target: dict[str, Any],
    ) -> tuple[bool, str, PermissionDefinition, dict[str, Any] | None]:
        registry = build_permission_registry()
        definition = registry.get(str(permission_id or "").strip())
        if definition is None:
            return False, "unknown_permission", PermissionDefinition("", None, None, None, "read_only", "low"), None
        if not identity.enabled:
            return False, "skill_disabled", definition, None
        normalized_manifest = validate_skill_manifest(manifest, expected_skill_pack_id=identity.skill_pack_id, install_path=identity.install_path)
        if str(normalized_manifest.get("content_fingerprint") or "") != identity.content_fingerprint:
            return False, "manifest_fingerprint_changed", definition, None
        declared = set(normalized_manifest.get("declared_permissions", []))
        if permission_id not in declared:
            return False, "permission_not_declared", definition, None
        if identity.bundled_or_external == "external" and not definition.external_allowed:
            return False, "external_skill_permission_denied", definition, None
        grant, reason = self.grant_store.effective_grant(identity=identity, permission_id=permission_id, target=target)
        if grant is None:
            return False, reason, definition, None
        return True, "allowed", definition, grant
