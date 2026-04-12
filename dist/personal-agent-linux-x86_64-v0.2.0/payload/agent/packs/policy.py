from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PackPolicyDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class PackPermissionDenied(RuntimeError):
    pack_id: str
    iface: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial wrapper
        return f"pack={self.pack_id} iface={self.iface} reason={self.reason}"


def is_iface_allowed(
    *,
    pack_id: str,
    iface: str,
    fallback_iface: str | None,
    pack_record: dict[str, Any] | None,
    trust: str,
    expected_permissions_hash: str,
) -> PackPolicyDecision:
    if not pack_record:
        if trust == "native":
            return PackPolicyDecision(True, "native_implicit_allow")
        return PackPolicyDecision(False, "pack_not_installed")

    if not bool(pack_record.get("enabled")):
        return PackPolicyDecision(False, "pack_disabled")

    approved_hash = str(pack_record.get("approved_permissions_hash") or "").strip()
    if not approved_hash:
        return PackPolicyDecision(False, "pack_not_approved")
    if approved_hash != str(expected_permissions_hash or "").strip():
        return PackPolicyDecision(False, "approval_hash_mismatch")

    permissions = pack_record.get("permissions")
    if not isinstance(permissions, dict):
        return PackPolicyDecision(False, "invalid_permissions")

    ifaces = permissions.get("ifaces")
    if not isinstance(ifaces, list):
        return PackPolicyDecision(False, "invalid_permissions")

    candidates: list[str] = [str(iface or "").strip()]
    fallback = str(fallback_iface or "").strip()
    if fallback and fallback not in candidates:
        candidates.append(fallback)

    allowed = any(candidate and candidate in ifaces for candidate in candidates)
    if not allowed:
        return PackPolicyDecision(False, "iface_not_allowed")
    return PackPolicyDecision(True, "allowed")


def enforce_iface_allowed(
    *,
    pack_id: str,
    iface: str,
    fallback_iface: str | None,
    pack_record: dict[str, Any] | None,
    trust: str,
    expected_permissions_hash: str,
) -> None:
    decision = is_iface_allowed(
        pack_id=pack_id,
        iface=iface,
        fallback_iface=fallback_iface,
        pack_record=pack_record,
        trust=trust,
        expected_permissions_hash=expected_permissions_hash,
    )
    if not decision.allowed:
        raise PackPermissionDenied(pack_id=pack_id, iface=iface, reason=decision.reason)
